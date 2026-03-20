module RLT_SDP_Comp_Combo

using MosekTools
using JuMP
using LinearAlgebra
using Parameters: @unpack
using MathOptInterface
const MOI = MathOptInterface

# ------------------------------------------------------------------
# Single source of truth:
# - instance preparation for complementarity (general bounds xL<=x<=xU)
# - RLT blocks (FC/FE/FB/FU) already updated
# ------------------------------------------------------------------
using ..RLTComp: prepare_instance_comp, add_FC_comp!, add_FE_v!, add_FB_v!, add_FU_v!

# --------------------------------------------------------------
# Relaxation modes (same structure as BigM combo)
# --------------------------------------------------------------
const RelaxationModes = (
    :RLT,

    :RLT_SOC2x2_diag_X,
    :RLT_SOC2x2_diag_V,
    :RLT_SOC2x2_diag_XV,

    :RLT_SOC2x2_full_X,
    :RLT_SOC2x2_full_V,
    :RLT_SOC2x2_full_XV,

    :RLT_SOC_directional_X,
    :RLT_SOC_directional_V,
    :RLT_SOC_directional_XV,

    :RLT_SOC_hybrid_X,
    :RLT_SOC_hybrid_V,
    :RLT_SOC_hybrid_XV,

    :RLT_PSD3x3_X,
    :RLT_PSD3x3_V,
    :RLT_PSD3x3_XV,

    :RLT_blockSDP_X,
    :RLT_blockSDP_V,
    :RLT_blockSDP_XV,

    :RLT_full_SDP,   # full moment matrix Y ⪰ 0
)

all_pairs(n::Int) = [(i,j) for i in 1:n for j in i+1:n]

# ==============================================================
# Generic strengthening blocks (re-used for X and/or V)
# ==============================================================

# 1) diag 2x2 SOC: Zii ≥ zi^2   via RSOC [Zii, 0.5, zi]
function add_SOC2x2_diag!(m::Model, z, Z)
    n = length(z)
    for i in 1:n
        @constraint(m, [Z[i,i], 0.5, z[i]] in MOI.RotatedSecondOrderCone(3))
    end
    return nothing
end

# 2) full 2x2 minors: Zii*Zjj ≥ Zij^2 via RSOC [Zii, Zjj, sqrt(2)*Zij]
function add_SOC2x2_full!(m::Model, z, Z)
    n = length(z)
    add_SOC2x2_diag!(m, z, Z)
    for i in 1:n, j in i+1:n
        @constraint(m, [Z[i,i], Z[j,j], sqrt(2.0)*Z[i,j]] in MOI.RotatedSecondOrderCone(3))
    end
    return nothing
end

# 3) directional core: d'Z d ≥ (z'd)^2 for d = e_i ± e_j
function add_SOC_directional_core!(m::Model, z, Z)
    n = length(z)
    for i in 1:n, j in i+1:n
        @constraint(m, [Z[i,i] + 2*Z[i,j] + Z[j,j], 0.5, z[i] + z[j]] in MOI.RotatedSecondOrderCone(3))
        @constraint(m, [Z[i,i] - 2*Z[i,j] + Z[j,j], 0.5, z[i] - z[j]] in MOI.RotatedSecondOrderCone(3))
    end
    return nothing
end

function add_SOC_directional!(m::Model, z, Z)
    add_SOC2x2_diag!(m, z, Z)
    add_SOC_directional_core!(m, z, Z)
    return nothing
end

function add_SOC_hybrid!(m::Model, z, Z)
    add_SOC2x2_full!(m, z, Z)
    add_SOC_directional_core!(m, z, Z)
    return nothing
end

# 4) PSD 3x3 principals of [1 z'; z Z]
function add_PSD3x3!(m::Model, z, Z)
    n = length(z)
    for (i,j) in all_pairs(n)
        @constraint(m, Symmetric([
            1.0   z[i]     z[j];
            z[i]  Z[i,i]   Z[i,j];
            z[j]  Z[i,j]   Z[j,j]
        ]) in PSDCone())
    end
    return nothing
end

# 5) block SDP: [1 z'; z Z] ⪰ 0
function add_blockSDP!(m::Model, z, Z)
    @constraint(m, Symmetric([1.0 z'; z Z]) in PSDCone())
    return nothing
end

# Full moment SDP: Y = [1 x' v'; x X W; v W' V] ⪰ 0
function add_fullSDP_comp!(m::Model, x, v, X, W, V; n::Int)
    @variable(m, Y[1:(2n+1), 1:(2n+1)], Symmetric)
    @constraint(m, Y .== [1.0 x' v'; x X W; v W' V])
    @constraint(m, Y in PSDCone())
    return Y
end

# ==============================================================
# Build model: base RLTComp blocks + optional strengthening
# ==============================================================

"""
    build_RLT_SDP_comp_model(data; variant="RLT_CompBinBou", relaxation=:RLT, optimizer)

Variants (preferred naming):
- "RLT_CompBin"      : FC_comp + FE_v + FB_v
- "RLT_CompBinBou"   : FC_comp + FE_v + FB_v + FU_v
- "RLT_CompBou"      : FC_comp + FE_v + FU_v

Backward-compatible aliases:
- "RLT_S2"  -> "RLT_CompBin"
- "RLT_S2U" -> "RLT_CompBinBou"
- "RLT_S3"  -> "RLT_CompBou"
"""
function build_RLT_SDP_comp_model(
    data::Dict{String,Any};
    variant::String="RLT_CompBinBou",
    optimizer = MosekTools.Optimizer,
    build_only::Bool=false,
    relaxation::Symbol = :RLT,
    verbose::Bool=false
)
    @assert relaxation in RelaxationModes "Unknown relaxation mode: $relaxation"

    # aliases
    if variant == "RLT_S2";  variant = "RLT_CompBin"; end
    if variant == "RLT_S2U"; variant = "RLT_CompBinBou"; end
    if variant == "RLT_S3";  variant = "RLT_CompBou"; end

    params = prepare_instance_comp(data)  # includes eeT, xL, xU, etc.
    @unpack n, ρ, Q0, q0 = params

    m = Model(optimizer)
    verbose ? nothing : set_silent(m)

    # Decision variables (names match PrettyPrint auto-detection)
    @variable(m, x[1:n])
    @variable(m, v[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, V[1:n,1:n], Symmetric)
    @variable(m, W[1:n,1:n])

    # Objective: 0.5⟨Q0,X⟩ + q0ᵀx
    @objective(m, Min, 0.5 * sum(Q0[i,j]*X[i,j] for i=1:n, j=1:n) + q0' * x)

    # ---- Base RLT blocks from complementarity side ----
    add_FC_comp!(m, x, v, X, V, W, params)
    add_FE_v!(m, x, v, X, V, W, params)

    if variant == "RLT_CompBin"
        add_FB_v!(m, v, V, params)
    elseif variant == "RLT_CompBinBou"
        add_FB_v!(m, v, V, params)
        add_FU_v!(m, x, v, X, V, W, params)
    elseif variant == "RLT_CompBou"
        add_FU_v!(m, x, v, X, V, W, params)
    else
        error("Unknown variant: $variant. Use RLT_CompBin, RLT_CompBinBou, or RLT_CompBou.")
    end

    # ---- Strengthening on top of RLT ----
    if relaxation == :RLT
        nothing

    elseif relaxation == :RLT_SOC2x2_diag_X
        add_SOC2x2_diag!(m, x, X)
    elseif relaxation == :RLT_SOC2x2_diag_V
        add_SOC2x2_diag!(m, v, V)
    elseif relaxation == :RLT_SOC2x2_diag_XV
        add_SOC2x2_diag!(m, x, X); add_SOC2x2_diag!(m, v, V)

    elseif relaxation == :RLT_SOC2x2_full_X
        add_SOC2x2_full!(m, x, X)
    elseif relaxation == :RLT_SOC2x2_full_V
        add_SOC2x2_full!(m, v, V)
    elseif relaxation == :RLT_SOC2x2_full_XV
        add_SOC2x2_full!(m, x, X); add_SOC2x2_full!(m, v, V)

    elseif relaxation == :RLT_SOC_directional_X
        add_SOC_directional!(m, x, X)
    elseif relaxation == :RLT_SOC_directional_V
        add_SOC_directional!(m, v, V)
    elseif relaxation == :RLT_SOC_directional_XV
        add_SOC_directional!(m, x, X); add_SOC_directional!(m, v, V)

    elseif relaxation == :RLT_SOC_hybrid_X
        add_SOC_hybrid!(m, x, X)
    elseif relaxation == :RLT_SOC_hybrid_V
        add_SOC_hybrid!(m, v, V)
    elseif relaxation == :RLT_SOC_hybrid_XV
        add_SOC_hybrid!(m, x, X); add_SOC_hybrid!(m, v, V)

    elseif relaxation == :RLT_PSD3x3_X
        add_PSD3x3!(m, x, X)
    elseif relaxation == :RLT_PSD3x3_V
        add_PSD3x3!(m, v, V)
    elseif relaxation == :RLT_PSD3x3_XV
        add_PSD3x3!(m, x, X); add_PSD3x3!(m, v, V)

    elseif relaxation == :RLT_blockSDP_X
        add_blockSDP!(m, x, X)
    elseif relaxation == :RLT_blockSDP_V
        add_blockSDP!(m, v, V)
    elseif relaxation == :RLT_blockSDP_XV
        add_blockSDP!(m, x, X); add_blockSDP!(m, v, V)

    elseif relaxation == :RLT_full_SDP
        add_fullSDP_comp!(m, x, v, X, W, V; n=n)

    else
        error("Unhandled relaxation mode: $relaxation")
    end

    return m
end

# ==============================================================
# PrettyPrint wrapper (model-based)
# ==============================================================

function solve_and_print(
    data::Dict{String,Any};
    variant::String="RLT_CompBinBou",
    relaxation::Symbol=:RLT,
    optimizer = MosekTools.Optimizer,
    verbose::Bool=false,
    show_mats::Bool=true,
    pp_kwargs...
)
    if !isdefined(Main, :PrettyPrint)
        error("PrettyPrint is not loaded. Please include(\"instances/prettyprint.jl\") before calling RLT_SDP_Comp_Combo.solve_and_print.")
    end

    m = build_RLT_SDP_comp_model(data;
        variant=variant,
        relaxation=relaxation,
        optimizer=optimizer,
        build_only=true,
        verbose=verbose
    )

    optimize!(m)

    Main.PrettyPrint.print_model_solution(
        m;
        variant=variant,
        relaxation=relaxation,
        show_mats=show_mats,
        pp_kwargs...
    )

    return m
end

# --------------------------------------------------------------
# Solve wrapper (returns values + time)
# --------------------------------------------------------------
function solve_RLT_SDP_comp(
    data::Dict{String,Any};
    variant::String="RLT_CompBinBou",
    optimizer = MosekTools.Optimizer,
    relaxation::Symbol = :RLT,
    verbose::Bool=false
)
    m = build_RLT_SDP_comp_model(
        data; variant=variant, optimizer=optimizer, relaxation=relaxation, verbose=verbose
    )

    t0 = time_ns()
    optimize!(m)
    t = round((time_ns() - t0) / 1e9; digits=4)

    st = termination_status(m)
    if st in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
        x = value.(m[:x])
        v = value.(m[:v])
        X = value.(m[:X])
        V = value.(m[:V])
        W = value.(m[:W])
        return (st, objective_value(m), x, v, X, V, W, t)
    else
        return (st, nothing, nothing, nothing, nothing, nothing, nothing, t)
    end
end

export build_RLT_SDP_comp_model, solve_RLT_SDP_comp, solve_and_print, RelaxationModes

end # module RLT_SDP_Comp_Combo

# ==============================================================
# Interactive demo (optional)
# ==============================================================
if isinteractive()
    using MosekTools

    include(normpath(joinpath(@__DIR__, "..", "instances", "prettyprint.jl")))
    using .PrettyPrint

    include(normpath(joinpath(@__DIR__, "..", "instances", "alper_stqp_instance.jl")))
    using .AlperStqpInstances
    inst = alper_stqp_rho3_instance()

    # Example calls:
    RLT_SDP_Comp_Combo.solve_and_print(
        inst;
        variant="RLT_CompBou",
        relaxation=:RLT_full_SDP,
        optimizer=MosekTools.Optimizer,
        show_mats=true
    )
end