module RLT_SDP_Comp_Combo

using MosekTools
using JuMP
using LinearAlgebra
using Parameters: @unpack
using MathOptInterface
const MOI = MathOptInterface

# ==============================================================
# Instance preparation + NO-BOUNDS E/I complementarity RLT blocks
# ==============================================================
using ..RLTBigM: prepare_instance

using ..RLTCompNoBounds:
    additional_params,
    add_FC_comp_nobounds!,
    add_FE_v!,
    add_FI_v!,
    add_FB_v!,
    add_FU_v_nobounds!,
    add_FIU_v!


# Single source of truth for the full complementarity PSD lift.
# SDPComplementarity must be loaded before this module.
using ..SDPComplementarity: add_PSD_comp!


# --------------------------------------------------------------
# Relaxation modes
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

    :RLT_full_SDP,
)

all_pairs(n::Int) = [(i,j) for i in 1:n for j in i+1:n]


# ==============================================================
# Generic strengthening blocks
# ==============================================================

# 1) diagonal 2x2 SOC:
#       Z_ii >= z_i^2
function add_SOC2x2_diag!(m::Model, z, Z)
    n = length(z)
    for i in 1:n
        @constraint(
            m,
            [Z[i,i], 0.5, z[i]]
            in MOI.RotatedSecondOrderCone(3)
        )
    end
    return nothing
end


# 2) all 2x2 PSD minors of Z, together with the diagonal SOCs
function add_SOC2x2_full!(m::Model, z, Z)
    n = length(z)

    add_SOC2x2_diag!(m, z, Z)

    for i in 1:n, j in i+1:n
        @constraint(
            m,
            [Z[i,i], Z[j,j], sqrt(2.0) * Z[i,j]]
            in MOI.RotatedSecondOrderCone(3)
        )
    end
    return nothing
end


# 3) directional SOC core:
#       d' Z d >= (z' d)^2
# for d = e_i +/- e_j
function add_SOC_directional_core!(m::Model, z, Z)
    n = length(z)

    for i in 1:n, j in i+1:n
        @constraint(
            m,
            [
                Z[i,i] + 2*Z[i,j] + Z[j,j],
                0.5,
                z[i] + z[j],
            ]
            in MOI.RotatedSecondOrderCone(3)
        )

        @constraint(
            m,
            [
                Z[i,i] - 2*Z[i,j] + Z[j,j],
                0.5,
                z[i] - z[j],
            ]
            in MOI.RotatedSecondOrderCone(3)
        )
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


# 4) 3x3 principal PSD blocks of [1 z'; z Z]
function add_PSD3x3!(m::Model, z, Z)
    n = length(z)

    for (i,j) in all_pairs(n)
        @constraint(
            m,
            Symmetric([
                1.0   z[i]     z[j];
                z[i]  Z[i,i]   Z[i,j];
                z[j]  Z[i,j]   Z[j,j]
            ])
            in PSDCone()
        )
    end

    return nothing
end


# 5) block SDP [1 z'; z Z] >= 0
function add_blockSDP!(m::Model, z, Z)
    @constraint(
        m,
        Symmetric([
            1.0  z';
            z    Z
        ])
        in PSDCone()
    )
    return nothing
end



# ==============================================================
# Variant handling
# ==============================================================

"""
    canonical_variant(variant)

Canonical NO-BOUNDS E/I variants:

    RLT_CompBin_E
    RLT_CompBinBou_E
    RLT_CompBou_E

    RLT_CompBin_I
    RLT_CompBinBou_I
    RLT_CompBou_I

Old names are retained as aliases for the E-cardinality versions.

IMPORTANT:
At the RLT level, CompBinBou is NOT aliased to CompBin.
Although the explicit bounds 0 <= v <= 1 are redundant under the
full SDP lift together with diag(V)=v, their first-level RLT products
can strengthen the SDP--RLT relaxation. Therefore CompBin and
CompBinBou remain genuinely separate RLT variants.
"""
function canonical_variant(variant::String)
    aliases = Dict(
        "RLT_CompBin"    => "RLT_CompBin_E",
        "RLT_CompBinBou" => "RLT_CompBinBou_E",
        "RLT_CompBou"    => "RLT_CompBou_E",
        "RLT_S2"         => "RLT_CompBin_E",
        "RLT_S2U"        => "RLT_CompBinBou_E",
        "RLT_S3"         => "RLT_CompBou_E",
    )

    return get(aliases, variant, variant)
end


# ==============================================================
# Build model
#
# Common NO-BOUNDS complementarity RLT core:
#
#   C_Comp =
#       lifted original QCQP rows
#       Ax <= b, Hx = h
#       diag(W) = 0
#       A-self RLT
#       HX = h x'
#       HW = h v'
#
# No additional/derived x-bounds are constructed.
# No RLT products from such bounds are present.
#
# Cardinality closures:
#
#   E:
#       e'v = n-rho
#       We = (n-rho)x
#       Ve = (n-rho)v
#
#   I:
#       e'v >= n-rho
#       cardinality-slack square
#       cardinality-slack x (b-Ax)
#
# U^v adds 0 <= v <= 1 and its first-level RLT products.
# For I+U, IU^v also adds the joint products of the
# cardinality slack with v and e-v.
# ==============================================================
function build_RLT_SDP_comp_model(
    data::Dict{String,Any};
    variant::String="RLT_CompBinBou_E",
    optimizer=MosekTools.Optimizer,
    relaxation::Symbol=:RLT,
    verbose::Bool=false,
)
    @assert relaxation in RelaxationModes "Unknown relaxation mode: $relaxation"

    variant = canonical_variant(variant)

    allowed = (
        "RLT_CompBin_E",
        "RLT_CompBinBou_E",
        "RLT_CompBou_E",
        "RLT_CompBin_I",
        "RLT_CompBinBou_I",
        "RLT_CompBou_I",
    )

    if !(variant in allowed)
        error(
            "Unknown variant: $variant. Use one of " *
            join(allowed, ", ")
        )
    end

    # IMPORTANT:
    # RLTCompNoBounds.additional_params adds only eeT.
    # It does NOT construct xL/xU.
    params = additional_params(prepare_instance(data))
    @unpack n, Q0, q0 = params

    m = Model(optimizer)

    if !verbose
        set_silent(m)
    end

    set_name(
        m,
        "SQCQP_RLT_SDP_Complementarity_NoBounds_$(variant)_$(relaxation)"
    )

    # Decision variables
    @variable(m, x[1:n])
    @variable(m, v[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, V[1:n,1:n], Symmetric)
    @variable(m, W[1:n,1:n])

    # Objective
    @objective(
        m,
        Min,
        0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n)
        + q0' * x
    )

    # ----------------------------------------------------------
    # Common NO-BOUNDS complementarity RLT block
    # ----------------------------------------------------------
    add_FC_comp_nobounds!(
        m, x, v, X, V, W, params
    )

    # ----------------------------------------------------------
    # Cardinality block: E or I
    # ----------------------------------------------------------
    if endswith(variant, "_E")
        add_FE_v!(
            m, x, v, X, V, W, params
        )
    elseif endswith(variant, "_I")
        add_FI_v!(
            m, x, v, X, V, W, params
        )
    end

    # ----------------------------------------------------------
    # Binary-lift block B^v: diag(V)=v
    # ----------------------------------------------------------
    if variant in (
        "RLT_CompBin_E",
        "RLT_CompBinBou_E",
        "RLT_CompBin_I",
        "RLT_CompBinBou_I",
    )
        add_FB_v!(m, v, V, params)
    end

    # ----------------------------------------------------------
    # Explicit support-bound block U^v
    #
    # At RLT level this is NOT redundant, because its products
    # generate additional constraints in (v,V,W).
    # ----------------------------------------------------------
    if variant in (
        "RLT_CompBinBou_E",
        "RLT_CompBou_E",
        "RLT_CompBinBou_I",
        "RLT_CompBou_I",
    )
        add_FU_v_nobounds!(
            m, x, v, X, V, W, params
        )
    end

    # ----------------------------------------------------------
    # Joint I x U products
    # ----------------------------------------------------------
    if variant in (
        "RLT_CompBinBou_I",
        "RLT_CompBou_I",
    )
        add_FIU_v!(
            m, v, V, params
        )
    end

    # ----------------------------------------------------------
    # Optional conic strengthening on top of the SAME RLT block
    # ----------------------------------------------------------
    if relaxation == :RLT
        nothing

    elseif relaxation == :RLT_SOC2x2_diag_X
        add_SOC2x2_diag!(m, x, X)

    elseif relaxation == :RLT_SOC2x2_diag_V
        add_SOC2x2_diag!(m, v, V)

    elseif relaxation == :RLT_SOC2x2_diag_XV
        add_SOC2x2_diag!(m, x, X)
        add_SOC2x2_diag!(m, v, V)

    elseif relaxation == :RLT_SOC2x2_full_X
        add_SOC2x2_full!(m, x, X)

    elseif relaxation == :RLT_SOC2x2_full_V
        add_SOC2x2_full!(m, v, V)

    elseif relaxation == :RLT_SOC2x2_full_XV
        add_SOC2x2_full!(m, x, X)
        add_SOC2x2_full!(m, v, V)

    elseif relaxation == :RLT_SOC_directional_X
        add_SOC_directional!(m, x, X)

    elseif relaxation == :RLT_SOC_directional_V
        add_SOC_directional!(m, v, V)

    elseif relaxation == :RLT_SOC_directional_XV
        add_SOC_directional!(m, x, X)
        add_SOC_directional!(m, v, V)

    elseif relaxation == :RLT_SOC_hybrid_X
        add_SOC_hybrid!(m, x, X)

    elseif relaxation == :RLT_SOC_hybrid_V
        add_SOC_hybrid!(m, v, V)

    elseif relaxation == :RLT_SOC_hybrid_XV
        add_SOC_hybrid!(m, x, X)
        add_SOC_hybrid!(m, v, V)

    elseif relaxation == :RLT_PSD3x3_X
        add_PSD3x3!(m, x, X)

    elseif relaxation == :RLT_PSD3x3_V
        add_PSD3x3!(m, v, V)

    elseif relaxation == :RLT_PSD3x3_XV
        add_PSD3x3!(m, x, X)
        add_PSD3x3!(m, v, V)

    elseif relaxation == :RLT_blockSDP_X
        add_blockSDP!(m, x, X)

    elseif relaxation == :RLT_blockSDP_V
        add_blockSDP!(m, v, V)

    elseif relaxation == :RLT_blockSDP_XV
        add_blockSDP!(m, x, X)
        add_blockSDP!(m, v, V)

    elseif relaxation == :RLT_full_SDP
        add_PSD_comp!(m, x, v, X, V, W)

    else
        error("Unhandled relaxation mode: $relaxation")
    end

    return m
end


# ==============================================================
# PrettyPrint wrapper
# ==============================================================
function solve_and_print(
    data::Dict{String,Any};
    variant::String="RLT_CompBinBou_E",
    relaxation::Symbol=:RLT,
    optimizer=MosekTools.Optimizer,
    verbose::Bool=false,
    show_mats::Bool=true,
    pp_kwargs...,
)
    if !isdefined(Main, :PrettyPrint)
        error(
            "PrettyPrint is not loaded. Please include(" *
            "\"instances/prettyprint.jl\") before calling " *
            "RLT_SDP_Comp_Combo.solve_and_print."
        )
    end

    m = build_RLT_SDP_comp_model(
        data;
        variant=variant,
        relaxation=relaxation,
        optimizer=optimizer,
        verbose=verbose,
    )

    optimize!(m)

    Main.PrettyPrint.print_model_solution(
        m;
        variant=canonical_variant(variant),
        relaxation=relaxation,
        show_mats=show_mats,
        pp_kwargs...,
    )

    return m
end


# ==============================================================
# Solve wrapper (returns primal values + optimization time)
# ==============================================================
function solve_RLT_SDP_comp(
    data::Dict{String,Any};
    variant::String="RLT_CompBinBou_E",
    optimizer=MosekTools.Optimizer,
    relaxation::Symbol=:RLT,
    verbose::Bool=false,
)
    m = build_RLT_SDP_comp_model(
        data;
        variant=variant,
        optimizer=optimizer,
        relaxation=relaxation,
        verbose=verbose,
    )

    t = @elapsed optimize!(m)

    st = termination_status(m)

    if st in (
        MOI.OPTIMAL,
        MOI.LOCALLY_SOLVED,
        MOI.ALMOST_OPTIMAL,
    )
        x = value.(m[:x])
        v = value.(m[:v])
        X = value.(m[:X])
        V = value.(m[:V])
        W = value.(m[:W])

        return (
            st,
            objective_value(m),
            x,
            v,
            X,
            V,
            W,
            t,
        )
    else
        return (
            st,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            t,
        )
    end
end


export build_RLT_SDP_comp_model,
       solve_RLT_SDP_comp,
       solve_and_print,
       RelaxationModes,
       canonical_variant

end # module RLT_SDP_Comp_Combo


# ==============================================================
# Optional interactive demo
# ==============================================================
if isinteractive()
    using MosekTools

    include(
        normpath(
            joinpath(
                @__DIR__,
                "..",
                "instances",
                "prettyprint.jl",
            )
        )
    )
    using .PrettyPrint

    include(
        normpath(
            joinpath(
                @__DIR__,
                "..",
                "instances",
                "alper_stqp_instance.jl",
            )
        )
    )
    using .AlperStqpInstances

    inst = alper_stqp_rho3_instance()

    # Example: strongest E and I complementarity SDP--RLT models.
    for var in (
        "RLT_CompBinBou_E",
        "RLT_CompBinBou_I",
    )
        RLT_SDP_Comp_Combo.solve_and_print(
            inst;
            variant=var,
            relaxation=:RLT_full_SDP,
            optimizer=MosekTools.Optimizer,
            show_mats=true,
        )
    end
end
