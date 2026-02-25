module RLT_SDP_Combo

using MosekTools
using JuMP, LinearAlgebra, Parameters
const MOI = JuMP.MOI

using ..RLTBigM: prepare_instance, add_FC!, add_FE!, add_FI!, add_FU!, add_FIU!
using ..SDPBigM: add_SDP_lift!

# --------------------------------------------------------------
# Available relaxation modes
# --------------------------------------------------------------
const RelaxationModes = (
    :RLT,

    :RLT_SOC2x2_diag_X,
    :RLT_SOC2x2_diag_U,
    :RLT_SOC2x2_diag_XU,

    :RLT_SOC2x2_full_X,
    :RLT_SOC2x2_full_U,
    :RLT_SOC2x2_full_XU,

    :RLT_SOC_directional_X,
    :RLT_SOC_directional_U,
    :RLT_SOC_directional_XU,

    :RLT_SOC_hybrid_X,
    :RLT_SOC_hybrid_U,
    :RLT_SOC_hybrid_XU,

    :RLT_PSD3x3_X,
    :RLT_PSD3x3_U,
    :RLT_PSD3x3_XU,

    :RLT_blockSDP_X,
    :RLT_blockSDP_U,
    :RLT_blockSDP_XU,

    :RLT_full_SDP,
)

all_pairs(n::Int) = [(i,j) for i in 1:n for j in i+1:n]

# ------------------------------------------------------------------
# 1) diagonal 2×2 SOC: Xii ≥ xi², Ujj ≥ uj²
#    [Xii, 0.5, xi] ∈ RSOC, [Ujj, 0.5, uj] ∈ RSOC
# ------------------------------------------------------------------
function add_SOC2x2_diag_X!(model::Model, x, X)
    n = length(x)
    for i in 1:n
        @constraint(model, [X[i,i], 0.5, x[i]] in MOI.RotatedSecondOrderCone(3))
    end
    return nothing
end

function add_SOC2x2_diag_U!(model::Model, u, U)
    n = length(u)
    for j in 1:n
        @constraint(model, [U[j,j], 0.5, u[j]] in MOI.RotatedSecondOrderCone(3))
    end
    return nothing
end

add_SOC2x2_diag_XU!(m, x, X, u, U) = begin
    add_SOC2x2_diag_X!(m, x, X)
    add_SOC2x2_diag_U!(m, u, U)
    nothing
end

# ------------------------------------------------------------------
# 2) full 2×2 SOC: diagonal + all 2×2 minors
#    Xii Xjj ≥ Xij² via RSOC; same for U
# ------------------------------------------------------------------
function add_SOC2x2_full_X!(model::Model, x, X)
    n = length(x)
    add_SOC2x2_diag_X!(model, x, X)
    for i in 1:n, j in i+1:n
        @constraint(model,
            [X[i,i], X[j,j], sqrt(2.0) * X[i,j]] in MOI.RotatedSecondOrderCone(3)
        )
    end
    return nothing
end

function add_SOC2x2_full_U!(model::Model, u, U)
    n = length(u)
    add_SOC2x2_diag_U!(model, u, U)
    for i in 1:n, j in i+1:n
        @constraint(model,
            [U[i,i], U[j,j], sqrt(2.0) * U[i,j]] in MOI.RotatedSecondOrderCone(3)
        )
    end
    return nothing
end

add_SOC2x2_full_XU!(m, x, X, u, U) = begin
    add_SOC2x2_full_X!(m, x, X)
    add_SOC2x2_full_U!(m, u, U)
    nothing
end

# ------------------------------------------------------------------
# 3) directional SOC: d'Z d ≥ (z'd)^2 for d = e_i ± e_j
#    "core" directional constraints, without diagonals
# ------------------------------------------------------------------
function add_SOC_directional_core_X!(model::Model, x, X)
    n = length(x)
    for i in 1:n, j in i+1:n
        # d = e_i + e_j
        @constraint(model,
            [X[i,i] + 2*X[i,j] + X[j,j],
             0.5,
             x[i] + x[j]] in MOI.RotatedSecondOrderCone(3)
        )

        # d = e_i - e_j
        @constraint(model,
            [X[i,i] - 2*X[i,j] + X[j,j],
             0.5,
             x[i] - x[j]] in MOI.RotatedSecondOrderCone(3)
        )
    end
    return nothing
end

function add_SOC_directional_core_U!(model::Model, u, U)
    n = length(u)
    for i in 1:n, j in i+1:n
        # d = e_i + e_j
        @constraint(model,
            [U[i,i] + 2*U[i,j] + U[j,j],
             0.5,
             u[i] + u[j]] in MOI.RotatedSecondOrderCone(3)
        )

        # d = e_i - e_j
        @constraint(model,
            [U[i,i] - 2*U[i,j] + U[j,j],
             0.5,
             u[i] - u[j]] in MOI.RotatedSecondOrderCone(3)
        )
    end
    return nothing
end

# directional family = diagonals (d = e_i) + directional_core (d = e_i ± e_j)
function add_SOC_directional_X!(model::Model, x, X)
    add_SOC2x2_diag_X!(model, x, X)
    add_SOC_directional_core_X!(model, x, X)
    return nothing
end

function add_SOC_directional_U!(model::Model, u, U)
    add_SOC2x2_diag_U!(model, u, U)
    add_SOC_directional_core_U!(model, u, U)
    return nothing
end

add_SOC_directional_XU!(m, x, X, u, U) = begin
    add_SOC_directional_X!(m, x, X)
    add_SOC_directional_U!(m, u, U)
    nothing
end

# ------------------------------------------------------------------
# 3b) hybrid SOC = 2×2 full + directional_core (no duplicate diagonals)
# ------------------------------------------------------------------
function add_SOC_hybrid_X!(model::Model, x, X)
    add_SOC2x2_full_X!(model, x, X)          # diag + all 2×2 minors
    add_SOC_directional_core_X!(model, x, X) # d = e_i ± e_j
    return nothing
end

function add_SOC_hybrid_U!(model::Model, u, U)
    add_SOC2x2_full_U!(model, u, U)
    add_SOC_directional_core_U!(model, u, U)
    return nothing
end

add_SOC_hybrid_XU!(m, x, X, u, U) = begin
    add_SOC_hybrid_X!(m, x, X)
    add_SOC_hybrid_U!(m, u, U)
    nothing
end

# ------------------------------------------------------------------
# 4) PSD 3×3 principals of [1 x'; x X] and [1 u'; u U]
# ------------------------------------------------------------------
function add_PSD3x3_X!(model::Model, x, X)
    n = length(x)
    for (i,j) in all_pairs(n)
        @constraint(model, Symmetric([
            1.0     x[i]         x[j];
            x[i]    X[i,i]       X[i,j];
            x[j]    X[i,j]       X[j,j]
        ]) in PSDCone())
    end
    return nothing
end

function add_PSD3x3_U!(model::Model, u, U)
    n = length(u)
    for (i,j) in all_pairs(n)
        @constraint(model, Symmetric([
            1.0     u[i]         u[j];
            u[i]    U[i,i]       U[i,j];
            u[j]    U[i,j]       U[j,j]
        ]) in PSDCone())
    end
    return nothing
end

add_PSD3x3_XU!(m, x, X, u, U) = begin
    add_PSD3x3_X!(m, x, X)
    add_PSD3x3_U!(m, u, U)
    nothing
end

# ------------------------------------------------------------------
# 5) block SDP constraints: [1  x'; x  X]  and  [1  u'; u  U]
# ------------------------------------------------------------------
function add_blockSDP_X!(model::Model, x, X)
    @constraint(model,
        Symmetric([ 1.0   x';
                    x     X ]) in PSDCone())
    return nothing
end

function add_blockSDP_U!(model::Model, u, U)
    @constraint(model,
        Symmetric([ 1.0   u';
                    u     U ]) in PSDCone())
    return nothing
end

add_blockSDP_XU!(m, x, X, u, U) = begin
    add_blockSDP_X!(m, x, X)
    add_blockSDP_U!(m, u, U)
    nothing
end

# ------------------------------------------------------------------
# Build and solve
# ------------------------------------------------------------------
function build_RLT_SDP_model(
    data::Dict{String,Any};
    variant::String = "EU",
    optimizer = MosekTools.Optimizer,
    build_only::Bool = false,
    relaxation::Symbol = :RLT,
)
    @assert relaxation in RelaxationModes "Unknown relaxation mode: $relaxation"

    params = prepare_instance(data)
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, e = params

    m = Model(optimizer)

    @variable(m, x[1:n])
    @variable(m, u[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, R[1:n,1:n])
    @variable(m, U[1:n,1:n], Symmetric)

    @objective(m, Min,
        0.5 * sum(Q0[i,j]*X[i,j] for i=1:n, j=1:n) +
        sum(q0[i]*x[i] for i=1:n)
    )

    # base RLT blocks
    add_FC!(m, x, u, X, R, U, params)

    if startswith(variant, "E")
        add_FE!(m, x, u, X, R, U, params)
    elseif startswith(variant, "I")
        add_FI!(m, x, u, X, R, U, params)
    else
        error("Unknown variant: $variant (expected prefix 'E' or 'I')")
    end

    if endswith(variant, "U")
        add_FU!(m, x, u, X, R, U, params)
        if startswith(variant, "I")
            add_FIU!(m, u, U, params)
        end
    end

    # gradual strengthening on top of RLT
    if relaxation == :RLT
        # pure RLT

    # ---- SOC 2x2 diag ----
    elseif relaxation == :RLT_SOC2x2_diag_X
        add_SOC2x2_diag_X!(m, x, X)
    elseif relaxation == :RLT_SOC2x2_diag_U
        add_SOC2x2_diag_U!(m, u, U)
    elseif relaxation == :RLT_SOC2x2_diag_XU
        add_SOC2x2_diag_XU!(m, x, X, u, U)

    # ---- SOC 2x2 full ----
    elseif relaxation == :RLT_SOC2x2_full_X
        add_SOC2x2_full_X!(m, x, X)
    elseif relaxation == :RLT_SOC2x2_full_U
        add_SOC2x2_full_U!(m, u, U)
    elseif relaxation == :RLT_SOC2x2_full_XU
        add_SOC2x2_full_XU!(m, x, X, u, U)

    # ---- SOC directional (diag + directional_core) ----
    elseif relaxation == :RLT_SOC_directional_X
        add_SOC_directional_X!(m, x, X)
    elseif relaxation == :RLT_SOC_directional_U
        add_SOC_directional_U!(m, u, U)
    elseif relaxation == :RLT_SOC_directional_XU
        add_SOC_directional_XU!(m, x, X, u, U)

    # ---- SOC hybrid = 2x2 full + directional_core ----
    elseif relaxation == :RLT_SOC_hybrid_X
        add_SOC_hybrid_X!(m, x, X)
    elseif relaxation == :RLT_SOC_hybrid_U
        add_SOC_hybrid_U!(m, u, U)
    elseif relaxation == :RLT_SOC_hybrid_XU
        add_SOC_hybrid_XU!(m, x, X, u, U)

    # ---- PSD 3x3 ----
    elseif relaxation == :RLT_PSD3x3_X
        add_PSD3x3_X!(m, x, X)
    elseif relaxation == :RLT_PSD3x3_U
        add_PSD3x3_U!(m, u, U)
    elseif relaxation == :RLT_PSD3x3_XU
        add_PSD3x3_XU!(m, x, X, u, U)

    # ---- block SDP ----
    elseif relaxation == :RLT_blockSDP_X
        add_blockSDP_X!(m, x, X)
    elseif relaxation == :RLT_blockSDP_U
        add_blockSDP_U!(m, u, U)
    elseif relaxation == :RLT_blockSDP_XU
        add_blockSDP_XU!(m, x, X, u, U)

    # ---- full SDP (single big block, both x and u together) ----
    elseif relaxation == :RLT_full_SDP
        add_SDP_lift!(m, x, u, X, R, U; n = n)

    else
        error("Unhandled relaxation mode: $relaxation")
    end

    if build_only
        return m
    end

    t0 = time_ns()
    optimize!(m)
    solve_time_sec = round((time_ns() - t0) / 1e9; digits = 4)

    return m, solve_time_sec
end

function solve_RLT_SDP(
    data::Dict{String,Any};
    variant::String = "EU",
    optimizer = MosekTools.Optimizer,
    relaxation::Symbol = :RLT,
)
    m, t = build_RLT_SDP_model(
        data; variant=variant, optimizer=optimizer,
        build_only=false, relaxation=relaxation
    )
    st = termination_status(m)

    if st in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
        x = value.(m[:x])
        u = value.(m[:u])
        X = value.(m[:X])
        R = value.(m[:R])
        U = value.(m[:U])
        return (st, objective_value(m), x, u, X, R, U, t)
    else
        return (st, nothing, nothing, nothing, nothing, nothing, nothing, t)
    end
end

# ------------------------------------------------------------------
# PrettyPrint wrapper (model-based)
# ------------------------------------------------------------------
function solve_and_print(
    data::Dict{String,Any};
    variant::String = "EU",
    relaxation::Symbol = :RLT,
    optimizer = MosekTools.Optimizer,
    show_mats::Bool = true,
    show_Z::Bool = false,
    pp_kwargs...
)
    if !isdefined(Main, :PrettyPrint)
        error("PrettyPrint is not loaded. Please include(\"instances/prettyprint.jl\") before calling RLT_SDP_Combo.solve_and_print.")
    end

    m = build_RLT_SDP_model(
        data;
        variant=variant,
        optimizer=optimizer,
        build_only=true,
        relaxation=relaxation
    )

    optimize!(m)

    Main.PrettyPrint.print_model_solution(
        m;
        variant=variant,
        relaxation=relaxation,
        show_mats=show_mats,
        show_Z=show_Z,
        pp_kwargs...
    )

    return m
end

end # module


if isinteractive()
    # ---- utilities first ----
    include("instances/prettyprint.jl")
    using .PrettyPrint

    using .RLTBigM
    using .SDPBigM
    using .RLT_SDP_Combo

    # ---- instances ----
    include("instances/alper_stqp_instance.jl")
    using .AlperStqpInstances
    alp_inst = alper_stqp_rho3_instance()

    include("instances/diff_RLTEU_RLTIU_bigM_instance.jl")
    using .EUIUdiffinstance
    diff_inst = euiu_diff_instance()

    # 1) Alper's instance: RLT + full SDP
    RLT_SDP_Combo.solve_and_print(alp_inst; variant="E", relaxation=:RLT_full_SDP, show_mats=true, show_Z=false)

    # 2) Diff instance: RLT + full SDP
    #RLT_SDP_Combo.solve_and_print(diff_inst; variant="EU", relaxation=:RLT_full_SDP, show_mats=true, show_Z=false)

end