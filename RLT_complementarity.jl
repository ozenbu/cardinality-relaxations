module RLTComp

using LinearAlgebra
using Parameters
using JuMP
using MathOptInterface
using Dates, Printf
const MOI = MathOptInterface

# Re-use instance parsing from BigM side (must be loaded before this module).
using ..RLTBigM: prepare_instance

# ---------- helpers ----------
outer_xy(a, b) = a * transpose(b)

# ============================================================
#  Block: FC_comp (common; uses only original linear rows for RLT generation)
# ============================================================
function add_FC_comp!(m, x, v, X, V, W, params)
    @unpack n, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, e, eeT = params

    # Original QCQP rows (lifted via X)
    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(m,
            0.5 * sum(Qmat[i,j] * X[i,j] for i=1:n, j=1:n) + qvec' * x + rterm <= 0
        )
    end
    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(m,
            0.5 * sum(Pmat[i,j] * X[i,j] for i=1:n, j=1:n) + pvec' * x + sterm == 0
        )
    end

    # Original linear rows
    if ℓ > 0
        @constraint(m, A * x .<= b)
    end
    if η > 0
        @constraint(m, H * x .== h)
    end

    # Boundedness rows for complementarity: 0 <= x <= e
    @constraint(m, 0 .<= x)
    @constraint(m, x .<= e)

    # Relaxed complementarity in the lift: diag(W)=0
    @constraint(m, [i=1:n], W[i,i] == 0)

    # First-level RLT generated only from original linear rows

    # x-box self-RLT (0<=x<=e)
    @constraint(m, X .>= 0)
    @constraint(m, x*e' .- X .>= 0)
    @constraint(m, X .- x*e' .- e*x' .+ eeT .>= 0)

    # (Ax<=b) × (Ax<=b): self-RLT and cross terms with x-bounds
    if ℓ > 0
        @constraint(m, A*X*A' .- A*(x*b') .- (b*x')*A' .+ (b*b') .>= 0)
        @constraint(m, -A*X .+ (b*x') .>= 0)
        @constraint(m, A*X .- A*(x*e') .- (b*x') .+ (b*e') .>= 0)
    end

    # (Hx=h) × variables: equality-products
    if η > 0
        @constraint(m, H * X .== outer_xy(h, x))
        @constraint(m, H * W .== outer_xy(h, v))
    end

    return nothing
end

# ============================================================
#  Block: FE_v (cardinality equality and its equality×variable products)
# ============================================================
function add_FE_v!(m, x, v, X, V, W, params)
    @unpack n, ρ, e = params

    # e'v = n - ρ
    @constraint(m, sum(v) == n - ρ)

    # e'V = (n-ρ) v'
    @constraint(m, (e' * V) .== (n - ρ) .* (v'))

    # e'W' = (n-ρ) x'
    @constraint(m, (e' * W') .== (n - ρ) .* (x'))

    return nothing
end

# ============================================================
#  Block: FB_v (binary lift diag(V)=v)
# ============================================================
function add_FB_v!(m, v, V, params)
    @unpack n = params
    @constraint(m, [i=1:n], V[i,i] == v[i])
    return nothing
end

# ============================================================
#  Block: FU_v (explicit 0<=v<=e and its self-RLT + cross-RLT with original inequalities)
# ============================================================
function add_FU_v!(m, x, v, X, V, W, params)
    @unpack n, A, b, ℓ, e, eeT = params

    # New bound rows: 0 <= v <= e
    @constraint(m, 0 .<= v)
    @constraint(m, v .<= e)

    # Self-RLT of 0<=v<=e (V-block)
    @constraint(m, V .>= 0)
    @constraint(m, v*e' .- V .>= 0)
    @constraint(m, V .- v*e' .- e*v' .+ eeT .>= 0)

    # Cross-RLT with original x-bounds: 0<=x<=e (W-block)
    @constraint(m, W .>= 0)
    @constraint(m, x*e' .- W .>= 0)
    @constraint(m, e*v' .- W .>= 0)
    @constraint(m, eeT .- x*e' .- e*v' .+ W .>= 0)

    # Cross-RLT with original linear inequalities: Ax<=b
    if ℓ > 0
        @constraint(m, b*v' .- A*W .>= 0)
        @constraint(m, b*e' .- (A*x)*e' .- b*v' .+ A*W .>= 0)
    end

    return nothing
end

# ============================================================
#  Build & solve (exact + RLT)  (objective/constraints/variants unchanged)
# ============================================================
"""
    build_and_solve(data; variant="RLT_S2", build_only=false, optimizer, verbose=false)

Variants:
- "EXACT_S2" : v binary, uses x <= e - v
- "EXACT_S3" : v continuous, uses nonconvex bilinear constraint sum_i x_i v_i = 0
- "RLT_S2"   : FC_comp + FE_v + FB_v
- "RLT_S2U"  : FC_comp + FE_v + FB_v + FU_v
- "RLT_S3"   : FC_comp + FE_v + FU_v
"""
function build_and_solve(data;
    variant::String="RLT_S2",
    build_only::Bool=false,
    optimizer,
    verbose::Bool=false
)
    # Base parser from RLTBigM + add eeT for box-RLT expansions.
    base   = prepare_instance(data)
    params = merge(base, (eeT = ones(base.n, base.n),))

    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η, e, eeT = params

    m = Model(optimizer)
    verbose ? set_optimizer_attribute(m, "OutputFlag", 1) : set_silent(m)

    # Useful for EXACT_S3 and nonconvex QCQP rows (if present)
    try
        set_optimizer_attribute(m, "NonConvex", 2)
    catch
    end

    set_name(m, "SQCQP_Complementarity")

    if variant == "EXACT_S2"
        @variable(m, x[1:n])
        @variable(m, v[1:n], Bin)

        @objective(m, Min, 0.5 * x' * Q0 * x + q0' * x)

        if ℓ > 0; @constraint(m, A * x .<= b); end
        if η > 0; @constraint(m, H * x .== h); end

        @constraint(m, 0 .<= x)
        @constraint(m, x .<= e)

        @constraint(m, sum(v) == n - ρ)
        @constraint(m, x .<= e .- v)

        for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
            @constraint(m, 0.5 * x' * Qmat * x + qvec' * x + rterm <= 0)
        end
        for (Pmat, pvec, sterm) in zip(Pi, pi, si)
            @constraint(m, 0.5 * x' * Pmat * x + pvec' * x + sterm == 0)
        end

        if build_only
            return m
        end

        optimize!(m)
        term = termination_status(m)
        if term == MOI.OPTIMAL
            return (:OPTIMAL, objective_value(m), value.(x), value.(v))
        else
            return (Symbol(term), nothing, nothing, nothing)
        end

    elseif variant == "EXACT_S3"
        @variable(m, x[1:n])
        @variable(m, 0 <= v[1:n] <= 1)

        @objective(m, Min, 0.5 * x' * Q0 * x + q0' * x)

        if ℓ > 0; @constraint(m, A * x .<= b); end
        if η > 0; @constraint(m, H * x .== h); end

        @constraint(m, 0 .<= x)
        @constraint(m, x .<= e)

        @constraint(m, sum(v) == n - ρ)
        @constraint(m, sum(x[i] * v[i] for i=1:n) == 0)

        for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
            @constraint(m, 0.5 * x' * Qmat * x + qvec' * x + rterm <= 0)
        end
        for (Pmat, pvec, sterm) in zip(Pi, pi, si)
            @constraint(m, 0.5 * x' * Pmat * x + pvec' * x + sterm == 0)
        end

        if build_only
            return m
        end

        optimize!(m)
        term = termination_status(m)
        if term == MOI.OPTIMAL
            return (:OPTIMAL, objective_value(m), value.(x), value.(v))
        else
            return (Symbol(term), nothing, nothing, nothing)
        end

    else
        # RLT relaxations
        @variable(m, x[1:n])
        @variable(m, v[1:n])                       # continuous in RLT
        @variable(m, X[1:n,1:n], Symmetric)
        @variable(m, V[1:n,1:n], Symmetric)
        @variable(m, W[1:n,1:n])

        @objective(m, Min,
            0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n) + q0' * x
        )

        add_FC_comp!(m, x, v, X, V, W, params)
        add_FE_v!(m, x, v, X, V, W, params)

        if variant == "RLT_S2"
            add_FB_v!(m, v, V, params)
        elseif variant == "RLT_S2U"
            add_FB_v!(m, v, V, params)
            add_FU_v!(m, x, v, X, V, W, params)
        elseif variant == "RLT_S3"
            add_FU_v!(m, x, v, X, V, W, params)
        else
            error("Unknown variant: $variant. Use EXACT_S2, EXACT_S3, RLT_S2, RLT_S2U, or RLT_S3.")
        end

        if build_only
            return m
        end

        optimize!(m)
        term = termination_status(m)
        if term == MOI.OPTIMAL
            return (:OPTIMAL, objective_value(m),
                    value.(x), value.(v), value.(X), value.(V), value.(W))
        else
            return (Symbol(term), nothing, nothing, nothing, nothing, nothing, nothing)
        end
    end
end

# ============================================================
# PrettyPrint wrapper (single call; no loops)
# ============================================================
function solve_and_print(data::Dict{String,Any};
    variant::String="RLT_S2U",
    relaxation = nothing,
    optimizer,
    verbose::Bool=false,
    show_mats::Bool=true,
    pp_kwargs...
)
    if !isdefined(Main, :PrettyPrint)
        error("PrettyPrint is not loaded. Please include(\"instances/prettyprint.jl\") before calling RLTComp.solve_and_print.")
    end

    rel = relaxation === nothing ? (startswith(variant, "EXACT") ? "EXACT" : :RLT) : relaxation

    m = build_and_solve(data;
        variant=variant,
        build_only=true,
        optimizer=optimizer,
        verbose=verbose
    )
    optimize!(m)

    Main.PrettyPrint.print_model_solution(m;
        variant=variant,
        relaxation=rel,
        show_mats=show_mats,
        pp_kwargs...
    )

    return m
end

function demo(data::Dict{String,Any};
    variant::String="RLT_S2U",
    optimizer,
    verbose::Bool=false,
    show_mats::Bool=true
)
    solve_and_print(data; variant=variant, optimizer=optimizer, verbose=verbose, show_mats=show_mats)
    return nothing
end

export build_and_solve, solve_and_print, demo

end # module RLTComp

if isinteractive()
    # ---- PrettyPrint first ----
    include("instances/prettyprint.jl")
    using .PrettyPrint
    # ---- Optimizer ----
    using Gurobi

    # ---- Instances (from .jl files) ----
    include("instances/alper_stqp_instance.jl")
    using .AlperStqpInstances
    alp_inst = alper_stqp_rho3_instance()

    include("instances/diff_RLTEU_RLTIU_bigM_instance.jl")
    using .EUIUdiffinstance
    diff_inst = euiu_diff_instance()


    #inst = alp_inst
    inst = alp_inst

    # =========================================================
    # Run all variants (loop)
    # =========================================================
    variants = ["EXACT_S2", "EXACT_S3", "RLT_S2", "RLT_S2U", "RLT_S3"]

    println("\n==============================")
    println("instance id = ", get(inst, "id", "unknown"), "\n")

    for var in variants
        println("\n------------------------------")
        println("variant = ", var)
        println("------------------------------")

        # Build model (no printing inside)
        m = RLTComp.build_and_solve(
            inst;
            variant=var,
            build_only=true,
            optimizer=Gurobi.Optimizer,
            verbose=false
        )

        optimize!(m)

        # PrettyPrint auto-detects x/v/X/V/W and prints what exists
        PrettyPrint.print_model_solution(
            m;
            variant=var,
            relaxation=(startswith(var, "EXACT") ? "EXACT" : :RLT),
            show_mats=true
        )
    end
end