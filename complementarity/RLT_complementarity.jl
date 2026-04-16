module RLTComp

using LinearAlgebra: diag
using JuMP
using MathOptInterface
const MOI = MathOptInterface
using Parameters: @unpack

# Single source of truth for instance parsing (BigM + general bounds)
using ..RLTBigM: prepare_instance

# ---------- helpers ----------
outer_xy(a, b) = a * transpose(b)

"""
    enrich_params(params)

Add complementarity-side derived constants once:
- xL, xU : vectors for the general box bounds on x
- eeT    : all-ones matrix (e eᵀ)
"""
function additional_params(params)
    xL  = -diag(params.Mminus)      # vector
    xU  =  diag(params.Mplus)       # vector
    eeT = params.e * params.e'      # matrix
    return merge(params, (; xL, xU, eeT))
end

# ============================================================
#  Block: FC_comp (common)
#   - only ORIGINAL linear rows generate 1st-level RLT constraints
#   - includes general x-box self-RLT to control X (analogue of 0<=x<=e case)
# ============================================================
function add_FC_comp!(m, x, v, X, V, W, params)
    @unpack n, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, xL, xU = params

    # ---------- original QCQP rows (lifted via X) ----------
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

    # ---------- original linear rows ----------
    if ℓ > 0
        @constraint(m, A * x .<= b)
    end
    if η > 0
        @constraint(m, H * x .== h)
    end

    # ---------- general x-bounds (part of the original model) ----------
    @constraint(m, xL .<= x)
    @constraint(m, x .<= xU)

    # ---------- relaxed complementarity in the lift ----------
    @constraint(m, [i=1:n], W[i,i] == 0)

    # ============================================================
    # First-level RLT constraints generated ONLY from original linear rows
    # ============================================================

    # --- x-box self-RLT (general bounds) to control X ---
    # (x - xL)(x - xL)^T >= 0
    @constraint(m, X .- x*xL' .- xL*x' .+ xL*xL' .>= 0)
    # (x - xL)(xU - x)^T >= 0
    @constraint(m, x*xU' .- X .- xL*xU' .+ xL*x' .>= 0)
    # (xU - x)(xU - x)^T >= 0
    @constraint(m, X .- xU*x' .- x*xU' .+ xU*xU' .>= 0)

    # --- (Ax<=b) × (Ax<=b), and cross with x-bounds ---
    if ℓ > 0
        # (Ax<=b)×(Ax<=b)
        @constraint(m, A*X*A' .- A*(x*b') .- (b*x')*A' .+ (b*b') .>= 0)

        # (b-Ax)(x-xL)^T >= 0  ->  -AX + b x^T + A x xL^T - b xL^T >= 0
        @constraint(m, -A*X .+ (b*x') .+ A*(x*xL') .- (b*xL') .>= 0)

        # (b-Ax)(xU-x)^T >= 0  ->  AX - A x xU^T - b x^T + b xU^T >= 0
        @constraint(m, A*X .- A*(x*xU') .- (b*x') .+ (b*xU') .>= 0)
    end

    # --- (Hx=h) × variables: equality-products ---
    if η > 0
        @constraint(m, H * X .== outer_xy(h, x))
        @constraint(m, H * W .== outer_xy(h, v))
    end

    return nothing
end

# ============================================================
#  Block: FE_v (cardinality equality + equality×variable products)
# ============================================================
function add_FE_v!(m, x, v, X, V, W, params)
    @unpack n, ρ, e = params
    @constraint(m, sum(v) == n - ρ)                 # eᵀ v = n-ρ
    @constraint(m, (e' * V) .== (n - ρ) .* (v'))    # eᵀ V = (n-ρ) vᵀ
    @constraint(m, (e' * W') .== (n - ρ) .* (x'))   # eᵀ Wᵀ = (n-ρ) xᵀ
    return nothing
end

# ============================================================
#  Block: FB_v (binary lift surrogate: diag(V)=v)
# ============================================================
function add_FB_v!(m, v, V, params)
    @unpack n = params
    @constraint(m, [i=1:n], V[i,i] == v[i])
    return nothing
end

# ============================================================
#  Block: FU_v (explicit 0<=v<=e and its self-RLT + cross-RLT)
#   Cross-RLT uses GENERAL x-bounds xL<=x<=xU.
# ============================================================
function add_FU_v!(m, x, v, X, V, W, params)
    @unpack n, A, b, ℓ, e, eeT, xL, xU = params

    # Bound rows: 0 <= v <= e
    @constraint(m, 0 .<= v)
    @constraint(m, v .<= e)

    # Self-RLT of 0<=v<=e (V-block)
    @constraint(m, V .>= 0)
    @constraint(m, v*e' .- V .>= 0)
    @constraint(m, V .- v*e' .- e*v' .+ eeT .>= 0)

    # Cross-RLT with general x-bounds:
    @constraint(m, W .- xL*v' .>= 0)                       # (x-xL)v^T >= 0
    @constraint(m, x*e' .- W .- xL*e' .+ xL*v' .>= 0)      # (x-xL)(e-v)^T >= 0
    @constraint(m, xU*v' .- W .>= 0)                       # (xU-x)v^T >= 0
    @constraint(m, xU*e' .- xU*v' .- x*e' .+ W .>= 0)      # (xU-x)(e-v)^T >= 0

    # Cross-RLT with Ax<=b:
    if ℓ > 0
        @constraint(m, b*v' .- A*W .>= 0)
        @constraint(m, b*e' .- (A*x)*e' .- b*v' .+ A*W .>= 0)
    end

    return nothing
end

# ============================================================
#  Build & solve
# ============================================================
"""
    build_and_solve(data; variant="RLT_CompBinBou", build_only=false, optimizer, verbose=false)

Preferred variants:
- "EXACT_CompBin"      : v binary; indicator bounds xL(1-v) <= x <= xU(1-v)
- "EXACT_CompBou"      : v ∈ [0,1]; bilinear x[i]*v[i]==0
- "RLT_CompBin"        : FC_comp + FE_v + FB_v
- "RLT_CompBinBou"     : FC_comp + FE_v + FB_v + FU_v
- "RLT_CompBou"        : FC_comp + FE_v + FU_v

Aliases (backward-compatible):
- "EXACT_S2" -> "EXACT_CompBin"
- "EXACT_S3" -> "EXACT_CompBou"
- "RLT_S2"   -> "RLT_CompBin"
- "RLT_S2U"  -> "RLT_CompBinBou"
- "RLT_S3"   -> "RLT_CompBou"
"""
function build_and_solve(data;
    variant::String="RLT_CompBinBou",
    build_only::Bool=false,
    optimizer,
    verbose::Bool=false
)
    params = additional_params(prepare_instance(data))
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, e, xL, xU = params

    # Aliases
    if variant == "EXACT_S2"; variant = "EXACT_CompBin"; end
    if variant == "EXACT_S3"; variant = "EXACT_CompBou"; end
    if variant == "RLT_S2";   variant = "RLT_CompBin"; end
    if variant == "RLT_S2U";  variant = "RLT_CompBinBou"; end
    if variant == "RLT_S3";   variant = "RLT_CompBou"; end

    m = Model(optimizer)

    # Silence / verbosity (solver-dependent)
    if !verbose
        set_silent(m)
    else
        try
            set_optimizer_attribute(m, "OutputFlag", 1)  # Gurobi-style
        catch
        end
    end

    # For nonconvex quadratics (Gurobi)
    try
        set_optimizer_attribute(m, "NonConvex", 2)
    catch
    end

    set_name(m, "SQCQP_Complementarity")

    # ---------------- EXACT_CompBin ----------------
    if variant == "EXACT_CompBin"
        @variable(m, x[1:n])
        @variable(m, v[1:n], Bin)

        @objective(m, Min, 0.5 * x' * Q0 * x + q0' * x)

        if ℓ > 0; @constraint(m, A * x .<= b); end
        if η > 0; @constraint(m, H * x .== h); end

        @constraint(m, xL .<= x)
        @constraint(m, x .<= xU)

        @constraint(m, sum(v) == n - ρ)

        # indicator bounds: if v[i]=1 -> x[i]=0; if v[i]=0 -> x in [xL[i],xU[i]]
        @constraint(m, [i=1:n], x[i] >= xL[i] * (1 - v[i]))
        @constraint(m, [i=1:n], x[i] <= xU[i] * (1 - v[i]))

        for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
            @constraint(m, 0.5 * x' * Qmat * x + qvec' * x + rterm <= 0)
        end
        for (Pmat, pvec, sterm) in zip(Pi, pi, si)
            @constraint(m, 0.5 * x' * Pmat * x + pvec' * x + sterm == 0)
        end

        if build_only; return m; end
        optimize!(m)
        term = termination_status(m)
        return term == MOI.OPTIMAL ? (:OPTIMAL, objective_value(m), value.(x), value.(v)) :
                                     (Symbol(term), nothing, nothing, nothing)
    end

    # ---------------- EXACT_CompBou ----------------
    if variant == "EXACT_CompBou"
        @variable(m, x[1:n])
        @variable(m, 0 <= v[1:n] <= 1)

        @objective(m, Min, 0.5 * x' * Q0 * x + q0' * x)

        if ℓ > 0; @constraint(m, A * x .<= b); end
        if η > 0; @constraint(m, H * x .== h); end

        @constraint(m, xL .<= x)
        @constraint(m, x .<= xU)

        @constraint(m, sum(v) == n - ρ)
        @constraint(m, [i=1:n], x[i] * v[i] == 0)

        for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
            @constraint(m, 0.5 * x' * Qmat * x + qvec' * x + rterm <= 0)
        end
        for (Pmat, pvec, sterm) in zip(Pi, pi, si)
            @constraint(m, 0.5 * x' * Pmat * x + pvec' * x + sterm == 0)
        end

        if build_only; return m; end
        optimize!(m)
        term = termination_status(m)
        return term == MOI.OPTIMAL ? (:OPTIMAL, objective_value(m), value.(x), value.(v)) :
                                     (Symbol(term), nothing, nothing, nothing)
    end

    # ---------------- RLT relaxations ----------------
    @variable(m, x[1:n])
    @variable(m, v[1:n])                       # continuous in RLT
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, V[1:n,1:n], Symmetric)
    @variable(m, W[1:n,1:n])

    @objective(m, Min, 0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n) + q0' * x)

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
        error("Unknown variant: $variant.")
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

export build_and_solve, add_FC_comp!, add_FE_v!, add_FB_v!, add_FU_v!, additional_params

end # module RLTComp


# --------------------------------------------------------------------
# Optional local demo (kept outside the module on purpose)
# --------------------------------------------------------------------
if isinteractive()
    using Gurobi
    using JuMP

    # PrettyPrint (your helper)
    include(normpath(joinpath(@__DIR__, "..", "instances", "prettyprint.jl")))
    using .PrettyPrint

    # Instance
    include(normpath(joinpath(@__DIR__, "..", "instances", "alper_stqp_instance.jl")))
    using .AlperStqpInstances
    inst = alper_stqp_rho3_instance()

    variants = [
        "EXACT_CompBin",
        "EXACT_CompBou",
        "RLT_CompBin",
        "RLT_CompBinBou",
        "RLT_CompBou",
    ]

    println("\n==============================")
    println("instance id = ", get(inst, "id", "unknown"))
    println("==============================\n")

    for var in variants
        println("\n------------------------------")
        println("variant = ", var)
        println("------------------------------")

        m = Main.RLTComp.build_and_solve(
            inst;
            variant=var,
            build_only=true,
            optimizer=Gurobi.Optimizer,
            verbose=false
        )
        optimize!(m)

        PrettyPrint.print_model_solution(
            m;
            variant=var,
            relaxation=(startswith(var, "EXACT") ? "EXACT" : :RLT),
            show_mats=true
        )
    end
end

using Gurobi
using JuMP

# --- PrettyPrint (your existing helper) ---
include(normpath(joinpath(@__DIR__, "..", "instances", "prettyprint.jl")))
using .PrettyPrint

# --- Instance ---
include(normpath(joinpath(@__DIR__, "..", "instances", "alper_stqp_instance.jl")))
using .AlperStqpInstances
inst = alper_stqp_rho3_instance()

variants = [
    "EXACT_CompBin",
    "EXACT_CompBou",
    "RLT_CompBin",
    "RLT_CompBinBou",
    "RLT_CompBou",
]

println("\n==============================")
println("instance id = ", get(inst, "id", "unknown"))
println("==============================\n")

for var in variants
    println("\n------------------------------")
    println("variant = ", var)
    println("------------------------------")

    m = RLTComp.build_and_solve(
        inst;
        variant=var,
        build_only=true,
        optimizer=Gurobi.Optimizer,
        verbose=false
    )
    optimize!(m)

    PrettyPrint.print_model_solution(
        m;
        variant=var,
        relaxation=(startswith(var, "EXACT") ? "EXACT" : :RLT),
        show_mats=true
    )
end