# ============================================================
#  Complementarity formulations + first-level RLT relaxations
#  Variants:
#    EXACT_S2     : v binary, exact (linearized complementarity via x <= 1 - v)
#    EXACT_S3     : v continuous in [0,1], exact (nonconvex bilinear x'v = 0)
#    RLT_S2       : FC_comp + FE_v + FB_v
#    RLT_S2U      : FC_comp + FE_v + FB_v + FU_v   (this is your S2^+ at RLT level)
#    RLT_S3       : FC_comp + FE_v + FU_v
#
#  Notes:
#   - We assume the boundedness rows 0 <= x <= e are part of the original model.
#   - Complementarity is relaxed in the lift via diag(W)=0 (componentwise x_i v_i = 0).
#   - In FC_comp we generate first-level RLT only from ORIGINAL linear rows:
#       Ax <= b, 0 <= x <= e, and Hx = h (equalities × variables).
#   - FU_v contains the additional RLT cuts induced by explicitly adding 0 <= v <= e,
#     including cross-products with the original linear inequalities (Ax<=b, 0<=x<=e).
# ============================================================

module RLTComp

using LinearAlgebra
using Parameters
using JuMP
using MathOptInterface
using Dates, Printf

const MOI = MathOptInterface

# ---------- helpers ----------
outer_xy(a, b) = a * transpose(b)

ensure_tuple(x) =
    x === nothing ? () :
    (x isa AbstractVector || x isa Tuple ? x : (x,))

"""
    prepare_instance(data::Dict)

Unifies instance fields for the complementarity formulations.

Required:
- "n", "rho"
Optional (defaults to empty if missing):
- "Q0", "q0"
- "Qi","qi","ri" (quadratic inequalities)
- "Pi","pi","si" (quadratic equalities)
- "A","b" (linear inequalities)
- "H","h" (linear equalities)

Assumes the boundedness rows for complementarity formulations:
- 0 <= x <= e (with e = ones(n))
"""
function prepare_instance(data::Dict)
    n = data["n"]
    ρ = data["rho"]

    Q0 = get(data, "Q0", zeros(n, n))::AbstractMatrix
    q0 = get(data, "q0", zeros(n))::AbstractVector

    Qi = ensure_tuple(get(data, "Qi", ()))
    qi = ensure_tuple(get(data, "qi", ()))
    ri = ensure_tuple(get(data, "ri", ()))

    Pi = ensure_tuple(get(data, "Pi", ()))
    pi = ensure_tuple(get(data, "pi", ()))
    si = ensure_tuple(get(data, "si", ()))

    A = get(data, "A", nothing)
    b = get(data, "b", nothing)
    if A === nothing
        A = zeros(0, n); b = zeros(0)
    end
    ℓ = size(A, 1)

    H = get(data, "H", nothing)
    h = get(data, "h", nothing)
    if H === nothing
        H = zeros(0, n); h = zeros(0)
    end
    η = size(H, 1)

    e = ones(n)
    eeT = ones(n, n)

    return (; n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, e, eeT)
end


# ============================================================
#  Block: FC_comp (common, uses only original linear rows for RLT generation)
# ============================================================
function add_FC_comp!(m, x, v, X, V, W, params)
    @unpack n, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, e, eeT = params

    # ---- original QCQP rows (lifted via X) ----
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

    # ---- original linear rows ----
    if ℓ > 0
        @constraint(m, A * x .<= b)
    end
    if η > 0
        @constraint(m, H * x .== h)
    end

    # ---- boundedness rows (assumed part of the original model here): 0 <= x <= e ----
    @constraint(m, 0 .<= x)
    @constraint(m, x .<= e)

    # ---- relaxed complementarity in the lift ----
    # diag(W)=0 (componentwise x_i v_i = 0 in the intended lift)
    @constraint(m, [i=1:n], W[i,i] == 0)

    # ============================================================
    # First-level RLT generated ONLY from original linear rows:
    # inequalities×inequalities and equalities×variables
    # ============================================================

    # --- x-box self-RLT (0<=x<=e), minimal with X symmetric ---
    # from x >= 0:  xx' >= 0  -> X >= 0 (entrywise)
    @constraint(m, X .>= 0)

    # from x >= 0 and (e-x) >= 0: x(e-x)' >= 0  -> x e' - X >= 0
    @constraint(m, x*e' .- X .>= 0)

    # from (e-x) >= 0: (e-x)(e-x)' >= 0 -> X - x e' - e x' + ee' >= 0
    @constraint(m, X .- x*e' .- e*x' .+ eeT .>= 0)

    # --- (Ax<=b) × (Ax<=b): self-RLT ---
    if ℓ > 0
        @constraint(m, A*X*A' .- A*(x*b') .- (b*x')*A' .+ (b*b') .>= 0)

        # --- (b-Ax) x' >= 0 ---
        @constraint(m, -A*X .+ (b*x') .>= 0)

        # --- (b-Ax)(e-x)' >= 0 ---
        @constraint(m, A*X .- A*(x*e') .- (b*x') .+ (b*e') .>= 0)
    end

    # --- (Hx=h) × variables: equality-products ---
    if η > 0
        @constraint(m, H * X .== outer_xy(h, x))
        # If W is intended as x v', then (Hx-h) v' = 0 linearises to H W = h v'
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

    # (e'v - (n-ρ)) v' = 0  ->  e'V = (n-ρ) v'
    @constraint(m, (e' * V) .== (n - ρ) .* (v'))

    # (e'v - (n-ρ)) x' = 0  ->  e'(v x') = (n-ρ) x'
    # with W' intended as v x'
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

    # ---- new bound row itself ----
    @constraint(m, 0 .<= v)
    @constraint(m, v .<= e)

    # ---- self-RLT of 0<=v<=e (V-block), minimal using V symmetric ----
    @constraint(m, V .>= 0)
    @constraint(m, v*e' .- V .>= 0)
    @constraint(m, V .- v*e' .- e*v' .+ eeT .>= 0)

    # ---- cross-RLT with original x-bounds: 0<=x<=e (W-block) ----
    @constraint(m, W .>= 0)
    @constraint(m, x*e' .- W .>= 0)
    @constraint(m, e*v' .- W .>= 0)
    @constraint(m, eeT .- x*e' .- e*v' .+ W .>= 0)

    # ---- cross-RLT with original linear inequalities: Ax<=b ----
    if ℓ > 0
        # (b-Ax) v' >= 0  ->  b v' - A W >= 0
        @constraint(m, b*v' .- A*W .>= 0)

        # (b-Ax)(e-v)' >= 0 -> b e' - (Ax)e' - b v' + A W >= 0
        @constraint(m, b*e' .- (A*x)*e' .- b*v' .+ A*W .>= 0)
    end

    return nothing
end


# ============================================================
#  Build & solve (exact + RLT)
# ============================================================
"""
    build_and_solve(data; variant="RLT_S2", build_only=false, optimizer, verbose=false)

Variants:
- "EXACT_S2" : v binary, uses x_i <= 1 - v_i (exact under 0<=x<=1)
- "EXACT_S3" : v continuous, uses nonconvex bilinear constraint x'v = 0
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
    params = prepare_instance(data)
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, e, eeT = params

    m = Model(optimizer)
    verbose ? set_optimizer_attribute(m, "OutputFlag", 1) : set_silent(m)

    # Gurobi/other nonconvex-capable solvers may need this for bilinear/quadratic nonconvexities.
    # Safe to set if supported; if not supported, it will error (user can disable manually).
    try
        set_optimizer_attribute(m, "NonConvex", 2)
    catch
        # ignore if optimizer does not support it
    end

    set_name(m, "SQCQP_Complementarity")

    if variant == "EXACT_S2"
        # ----- exact S2: binary v, linearized complementarity via x <= 1 - v -----
        @variable(m, x[1:n])
        @variable(m, v[1:n], Bin)

        @objective(m, Min, 0.5 * x' * Q0 * x + q0' * x)

        # original linear rows
        if ℓ > 0; @constraint(m, A * x .<= b); end
        if η > 0; @constraint(m, H * x .== h); end

        # boundedness
        @constraint(m, 0 .<= x)
        @constraint(m, x .<= e)

        # cardinality
        @constraint(m, sum(v) == n - ρ)

        # exact complementarity under 0<=x<=1:
        # v_i = 1 forces x_i = 0; v_i = 0 leaves 0<=x_i<=1
        @constraint(m, x .<= e .- v)

        # original QCQP rows
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
        # ----- exact S3: continuous v in [0,1], nonconvex bilinear complementarity x'v=0 -----
        @variable(m, x[1:n])
        @variable(m, 0 <= v[1:n] <= 1)

        @objective(m, Min, 0.5 * x' * Q0 * x + q0' * x)

        # original linear rows
        if ℓ > 0; @constraint(m, A * x .<= b); end
        if η > 0; @constraint(m, H * x .== h); end

        # boundedness
        @constraint(m, 0 .<= x)
        @constraint(m, x .<= e)

        # cardinality
        @constraint(m, sum(v) == n - ρ)

        # complementarity (scalar, but exact under x>=0,v>=0):
        @constraint(m, sum(x[i] * v[i] for i=1:n) == 0)

        # original QCQP rows
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
        # ----- RLT relaxations -----
        @variable(m, x[1:n])
        @variable(m, v[1:n])                       # continuous in RLT
        @variable(m, X[1:n,1:n], Symmetric)
        @variable(m, V[1:n,1:n], Symmetric)
        @variable(m, W[1:n,1:n])

        @objective(m, Min,
            0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n) + q0' * x
        )

        # common block
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
#  Inspect helper (writes LP)
# ============================================================
function inspect_model(data, variant::String; optimizer)
    m = build_and_solve(data; variant=variant, build_only=true, optimizer=optimizer, verbose=false)
    fname = "$(variant)_model.lp"
    write_to_file(m, fname)
    println("\n$fname written.\n")
    println("Number of constraints: ",
        num_constraints(m; count_variable_in_set_constraints=true))
    return nothing
end

function make_data_test4()
    Q0 = [
        3//10000    127//1250   79//2500    867//10000;
        127//1250   1//500      1001//10000 1059//10000;
        79//2500    1001//10000 -1//2000    -703//10000;
        867//10000  1059//10000 -703//10000 -1063//10000
    ]
    Q0d = Q0 * 20000

    q0  = [-1973//10000, -2535//10000, -1967//10000, -973//10000]
    q0d = q0 * 20000

    A2 = [
         1.0  0.0  0.0  0.0
         0.0  1.0  0.0  0.0
         0.0  0.0  1.0  0.0
         0.0  0.0  0.0  1.0
        -1.0  0.0  0.0  0.0
         0.0 -1.0  0.0  0.0
         0.0  0.0 -1.0  0.0
         0.0  0.0  0.0 -1.0
    ]

    b2 = [1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0]

    return Dict(
        "n"   => 4,
        "rho" => 3.0,
        "Q0"  => Q0d,
        "q0"  => q0d,
        "Qi"  => nothing, "qi" => nothing, "ri" => nothing,
        "Pi"  => nothing, "pi" => nothing, "si" => nothing,
        "A"   => A2, "b" => b2,
        "H"   => nothing, "h" => nothing,
        "M"   => I(4)
    )
end

export make_data_test4
end # module RLTComp

# ============================================================
# Runner for EXACT_S2, EXACT_S3, RLT_S2, RLT_S2U (=RLT_S2+), RLT_S3
# Assumes everything (module RLTComp + this runner) is in the SAME file.
# ============================================================

using LinearAlgebra
using Gurobi
using .RLTComp



function run_comp_suite(data; optimizer=Gurobi.Optimizer, verbose::Bool=false)
    variants = ["EXACT_S2", "EXACT_S3", "RLT_S2", "RLT_S2U", "RLT_S3"]

    println("\n==============================")
    println(" Complementarity suite runner ")
    println("==============================\n")

    for var in variants
        println("---- variant = $var ----")
        res = nothing
        try
            res = RLTComp.build_and_solve(
                data;
                variant=var,
                build_only=false,
                optimizer=optimizer,
                verbose=verbose
            )
        catch err
            println("ERROR running $var:\n  ", err)
            println()
            continue
        end

        status = res[1]
        obj    = res[2]

        println("status = ", status)
        if obj !== nothing
            println("obj    = ", obj)
        end

        if status == :OPTIMAL
            if startswith(var, "EXACT")
                x = res[3]; v = res[4]
                println("x = ", x)
                println("v = ", v)
            else
                x = res[3]; v = res[4]
                println("x = ", x)
                println("v = ", v)
                # Uncomment if you want the lifted variables:
                # X = res[5]; V = res[6]; W = res[7]
                # println("X = "); show(stdout, "text/plain", X); println()
                # println("V = "); show(stdout, "text/plain", V); println()
                # println("W = "); show(stdout, "text/plain", W); println()
            end
        end
        println()
    end

    return nothing
end

function bomze_stqp_rho3_instance(; bigM_scale::Real = 1.0)
    n   = 6
    rho = 3

    # Q taken from Alper's slide (Bomze et al. example)
    Q = [
        2.6947  -0.2028  -1.1144  -2.4230  -2.1633   0.7710
       -0.2028   5.1998   0.5005  -2.0941   0.7828  -3.3611
       -1.1144   0.5005   5.2918   1.4119  -1.1526  -1.9723
       -2.4230  -2.0941   1.4119   4.1140  -0.2025   0.3132
       -2.1633   0.7828  -1.1526  -0.2025   7.6645  -0.0170
        0.7710  -3.3611  -1.9723   0.3132  -0.0170   3.3040
    ]

    # Our convention is obj = 0.5 x'Q0 x + q0'x, so choose Q0 = 2Q, q0 = 0
    Q0 = 2.0 .* Q
    q0 = zeros(Float64, n)

    # Base region: simplex  x ≥ 0, eᵀx = 1
    A, b, H, h = UserInstanceGen.build_simplex_bounds(n)

    # Anchor point: barycenter of the simplex
    xbar = fill(1.0 / n, n)

    # Bounds on simplex: each coordinate in [0,1]
    ell = zeros(Float64, n)
    u   = ones(Float64, n)
    UserInstanceGen.clean_bounds!(ell, u; digits = 4)

    # Big-M from bounds
    M_common, M_minus, M_plus = UserInstanceGen.bigM_from_bounds(ell, u; scale = bigM_scale)

    # Pack into Dict consistent with generate_instance output
    inst = Dict{String,Any}()

    inst["n"]   = n
    inst["rho"] = rho

    inst["Q0"] = Q0
    inst["q0"] = q0

    # No quadratic (in)equality constraints
    inst["Qi"] = nothing
    inst["qi"] = nothing
    inst["ri"] = nothing

    inst["Pi"] = nothing
    inst["pi"] = nothing
    inst["si"] = nothing

    # Linear constraints of the simplex
    inst["A"] = A
    inst["b"] = b
    inst["H"] = H
    inst["h"] = h

    inst["ell"]     = ell
    inst["u"]       = u
    inst["M"]       = M_common
    inst["M_minus"] = M_minus
    inst["M_plus"]  = M_plus

    inst["xbar"] = xbar

    inst["base_type"] = "simplex"
    inst["want_convex"] = true

    # Eigenvalue pattern: Q0 is PSD, no QI
    inst["neg_eig_counts_input"] = [0]
    inst["neg_eig_counts_used"]  = [0]

    # Seed is arbitrary for this deterministic instance
    inst["seed"] = 0

    # Meta tags
    inst["base_tag"]  = "S"
    inst["convexity"] = "CVX"

    inst["n_LI"] = 0
    inst["n_LE"] = 0
    inst["n_QI"] = 0
    inst["n_QE"] = 0
    inst["bigM_scale"] = bigM_scale

    # Final ID with "fv" tag (final version)
    inst["id"] = "n6_rho3_S_StQP_rho3_CVX_fv"

    return inst
end


# ============================================================
# Call this at the bottom of the same file (or inside isinteractive()).
# ============================================================
function demo_comp(data)
    run_comp_suite(data; optimizer=Gurobi.Optimizer, verbose=false)
end

# Uncomment to auto-run when executing the file:
data_test4 = make_data_test4()
demo_comp(data_test4)

