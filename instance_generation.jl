###############################
# InstanceGen.jl
###############################

module InstanceGen

using Random, LinearAlgebra
using JuMP, MosekTools
using JSON

const MOI = JuMP.MOI

# ---------------------------
# Integer / basic helpers
# ---------------------------

rand_vec_int(rng::AbstractRNG, n::Int; vals=-10:10) = rand(rng, vals, n)

function symmetrize_int!(Q::AbstractMatrix)
    Q .= (Q .+ Q') .÷ 2  # integer division
    return Q
end

"Random symmetric integer matrix with even entries."
function rand_symm_int_even(rng::AbstractRNG, n::Int; vals=-10:10)
    Q = rand(rng, vals, n, n)
    symmetrize_int!(Q)
    return 2 .* Q
end

"Random positive integer slack δ ∈ [lo, hi]."
rand_slack(rng::AbstractRNG; lo::Int = 1, hi::Int = 5) = rand(rng, lo:hi)

# ---------------------------
# Eigenvalue-controlled symmetric matrices
# ---------------------------

"""
rand_symm_from_eigs(rng, n; num_neg, nonneg_vals, neg_vals)

Builds a symmetric matrix Q = Qmat * Diag(eigs) * Qmat'
with integer eigenvalues:
- num_neg entries drawn from neg_vals (default -10:-1),
- the rest from nonneg_vals (default 0:10).
"""
function rand_symm_from_eigs(rng::AbstractRNG, n::Int;
                             num_neg::Int=0,
                             nonneg_vals::UnitRange{Int}=0:10,
                             neg_vals::UnitRange{Int}=-10:-1)
    @assert 0 ≤ num_neg ≤ n
    num_nonneg = n - num_neg

    negs    = num_neg == 0     ? Int[] : rand(rng, neg_vals,    num_neg)
    nonnegs = num_nonneg == 0  ? Int[] : rand(rng, nonneg_vals, num_nonneg)

    eigs = vcat(negs, nonnegs)
    eigs = eigs[randperm(rng, length(eigs))]

    R        = randn(rng, n, n)
    Qfac, _  = qr(R)
    Qmat     = Matrix(Qfac)
    Λ        = Diagonal(Float64.(eigs))

    return Qmat * Λ * Qmat'
end

"""
rand_objective(n; rng, num_neg, ...)

Objective matrix Q0 with integer eigenvalues, num_neg of them negative.
q0 is an integer vector.
"""
function rand_objective(n::Int; rng::AbstractRNG,
                        num_neg::Int=0,
                        nonneg_vals::UnitRange{Int}=0:10,
                        neg_vals::UnitRange{Int}=-10:-1,
                        qvals=-10:10)
    Q0 = rand_symm_from_eigs(rng, n;
                             num_neg=num_neg,
                             nonneg_vals=nonneg_vals,
                             neg_vals=neg_vals)
    q0 = rand_vec_int(rng, n; vals=qvals)
    return Q0, q0
end

# ---------------------------
# A that positively spans R^n
# ---------------------------

function build_bounded_A(; n::Int, mode::Symbol=:minimal, rng::AbstractRNG)
    if mode === :minimal
        A = vcat(Matrix{Int}(I, n, n), -ones(Int, 1, n))
    elseif mode === :symmetric
        A = vcat(Matrix{Int}(I, n, n), -Matrix{Int}(I, n, n))
    else
        error("unknown mode = $mode (use :minimal or :symmetric)")
    end
    xhat = rand_vec_int(rng, n; vals=-2:2)
    δ    = rand_slack(rng)
    b    = A * xhat .+ δ
    return A, b, collect(xhat)
end

# ---------------------------
# Bounds via LP
# ---------------------------

function bounds_via_LP(A::AbstractMatrix, b::AbstractVector;
                       H::Union{Nothing,AbstractMatrix}=nothing,
                       h::Union{Nothing,AbstractVector}=nothing,
                       optimizer = MosekTools.Optimizer)
    _, n = size(A)
    ℓ = zeros(Float64, n)
    u = zeros(Float64, n)
    for i in 1:n
        # lower bound
        model = Model(optimizer)
        @variable(model, x[1:n])
        @constraint(model, A * x .<= b)
        if H !== nothing && h !== nothing
            @constraint(model, H * x .== h)
        end
        @objective(model, Min, x[i])
        optimize!(model)
        @assert termination_status(model) == MOI.OPTIMAL "LP lower bound not optimal"
        ℓ[i] = value.(x)[i]

        # upper bound
        model = Model(optimizer)
        @variable(model, x2[1:n])
        @constraint(model, A * x2 .<= b)
        if H !== nothing && h !== nothing
            @constraint(model, H * x2 .== h)
        end
        @objective(model, Max, x2[i])
        optimize!(model)
        @assert termination_status(model) == MOI.OPTIMAL "LP upper bound not optimal"
        u[i] = value.(x2)[i]
    end
    return ℓ, u
end

# ---------------------------
# Big-M from bounds
# ---------------------------

"""
bigM_from_bounds(ℓ, u; scale, common)

If common=true:
    returns a single vector M with M_i ≥ max(|ℓ_i|, |u_i|), scaled by 'scale'.

If common=false:
    returns (Mminus, Mplus) with:
        Mminus_i ≥ max(0, -ℓ_i), Mplus_i ≥ max(0, u_i),
    both scaled by 'scale'.
"""
function bigM_from_bounds(ℓ::AbstractVector, u::AbstractVector;
                          scale::Real = 1.0,
                          common::Bool = true)
    @assert length(ℓ) == length(u)
    @assert scale ≥ 1.0 "scale should be ≥ 1.0 to keep Big-M safe"

    if common
        base = max.(abs.(ℓ), abs.(u))
        return scale .* base
    else
        Mminus = scale .* max.(0.0, -ℓ)
        Mplus  = scale .* max.(0.0,  u)
        return Mminus, Mplus
    end
end

# ---------------------------
# Simplex and box instances
# ---------------------------

"""
Simplex:  x ≥ 0,  eᵀx = 1.

Returns (A, b, H, h, M) with:
- A x ≤ b   encoding x ≥ 0 (so A = -I, b = 0),
- H x = h   encoding eᵀx = 1,
- M        from LP-based bounds.
"""
function simplex_instance(n::Int; optimizer=MosekTools.Optimizer)
    A = -Matrix{Float64}(I, n, n)
    b = zeros(Float64, n)
    H = ones(Float64, 1, n)
    h = [1.0]
    ℓ, u = bounds_via_LP(A, b; H=H, h=h, optimizer=optimizer)
    M    = bigM_from_bounds(ℓ, u; scale=1.0, common=true)
    return A, b, H, h, M
end

"""
Box: ℓ ≤ x ≤ u with integer bounds.

Returns (A, b, M, ℓ, u, xhat) where:
- A x ≤ b encodes ℓ ≤ x ≤ u,
- M is a Big-M vector from bounds,
- xhat is a random integer point in the box.
"""
function box_instance(n::Int; rng::AbstractRNG, low::Int=-8, high::Int=8, min_width::Int=3)
    @assert low < high
    ℓ = zeros(Int, n)
    u = zeros(Int, n)
    for i in 1:n
        a = rand(rng, low:(high-min_width))
        w = rand(rng, min_width:(high-a))
        ℓ[i] = a
        u[i] = a + w
    end
    A = vcat(Matrix{Int}(I, n, n), -Matrix{Int}(I, n, n))
    b = vcat(u, -ℓ)
    ℓf = Float64.(ℓ)
    uf = Float64.(u)
    M  = bigM_from_bounds(ℓf, uf; scale=1.0, common=true)
    xhat = [rand(rng, ℓ[i]:u[i]) for i in 1:n]
    return A, b, M, ℓ, u, xhat
end

# ---------------------------
# Quadratic rows with anchor
# ---------------------------

"""
Quadratic inequality:

    0.5 xᵀ Q x + qᵀ x + r ≤ 0

constructed so that g(x̄) = -δ < 0 at the anchor x̄, with δ > 0.

num_neg controls the number of negative eigenvalues of Q.
If num_neg = 0, Q is PSD (inequality convex in x).
"""
function make_qineq_with_anchor(
    rng::AbstractRNG,
    xbar::AbstractVector{<:Integer};
    num_neg::Int = 0,
    nonneg_vals::UnitRange{Int} = 0:10,
    neg_vals::UnitRange{Int} = -10:-1,
    qvals::UnitRange{Int} = -10:10,
    slack_lo::Int = 1,
    slack_hi::Int = 5,
)
    n = length(xbar)

    Q = rand_symm_from_eigs(rng, n;
                            num_neg     = num_neg,
                            nonneg_vals = nonneg_vals,
                            neg_vals    = neg_vals)
    q = rand_vec_int(rng, n; vals = qvals)

    g0 = 0.5 * (xbar' * Q * xbar) + dot(q, xbar)
    δ  = rand_slack(rng; lo = slack_lo, hi = slack_hi)

    r = -g0 - δ

    return Matrix{Float64}(Q), Float64.(q), Float64(r)
end

"""
Quadratic equality:

    0.5 xᵀ P x + pᵀ x + s = 0

constructed so that g(x̄) = 0 at the anchor x̄.

Any genuinely quadratic equality (with P ≠ 0) defines a nonconvex feasible set.
"""
function make_qeq_with_anchor(
    rng::AbstractRNG,
    xbar::AbstractVector{<:Integer};
    num_neg::Int = 0,
    nonneg_vals::UnitRange{Int} = 0:10,
    neg_vals::UnitRange{Int} = -10:-1,
    pvals::UnitRange{Int} = -3:3,
)
    n = length(xbar)

    P = rand_symm_from_eigs(rng, n;
                            num_neg     = num_neg,
                            nonneg_vals = nonneg_vals,
                            neg_vals    = neg_vals)
    p = rand_vec_int(rng, n; vals = pvals)

    g0 = 0.5 * (xbar' * P * xbar) + dot(p, xbar)
    s  = -g0

    return Matrix{Float64}(P), Float64.(p), Float64(s)
end

# ---------------------------
# Dict packer
# ---------------------------

function pack(; n::Int, rho::Real=3.0, Q0=nothing, q0=nothing,
               Qi=nothing, qi=nothing, ri=nothing,
               Pi=nothing, pi=nothing, si=nothing,
               A=nothing, b=nothing, H=nothing, h=nothing,
               M::Union{Nothing,AbstractVector}=nothing,
               seed::Union{Nothing,Int}=nothing,
               base_tag::AbstractString = "",
               convexity::AbstractString = "CVX",
               n_LI::Int = 0, n_LE::Int = 0,
               n_QI::Int = 0, n_QE::Int = 0,
               bigM_scale::Real = 1.0)
    if M === nothing
        M = ones(Float64, n)
    end
    d = Dict{String,Any}(
        "n"          => n,
        "rho"        => rho,
        "Q0"         => Q0,
        "q0"         => q0,
        "Qi"         => Qi,
        "qi"         => qi,
        "ri"         => ri,
        "Pi"         => Pi,
        "pi"         => pi,
        "si"         => si,
        "A"          => A,
        "b"          => b,
        "H"          => H,
        "h"          => h,
        "M"          => M,
        "base_tag"   => String(base_tag),
        "convexity"  => String(convexity),
        "n_LI"       => n_LI,
        "n_LE"       => n_LE,
        "n_QI"       => n_QI,
        "n_QE"       => n_QE,
        "bigM_scale" => bigM_scale,
    )
    if seed !== nothing
        d["seed"] = seed
    end
    return d
end

# ---------------------------
# Constraint-type helpers
# ---------------------------

has_lin_ineq(d::Dict{String,Any}) = get(d, "A", nothing) !== nothing
has_lin_eq(d::Dict{String,Any})   = get(d, "H", nothing) !== nothing
has_qineq(d::Dict{String,Any})    = get(d, "Qi", nothing) !== nothing
has_qeq(d::Dict{String,Any})      = get(d, "Pi", nothing) !== nothing

"""
Fallback convexity check (not used for tagging, but kept for sanity checks).
"""
function convex_tag_eig(d::Dict{String,Any}; tol::Real=1e-6)
    Q0 = get(d, "Q0", nothing)
    Qi = get(d, "Qi", nothing)
    Pi = get(d, "Pi", nothing)

    if (Qi !== nothing && length(Qi) > 0) || (Pi !== nothing && length(Pi) > 0)
        return "NCVX"
    end
    Q0 === nothing && return "NCVX"

    vals = eigvals(Symmetric(Matrix(Q0)))
    return minimum(vals) ≥ -tol ? "CVX" : "NCVX"
end

"Build human-readable instance id string, with counts and Big-M scale."
function make_id(d::Dict{String,Any})
    n    = Int(d["n"])
    rho  = Int(round(d["rho"]))
    base = get(d, "base_tag", "")

    has_counts = haskey(d, "n_LI") || haskey(d, "n_LE") ||
                 haskey(d, "n_QI") || haskey(d, "n_QE")
    n_LI = has_counts ? get(d, "n_LI", 0) : 0
    n_LE = has_counts ? get(d, "n_LE", 0) : 0
    n_QI = has_counts ? get(d, "n_QI", 0) : 0
    n_QE = has_counts ? get(d, "n_QE", 0) : 0

    convexity = get(d, "convexity", nothing)
    if convexity === nothing
        convexity = convex_tag_eig(d)
    end

    parts = String[]
    push!(parts, "n$(n)")
    push!(parts, "rho$(rho)")
    base != "" && push!(parts, String(base))

    if has_counts
        if n_LI > 0
            push!(parts, "LI$(n_LI)")
        elseif has_lin_ineq(d)
            push!(parts, "LI")
        end
        if n_LE > 0
            push!(parts, "LE$(n_LE)")
        elseif has_lin_eq(d)
            push!(parts, "LE")
        end
        if n_QI > 0
            push!(parts, "QI$(n_QI)")
        elseif has_qineq(d)
            push!(parts, "QI")
        end
        if n_QE > 0
            push!(parts, "QE$(n_QE)")
        elseif has_qeq(d)
            push!(parts, "QE")
        end
    else
        has_lin_ineq(d) && push!(parts, "LI")
        has_lin_eq(d)   && push!(parts, "LE")
        has_qineq(d)    && push!(parts, "QI")
        has_qeq(d)      && push!(parts, "QE")
    end

    push!(parts, String(convexity))

    bigM_scale = get(d, "bigM_scale", 1.0)
    if abs(bigM_scale - 1.0) > 1e-8
        k = round(bigM_scale; digits=6)
        if isapprox(k, round(k); atol=1e-8)
            push!(parts, "Mx$(Int(round(k)))")
        else
            push!(parts, "Mx$(k)")
        end
    end

    seed = get(d, "seed", nothing)
    if seed !== nothing
        push!(parts, "seed$(seed)")
    end

    return join(parts, "_")
end

# ---------------------------
# Build a small test suite
# ---------------------------

function build_instances(; objective_scale::Real=10.0, optimizer=MosekTools.Optimizer)
    insts = Vector{Dict{String,Any}}(undef, 8)

    # 1) Simplex (n=7), convex objective (num_neg=0)
    let i = 1, rng = MersenneTwister(1)
        is_cvx = true
        n   = 7
        rho = 4.0
        A, b, H, h, M = simplex_instance(n; optimizer=optimizer)

        num_neg_q0 = 0
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, H=H, h=h, M=M,
                  Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                  seed=i, base_tag="S", convexity=convexity)
        insts[i] = d
    end

    # 2) Box (n=4), convex objective (num_neg=0)
    let i = 2, rng = MersenneTwister(2)
        is_cvx = true
        n   = 4
        rho = 3.0
        A, b, M, ℓ, u, xhat = box_instance(n; rng=rng)

        num_neg_q0 = 0
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=convexity)
        insts[i] = d
    end

    # 3) Box + 2 linear equalities, convex objective
    let i = 3, rng = MersenneTwister(3)
        is_cvx = true
        n   = 15
        rho = 10.0
        A, b, M, ℓ, u, xhat = box_instance(n; rng=rng)
        H = rand(rng, -2:2, 2, n)
        h = H * xhat

        num_neg_q0 = 0
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, H=H, h=h, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=convexity)
        insts[i] = d
    end

    # 4) A-bounded minimal (positive spanning), convex objective
    let i = 4, rng = MersenneTwister(4)
        is_cvx = true
        n   = 5
        rho = 4.0
        A, b, xhat = build_bounded_A(n=n, mode=:minimal, rng=rng)
        ℓ, u = bounds_via_LP(A, b; optimizer=optimizer)
        M    = bigM_from_bounds(ℓ, u; scale=1.0, common=true)

        num_neg_q0 = 0
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="P", convexity=convexity)
        insts[i] = d
    end

    # 5) A-bounded symmetric, nonconvex objective (num_neg > 0)
    let i = 5, rng = MersenneTwister(5)
        is_cvx = true
        n   = 4
        rho = 3.0
        A, b, xhat = build_bounded_A(n=n, mode=:symmetric, rng=rng)
        ℓ, u = bounds_via_LP(A, b; optimizer=optimizer)
        M    = bigM_from_bounds(ℓ, u; scale=1.0, common=true)

        num_neg_q0 = 2
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="P", convexity=convexity)
        insts[i] = d
    end

    # 6) Box + one quadratic inequality (convex QI, nonconvex objective)
    let i = 6, rng = MersenneTwister(6)
        is_cvx = true
        n   = 4
        rho = 3.0
        A, b, M, ℓ, u, xhat = box_instance(n; rng=rng)

        # convex QI
        Q6, q6, r6 = make_qineq_with_anchor(rng, xhat; num_neg=0)

        # nonconvex objective
        num_neg_q0 = 1
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                 Qi=(Q6,), qi=(q6,), ri=(r6,),
                 M=M, Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=convexity)
        insts[i] = d
    end

    # 7) Box + one quadratic equality (always nonconvex unless linear)
    let i = 7, rng = MersenneTwister(7)
        is_cvx = false
        n   = 4
        rho = 3.0
        A, b, M, ℓ, u, xhat = box_instance(n; rng=rng)

        P7, p7, s7 = make_qeq_with_anchor(rng, xhat; num_neg=0)

        num_neg_q0 = 1
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                 Pi=(P7,), pi=(p7,), si=(s7,),
                 M=M, Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=convexity)
        insts[i] = d
    end

    # 8) Box + two q-inequalities + one q-equality + one linear equality
    let i = 8, rng = MersenneTwister(8)
        is_cvx = false
        n   = 10
        rho = 8.0
        A, b, M, ℓ, u, xhat = box_instance(n; rng=rng)

        # First QI: explicit small matrix
        Qqi1 = zeros(Float64, n, n)
        Qqi1[1,1] = 2.0; Qqi1[2,2] = 2.0; Qqi1[1,2] = -2.0; Qqi1[2,1] = -2.0
        qqi1 = zeros(Float64, n)
        τ1   = rand(rng, 0:2)
        g01  = 0.5 * (xhat' * Qqi1 * xhat) + dot(qqi1, xhat)
        rqi1 = -g01 - τ1

        # Second QI: eigenvalue-controlled
        Qqi2, qqi2, rqi2 = make_qineq_with_anchor(rng, xhat; num_neg=1)

        # One quadratic equality
        Pqe, pqe, sqe = make_qeq_with_anchor(rng, xhat; num_neg=0)

        # One linear equality Hx = h
        H = rand(rng, 1, n)
        h = H * xhat

        num_neg_q0 = 2
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                 Qi=(Qqi1, Qqi2), qi=(qqi1, qqi2), ri=(rqi1, rqi2),
                 Pi=(Pqe,), pi=(pqe,), si=(sqe,),
                 H=H, h=h, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=convexity)
        insts[i] = d
    end

    # 9) Take instance 1 and scale Big-M by a factor of 10
    base_idx = 1
    base = insts[base_idx]
    new_inst = deepcopy(base)
    new_inst["M"] = 10.0 .* base["M"]
    new_inst["bigM_scale"] = get(base, "bigM_scale", 1.0) * 10.0
    push!(insts, new_inst)

    # Assign IDs
    for d in insts
        d["id"] = make_id(d)
    end

    return insts
end

# ---------------------------
# JSON I/O
# ---------------------------

function save_instances_json(path::AbstractString, insts)
    open(path, "w") do io
        JSON.print(io, insts, 2)
    end
end

function load_instances_json(path::AbstractString)::Vector{Dict{String,Any}}
    return JSON.parsefile(path)
end

end # module InstanceGen




###############################
# UserInstanceGen
###############################

module UserInstanceGen

using Random, LinearAlgebra
using JuMP, MosekTools

const MOI = JuMP.MOI

export generate_instance

# ============================
# Basic random helpers
# ============================

"Random integer vector of length n with entries in `vals`."
rand_vec_int(rng::AbstractRNG, n::Int; vals = -5:5) = rand(rng, vals, n)

"Random positive integer slack δ in [lo, hi]."
rand_slack(rng::AbstractRNG; lo::Int = 1, hi::Int = 5) = rand(rng, lo:hi)

"""
Random symmetric matrix with prescribed numbers of negative eigenvalues.

Builds Q = Qmat * Diag(eigs) * Qmat', where:
- `num_neg` eigenvalues are drawn from `neg_vals` (negative integers),
- the remaining eigenvalues are drawn from `nonneg_vals` (nonnegative integers).
"""
function rand_symm_from_eigs(
    rng::AbstractRNG,
    n::Int;
    num_neg::Int,
    nonneg_vals::UnitRange{Int} = 0:5,
    neg_vals::UnitRange{Int} = -5:-1,
)
    @assert 0 ≤ num_neg ≤ n "num_neg must be between 0 and n"
    num_nonneg = n - num_neg

    negs    = num_neg == 0    ? Int[] : rand(rng, neg_vals,    num_neg)
    nonnegs = num_nonneg == 0 ? Int[] : rand(rng, nonneg_vals, num_nonneg)

    eigs = vcat(negs, nonnegs)
    eigs = eigs[randperm(rng, length(eigs))]

    R = randn(rng, n, n)
    F = qr(R)
    Qmat = Matrix(F.Q)
    Λ = Diagonal(Float64.(eigs))

    return Qmat * Λ * Qmat'
end

"Random quadratic objective: 0.5 x'Q0 x + q0'x, with controlled negative eigenvalues."
function rand_objective(
    rng::AbstractRNG,
    n::Int;
    num_neg::Int = 0,
    nonneg_vals::UnitRange{Int} = 0:5,
    neg_vals::UnitRange{Int} = -5:-1,
    qvals::UnitRange{Int} = -5:5,
    scale_Q::Real = 1.0,
    scale_q::Real = 1.0,
)
    Q0 = rand_symm_from_eigs(rng, n;
                             num_neg     = num_neg,
                             nonneg_vals = nonneg_vals,
                             neg_vals    = neg_vals)
    q0 = rand_vec_int(rng, n; vals = qvals)
    return scale_Q .* Matrix{Float64}(Q0), scale_q .* Float64.(q0)
end


# ============================
# Big-M from bounds
# ============================

"""
Compute Big-M vectors from componentwise bounds ell ≤ x ≤ u.

- M[i]       = scale * max(|ell[i]|, |u[i]|)
- M_minus[i] = scale * max(0, -ell[i])
- M_plus[i]  = scale * max(0,  u[i])
"""
function bigM_from_bounds(ell::AbstractVector, u::AbstractVector; scale::Real = 1.0)
    @assert length(ell) == length(u)
    @assert scale ≥ 1.0 "scale should be ≥ 1.0."

    base   = max.(abs.(ell), abs.(u))
    M      = scale .* base
    Mminus = scale .* max.(0.0, -ell)
    Mplus  = scale .* max.(0.0,  u)

    return M, Mminus, Mplus
end

# ============================
# Anchor generators
# ============================

"Generate a sparse anchor xbar with ‖xbar‖₀ ≤ rho in R^n."
function generate_sparse_anchor(
    rng::AbstractRNG,
    n::Int,
    rho::Int;
    vals::UnitRange{Int} = -3:3,
)
    x = zeros(Float64, n)
    rho == 0 && return x

    k = rand(rng, 1:rho)            # support size
    idx = randperm(rng, n)[1:k]
    for i in idx
        v = 0
        while v == 0
            v = rand(rng, vals)
        end
        x[i] = Float64(v)
    end
    return x
end

"Generate an anchor xbar on the unit simplex with support size at most rho."
function generate_simplex_anchor(rng::AbstractRNG, n::Int, rho::Int)
    @assert rho ≥ 1 "On the unit simplex we need rho ≥ 1 (at least one nonzero entry)."
    s = rand(rng, 1:min(rho, n))    # support size
    idx = randperm(rng, n)[1:s]

    x = zeros(Float64, n)
    vals = rand(rng, s)
    vals ./= sum(vals)

    for (j, i) in enumerate(idx)
        x[i] = vals[j]
    end
    return x
end

# ============================
# Base linear bounded regions
# ============================

"""
Unit simplex base:

    x >= 0,  e'x = 1.

Returns:
- A, b for x >= 0 encoded as -I x <= 0,
- H, h for e'x = 1.
"""

function build_simplex_bounds(n::Int)
    A   = -Matrix{Float64}(I, n, n)
    b   = zeros(Float64, n)
    H   = ones(Float64, 1, n)
    h   = [1.0]
    return A, b, H, h
end

"""
Box base: a random box around xbar.

We choose random positive margins around each coordinate of xbar so that
    ell[i] < xbar[i] < u[i],
and encode ell <= x <= u as [I; -I] x <= [u; -ell].
"""

function build_box_bounds(
    rng::AbstractRNG,
    xbar::AbstractVector;
    min_margin::Real = 1.0,
    max_margin::Real = 3.0,
)
    n = length(xbar)
    ell = zeros(Float64, n)
    u   = zeros(Float64, n)

    for i in 1:n
        m_low = rand(rng) * (max_margin - min_margin) + min_margin
        m_up  = rand(rng) * (max_margin - min_margin) + min_margin
        ell[i] = xbar[i] - m_low
        u[i]   = xbar[i] + m_up
    end

    A = vcat(Matrix{Float64}(I, n, n), -Matrix{Float64}(I, n, n))
    b = vcat(u, -ell)

    return A, b
end


"""
General polytope base using Todd-style construction.

We build a bounded polytope of the form
    P = { x : A x ≤ b }
such that:
  - rows(B) are unbiased directions on the sphere (Todd, 1991),
  - the last row is a normalized negative conic combination of rows(B),
  - xbar is strictly feasible: A * xbar < b.

Returns:
- A, b for Ax ≤ b with x̄ strictly feasible.
"""
function build_polytope_bounds(
    rng::AbstractRNG,
    xbar::AbstractVector;
    slack_min::Real = 1.0,
    slack_max::Real = 3.0,
    max_rank_tries::Int = 50,
)
    n = length(xbar)
    @assert slack_min > 0 "slack_min must be > 0."
    @assert slack_max ≥ slack_min "slack_max must be ≥ slack_min."

    # --------------------------------------------------
    # 1) Todd-style B: unbiased row directions + rank(B) = n
    # --------------------------------------------------
    B = zeros(Float64, n, n)
    got_full_rank = false

    for _try in 1:max_rank_tries
        for i in 1:n
            g = randn(rng, n)          # g_i ~ N(0, I_n)
            a = g / norm(g)            # normalize ⇒ a ∈ S^{n-1}
            B[i, :] .= a               # row i of B
        end

        if rank(B) == n
            got_full_rank = true
            break
        end
    end

    if !got_full_rank
        error("build_polytope_bounds: could not generate full-rank B after $max_rank_tries tries.")
    end

    # --------------------------------------------------
    # 2) Positive vector w ∈ Rⁿ_{++}
    # --------------------------------------------------
    # Example: w_i ∈ (1,2), so strictly positive and O(1) scaled.
    w = 1 .+ rand(rng, n)

    # --------------------------------------------------
    # 3) Last row r = normalized ( - wᵀ B )
    #
    # r0 = - wᵀ B  (row combination of rows(B) with positive weights)
    # normalize ⇒ r = r0 / ‖r0‖
    # still r = - ŵᵀ B with ŵ > 0 (just rescaled).
    # --------------------------------------------------
    r0_row = -(w' * B)          # 1×n row, equals -wᵀ B
    norm_r0 = norm(r0_row)
    @assert norm_r0 > 0 "Norm of -wᵀB is zero, something is wrong."
    r_row = r0_row ./ norm_r0   # 1×n row with ‖r_row‖₂ = 1

    # Stack rows: A = [B; r_row]
    A = vcat(B, Array(r_row))   # (n+1)×n

    # --------------------------------------------------
    # 4) RHS: b = A * xbar + [s; δ], with s > 0, δ > 0
    # so that xbar is strictly feasible.
    # --------------------------------------------------
    s = slack_min .+ (slack_max - slack_min) .* rand(rng, n)   # s ∈ Rⁿ_{++}
    δ = slack_min + (slack_max - slack_min) * rand(rng)        # δ > 0

    shift = vcat(s, [δ])
    b = A * xbar + shift

    return A, b
end

# ============================
# Vertex LP helpers
# ============================

"""
Solve a small LP to get a vertex v of {x : A x <= b, H x = h}
with a random linear objective c'x.

Returns (v, c).
"""
function find_vertex_lp(
    rng::AbstractRNG,
    A::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    H::AbstractMatrix{<:Real},
    h::AbstractVector{<:Real};
    optimizer = MosekTools.Optimizer,
)
    m, n = size(A)
    @assert length(b) == m
    @assert size(H, 2) == n || size(H, 1) == 0

    model = Model(optimizer)
    @variable(model, x[1:n])

    if m > 0
        @constraint(model, A * x .<= b)
    end
    if size(H, 1) > 0
        @constraint(model, H * x .== h)
    end

    # random nonzero objective c
    c_int = rand(rng, -2:2, n)
    while all(c_int .== 0)
        c_int = rand(rng, -2:2, n)
    end
    c = Float64.(c_int)

    @objective(model, Min, dot(c, x))
    optimize!(model)

    status = termination_status(model)
    @assert status == MOI.OPTIMAL "LP for vertex is not optimal: $status"

    v = [value(x[i]) for i in 1:n]
    return v, c
end

"Given xbar, v, and a normal a, build a halfspace a'x <= β that keeps xbar and cuts off v."
function build_cut_halfspace_from_normal(
    xbar::AbstractVector,
    v::AbstractVector,
    a::AbstractVector;
    θ::Real = 0.5,
)
    n = length(xbar)
    @assert length(v) == n
    @assert length(a) == n

    αx = dot(a, xbar)
    αv = dot(a, v)

    if isapprox(αx, αv; atol=1e-8)
        error("Cannot separate: projections coincide (a'xbar ≈ a'v).")
    end

    # ensure αv > αx by flipping sign if needed
    if αv ≤ αx
        a = -a
        αx = -αx
        αv = -αv
    end

    if αv ≤ αx + 1e-8
        error("Cannot separate safely: a'v not strictly larger than a'xbar.")
    end

    β = αx + θ * (αv - αx)

    return a, β, αx, αv
end

"Given xbar, v and normal a, build an equality a'x = β passing through xbar but not v."
function build_equality_from_normal(
    xbar::AbstractVector,
    v::AbstractVector,
    a::AbstractVector,
)
    n = length(xbar)
    @assert length(v) == n
    @assert length(a) == n

    β = dot(a, xbar)
    αv = dot(a, v)

    if isapprox(β, αv; atol=1e-8)
        error("Cannot build nonredundant equality: a'xbar ≈ a'v.")
    end

    return a, β, β, αv
end

# ============================
# Extra linear constraints
# ============================

"""
Add n_extra linear inequalities via vertex cuts (nonredundant, keep xbar feasible),
using the objective vector c of the vertex LP as the normal direction.
"""
function add_extra_LI_vertex_cuts!(
    rng::AbstractRNG,
    A::Matrix{Float64},
    b::Vector{Float64},
    H::Matrix{Float64},
    h::Vector{Float64},
    xbar::AbstractVector,
    n_extra::Int;
    optimizer = MosekTools.Optimizer,
)
    n = length(xbar)
    for k in 1:n_extra
        println()
        println("---- Extra linear inequality $k / $n_extra ----")

        # 1) find a vertex and its objective direction c
        v, c = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
        println("Vertex v:")
        println(v)

        # avoid degenerate case v ≈ xbar
        if norm(v .- xbar, Inf) < 1e-6
            println("Vertex is too close to anchor; trying alternative random objectives.")
            found = false
            for _try in 1:10
                v2, c2 = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
                if norm(v2 .- xbar, Inf) ≥ 1e-6
                    v = v2
                    c = c2
                    found = true
                    println("Replaced vertex with:")
                    println(v)
                    break
                end
            end
            if !found
                @warn "Polytope seems to have collapsed to xbar; cannot add more nonredundant LI."
                return A, b
            end
        end

        # 2) build halfspace using a = c (may be flipped internally)
        try
            a, β, αx, αv = build_cut_halfspace_from_normal(xbar, v, c; θ = 0.5)

            println("New inequality: a' x ≤ β")
            println("  a = ", a)
            println("  β = ", β)
            println("  a' x̄ = ", αx, "   (should be < β)")
            println("  a' v  = ", αv, "   (should be > β)")
            println("  a' x̄ - β = ", αx - β)
            println("  a' v  - β = ", αv - β)

            # 3) append to A, b
            A = vcat(A, a')
            b = vcat(b, β)
        catch err
            @warn "Failed to construct separating inequality from c; skipping this LI. Error: $err"
        end
    end

    return A, b
end

"""
Add n_LE linear equalities of the form a'x = β using vertex + objective vector.
"""
function add_extra_LE!(
    rng::AbstractRNG,
    A::Matrix{Float64},
    b::Vector{Float64},
    H::Matrix{Float64},
    h::Vector{Float64},
    xbar::AbstractVector,
    n_LE::Int;
    optimizer = MosekTools.Optimizer,
)
    n = length(xbar)
    if n_LE == 0
        return H, h
    end

    for k in 1:n_LE
        println()
        println("---- Extra linear equality $k / $n_LE ----")

        v, c = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
        println("Vertex v used for equality:")
        println(v)

        # avoid degenerate case v ≈ xbar or cᵀv ≈ cᵀx̄
        ok = false
        for _try in 1:10
            if norm(v .- xbar, Inf) < 1e-6 || isapprox(dot(c, v), dot(c, xbar); atol = 1e-8)
                v2, c2 = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
                v, c = v2, c2
            else
                ok = true
                break
            end
        end
        if !ok
            @warn "Could not find a vertex with distinct projection for equality; skipping this LE."
            continue
        end

        try
            a, β, αx, αv = build_equality_from_normal(xbar, v, c)

            println("New equality: a' x = β")
            println("  a = ", a)
            println("  β = ", β)
            println("  a' x̄ = ", αx, "   (x̄ lies on the hyperplane)")
            println("  a' v  = ", αv, "   (vertex does NOT lie on the hyperplane)")
            println("  a' x̄ - β = ", αx - β)
            println("  a' v  - β = ", αv - β)

            H = vcat(H, a')
            h = vcat(h, β)
        catch err
            @warn "Failed to construct separating equality from c; skipping this LE. Error: $err"
        end
    end

    return H, h
end

"Random orthogonal matrix in R^{n×n}."
function random_orthogonal(rng::AbstractRNG, n::Int)
    R = randn(rng, n, n)
    F = qr(R)
    return Matrix(F.Q)
end

"""
Build one quadratic inequality

    g(x) = 0.5 x'Q x + q'x + r <= 0

from a *given* vertex set:

- Anchor x̄ and all `feas_verts` are strictly feasible: g(x) <= -eps_feas
- `v_cut` is strictly cut: g(v_cut) >= eps_cut

Eigenpattern is controlled by (want_convex, num_neg):

- want_convex = true  => all lambda_i >= 0, num_neg must be 0
- want_convex = false => choose num_neg indices with lambda_i <= -eig_eps,
                         others lambda_j >= 0

All constraints are linear in (lambda, q, r), so this is an LP.
"""
function build_QI_from_vertex_set(
    rng::AbstractRNG,
    xbar::AbstractVector,
    feas_verts::Vector{Vector{Float64}},
    v_cut::Vector{Float64};
    want_convex::Bool,
    num_neg::Int,
    eps_feas::Real = 0.5,
    eps_cut::Real  = 0.5,
    eig_eps::Real  = 0.1,
    coef_bound::Real = 5.0,
    optimizer = MosekTools.Optimizer,
)
    n = length(xbar)
    @assert length(v_cut) == n "v_cut must have length n"
    for v in feas_verts
        @assert length(v) == n "All feasible vertices must have length n"
    end

    @assert 0 ≤ num_neg ≤ n "build_QI_from_vertex_set: num_neg must be between 0 and n."
    if want_convex
        @assert num_neg == 0 "In convex mode, num_neg must be 0 for QI."
    end

    # Random orthogonal basis
    Qmat = random_orthogonal(rng, n)

    model = Model(optimizer)
    @variable(model, lambda[1:n])
    @variable(model, q[1:n])
    @variable(model, r)

    # Eigenvalue pattern
    if want_convex
        @constraint(model, lambda .>= 0.0)
    else
        neg_idx = num_neg == 0 ? Int[] : randperm(rng, n)[1:num_neg]
        pos_idx = setdiff(1:n, neg_idx)
        for i in neg_idx
            @constraint(model, lambda[i] <= -eig_eps)
        end
        for j in pos_idx
            @constraint(model, lambda[j] >= 0.0)
        end
    end

    # Coefficient bounds
    for i in 1:n
        @constraint(model, -coef_bound <= lambda[i] <= coef_bound)
        @constraint(model, -coef_bound <= q[i]      <= coef_bound)
    end
    @constraint(model, -coef_bound <= r <= coef_bound)

    # Helper: add g(x) (<= or >= rhs)
    function add_qi_constraint!(x::AbstractVector, sense::Symbol, rhs::Real)
        @assert length(x) == n
        y = Qmat' * x
        if sense === :le
            @constraint(
                model,
                sum(0.5 * (y[i]^2) * lambda[i] for i in 1:n) +
                sum(x[j] * q[j] for j in 1:n) + r <= rhs
            )
        elseif sense === :ge
            @constraint(
                model,
                sum(0.5 * (y[i]^2) * lambda[i] for i in 1:n) +
                sum(x[j] * q[j] for j in 1:n) + r >= rhs
            )
        else
            error("Unsupported sense $sense in add_qi_constraint! (expected :le or :ge)")
        end
    end

    # Anchor + feasible vertices: strictly feasible
    add_qi_constraint!(xbar, :le, -eps_feas)
    for v in feas_verts
        add_qi_constraint!(v, :le, -eps_feas)
    end

    # Cut vertex: strictly infeasible
    add_qi_constraint!(v_cut, :ge, eps_cut)

    @objective(model, Min, 0.0)
    optimize!(model)
    status = termination_status(model)
    @assert status == MOI.OPTIMAL "build_QI_from_vertex_set: LP not solved to OPTIMAL, status = $status"

    λ_opt = value.(lambda)
    q_opt = value.(q)
    r_opt = value(r)

    # Build Q = Qmat * Diag(λ) * Qmat'
    Q = Qmat * Diagonal(λ_opt) * Qmat'
    Q = 0.5 * (Q + Q')  # symmetrize

    return Q, Vector{Float64}(q_opt), Float64(r_opt)
end

"""
Collect k distinct vertices of {x : A x <= b, H x = h} via repeated LP solves.
Anchor xbar is excluded from the vertex set (we keep it separate).

Throws an error if not enough distinct vertices can be found.
"""
function collect_vertices_for_quad(
    rng::AbstractRNG,
    A::Matrix{Float64},
    b::Vector{Float64},
    H::Matrix{Float64},
    h::Vector{Float64},
    xbar::AbstractVector;
    k_vertices::Int = 3,
    max_tries_per_vertex::Int = 20,
    tol::Real = 1e-6,
    optimizer = MosekTools.Optimizer,
)
    verts = Vector{Vector{Float64}}()

    while length(verts) < k_vertices
        found = false
        for _try in 1:max_tries_per_vertex
            v, _ = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)

            # distinct from anchor and existing vertices
            if norm(v .- xbar, Inf) < tol
                continue
            end
            if any(norm(v .- w, Inf) < tol for w in verts)
                continue
            end

            push!(verts, v)
            found = true
            break
        end

        if !found
            error("collect_vertices_for_quad: could not find enough distinct vertices.")
        end
    end

    return verts
end


"""
Build a quadratic inequality

    g(x) = 0.5 x'Q x + q'x + r <= 0

that cuts exactly one vertex from the given vertex set.

Algorithm:
  1. Sample Q and q with the requested number of negative eigenvalues (num_neg).
  2. Evaluate phi(x) = 0.5 x'Q x + q'x at xbar and at all vertices.
  3. Require that xbar is not the maximizer and that there is a positive gap
     between the largest and second largest values.
  4. Set r so that the unique maximizer vertex is cut (g > 0) and all others
     (including xbar) are strictly feasible (g < 0).

Returns (Q, q, r, cut_idx_local), where cut_idx_local is in 1:length(verts).
"""
function build_QI_cut_max_vertex(
    rng::AbstractRNG,
    xbar::AbstractVector,
    verts::Vector{Vector{Float64}};
    want_convex::Bool,
    num_neg::Int,
    nonneg_vals::UnitRange{Int} = 0:5,
    neg_vals::UnitRange{Int}    = -5:-1,
    qvals::UnitRange{Int}       = -5:5,
    scale_Q::Real               = 1.0,
    scale_q::Real               = 1.0,
    gap_tol::Real               = 1e-4,
    max_tries::Int              = 50,
)
    n = length(xbar)
    k = length(verts)
    @assert k ≥ 1 "build_QI_cut_max_vertex: need at least one vertex."
    @assert 0 ≤ num_neg ≤ n "build_QI_cut_max_vertex: num_neg must be between 0 and n."

    if want_convex
        @assert num_neg == 0 "In convex mode, num_neg must be 0 for QI."
    end

    for attempt in 1:max_tries
        # 1) Sample Q and q with given eigenpattern
        Q_raw = rand_symm_from_eigs(
            rng, n;
            num_neg     = num_neg,
            nonneg_vals = nonneg_vals,
            neg_vals    = neg_vals,
        )
        Q = scale_Q .* Matrix{Float64}(Q_raw)
        q = scale_q .* Float64.(rand_vec_int(rng, n; vals = qvals))

        # 2) Evaluate phi(x) = 0.5 x'Qx + q'x on {xbar} ∪ verts
        phi_xbar  = 0.5 * (xbar' * Q * xbar) + dot(q, xbar)
        phi_verts = [0.5 * (v' * Q * v) + dot(q, v) for v in verts]

        # all_vals[1] corresponds to xbar,
        # all_vals[j+1] corresponds to verts[j]
        all_vals   = vcat(phi_xbar, phi_verts)
        idx_sorted = sortperm(all_vals; rev = true)

        j1 = idx_sorted[1]  # index of maximum in all_vals
        j2 = idx_sorted[2]  # index of second maximum

        # xbar must NOT be the maximizer
        if j1 == 1
            continue
        end

        val1 = all_vals[j1]
        val2 = all_vals[j2]

        # Require a nontrivial gap
        if val1 - val2 ≤ gap_tol
            continue
        end

        # Threshold t = average of top two values
        t = 0.5 * (val1 + val2)
        r = -t

        # Local index of cut vertex in `verts`
        cut_idx_local = j1 - 1  # because all_vals[1] = xbar

        return Matrix{Float64}(Q), Vector{Float64}(q), Float64(r), cut_idx_local
    end

    error("build_QI_cut_max_vertex: could not find suitable (Q, q) after $max_tries attempts.")
end


"""
Build one quadratic equality

    h(x) = 0.5 x'P x + p'x + s = 0

from a *given* vertex set:

- Anchor x̄ and all `eq_verts` satisfy h(x) = 0
- `v_cut` is strictly on one side: h(v_cut) ≥ eps_cut or ≤ -eps_cut

No eigenvalue pattern imposed on λ (P can be indefinite).
"""
function build_QE_from_vertex_set(
    rng::AbstractRNG,
    xbar::AbstractVector,
    eq_verts::Vector{Vector{Float64}},
    v_cut::Vector{Float64};
    eps_cut::Real = 0.5,
    coef_bound::Real = 5.0,
    optimizer = MosekTools.Optimizer,
)
    n = length(xbar)
    @assert length(v_cut) == n "v_cut must have length n"
    for v in eq_verts
        @assert length(v) == n "All equality vertices must have length n"
    end

    # Random orthogonal basis
    Qmat = random_orthogonal(rng, n)

    model = Model(optimizer)
    @variable(model, lambda[1:n])
    @variable(model, p[1:n])
    @variable(model, s)

    # Coefficient bounds
    for i in 1:n
        @constraint(model, -coef_bound <= lambda[i] <= coef_bound)
        @constraint(model, -coef_bound <= p[i]      <= coef_bound)
    end
    @constraint(model, -coef_bound <= s <= coef_bound)

    # Helper: add h(x) == 0 or ≤/≥ rhs
    function add_qe_constraint!(x::AbstractVector, sense::Symbol, rhs::Real)
        @assert length(x) == n
        y = Qmat' * x
        if sense === :eq
            @constraint(model,
                sum(0.5 * (y[i]^2) * lambda[i] for i in 1:n) +
                sum(x[j] * p[j] for j in 1:n) + s == rhs
            )
        elseif sense === :le
            @constraint(model,
                sum(0.5 * (y[i]^2) * lambda[i] for i in 1:n) +
                sum(x[j] * p[j] for j in 1:n) + s <= rhs
            )
        elseif sense === :ge
            @constraint(model,
                sum(0.5 * (y[i]^2) * lambda[i] for i in 1:n) +
                sum(x[j] * p[j] for j in 1:n) + s >= rhs
            )
        else
            error("Unsupported sense $sense in add_qe_constraint! (expected :eq, :le, :ge)")
        end
    end

    # Equality points: anchor + eq_verts
    add_qe_constraint!(xbar, :eq, 0.0)
    for v in eq_verts
        add_qe_constraint!(v, :eq, 0.0)
    end

    # Cut vertex: choose side randomly
    if rand(rng) < 0.5
        add_qe_constraint!(v_cut, :ge, eps_cut)
    else
        add_qe_constraint!(v_cut, :le, -eps_cut)
    end

    @objective(model, Min, 0.0)
    optimize!(model)
    status = termination_status(model)
    @assert status == MOI.OPTIMAL "build_QE_from_vertex_set: LP not solved to OPTIMAL, status = $status"

    λ_opt = value.(lambda)
    p_opt = value.(p)
    s_opt = value(s)

    P = Qmat * Diagonal(λ_opt) * Qmat'
    P = 0.5 * (P + P')   # symmetrize

    return P, Vector{Float64}(p_opt), Float64(s_opt)
end


"""
Build all quadratic inequalities and equalities from a precomputed vertex set,
using a sequential cut-and-remove rule.

Inputs:
- rng, xbar
- verts      : vector of K vertices in R^n; we require K = n_QI + n_QE.
- n_QI, n_QE : number of quadratic inequalities / equalities
- want_convex: global convexity flag (QE only allowed if false)
- neg_QI     : length-n_QI vector; neg_QI[i] = number of negative eigenvalues in QI_i

Rule:
- Maintain an active index set A ⊆ {1,...,K}.
- For each QI_i:
    * consider the active vertices,
    * build QI_i via build_QI_cut_max_vertex so that exactly one active vertex
      (the maximizer of phi) is cut and all others (and xbar) are strictly feasible,
    * remove that cut vertex from A.
- For each QE_j:
    * pick one remaining active vertex,
    * build a QE that has h(xbar) = 0 and cuts that vertex on one side,
    * remove that vertex from A.
"""
function build_quadratics_from_vertices(
    rng::AbstractRNG,
    xbar::AbstractVector,
    verts::Vector{Vector{Float64}};
    n_QI::Int,
    n_QE::Int,
    want_convex::Bool,
    neg_QI::Vector{Int},
    eps_feas::Real    = 0.5,   # kept for compatibility (not used by QI now)
    eps_cut::Real     = 0.5,
    eig_eps::Real     = 0.1,   # kept for compatibility (not used by QI now)
    coef_bound::Real  = 5.0,
    optimizer = MosekTools.Optimizer,
)
    K = length(verts)
    @assert K == n_QI + n_QE "build_quadratics_from_vertices: expected n_QI + n_QE = $(n_QI + n_QE) vertices, got $K."
    n = length(xbar)
    for v in verts
        @assert length(v) == n "All vertices in `verts` must have length n."
    end

    Qi_list = Matrix{Float64}[]
    qi_list = Vector{Vector{Float64}}()
    ri_list = Float64[]

    Pi_list = Matrix{Float64}[]
    pi_list = Vector{Vector{Float64}}()
    si_list = Float64[]

    # Active indices into `verts`, shrunk after each quadratic constraint
    active = collect(1:K)

    # ---- Quadratic inequalities: new max-vertex construction ----
    if n_QI > 0
        @assert length(neg_QI) == n_QI
        for i in 1:n_QI
            @assert !isempty(active) "No active vertices left for quadratic inequality $i."

            # Current active vertices (coordinates)
            verts_active = [verts[j] for j in active]

            num_neg   = neg_QI[i]
            qi_convex = (num_neg == 0)
            if want_convex
                @assert qi_convex "In convex mode, QI $i must be PSD (num_neg = 0)."
            end

            Q_i, q_i, r_i, cut_local = build_QI_cut_max_vertex(
                rng,
                xbar,
                verts_active;
                want_convex = qi_convex,
                num_neg     = num_neg,
                # you can expose further keyword args here if you want
            )

            push!(Qi_list, Q_i)
            push!(qi_list, q_i)
            push!(ri_list, r_i)

            # Map local index back to global index and remove it from active set
            cut_global = active[cut_local]
            deleteat!(active, cut_local)
        end
    end

    # ---- Quadratic equalities: keep old LP-based construction ----
    if n_QE > 0
        @assert !want_convex "Quadratic equalities are only allowed in nonconvex mode."
        for j in 1:n_QE
            @assert !isempty(active) "No active vertices left for quadratic equality $j."

            # Seçilecek kesilecek vertex
            cut_idx = active[1]
            v_cut   = verts[cut_idx]

            # Diğer aktif vertexler bu QE için h(x) = 0 sağlayacak
            other_active = setdiff(active, [cut_idx])
            eq_verts = [verts[k] for k in other_active]

            P_j, p_j, s_j = build_QE_from_vertex_set(
                rng, xbar, eq_verts, v_cut;
                eps_cut    = eps_cut,
                coef_bound = coef_bound,
                optimizer  = optimizer,
            )

            push!(Pi_list, P_j)
            push!(pi_list, p_j)
            push!(si_list, s_j)

            # Artık cut edilen vertex aktif değil, diğerleri kalıyor
            active = other_active
        end
    end


    return Qi_list, qi_list, ri_list, Pi_list, pi_list, si_list
end


# ============================
# Spectral control for Q0 and QI
# ============================

"""
Build (neg_obj, neg_QI) from user-specified neg_eig_counts (or defaults),
and check consistency with convex / nonconvex mode and dimension.
"""
function build_neg_eig_counts(
    n::Int,
    n_QI::Int,
    n_QE::Int,
    want_convex::Bool,
    neg_eig_counts::Union{Nothing,Vector{Int}},
)
    # No vector provided: choose defaults
    if neg_eig_counts === nothing
        if want_convex
            return 0, fill(0, n_QI)
        else
            if n ≥ 2
                # introduce nonconvexity via the objective
                return 1, fill(0, n_QI)
            else
                # n = 1; nonconvexity must come from QE (if any).
                if n_QE > 0
                    return 0, fill(0, n_QI)
                else
                    error("Cannot realize a nonconvex instance in dimension n=1 without QE or explicit neg_eig_counts.")
                end
            end
        end
    end

    counts = neg_eig_counts::Vector{Int}
    expected_len = 1 + n_QI
    @assert length(counts) == expected_len "neg_eig_counts must have length 1 + n_QI."

    for k in counts
        @assert 0 ≤ k < n "Each entry in neg_eig_counts must satisfy 0 ≤ k < n."
    end

    neg_obj = counts[1]
    neg_QI  = counts[2:end]

    if want_convex
        @assert all(counts .== 0) "In convex mode, all entries of neg_eig_counts must be zero."
    else
        if all(counts .== 0) && n_QE == 0
            error("In nonconvex mode, at least one entry of neg_eig_counts must be positive (or n_QE > 0).")
        end
    end

    return neg_obj, neg_QI
end

# ============================
# Bounds (for Big-M)
# ============================

"""
Compute coordinate-wise bounds (ell, u) of the linear region:

    P_lin := { x : A x <= b, H x = h }.

Used both for linear-only and for Big-M (quadratic constraints ignored).
"""
function bounds_from_linear_part(
    A::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    H::AbstractMatrix{<:Real},
    h::AbstractVector{<:Real};
    optimizer = MosekTools.Optimizer,
)
    m, n = size(A)
    @assert length(b) == m
    @assert size(H, 2) == n || size(H, 1) == 0

    use_eq = (size(H, 1) > 0)

    ell = zeros(Float64, n)
    u   = zeros(Float64, n)

    for i in 1:n
        # lower bound on x[i]
        model = Model(optimizer)
        @variable(model, x[1:n])
        if m > 0
            @constraint(model, A * x .<= b)
        end
        if use_eq
            @constraint(model, H * x .== h)
        end
        @objective(model, Min, x[i])
        optimize!(model)
        status = termination_status(model)
        @assert status == MOI.OPTIMAL "Lower-bound LP for x[$i] not optimal: $status"
        ell[i] = value(x[i])

        # upper bound on x[i]
        model = Model(optimizer)
        @variable(model, x2[1:n])
        if m > 0
            @constraint(model, A * x2 .<= b)
        end
        if use_eq
            @constraint(model, H * x2 .== h)
        end
        @objective(model, Max, x2[i])
        optimize!(model)
        status = termination_status(model)
        @assert status == MOI.OPTIMAL "Upper-bound LP for x[$i] not optimal: $status"
        u[i] = value(x2[i])
    end

    return ell, u
end

"""
(Kept for completeness but no longer used in Big-M.)

Compute bounds including convex quadratic inequalities, if needed in future.
"""
function bounds_with_convex_quadratics(
    A::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    H::AbstractMatrix{<:Real},
    h::AbstractVector{<:Real},
    Qi_list::Vector{Matrix{Float64}},
    qi_list::Vector{Vector{Float64}},
    ri_list::Vector{Float64},
    convex_qi_indices::Vector{Int};
    optimizer = MosekTools.Optimizer,
)
    m, n = size(A)
    @assert length(b) == m
    @assert size(H, 2) == n || size(H, 1) == 0

    use_eq = (size(H, 1) > 0)

    ell = zeros(Float64, n)
    u   = zeros(Float64, n)

    for i in 1:n
        # ----- lower bound: min x[i] -----
        model = Model(optimizer)
        @variable(model, x[1:n])

        if m > 0
            @constraint(model, A * x .<= b)
        end
        if use_eq
            @constraint(model, H * x .== h)
        end

        for j in convex_qi_indices
            Q = Qi_list[j]
            q = qi_list[j]
            r = ri_list[j]
            @constraint(model,
                0.5 * sum(Q[p, q_] * x[p] * x[q_] for p in 1:n, q_ in 1:n) +
                sum(q[p] * x[p] for p in 1:n) +
                r <= 0.0
            )
        end

        @objective(model, Min, x[i])
        optimize!(model)
        status = termination_status(model)
        @assert status == MOI.OPTIMAL "Lower-bound QP for x[$i] not optimal: $status"
        ell[i] = value(x[i])

        # ----- upper bound: max x[i] -----
        model = Model(optimizer)
        @variable(model, x2[1:n])

        if m > 0
            @constraint(model, A * x2 .<= b)
        end
        if use_eq
            @constraint(model, H * x2 .== h)
        end

        for j in convex_qi_indices
            Q = Qi_list[j]
            q = qi_list[j]
            r = ri_list[j]
            @constraint(model,
                0.5 * sum(Q[p, q_] * x2[p] * x2[q_] for p in 1:n, q_ in 1:n) +
                sum(q[p] * x2[p] for p in 1:n) +
                r <= 0.0
            )
        end

        @objective(model, Max, x2[i])
        optimize!(model)
        status = termination_status(model)
        @assert status == MOI.OPTIMAL "Upper-bound QP for x[$i] not optimal: $status"
        u[i] = value(x2[i])
    end

    return ell, u
end

"""
Clean bounds (ell, u) for nicer numerics, in a *safe* way for Big-M:

- Round ell[i] downwards and u[i] upwards to the given number of digits.
"""
function clean_bounds!(
    ell::AbstractVector,
    u::AbstractVector;
    digits::Int = 4,
)
    @assert length(ell) == length(u)
    fac = 10.0 ^ digits

    for i in eachindex(ell)
        # skip non-finite just in case
        if !isfinite(ell[i]) || !isfinite(u[i])
            continue
        end

        # outward rounding
        ell[i] = floor(ell[i] * fac) / fac
        u[i]   = ceil( u[i] * fac) / fac

        # tiny values → snap to exactly 0.0
        if abs(ell[i]) < 10.0^(-digits)
            ell[i] = 0.0
        end
        if abs(u[i]) < 10.0^(-digits)
            u[i] = 0.0
        end
    end

    return ell, u
end

"""
Compute bounds (ell, u) for Big-M using only the linear region

    { x : A x <= b, H x = h }.

Quadratic inequalities are deliberately ignored (supervisor's recommendation).
"""
function bounds_for_bigM(
    A::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    H::AbstractMatrix{<:Real},
    h::AbstractVector{<:Real},
    Qi_list::Vector{Matrix{Float64}},
    qi_list::Vector{Vector{Float64}},
    ri_list::Vector{Float64},
    neg_QI::Vector{Int};
    optimizer = MosekTools.Optimizer,
)
    return bounds_from_linear_part(A, b, H, h; optimizer = optimizer)
end

# ============================
# Final anchor feasibility check
# ============================

"""
Sanity check: verify that the anchor x̄ is feasible for all constraints
(up to a small tolerance).
"""
function check_anchor_feasible(
    xbar::AbstractVector,
    A::Matrix{Float64},
    b::Vector{Float64},
    H::Matrix{Float64},
    h::Vector{Float64},
    Qi_list::Vector{Matrix{Float64}},
    qi_list::Vector{Vector{Float64}},
    ri_list::Vector{Float64},
    Pi_list::Vector{Matrix{Float64}},
    pi_list::Vector{Vector{Float64}},
    si_list::Vector{Float64};
    tol::Real = 1e-6,
)
    # Linear inequalities: A x <= b
    if size(A, 1) > 0 && !isempty(b)
        viol = maximum(A * xbar .- b)
        @assert viol <= tol "Anchor x̄ violates a linear inequality: max(Ax̄ - b) = $(viol)"
    end

    # Linear equalities: H x = h
    if size(H, 1) > 0 && !isempty(h)
        viol = maximum(abs.(H * xbar .- h))
        @assert viol <= tol "Anchor x̄ violates a linear equality: max(|Hx̄ - h|) = $(viol)"
    end

    # Quadratic inequalities: g_i(x) <= 0
    if !isempty(Qi_list)
        for (Q, q, r) in zip(Qi_list, qi_list, ri_list)
            val = 0.5 * dot(xbar, Q * xbar) + dot(q, xbar) + r
            @assert val <= tol "Anchor x̄ violates a quadratic inequality: g(x̄) = $(val)"
        end
    end

    # Quadratic equalities: h_j(x) = 0
    if !isempty(Pi_list)
        for (P, p, s) in zip(Pi_list, pi_list, si_list)
            val = 0.5 * dot(xbar, P * xbar) + dot(p, xbar) + s
            @assert abs(val) <= tol "Anchor x̄ violates a quadratic equality: h(x̄) = $(val)"
        end
    end

    return nothing
end

# ============================
# Main instance generator
# ============================

"""
generate_instance(...)

Build a single QCQP instance according to a user specification.
"""
function generate_instance(;
    n::Int,
    rho::Int,
    base_type::Symbol = :box,
    n_LI::Int = 0,
    n_LE::Int = 0,
    n_QI::Int = 0,
    n_QE::Int = 0,
    want_convex::Bool = true,
    neg_eig_counts::Union{Nothing,Vector{Int}} = nothing,
    seed::Int = 1,
    bigM_scale::Real = 1.0,
    optimizer = MosekTools.Optimizer,
)
    rng = MersenneTwister(seed)

    # --- basic checks ---
    @assert n ≥ 1 "n must be at least 1."
    @assert 0 ≤ rho ≤ n "rho must satisfy 0 ≤ rho ≤ n."
    @assert n_LI ≥ 0 && n_LE ≥ 0 && n_QI ≥ 0 && n_QE ≥ 0
    @assert bigM_scale ≥ 1.0 "bigM_scale must be ≥ 1.0."
    @assert base_type in (:simplex, :box, :poly) "base_type must be :simplex, :box, or :poly."

    if base_type == :simplex
        @assert rho ≥ 1 "With base_type = :simplex we need rho ≥ 1."
    end

    if want_convex
        @assert n_QE == 0 "Quadratic equalities are only allowed in nonconvex mode."
    end

    # --- spectral pattern for objective and QI ---
    neg_obj, neg_QI = build_neg_eig_counts(n, n_QI, n_QE, want_convex, neg_eig_counts)

    # --- anchor xbar ---
    xbar = if base_type == :simplex
        generate_simplex_anchor(rng, n, rho)
    else
        generate_sparse_anchor(rng, n, rho)
    end

    # --- base linear region ---
    A = zeros(Float64, 0, n)
    b = Float64[]
    H = zeros(Float64, 0, n)
    h = Float64[]
    ell = zeros(Float64, n)
    u   = zeros(Float64, n)

    if base_type == :simplex
        A0, b0, H0, h0 = build_simplex_bounds(n)
        A = vcat(A, A0)
        b = vcat(b, b0)
        H = vcat(H, H0)
        h = vcat(h, h0)
    elseif base_type == :box
        A0, b0 = build_box_bounds(rng, xbar)
        A = vcat(A, A0)
        b = vcat(b, b0)
    elseif base_type == :poly
        A0, b0 = build_polytope_bounds(rng, xbar)
        A = vcat(A, A0)
        b = vcat(b, b0)
    end

    # --- extra LE (first fix the affine subspace) ---
    if n_LE > 0
        H, h = add_extra_LE!(rng, A, b, H, h, xbar, n_LE; optimizer = optimizer)
    end

    # --- extra LI via vertex cuts (nonredundant) ---
    if n_LI > 0
        println()
        println("=== Linear enrichment with vertex cuts ===")
        println("Base type      : ", base_type)
        println("Dimension n    : ", n)
        println("Anchor x̄:")
        println(xbar)
        println("Number of extra LI constraints to add: ", n_LI)
        A, b = add_extra_LI_vertex_cuts!(rng, A, b, H, h, xbar, n_LI; optimizer = optimizer)
        println("=== Finished adding extra LI constraints ===")
        println()
    end

    # --- quadratic constraints from vertices (sequential cut/remove) ---
    Qi_list = Matrix{Float64}[]
    qi_list = Vector{Vector{Float64}}()
    ri_list = Float64[]

    Pi_list = Matrix{Float64}[]
    pi_list = Vector{Vector{Float64}}()
    si_list = Float64[]

    k_quads = n_QI + n_QE
    if k_quads > 0
        verts_for_quads = collect_vertices_for_quad(
            rng, A, b, H, h, xbar;
            k_vertices = k_quads,
            optimizer  = optimizer,
        )

        Qi_list, qi_list, ri_list, Pi_list, pi_list, si_list =
            build_quadratics_from_vertices(
                rng,
                xbar,
                verts_for_quads;
                n_QI        = n_QI,
                n_QE        = n_QE,
                want_convex = want_convex,
                neg_QI      = neg_QI,
                eps_feas    = 0.5,
                eps_cut     = 0.5,
                eig_eps     = 0.1,
                coef_bound  = 5.0,
                optimizer   = optimizer,
            )
    end

    # --- final anchor feasibility sanity check (all constraints) ---
    check_anchor_feasible(
        xbar,
        A, b,
        H, h,
        Qi_list, qi_list, ri_list,
        Pi_list, pi_list, si_list;
        tol = 1e-6,
    )

    # --- bounds for Big-M: linear only (quadratic ignored) ---
    ell, u = bounds_for_bigM(A, b, H, h, Qi_list, qi_list, ri_list, neg_QI;
                             optimizer = optimizer)

    # numeric cleaning: outward rounding for safe Big-M
    clean_bounds!(ell, u; digits = 4)

    println("Final bounds (linear only) for Big-M (cleaned):")
    println("  ell = ", ell)
    println("  u   = ", u)

    # --- objective ---
    Q0, q0 = rand_objective(rng, n; num_neg = neg_obj,
                            scale_Q = 1.0, scale_q = 1.0)

    # --- Big-M from bounds ---
    M_common, M_minus, M_plus = bigM_from_bounds(ell, u; scale = bigM_scale)

    # --- pack into Dict ---
    inst = Dict{String,Any}()

    inst["n"]   = n
    inst["rho"] = rho

    inst["Q0"] = Q0
    inst["q0"] = q0

    inst["Qi"] = isempty(Qi_list) ? nothing : Tuple(Qi_list)
    inst["qi"] = isempty(qi_list) ? nothing : Tuple(qi_list)
    inst["ri"] = isempty(ri_list) ? nothing : ri_list

    inst["Pi"] = isempty(Pi_list) ? nothing : Tuple(Pi_list)
    inst["pi"] = isempty(pi_list) ? nothing : Tuple(pi_list)
    inst["si"] = isempty(si_list) ? nothing : si_list

    inst["A"] = size(A, 1) == 0 ? nothing : A
    inst["b"] = isempty(b)      ? nothing : b
    inst["H"] = size(H, 1) == 0 ? nothing : H
    inst["h"] = isempty(h)      ? nothing : h

    inst["ell"]     = ell
    inst["u"]       = u
    inst["M"]       = M_common
    inst["M_minus"] = M_minus
    inst["M_plus"]  = M_plus

    inst["xbar"]                  = xbar
    inst["base_type"]             = String(Symbol(base_type))
    inst["want_convex"]           = want_convex
    inst["neg_eig_counts_input"]  = neg_eig_counts
    inst["neg_eig_counts_used"]   = vcat(neg_obj, neg_QI)
    inst["seed"]                  = seed

    # base geometry tag S/B/P
    base_tag = base_type == :simplex ? "S" :
               base_type == :box     ? "B" :
               base_type == :poly    ? "P" : ""
    inst["base_tag"] = base_tag

    # convexity string
    convexity = want_convex ? "CVX" : "NCVX"
    inst["convexity"] = convexity

    # meta counts
    inst["n_LI"] = n_LI
    inst["n_LE"] = n_LE
    inst["n_QI"] = n_QI
    inst["n_QE"] = n_QE
    inst["bigM_scale"] = bigM_scale

    # --- ID: n{n}_rho{rho}_{S/B/P}_LI3_LE1_QI2_QE1_{CVX/NCVX}_Mx10_seed{seed}
    parts = String[]
    push!(parts, "n$(n)")
    push!(parts, "rho$(rho)")
    base_tag != "" && push!(parts, base_tag)

    if n_LI > 0
        push!(parts, "LI$(n_LI)")
    elseif size(A, 1) > 0
        push!(parts, "LI")
    end

    if n_LE > 0
        push!(parts, "LE$(n_LE)")
    elseif size(H, 1) > 0
        push!(parts, "LE")
    end

    if n_QI > 0
        push!(parts, "QI$(n_QI)")
    elseif !isempty(Qi_list)
        push!(parts, "QI")
    end

    if n_QE > 0
        push!(parts, "QE$(n_QE)")
    elseif !isempty(Pi_list)
        push!(parts, "QE")
    end

    push!(parts, convexity)

    if abs(bigM_scale - 1.0) > 1e-8
        k = round(bigM_scale; digits=6)
        if isapprox(k, round(k); atol=1e-8)
            push!(parts, "Mx$(Int(round(k)))")
        else
            push!(parts, "Mx$(k)")
        end
    end

    push!(parts, "seed$(seed)")
    inst["id"] = join(parts, "_")

    return inst
end

end # module UserInstanceGen


###############################
# instances_build_script.jl
###############################

using .InstanceGen
using .UserInstanceGen
using Random

# ------------------------------------------------
# Special small rational box instance (EU ≠ IU)
# ------------------------------------------------
function make_small_rational_instance()
    Q0 = [
        3//10000    127//1250   79//2500    867//10000;
        127//1250   1//500      1001//10000 1059//10000;
        79//2500    1001//10000 -1//2000    -703//10000;
        867//10000  1059//10000 -703//10000 -1063//10000
    ]
    Q0d = Q0 .* 20000

    q0  = [-1973//10000, -2535//10000, -1967//10000, -973//10000]
    q0d = q0 .* 20000

    A2 = [ 1.0   0.0   0.0   0.0
           0.0   1.0   0.0   0.0
           0.0   0.0   1.0   0.0
           0.0   0.0   0.0   1.0
          -1.0   0.0   0.0   0.0
           0.0  -1.0   0.0   0.0
           0.0   0.0  -1.0   0.0
           0.0   0.0   0.0  -1.0 ]

    b2 = fill(1.0, 8)   # -1 ≤ x_i ≤ 1

    n   = 4
    rho = 3.0

    d = Dict{String,Any}()

    d["n"]   = n
    d["rho"] = rho

    d["Q0"] = Float64.(Q0d)
    d["q0"] = Float64.(q0d)

    d["Qi"] = nothing
    d["qi"] = nothing
    d["ri"] = nothing

    d["Pi"] = nothing
    d["pi"] = nothing
    d["si"] = nothing

    d["A"] = A2
    d["b"] = b2
    d["H"] = nothing
    d["h"] = nothing

    # Box [-1,1]^4 → bounds and Big-M
    ell = fill(-1.0, n)
    u   = fill( 1.0, n)
    d["ell"] = ell
    d["u"]   = u
    d["M"]       = ones(Float64, n)
    d["M_minus"] = ones(Float64, n)
    d["M_plus"]  = ones(Float64, n)

    # Metadata so the rest of the pipeline understands it
    d["base_tag"]   = "B"        # box
    d["convexity"]  = "NCVX"     # we know EU/IU differ at RLT level
    d["n_LI"]       = 0          # only the box, no extra linear constraints
    d["n_LE"]       = 0
    d["n_QI"]       = 0
    d["n_QE"]       = 0
    d["bigM_scale"] = 1.0
    d["seed"]       = 0          # artificial seed

    # Manual ID so it is easy to spot in plots / tables
    d["id"] = "n4_rho3_B_LI_NCVX_rational"

    return d
end


# ------------------------------------------------
# Vertex–LP based test: 2 QI + 1 QE (sequential vertices)
# base type P (polytope), n = 4, rho = 3
# ------------------------------------------------
function make_vertex_QI2_QE1_instance(; bigM_scale::Real = 1.0)
    n   = 4
    rho = 3
    rng = Random.MersenneTwister(401)

    # Anchor
    xbar = UserInstanceGen.generate_sparse_anchor(rng, n, rho)

    # Polytope base from LI only (positive spanning style) + 1 LE + 1 extra LI
    A, b = UserInstanceGen.build_polytope_bounds(rng, xbar)
    H = zeros(Float64, 0, n)
    h = Float64[]

    # One extra LE then one extra LI
    H, h = UserInstanceGen.add_extra_LE!(rng, A, b, H, h, xbar, 1)
    A, b = UserInstanceGen.add_extra_LI_vertex_cuts!(rng, A, b, H, h, xbar, 1)

    # Total number of quadratic constraints
    n_QI = 2
    n_QE = 1
    K_total = n_QI + n_QE

    # Collect K_total distinct vertices of the linear polytope
    verts = UserInstanceGen.collect_vertices_for_quad(
        rng, A, b, H, h, xbar;
        k_vertices = K_total,
    )

    # Active vertex indices, to be shrunk sequentially
    active = collect(1:K_total)

    Qi_list = Matrix{Float64}[]
    qi_list = Vector{Vector{Float64}}()
    ri_list = Float64[]

    Pi_list = Matrix{Float64}[]
    pi_list = Vector{Vector{Float64}}()
    si_list = Float64[]

    # ---- QI1: cuts v1 (indefinite: 1 negative eigenvalue) ----
    cut_idx = active[1]                # deterministic: first active
    v_cut   = verts[cut_idx]
    feas_idx = setdiff(active, [cut_idx])
    feas_verts = [verts[j] for j in feas_idx]

    Q1, q1, r1 = UserInstanceGen.build_QI_from_vertex_set(
        rng, xbar, feas_verts, v_cut;
        want_convex = false,
        num_neg     = 1,
        eps_feas    = 0.5,
        eps_cut     = 0.5,
        coef_bound  = 5.0,
    )
    push!(Qi_list, Q1)
    push!(qi_list, q1)
    push!(ri_list, r1)

    active = feas_idx   # remove v1

    # ---- QI2: cuts one of the remaining vertices (PSD) ----
    cut_idx = active[1]      # now this is "v2" in hikâyemiz
    v_cut   = verts[cut_idx]
    feas_idx = setdiff(active, [cut_idx])   # one vertex left
    feas_verts = [verts[j] for j in feas_idx]

    Q2, q2, r2 = UserInstanceGen.build_QI_from_vertex_set(
        rng, xbar, feas_verts, v_cut;
        want_convex = true,   # PSD
        num_neg     = 0,
        eps_feas    = 0.5,
        eps_cut     = 0.5,
        coef_bound  = 5.0,
    )
    push!(Qi_list, Q2)
    push!(qi_list, q2)
    push!(ri_list, r2)

    active = feas_idx   # remove v2 → sadece bir vertex kaldı (v3)

    # ---- QE1: cuts the last remaining vertex, only x̄ satisfies equality ----
    @assert length(active) == 1
    cut_idx = active[1]
    v_cut   = verts[cut_idx]
    eq_verts = Vector{Vector{Float64}}()   # no extra equality vertices besides x̄

    P1, p1, s1 = UserInstanceGen.build_QE_from_vertex_set(
        rng, xbar, eq_verts, v_cut;
        eps_cut    = 0.5,
        coef_bound = 5.0,
    )
    push!(Pi_list, P1)
    push!(pi_list, p1)
    push!(si_list, s1)

    # Bounds from linear part only (for Big-M)
    ell, u = UserInstanceGen.bounds_from_linear_part(A, b, H, h)
    UserInstanceGen.clean_bounds!(ell, u; digits = 4)
    M_common, M_minus, M_plus = UserInstanceGen.bigM_from_bounds(ell, u; scale = bigM_scale)

    # Objective: 1 negative eigenvalue (additional nonconvexity)
    rng_obj = Random.MersenneTwister(402)
    Q0, q0 = UserInstanceGen.rand_objective(rng_obj, n; num_neg = 1)

    # Anchor feasibility check (all constraints)
    UserInstanceGen.check_anchor_feasible(
        xbar,
        A, b,
        H, h,
        Qi_list, qi_list, ri_list,
        Pi_list, pi_list, si_list;
        tol = 1e-6,
    )

    d = Dict{String,Any}()
    d["n"]   = n
    d["rho"] = rho

    d["Q0"] = Q0
    d["q0"] = q0

    d["Qi"] = Tuple(Qi_list)
    d["qi"] = Tuple(qi_list)
    d["ri"] = ri_list

    d["Pi"] = Tuple(Pi_list)
    d["pi"] = Tuple(pi_list)
    d["si"] = si_list

    d["A"] = A
    d["b"] = b
    d["H"] = H
    d["h"] = h

    d["ell"]     = ell
    d["u"]       = u
    d["M"]       = M_common
    d["M_minus"] = M_minus
    d["M_plus"]  = M_plus

    d["xbar"]        = xbar
    d["base_type"]   = "P"
    d["base_tag"]    = "P"
    d["want_convex"] = false

    # neg_eig_counts: [obj, QI1, QI2]
    d["neg_eig_counts_input"] = [1, 1, 0]
    d["neg_eig_counts_used"]  = [1, 1, 0]

    d["seed"]       = 401
    d["n_LI"]       = 1
    d["n_LE"]       = 1
    d["n_QI"]       = 2
    d["n_QE"]       = 1
    d["bigM_scale"] = bigM_scale

    d["convexity"] = "NCVX"

    d["id"] = "n4_rho3_P_LI1_LE1_QI2_QE1_NCVX_vertexLP_seq"

    return d
end

# ------------------------------------------------
# Vertex–LP based test: 4 QI (sequential vertices)
# base type B (box), n = 4, rho = 3
# ------------------------------------------------
function make_vertex_QI4_instance(; bigM_scale::Real = 1.0)
    n   = 4
    rho = 3
    rng = Random.MersenneTwister(501)

    # Anchor
    xbar = UserInstanceGen.generate_sparse_anchor(rng, n, rho)

    # Box base around anchor + 2 extra LI (vertex cuts)
    A, b = UserInstanceGen.build_box_bounds(rng, xbar)
    H = zeros(Float64, 0, n)
    h = Float64[]

    A, b = UserInstanceGen.add_extra_LI_vertex_cuts!(rng, A, b, H, h, xbar, 2)

    # Only QI constraints
    n_QI = 4
    n_QE = 0
    K_total = n_QI

    # Collect K_total distinct vertices of the linear polytope
    verts = UserInstanceGen.collect_vertices_for_quad(
        rng, A, b, H, h, xbar;
        k_vertices = K_total,
    )

    active = collect(1:K_total)

    Qi_list = Matrix{Float64}[]
    qi_list = Vector{Vector{Float64}}()
    ri_list = Float64[]

    Pi_list = Matrix{Float64}[]
    pi_list = Vector{Vector{Float64}}()
    si_list = Float64[]

    # Eigenpatterns: two indefinite (1 negative eigenvalue), then two PSD
    neg_pattern = [1, 1, 0, 0]

    for k in 1:n_QI
        @assert !isempty(active) "No active vertices left for QI $k"
        cut_idx = active[1]              # deterministik seçim
        v_cut   = verts[cut_idx]
        feas_idx = setdiff(active, [cut_idx])
        feas_verts = [verts[j] for j in feas_idx]

        want_convex = (neg_pattern[k] == 0)
        num_neg     = neg_pattern[k]

        Qk, qk, rk = UserInstanceGen.build_QI_from_vertex_set(
            rng, xbar, feas_verts, v_cut;
            want_convex = want_convex,
            num_neg     = num_neg,
            eps_feas    = 0.5,
            eps_cut     = 0.5,
            coef_bound  = 5.0,
        )

        push!(Qi_list, Qk)
        push!(qi_list, qk)
        push!(ri_list, rk)

        active = feas_idx   # cut edilen vertexi setten çıkar
    end

    # Bounds from linear part only
    ell, u = UserInstanceGen.bounds_from_linear_part(A, b, H, h)
    UserInstanceGen.clean_bounds!(ell, u; digits = 4)
    M_common, M_minus, M_plus = UserInstanceGen.bigM_from_bounds(ell, u; scale = bigM_scale)

    # Objective: PSD (nonconvexity only from QI's)
    rng_obj = Random.MersenneTwister(502)
    Q0, q0 = UserInstanceGen.rand_objective(rng_obj, n; num_neg = 0)

    # Anchor feasibility check
    UserInstanceGen.check_anchor_feasible(
        xbar,
        A, b,
        H, h,
        Qi_list, qi_list, ri_list,
        Pi_list, pi_list, si_list;
        tol = 1e-6,
    )

    d = Dict{String,Any}()
    d["n"]   = n
    d["rho"] = rho

    d["Q0"] = Q0
    d["q0"] = q0

    d["Qi"] = Tuple(Qi_list)
    d["qi"] = Tuple(qi_list)
    d["ri"] = ri_list

    d["Pi"] = nothing
    d["pi"] = nothing
    d["si"] = nothing

    d["A"] = A
    d["b"] = b
    d["H"] = H
    d["h"] = h

    d["ell"]     = ell
    d["u"]       = u
    d["M"]       = M_common
    d["M_minus"] = M_minus
    d["M_plus"]  = M_plus

    d["xbar"]        = xbar
    d["base_type"]   = "B"
    d["base_tag"]    = "B"
    d["want_convex"] = false    # overall NCVX due to QI's

    # neg_eig_counts: [obj, QI1, QI2, QI3, QI4]
    d["neg_eig_counts_input"] = [0; neg_pattern]
    d["neg_eig_counts_used"]  = [0; neg_pattern]

    d["seed"]       = 501
    d["n_LI"]       = 2
    d["n_LE"]       = 0
    d["n_QI"]       = n_QI
    d["n_QE"]       = 0
    d["bigM_scale"] = bigM_scale

    d["convexity"] = "NCVX"

    d["id"] = "n4_rho3_B_LI2_QI4_NCVX_vertexLP_seq"

    return d
end


function build_all_instances(; bigM_scale::Real = 1.0)
    # 0) Special rational box instance with EU ≠ IU at plain RLT
    special = make_small_rational_instance()

    # 1) Old small instances (8 + 1 scaled copy) → 9 instances
    base_insts = InstanceGen.build_instances()

    # start list with the special instance
    insts = vcat([special], base_insts)

    # 2) New user-specified instances (UserInstanceGen) → 11 more
    # Total will be 1 + 9 + 11 = 21.

    # 2.1  Large simplex instance: n=50, rho=10, 3 extra LI, only linear
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 50,
            rho = 10,
            base_type = :simplex,
            n_LI = 3,
            n_LE = 0,
            n_QI = 0,
            n_QE = 0,
            want_convex = true,
            neg_eig_counts = nothing,
            seed = 101,
            bigM_scale = bigM_scale,
        )
    )

    # 2.2  Simplex, n=10, convex, 1 quadratic inequality (PSD)
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 10,
            rho = 4,
            base_type = :simplex,
            n_LI = 0,
            n_LE = 0,
            n_QI = 1,
            n_QE = 0,
            want_convex = true,
            neg_eig_counts = nothing,
            seed = 102,
            bigM_scale = bigM_scale,
        )
    )

    # 2.3  Box, convex, 2 extra LI (vertex cuts), no QE/QI
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 8,
            rho = 3,
            base_type = :box,
            n_LI = 2,
            n_LE = 0,
            n_QI = 0,
            n_QE = 0,
            want_convex = true,
            neg_eig_counts = nothing,
            seed = 103,
            bigM_scale = bigM_scale,
        )
    )

    # 2.4  Box, convex, 2 LI + 2 LE
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 8,
            rho = 3,
            base_type = :box,
            n_LI = 2,
            n_LE = 2,
            n_QI = 0,
            n_QE = 0,
            want_convex = true,
            neg_eig_counts = nothing,
            seed = 104,
            bigM_scale = bigM_scale,
        )
    )

    # 2.5  Box, nonconvex only via objective (no QI/QE)
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 8,
            rho = 3,
            base_type = :box,
            n_LI = 0,
            n_LE = 0,
            n_QI = 0,
            n_QE = 0,
            want_convex = false,
            neg_eig_counts = [1],   # one negative eigenvalue in Q0
            seed = 105,
            bigM_scale = bigM_scale,
        )
    )

    # 2.6  Polytope base, convex, 2 LI, no QI/QE
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 6,
            rho = 4,
            base_type = :poly,
            n_LI = 2,
            n_LE = 0,
            n_QI = 0,
            n_QE = 0,
            want_convex = true,
            neg_eig_counts = nothing,
            seed = 106,
            bigM_scale = bigM_scale,
        )
    )

    # 2.7  Polytope base, nonconvex via objective + 1 convex QI
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 6,
            rho = 4,
            base_type = :poly,
            n_LI = 2,
            n_LE = 0,
            n_QI = 1,
            n_QE = 0,
            want_convex = false,
            neg_eig_counts = [1, 0],   # Q0 indefinite, Q1 PSD
            seed = 107,
            bigM_scale = bigM_scale,
        )
    )

    # 2.8  Polytope base, 1 quadratic equality (nonconvex via QE)
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 6,
            rho = 4,
            base_type = :poly,
            n_LI = 1,
            n_LE = 1,
            n_QI = 0,
            n_QE = 1,
            want_convex = false,
            neg_eig_counts = [0],      # Q0 PSD, nonconvexity from QE
            seed = 108,
            bigM_scale = bigM_scale,
        )
    )

    # 2.9  Box, medium size n=20, convex, 3 LI + 2 LE
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 20,
            rho = 5,
            base_type = :box,
            n_LI = 3,
            n_LE = 2,
            n_QI = 0,
            n_QE = 0,
            want_convex = true,
            neg_eig_counts = nothing,
            seed = 109,
            bigM_scale = bigM_scale,
        )
    )

    # 2.10 Box, n=20, nonconvex, 1 QI + 1 QE + some LI/LE
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 20,
            rho = 5,
            base_type = :box,
            n_LI = 2,
            n_LE = 2,
            n_QI = 1,
            n_QE = 1,
            want_convex = false,
            neg_eig_counts = [1, 0],   # Q0 indefinite, QI PSD
            seed = 110,
            bigM_scale = bigM_scale,
        )
    )

    # 2.11  Polytope, n=15, mixed: LI + LE + 2 QI + 1 QE
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 15,
            rho = 7,
            base_type = :poly,
            n_LI = 3,
            n_LE = 1,
            n_QI = 2,
            n_QE = 1,
            want_convex = false,
            neg_eig_counts = [1, 1, 0],  # Q0 indef, QI1 indef, QI2 PSD
            seed = 111,
            bigM_scale = bigM_scale,
        )
    )

    # 2.12  Vertex–LP test: P base, 2 QI + 1 QE (sequential vertices)
    push!(insts, make_vertex_QI2_QE1_instance(bigM_scale = bigM_scale))

    # 2.13  Vertex–LP test: B base, 4 QI (sequential vertices)
    push!(insts, make_vertex_QI4_instance(bigM_scale = bigM_scale))

    @assert length(insts) == 23 "We expect exactly 23 instances."

    return insts
end


# --- Build & save ---

insts = build_all_instances(; bigM_scale = 1.0)
@assert length(insts) == 23  # sanity check

instances_path = joinpath(@__DIR__, "instances.json")
InstanceGen.save_instances_json(instances_path, insts)

println("Saved $(length(insts)) instances to $(instances_path)")
println("Instance IDs:")
for d in insts
    println("  ", d["id"])
end


"""
Bomze et al. sparse StQP instance for ρ = 3 (final-version tag "fv").

Region:
    x ≥ 0,  eᵀx = 1,  ‖x‖₀ ≤ 3

Objective:
    min  xᵀ Q x   (implemented as 0.5 xᵀ Q0 x with Q0 = 2Q)

No extra linear or quadratic constraints are added; this is a pure StQP
over the simplex, convex in x (Q0 is positive definite), but will be
used with a cardinality constraint ρ = 3.
"""
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

# Read instances.json
instances_all = InstanceGen.load_instances_json("instances.json")

# Add instance
push!(instances_all, bomze_stqp_rho3_instance())

# Resave
InstanceGen.save_instances_json("instances.json", instances_all)
println("Total number of instances:", length(instances_all))