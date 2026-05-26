###############################
# build_instances_clean.jl
###############################

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
    Q .= (Q .+ Q') .÷ 2
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

function rand_symm_from_eigs(rng::AbstractRNG, n::Int;
                             num_neg::Int=0,
                             nonneg_vals::UnitRange{Int}=0:10,
                             neg_vals::UnitRange{Int}=-10:-1)
    @assert 0 ≤ num_neg ≤ n
    num_nonneg = n - num_neg

    negs    = num_neg == 0    ? Int[] : rand(rng, neg_vals,    num_neg)
    nonnegs = num_nonneg == 0 ? Int[] : rand(rng, nonneg_vals, num_nonneg)

    eigs = vcat(negs, nonnegs)
    eigs = eigs[randperm(rng, length(eigs))]

    R       = randn(rng, n, n)
    Qfac, _ = qr(R)
    Qmat    = Matrix(Qfac)
    Λ       = Diagonal(Float64.(eigs))

    return Qmat * Λ * Qmat'
end

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

    # Keep both xhat and 0 feasible.
    # Original code used one scalar δ and b = A*xhat + δ, which could exclude 0.
    δ = rand(rng, 1:5, size(A, 1))
    b = max.(A * xhat .+ δ, δ)

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
        model = Model(optimizer)
        @variable(model, x[1:n])
        @constraint(model, A * x .<= b)
        if H !== nothing && h !== nothing
            @constraint(model, H * x .== h)
        end
        @objective(model, Min, x[i])
        optimize!(model)
        @assert termination_status(model) == MOI.OPTIMAL "LP lower bound not optimal"
        ℓ[i] = value(x[i])

        model = Model(optimizer)
        @variable(model, x2[1:n])
        @constraint(model, A * x2 .<= b)
        if H !== nothing && h !== nothing
            @constraint(model, H * x2 .== h)
        end
        @objective(model, Max, x2[i])
        optimize!(model)
        @assert termination_status(model) == MOI.OPTIMAL "LP upper bound not optimal"
        u[i] = value(x2[i])
    end

    return ℓ, u
end

# ---------------------------
# Big-M from bounds
# ---------------------------

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

function simplex_instance(n::Int; optimizer=MosekTools.Optimizer)
    A = -Matrix{Float64}(I, n, n)
    b = zeros(Float64, n)
    H = ones(Float64, 1, n)
    h = [1.0]

    ℓ, u = bounds_via_LP(A, b; H=H, h=h, optimizer=optimizer)
    M    = bigM_from_bounds(ℓ, u; scale=1.0, common=true)
    Mminus, Mplus = bigM_from_bounds(ℓ, u; scale=1.0, common=false)

    return A, b, H, h, M, Mminus, Mplus, ℓ, u
end

function box_instance(n::Int; rng::AbstractRNG, low::Int=-8, high::Int=8, min_width::Int=3)
    @assert low < 0 < high
    @assert min_width ≥ 1
    @assert abs(low) + high ≥ min_width "The available range is too small for min_width."

    ℓ = zeros(Int, n)
    u = zeros(Int, n)

    for i in 1:n
        # Zero-safe box: always ℓ_i < 0 < u_i.
        # Also enforce u_i - ℓ_i >= min_width.
        while true
            left  = rand(rng, 1:abs(low))  # gives lower bound -left
            right = rand(rng, 1:high)      # gives upper bound +right

            if left + right ≥ min_width
                ℓ[i] = -left
                u[i] = right
                break
            end
        end

        @assert ℓ[i] < 0 < u[i]
        @assert u[i] - ℓ[i] ≥ min_width
    end

    A = vcat(Matrix{Int}(I, n, n), -Matrix{Int}(I, n, n))
    b = vcat(u, -ℓ)

    ℓf = Float64.(ℓ)
    uf = Float64.(u)

    M = bigM_from_bounds(ℓf, uf; scale=1.0, common=true)
    Mminus, Mplus = bigM_from_bounds(ℓf, uf; scale=1.0, common=false)

    xhat = [rand(rng, ℓ[i]:u[i]) for i in 1:n]

    return A, b, M, Mminus, Mplus, ℓ, u, xhat
end

# ---------------------------
# Quadratic rows with anchor
# ---------------------------

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
    r  = -g0 - δ

    return Matrix{Float64}(Q), Float64.(q), Float64(r)
end

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
               M_minus::Union{Nothing,AbstractVector}=nothing,
               M_plus::Union{Nothing,AbstractVector}=nothing,
               ell::Union{Nothing,AbstractVector}=nothing,
               u::Union{Nothing,AbstractVector}=nothing,
               seed::Union{Nothing,Int}=nothing,
               base_tag::AbstractString = "",
               convexity::AbstractString = "CVX",
               n_LI::Int = 0, n_LE::Int = 0,
               n_QI::Int = 0, n_QE::Int = 0,
               bigM_scale::Real = 1.0)

    if M === nothing
        M = ones(Float64, n)
    end

    if ell === nothing
        ell = -Float64.(M)
    else
        ell = Float64.(ell)
    end

    if u === nothing
        u = Float64.(M)
    else
        u = Float64.(u)
    end

    if M_minus === nothing || M_plus === nothing
        M_minus, M_plus = bigM_from_bounds(ell, u; scale=bigM_scale, common=false)
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
        "ell"        => Float64.(ell),
        "u"          => Float64.(u),
        "M"          => Float64.(M),
        "M_minus"    => Float64.(M_minus),
        "M_plus"     => Float64.(M_plus),
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

function random_zero_safe_H(rng::AbstractRNG, n_rows::Int, n::Int, xhat::AbstractVector)
    H = zeros(Float64, n_rows, n)
    for r in 1:n_rows
        a = Float64.(rand(rng, -2:2, n))
        while all(abs.(a) .< 1e-12)
            a = Float64.(rand(rng, -2:2, n))
        end

        # Project a to be orthogonal to xhat so that H*xhat = 0.
        nx = dot(xhat, xhat)
        if nx > 1e-12
            a = a .- (dot(a, xhat) / nx) .* xhat
        end

        if norm(a) < 1e-10
            a = zeros(Float64, n)
            a[r <= n ? r : 1] = 1.0
            if nx > 1e-12
                a = a .- (dot(a, xhat) / nx) .* xhat
            end
        end

        H[r, :] .= a
    end
    h = zeros(Float64, n_rows)
    return H, h
end

function build_instances(; objective_scale::Real=10.0, optimizer=MosekTools.Optimizer)
    insts = Vector{Dict{String,Any}}(undef, 8)

    # 1) Simplex (n=7), convex objective
    let i = 1, rng = MersenneTwister(1)
        is_cvx = true
        n   = 7
        rho = 4.0
        A, b, H, h, M, Mminus, Mplus, ℓ, u = simplex_instance(n; optimizer=optimizer)

        num_neg_q0 = 0
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, H=H, h=h, M=M,
                  M_minus=Mminus, M_plus=Mplus, ell=ℓ, u=u,
                  Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                  seed=i, base_tag="S", convexity=convexity)
        insts[i] = d
    end

    # 2) Box (n=4), convex objective
    let i = 2, rng = MersenneTwister(2)
        is_cvx = true
        n   = 4
        rho = 3.0
        A, b, M, Mminus, Mplus, ℓ, u, xhat = box_instance(n; rng=rng)

        num_neg_q0 = 0
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                  M_minus=Mminus, M_plus=Mplus, ell=ℓ, u=u,
                  Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                  seed=i, base_tag="B", convexity=convexity)
        insts[i] = d
    end

    # 3) Box + 2 linear equalities, convex objective
    let i = 3, rng = MersenneTwister(3)
        is_cvx = true
        n   = 15
        rho = 10.0
        A, b, M, Mminus, Mplus, ℓ, u, xhat = box_instance(n; rng=rng)

        H, h = random_zero_safe_H(rng, 2, n, xhat)

        num_neg_q0 = 0
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, H=H, h=h, M=M,
                  M_minus=Mminus, M_plus=Mplus, ell=ℓ, u=u,
                  Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                  seed=i, base_tag="B", convexity=convexity)
        insts[i] = d
    end

    # 4) A-bounded minimal, convex objective
    let i = 4, rng = MersenneTwister(4)
        is_cvx = true
        n   = 5
        rho = 4.0
        A, b, xhat = build_bounded_A(n=n, mode=:minimal, rng=rng)
        ℓ, u = bounds_via_LP(A, b; optimizer=optimizer)
        M    = bigM_from_bounds(ℓ, u; scale=1.0, common=true)
        Mminus, Mplus = bigM_from_bounds(ℓ, u; scale=1.0, common=false)

        num_neg_q0 = 0
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                  M_minus=Mminus, M_plus=Mplus, ell=ℓ, u=u,
                  Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                  seed=i, base_tag="P", convexity=convexity)
        insts[i] = d
    end

    # 5) A-bounded symmetric, nonconvex objective
    let i = 5, rng = MersenneTwister(5)
        is_cvx = true
        n   = 4
        rho = 3.0
        A, b, xhat = build_bounded_A(n=n, mode=:symmetric, rng=rng)
        ℓ, u = bounds_via_LP(A, b; optimizer=optimizer)
        M    = bigM_from_bounds(ℓ, u; scale=1.0, common=true)
        Mminus, Mplus = bigM_from_bounds(ℓ, u; scale=1.0, common=false)

        num_neg_q0 = 2
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                  M_minus=Mminus, M_plus=Mplus, ell=ℓ, u=u,
                  Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                  seed=i, base_tag="P", convexity=convexity)
        insts[i] = d
    end

    # 6) Box + one quadratic inequality
    let i = 6, rng = MersenneTwister(6)
        is_cvx = true
        n   = 4
        rho = 3.0
        A, b, M, Mminus, Mplus, ℓ, u, xhat = box_instance(n; rng=rng)

        Q6, q6, r6 = make_qineq_with_anchor(rng, xhat; num_neg=0)

        num_neg_q0 = 1
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                  Qi=(Q6,), qi=(q6,), ri=(r6,),
                  M=M, M_minus=Mminus, M_plus=Mplus, ell=ℓ, u=u,
                  Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                  seed=i, base_tag="B", convexity=convexity)
        insts[i] = d
    end

    # 7) Box + one quadratic equality
    let i = 7, rng = MersenneTwister(7)
        is_cvx = false
        n   = 4
        rho = 3.0
        A, b, M, Mminus, Mplus, ℓ, u, xhat = box_instance(n; rng=rng)

        P7, p7, s7 = make_qeq_with_anchor(rng, xhat; num_neg=0)

        num_neg_q0 = 1
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                  Pi=(P7,), pi=(p7,), si=(s7,),
                  M=M, M_minus=Mminus, M_plus=Mplus, ell=ℓ, u=u,
                  Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                  seed=i, base_tag="B", convexity=convexity)
        insts[i] = d
    end

    # 8) Box + two q-inequalities + one q-equality + one linear equality
    let i = 8, rng = MersenneTwister(8)
        is_cvx = false
        n   = 10
        rho = 8.0
        A, b, M, Mminus, Mplus, ℓ, u, xhat = box_instance(n; rng=rng)

        Qqi1 = zeros(Float64, n, n)
        Qqi1[1,1] = 2.0
        Qqi1[2,2] = 2.0
        Qqi1[1,2] = -2.0
        Qqi1[2,1] = -2.0
        qqi1 = zeros(Float64, n)
        τ1   = rand(rng, 0:2)
        g01  = 0.5 * (xhat' * Qqi1 * xhat) + dot(qqi1, xhat)
        rqi1 = -g01 - τ1

        Qqi2, qqi2, rqi2 = make_qineq_with_anchor(rng, xhat; num_neg=1)
        Pqe, pqe, sqe = make_qeq_with_anchor(rng, xhat; num_neg=0)

        H, h = random_zero_safe_H(rng, 1, n, xhat)

        num_neg_q0 = 2
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                  Qi=(Qqi1, Qqi2), qi=(qqi1, qqi2), ri=(rqi1, rqi2),
                  Pi=(Pqe,), pi=(pqe,), si=(sqe,),
                  H=H, h=h, M=M, M_minus=Mminus, M_plus=Mplus, ell=ℓ, u=u,
                  Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                  seed=i, base_tag="B", convexity=convexity)
        insts[i] = d
    end

    # 9) Take instance 1 and scale Big-M by a factor of 10
    base_idx = 1
    base = insts[base_idx]
    new_inst = deepcopy(base)
    new_inst["M"] = 10.0 .* base["M"]
    new_inst["M_minus"] = 10.0 .* base["M_minus"]
    new_inst["M_plus"] = 10.0 .* base["M_plus"]
    new_inst["bigM_scale"] = get(base, "bigM_scale", 1.0) * 10.0
    push!(insts, new_inst)

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

rand_vec_int(rng::AbstractRNG, n::Int; vals = -5:5) = rand(rng, vals, n)
rand_slack(rng::AbstractRNG; lo::Int = 1, hi::Int = 5) = rand(rng, lo:hi)

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

function generate_sparse_anchor(
    rng::AbstractRNG,
    n::Int,
    rho::Int;
    vals::UnitRange{Int} = -3:3,
)
    x = zeros(Float64, n)
    rho == 0 && return x

    k = rand(rng, 1:rho)
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

function generate_simplex_anchor(rng::AbstractRNG, n::Int, rho::Int)
    @assert rho ≥ 1 "On the unit simplex we need rho ≥ 1."

    s = rand(rng, 1:min(rho, n))
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

function build_simplex_bounds(n::Int)
    A = -Matrix{Float64}(I, n, n)
    b = zeros(Float64, n)
    H = ones(Float64, 1, n)
    h = [1.0]
    return A, b, H, h
end

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

        # Build the box around both 0 and xbar[i].
        ell[i] = min(0.0, xbar[i]) - m_low
        u[i]   = max(0.0, xbar[i]) + m_up
    end

    A = vcat(Matrix{Float64}(I, n, n), -Matrix{Float64}(I, n, n))
    b = vcat(u, -ell)

    return A, b
end

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

    B = zeros(Float64, n, n)
    got_full_rank = false

    for _try in 1:max_rank_tries
        for i in 1:n
            g = randn(rng, n)
            a = g / norm(g)
            B[i, :] .= a
        end

        if rank(B) == n
            got_full_rank = true
            break
        end
    end

    if !got_full_rank
        error("build_polytope_bounds: could not generate full-rank B after $max_rank_tries tries.")
    end

    w = 1 .+ rand(rng, n)

    r0_row = -(w' * B)
    norm_r0 = norm(r0_row)
    @assert norm_r0 > 0 "Norm of -wᵀB is zero, something is wrong."
    r_row = r0_row ./ norm_r0

    A = vcat(B, Array(r_row))

    s = slack_min .+ (slack_max - slack_min) .* rand(rng, n)
    δ = slack_min + (slack_max - slack_min) * rand(rng)

    shift = vcat(s, [δ])

    # Keep both xbar and 0 feasible.
    b = max.(A * xbar .+ shift, shift)

    return A, b
end

# ============================
# Vertex LP helpers
# ============================

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

function build_cut_halfspace_from_normal(
    xbar::AbstractVector,
    v::AbstractVector,
    a::AbstractVector;
    θ::Real = 0.5,
)
    n = length(xbar)
    @assert length(v) == n
    @assert length(a) == n

    # Need a'x <= β to keep both xbar and 0 feasible, and cut v.
    for sgn in (1.0, -1.0)
        aa = sgn .* Float64.(a)
        αx = dot(aa, xbar)
        α0 = 0.0
        αv = dot(aa, v)

        keep_level = max(αx, α0)

        if αv > keep_level + 1e-8
            β = keep_level + θ * (αv - keep_level)
            return aa, β, αx, αv
        end
    end

    error("Cannot separate vertex while keeping both xbar and 0 feasible.")
end

function build_equality_from_normal(
    xbar::AbstractVector,
    v::AbstractVector,
    a::AbstractVector,
)
    n = length(xbar)
    @assert length(v) == n
    @assert length(a) == n

    aa = Float64.(a)

    # Make equality pass through both xbar and 0:
    # aa'x = 0 with aa ⟂ xbar.
    nx = dot(xbar, xbar)
    if nx > 1e-12
        aa = aa .- (dot(aa, xbar) / nx) .* xbar
    end

    if norm(aa) < 1e-10
        # Deterministic fallback.
        aa = zeros(Float64, n)
        aa[1] = 1.0
        if nx > 1e-12
            aa = aa .- (dot(aa, xbar) / nx) .* xbar
        end
    end

    if norm(aa) < 1e-10 || isapprox(dot(aa, v), 0.0; atol=1e-8)
        error("Cannot build zero-safe nonredundant equality from this normal.")
    end

    β = 0.0
    αv = dot(aa, v)

    return aa, β, 0.0, αv
end

# ============================
# Extra linear constraints
# ============================

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
    for k in 1:n_extra
        println()
        println("---- Extra linear inequality $k / $n_extra ----")

        v, c = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
        println("Vertex v:")
        println(v)

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

        try
            a, β, αx, αv = build_cut_halfspace_from_normal(xbar, v, c; θ = 0.5)

            println("New inequality: a' x ≤ β")
            println("  a = ", a)
            println("  β = ", β)
            println("  a' x̄ = ", αx, "   (should be ≤ β)")
            println("  a' 0  = 0.0       (should be ≤ β)")
            println("  a' v  = ", αv, "   (should be > β)")
            println("  a' x̄ - β = ", αx - β)
            println("  a' v  - β = ", αv - β)

            A = vcat(A, a')
            b = vcat(b, β)
        catch err
            @warn "Failed to construct zero-safe separating inequality; skipping this LI. Error: $err"
        end
    end

    return A, b
end

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
    if n_LE == 0
        return H, h
    end

    for k in 1:n_LE
        println()
        println("---- Extra linear equality $k / $n_LE ----")

        v, c = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
        println("Vertex v used for equality:")
        println(v)

        ok = false
        for _try in 1:20
            if norm(v .- xbar, Inf) < 1e-6
                v, c = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
                continue
            end

            try
                a, β, αx, αv = build_equality_from_normal(xbar, v, c)

                println("New equality: a' x = β")
                println("  a = ", a)
                println("  β = ", β)
                println("  a' x̄ = ", αx, "   (x̄ lies on the hyperplane)")
                println("  a' 0  = 0.0       (0 lies on the hyperplane)")
                println("  a' v  = ", αv, "   (vertex does NOT lie on the hyperplane)")

                H = vcat(H, a')
                h = vcat(h, β)
                ok = true
                break
            catch
                v, c = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
            end
        end

        if !ok
            @warn "Could not construct zero-safe nonredundant equality; skipping this LE."
        end
    end

    return H, h
end

function random_orthogonal(rng::AbstractRNG, n::Int)
    R = randn(rng, n, n)
    F = qr(R)
    return Matrix(F.Q)
end

# ============================
# Quadratic constraints from vertices
# ============================

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
    @assert length(v_cut) == n
    for v in feas_verts
        @assert length(v) == n
    end

    @assert 0 ≤ num_neg ≤ n
    if want_convex
        @assert num_neg == 0
    end

    Qmat = random_orthogonal(rng, n)

    model = Model(optimizer)
    @variable(model, lambda[1:n])
    @variable(model, q[1:n])
    @variable(model, r)

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

    for i in 1:n
        @constraint(model, -coef_bound <= lambda[i] <= coef_bound)
        @constraint(model, -coef_bound <= q[i]      <= coef_bound)
    end
    @constraint(model, -coef_bound <= r <= coef_bound)

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
            error("Unsupported sense $sense.")
        end
    end

    add_qi_constraint!(xbar, :le, -eps_feas)
    for v in feas_verts
        add_qi_constraint!(v, :le, -eps_feas)
    end
    add_qi_constraint!(v_cut, :ge, eps_cut)

    @objective(model, Min, 0.0)
    optimize!(model)
    status = termination_status(model)
    @assert status == MOI.OPTIMAL "build_QI_from_vertex_set: LP status = $status"

    λ_opt = value.(lambda)
    q_opt = value.(q)
    r_opt = value(r)

    Q = Qmat * Diagonal(λ_opt) * Qmat'
    Q = 0.5 * (Q + Q')

    return Q, Vector{Float64}(q_opt), Float64(r_opt)
end

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
    @assert k ≥ 1
    @assert 0 ≤ num_neg ≤ n

    if want_convex
        @assert num_neg == 0
    end

    for _attempt in 1:max_tries
        Q_raw = rand_symm_from_eigs(
            rng, n;
            num_neg     = num_neg,
            nonneg_vals = nonneg_vals,
            neg_vals    = neg_vals,
        )
        Q = scale_Q .* Matrix{Float64}(Q_raw)
        q = scale_q .* Float64.(rand_vec_int(rng, n; vals = qvals))

        phi_xbar  = 0.5 * (xbar' * Q * xbar) + dot(q, xbar)
        phi_verts = [0.5 * (v' * Q * v) + dot(q, v) for v in verts]

        all_vals   = vcat(phi_xbar, phi_verts)
        idx_sorted = sortperm(all_vals; rev = true)

        j1 = idx_sorted[1]
        j2 = idx_sorted[2]

        if j1 == 1
            continue
        end

        val1 = all_vals[j1]
        val2 = all_vals[j2]

        if val1 - val2 ≤ gap_tol
            continue
        end

        t = 0.5 * (val1 + val2)
        r = -t

        cut_idx_local = j1 - 1

        return Matrix{Float64}(Q), Vector{Float64}(q), Float64(r), cut_idx_local
    end

    error("build_QI_cut_max_vertex: could not find suitable (Q, q).")
end

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
    @assert length(v_cut) == n
    for v in eq_verts
        @assert length(v) == n
    end

    Qmat = random_orthogonal(rng, n)

    model = Model(optimizer)
    @variable(model, lambda[1:n])
    @variable(model, p[1:n])
    @variable(model, s)

    for i in 1:n
        @constraint(model, -coef_bound <= lambda[i] <= coef_bound)
        @constraint(model, -coef_bound <= p[i]      <= coef_bound)
    end
    @constraint(model, -coef_bound <= s <= coef_bound)

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
            error("Unsupported sense $sense.")
        end
    end

    add_qe_constraint!(xbar, :eq, 0.0)
    for v in eq_verts
        add_qe_constraint!(v, :eq, 0.0)
    end

    if rand(rng) < 0.5
        add_qe_constraint!(v_cut, :ge, eps_cut)
    else
        add_qe_constraint!(v_cut, :le, -eps_cut)
    end

    @objective(model, Min, 0.0)
    optimize!(model)
    status = termination_status(model)
    @assert status == MOI.OPTIMAL "build_QE_from_vertex_set: LP status = $status"

    λ_opt = value.(lambda)
    p_opt = value.(p)
    s_opt = value(s)

    P = Qmat * Diagonal(λ_opt) * Qmat'
    P = 0.5 * (P + P')

    return P, Vector{Float64}(p_opt), Float64(s_opt)
end

function build_quadratics_from_vertices(
    rng::AbstractRNG,
    xbar::AbstractVector,
    verts::Vector{Vector{Float64}};
    n_QI::Int,
    n_QE::Int,
    want_convex::Bool,
    neg_QI::Vector{Int},
    eps_feas::Real    = 0.5,
    eps_cut::Real     = 0.5,
    eig_eps::Real     = 0.1,
    coef_bound::Real  = 5.0,
    optimizer = MosekTools.Optimizer,
)
    K = length(verts)
    @assert K == n_QI + n_QE "Expected n_QI + n_QE vertices."
    n = length(xbar)

    for v in verts
        @assert length(v) == n
    end

    Qi_list = Matrix{Float64}[]
    qi_list = Vector{Vector{Float64}}()
    ri_list = Float64[]

    Pi_list = Matrix{Float64}[]
    pi_list = Vector{Vector{Float64}}()
    si_list = Float64[]

    active = collect(1:K)

    if n_QI > 0
        @assert length(neg_QI) == n_QI
        for i in 1:n_QI
            @assert !isempty(active)

            verts_active = [verts[j] for j in active]

            num_neg   = neg_QI[i]
            qi_convex = (num_neg == 0)
            if want_convex
                @assert qi_convex
            end

            Q_i, q_i, r_i, cut_local = build_QI_cut_max_vertex(
                rng,
                xbar,
                verts_active;
                want_convex = qi_convex,
                num_neg     = num_neg,
            )

            push!(Qi_list, Q_i)
            push!(qi_list, q_i)
            push!(ri_list, r_i)

            deleteat!(active, cut_local)
        end
    end

    if n_QE > 0
        @assert !want_convex
        for _j in 1:n_QE
            @assert !isempty(active)

            cut_idx = active[1]
            v_cut   = verts[cut_idx]

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

            active = other_active
        end
    end

    return Qi_list, qi_list, ri_list, Pi_list, pi_list, si_list
end

# ============================
# Spectral control for Q0 and QI
# ============================

function build_neg_eig_counts(
    n::Int,
    n_QI::Int,
    n_QE::Int,
    want_convex::Bool,
    neg_eig_counts::Union{Nothing,Vector{Int}},
)
    if neg_eig_counts === nothing
        if want_convex
            return 0, fill(0, n_QI)
        else
            if n ≥ 2
                return 1, fill(0, n_QI)
            else
                if n_QE > 0
                    return 0, fill(0, n_QI)
                else
                    error("Cannot realize nonconvex n=1 instance without QE or explicit neg_eig_counts.")
                end
            end
        end
    end

    counts = neg_eig_counts::Vector{Int}
    expected_len = 1 + n_QI
    @assert length(counts) == expected_len "neg_eig_counts must have length 1 + n_QI."

    for k in counts
        @assert 0 ≤ k < n "Each neg_eig_counts entry must satisfy 0 ≤ k < n."
    end

    neg_obj = counts[1]
    neg_QI  = counts[2:end]

    if want_convex
        @assert all(counts .== 0)
    else
        if all(counts .== 0) && n_QE == 0
            error("Nonconvex mode needs at least one negative eigenvalue or n_QE > 0.")
        end
    end

    return neg_obj, neg_QI
end

# ============================
# Bounds
# ============================

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
                sum(q[p] * x[p] for p in 1:n) + r <= 0.0
            )
        end

        @objective(model, Min, x[i])
        optimize!(model)
        status = termination_status(model)
        @assert status == MOI.OPTIMAL "Lower-bound QP for x[$i] not optimal: $status"
        ell[i] = value(x[i])

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
                sum(q[p] * x2[p] for p in 1:n) + r <= 0.0
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

function clean_bounds!(
    ell::AbstractVector,
    u::AbstractVector;
    digits::Int = 4,
)
    @assert length(ell) == length(u)
    fac = 10.0 ^ digits

    for i in eachindex(ell)
        if !isfinite(ell[i]) || !isfinite(u[i])
            continue
        end

        ell[i] = floor(ell[i] * fac) / fac
        u[i]   = ceil( u[i] * fac) / fac

        if abs(ell[i]) < 10.0^(-digits)
            ell[i] = 0.0
        end
        if abs(u[i]) < 10.0^(-digits)
            u[i] = 0.0
        end
    end

    return ell, u
end

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
# Final checks
# ============================

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
    if size(A, 1) > 0 && !isempty(b)
        viol = maximum(A * xbar .- b)
        @assert viol <= tol "Anchor x̄ violates a linear inequality: max(Ax̄ - b) = $(viol)"
    end

    if size(H, 1) > 0 && !isempty(h)
        viol = maximum(abs.(H * xbar .- h))
        @assert viol <= tol "Anchor x̄ violates a linear equality: max(|Hx̄ - h|) = $(viol)"
    end

    if !isempty(Qi_list)
        for (Q, q, r) in zip(Qi_list, qi_list, ri_list)
            val = 0.5 * dot(xbar, Q * xbar) + dot(q, xbar) + r
            @assert val <= tol "Anchor x̄ violates a quadratic inequality: g(x̄) = $(val)"
        end
    end

    if !isempty(Pi_list)
        for (P, p, s) in zip(Pi_list, pi_list, si_list)
            val = 0.5 * dot(xbar, P * xbar) + dot(p, xbar) + s
            @assert abs(val) <= tol "Anchor x̄ violates a quadratic equality: h(x̄) = $(val)"
        end
    end

    return nothing
end

function assert_zero_allowed_by_bounds(ell, u; tol::Real = 1e-8, context::String = "")
    bad = findall(i -> ell[i] > tol || u[i] < -tol, eachindex(ell))

    if !isempty(bad)
        msg = "Some variables cannot take value zero"
        if context != ""
            msg *= " in $context"
        end
        msg *= ". Bad indices and bounds: "
        msg *= join(["x[$i] in [$(ell[i]), $(u[i])]" for i in bad], ", ")
        error(msg)
    end

    return nothing
end

# ============================
# Main instance generator
# ============================

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

    @assert n ≥ 1 "n must be at least 1."
    @assert 0 ≤ rho ≤ n "rho must satisfy 0 ≤ rho ≤ n."
    @assert n_LI ≥ 0 && n_LE ≥ 0 && n_QI ≥ 0 && n_QE ≥ 0
    @assert bigM_scale ≥ 1.0 "bigM_scale must be ≥ 1.0."
    @assert base_type in (:simplex, :box, :poly)

    if base_type == :simplex
        @assert rho ≥ 1 "With base_type = :simplex we need rho ≥ 1."
    end

    if want_convex
        @assert n_QE == 0 "Quadratic equalities are only allowed in nonconvex mode."
    end

    neg_obj, neg_QI = build_neg_eig_counts(n, n_QI, n_QE, want_convex, neg_eig_counts)

    xbar = if base_type == :simplex
        generate_simplex_anchor(rng, n, rho)
    else
        generate_sparse_anchor(rng, n, rho)
    end

    A = zeros(Float64, 0, n)
    b = Float64[]
    H = zeros(Float64, 0, n)
    h = Float64[]

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

    if n_LE > 0
        H, h = add_extra_LE!(rng, A, b, H, h, xbar, n_LE; optimizer = optimizer)
    end

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

    check_anchor_feasible(
        xbar,
        A, b,
        H, h,
        Qi_list, qi_list, ri_list,
        Pi_list, pi_list, si_list;
        tol = 1e-6,
    )

    ell, u = bounds_for_bigM(A, b, H, h, Qi_list, qi_list, ri_list, neg_QI;
                             optimizer = optimizer)

    clean_bounds!(ell, u; digits = 4)

    assert_zero_allowed_by_bounds(
        ell, u;
        context = "generated instance with n=$n, rho=$rho, base_type=$base_type, seed=$seed"
    )

    println("Final bounds (linear only) for Big-M (cleaned):")
    println("  ell = ", ell)
    println("  u   = ", u)

    Q0, q0 = rand_objective(rng, n; num_neg = neg_obj,
                            scale_Q = 1.0, scale_q = 1.0)

    M_common, M_minus, M_plus = bigM_from_bounds(ell, u; scale = bigM_scale)

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

    base_tag = base_type == :simplex ? "S" :
               base_type == :box     ? "B" :
               base_type == :poly    ? "P" : ""
    inst["base_tag"] = base_tag

    convexity = want_convex ? "CVX" : "NCVX"
    inst["convexity"] = convexity

    inst["n_LI"] = n_LI
    inst["n_LE"] = n_LE
    inst["n_QI"] = n_QI
    inst["n_QE"] = n_QE
    inst["bigM_scale"] = bigM_scale

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
using LinearAlgebra
using MosekTools

# ------------------------------------------------
# Utilities for driver validation
# ------------------------------------------------

function as_matrix_or_empty(A, n)
    A === nothing && return zeros(Float64, 0, n)
    return Matrix{Float64}(A)
end

function as_vector_or_empty(b)
    b === nothing && return Float64[]
    return Vector{Float64}(b)
end

function validate_zero_allowed_all!(insts; optimizer=MosekTools.Optimizer, tol=1e-8, digits=4)
    bad_instances = []

    for inst in insts
        id = get(inst, "id", "unknown")
        n = Int(inst["n"])

        A = as_matrix_or_empty(get(inst, "A", nothing), n)
        b = as_vector_or_empty(get(inst, "b", nothing))
        H = as_matrix_or_empty(get(inst, "H", nothing), n)
        h = as_vector_or_empty(get(inst, "h", nothing))

        ell, u = UserInstanceGen.bounds_from_linear_part(
            A, b, H, h;
            optimizer = optimizer,
        )

        UserInstanceGen.clean_bounds!(ell, u; digits = digits)

        bad = findall(i -> ell[i] > tol || u[i] < -tol, 1:n)

        if !isempty(bad)
            push!(bad_instances, (id = id, bad = bad, ell = ell, u = u))
            continue
        end

        inst["ell"] = ell
        inst["u"]   = u

        scale = get(inst, "bigM_scale", 1.0)
        M_common, M_minus, M_plus = UserInstanceGen.bigM_from_bounds(ell, u; scale = scale)

        inst["M"]       = M_common
        inst["M_minus"] = M_minus
        inst["M_plus"]  = M_plus
    end

    if !isempty(bad_instances)
        println("\n============================================================")
        println("Zero-allowed validation failed")
        println("============================================================")
        for item in bad_instances
            println("\nInstance: ", item.id)
            for i in item.bad
                println("  x[$i] in [", item.ell[i], ", ", item.u[i], "]")
            end
        end
        error("Do not save instances.json: some variables cannot take value zero.")
    end

    println("\nAll instances passed zero-allowed validation.")
    return insts
end

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

    b2 = fill(1.0, 8)

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

    ell = fill(-1.0, n)
    u   = fill( 1.0, n)

    d["ell"] = ell
    d["u"]   = u

    d["M"]       = ones(Float64, n)
    d["M_minus"] = ones(Float64, n)
    d["M_plus"]  = ones(Float64, n)

    d["base_tag"]   = "B"
    d["convexity"]  = "NCVX"
    d["n_LI"]       = 0
    d["n_LE"]       = 0
    d["n_QI"]       = 0
    d["n_QE"]       = 0
    d["bigM_scale"] = 1.0
    d["seed"]       = 0

    d["id"] = "n4_rho3_B_LI_NCVX_rational"

    return d
end

# ------------------------------------------------
# Vertex–LP based test: 2 QI + 1 QE
# ------------------------------------------------

function make_vertex_QI2_QE1_instance(; bigM_scale::Real = 1.0)
    n   = 4
    rho = 3
    rng = Random.MersenneTwister(401)

    xbar = UserInstanceGen.generate_sparse_anchor(rng, n, rho)

    A, b = UserInstanceGen.build_polytope_bounds(rng, xbar)
    H = zeros(Float64, 0, n)
    h = Float64[]

    H, h = UserInstanceGen.add_extra_LE!(rng, A, b, H, h, xbar, 1)
    A, b = UserInstanceGen.add_extra_LI_vertex_cuts!(rng, A, b, H, h, xbar, 1)

    n_QI = 2
    n_QE = 1
    K_total = n_QI + n_QE

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

    cut_idx = active[1]
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

    active = feas_idx

    cut_idx = active[1]
    v_cut   = verts[cut_idx]
    feas_idx = setdiff(active, [cut_idx])
    feas_verts = [verts[j] for j in feas_idx]

    Q2, q2, r2 = UserInstanceGen.build_QI_from_vertex_set(
        rng, xbar, feas_verts, v_cut;
        want_convex = true,
        num_neg     = 0,
        eps_feas    = 0.5,
        eps_cut     = 0.5,
        coef_bound  = 5.0,
    )
    push!(Qi_list, Q2)
    push!(qi_list, q2)
    push!(ri_list, r2)

    active = feas_idx

    @assert length(active) == 1
    cut_idx = active[1]
    v_cut   = verts[cut_idx]
    eq_verts = Vector{Vector{Float64}}()

    P1, p1, s1 = UserInstanceGen.build_QE_from_vertex_set(
        rng, xbar, eq_verts, v_cut;
        eps_cut    = 0.5,
        coef_bound = 5.0,
    )
    push!(Pi_list, P1)
    push!(pi_list, p1)
    push!(si_list, s1)

    ell, u = UserInstanceGen.bounds_from_linear_part(A, b, H, h)
    UserInstanceGen.clean_bounds!(ell, u; digits = 4)
    UserInstanceGen.assert_zero_allowed_by_bounds(ell, u; context = "vertex_QI2_QE1")

    M_common, M_minus, M_plus = UserInstanceGen.bigM_from_bounds(ell, u; scale = bigM_scale)

    rng_obj = Random.MersenneTwister(402)
    Q0, q0 = UserInstanceGen.rand_objective(rng_obj, n; num_neg = 1)

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
# Vertex–LP based test: 4 QI
# ------------------------------------------------

function make_vertex_QI4_instance(; bigM_scale::Real = 1.0)
    n   = 4
    rho = 3
    rng = Random.MersenneTwister(501)

    xbar = UserInstanceGen.generate_sparse_anchor(rng, n, rho)

    A, b = UserInstanceGen.build_box_bounds(rng, xbar)
    H = zeros(Float64, 0, n)
    h = Float64[]

    A, b = UserInstanceGen.add_extra_LI_vertex_cuts!(rng, A, b, H, h, xbar, 2)

    n_QI = 4
    n_QE = 0
    K_total = n_QI

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

    neg_pattern = [1, 1, 0, 0]

    for k in 1:n_QI
        @assert !isempty(active)
        cut_idx = active[1]
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

        active = feas_idx
    end

    ell, u = UserInstanceGen.bounds_from_linear_part(A, b, H, h)
    UserInstanceGen.clean_bounds!(ell, u; digits = 4)
    UserInstanceGen.assert_zero_allowed_by_bounds(ell, u; context = "vertex_QI4")

    M_common, M_minus, M_plus = UserInstanceGen.bigM_from_bounds(ell, u; scale = bigM_scale)

    rng_obj = Random.MersenneTwister(502)
    Q0, q0 = UserInstanceGen.rand_objective(rng_obj, n; num_neg = 0)

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
    d["want_convex"] = false

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

# ------------------------------------------------
# Bomze sparse StQP instance
# ------------------------------------------------

function bomze_stqp_rho3_instance(; bigM_scale::Real = 1.0)
    n   = 6
    rho = 3

    Q = [
        2.6947  -0.2028  -1.1144  -2.4230  -2.1633   0.7710
       -0.2028   5.1998   0.5005  -2.0941   0.7828  -3.3611
       -1.1144   0.5005   5.2918   1.4119  -1.1526  -1.9723
       -2.4230  -2.0941   1.4119   4.1140  -0.2025   0.3132
       -2.1633   0.7828  -1.1526  -0.2025   7.6645  -0.0170
        0.7710  -3.3611  -1.9723   0.3132  -0.0170   3.3040
    ]

    Q0 = 2.0 .* Q
    q0 = zeros(Float64, n)

    A, b, H, h = UserInstanceGen.build_simplex_bounds(n)

    xbar = fill(1.0 / n, n)

    ell = zeros(Float64, n)
    u   = ones(Float64, n)
    UserInstanceGen.clean_bounds!(ell, u; digits = 4)

    M_common, M_minus, M_plus = UserInstanceGen.bigM_from_bounds(ell, u; scale = bigM_scale)

    inst = Dict{String,Any}()

    inst["n"]   = n
    inst["rho"] = rho

    inst["Q0"] = Q0
    inst["q0"] = q0

    inst["Qi"] = nothing
    inst["qi"] = nothing
    inst["ri"] = nothing

    inst["Pi"] = nothing
    inst["pi"] = nothing
    inst["si"] = nothing

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

    inst["neg_eig_counts_input"] = [0]
    inst["neg_eig_counts_used"]  = [0]

    inst["seed"] = 0

    inst["base_tag"]  = "S"
    inst["convexity"] = "CVX"

    inst["n_LI"] = 0
    inst["n_LE"] = 0
    inst["n_QI"] = 0
    inst["n_QE"] = 0
    inst["bigM_scale"] = bigM_scale

    inst["id"] = "n6_rho3_S_StQP_rho3_CVX_fv"

    return inst
end

# ------------------------------------------------
# Build all instances
# ------------------------------------------------

function build_all_instances(; bigM_scale::Real = 1.0)
    special = make_small_rational_instance()

    base_insts = InstanceGen.build_instances()

    insts = vcat([special], base_insts)

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
            neg_eig_counts = [1],
            seed = 105,
            bigM_scale = bigM_scale,
        )
    )

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
            neg_eig_counts = [1, 0],
            seed = 107,
            bigM_scale = bigM_scale,
        )
    )

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
            neg_eig_counts = [0],
            seed = 108,
            bigM_scale = bigM_scale,
        )
    )

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
            neg_eig_counts = [1, 0],
            seed = 110,
            bigM_scale = bigM_scale,
        )
    )

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
            neg_eig_counts = [1, 1, 0],
            seed = 111,
            bigM_scale = bigM_scale,
        )
    )

    push!(insts, make_vertex_QI2_QE1_instance(bigM_scale = bigM_scale))
    push!(insts, make_vertex_QI4_instance(bigM_scale = bigM_scale))
    push!(insts, bomze_stqp_rho3_instance(bigM_scale = bigM_scale))

    @assert length(insts) == 24 "We expect exactly 24 instances."

    return insts
end

# ------------------------------------------------
# Run
# ------------------------------------------------

insts = build_all_instances(; bigM_scale = 1.0)
@assert length(insts) == 24

insts = validate_zero_allowed_all!(insts; optimizer = MosekTools.Optimizer)

instances_path = joinpath(@__DIR__, "instances_clean.json")
InstanceGen.save_instances_json(instances_path, insts)

println()
println("Saved $(length(insts)) clean instances to $(instances_path)")
println("Instance IDs:")
for d in insts
    println("  ", d["id"])
end