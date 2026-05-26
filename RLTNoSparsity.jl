# ============================================================
# RLTNoSparsity.jl
# Sparsity-free QCQP-RLT relaxation
# ============================================================

module RLTNoSparsity

using LinearAlgebra: diag
using JuMP
using MathOptInterface
const MOI = MathOptInterface
using Parameters: @unpack

using ..RLTBigM: prepare_instance

outer_xy(a, b) = a * transpose(b)

function additional_params(params)
    xL = -diag(params.Mminus)
    xU =  diag(params.Mplus)
    return merge(params, (; xL, xU))
end

function add_F_qcqp_rlt!(m, x, X, params)
    @unpack n, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, xL, xU = params

    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(m,
            0.5 * sum(Qmat[i,j] * X[i,j] for i in 1:n, j in 1:n)
            + qvec' * x + rterm <= 0
        )
    end

    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(m,
            0.5 * sum(Pmat[i,j] * X[i,j] for i in 1:n, j in 1:n)
            + pvec' * x + sterm == 0
        )
    end

    if ℓ > 0
        @constraint(m, A * x .<= b)
    end

    if η > 0
        @constraint(m, H * x .== h)
    end

    @constraint(m, xL .<= x)
    @constraint(m, x .<= xU)

    @constraint(m, X .- x * xL' .- xL * x' .+ xL * xL' .>= 0)
    @constraint(m, x * xU' .- X .- xL * xU' .+ xL * x' .>= 0)
    @constraint(m, X .- xU * x' .- x * xU' .+ xU * xU' .>= 0)

    if ℓ > 0
        @constraint(m,
            A * X * A' .- A * (x * b') .- (b * x') * A' .+ b * b' .>= 0
        )

        @constraint(m,
            -A * X .+ b * x' .+ A * (x * xL') .- b * xL' .>= 0
        )

        @constraint(m,
            A * X .- A * (x * xU') .- b * x' .+ b * xU' .>= 0
        )
    end

    if η > 0
        @constraint(m, H * X .== outer_xy(h, x))
    end

    return nothing
end

function build_and_solve(data;
    build_only::Bool = false,
    optimizer,
    verbose::Bool = false,
)
    params = additional_params(prepare_instance(data))
    @unpack n, Q0, q0 = params

    m = Model(optimizer)

    if !verbose
        set_silent(m)
    end

    set_name(m, "QCQP_RLT_NoSparsity")

    @variable(m, x[1:n])
    @variable(m, X[1:n, 1:n], Symmetric)

    @objective(m, Min,
        0.5 * sum(Q0[i,j] * X[i,j] for i in 1:n, j in 1:n)
        + q0' * x
    )

    add_F_qcqp_rlt!(m, x, X, params)

    if build_only
        return m
    end

    optimize!(m)
    term = termination_status(m)

    if term == MOI.OPTIMAL
        return (:OPTIMAL, objective_value(m), value.(x), value.(X))
    else
        return (Symbol(term), nothing, nothing, nothing)
    end
end

export build_and_solve, add_F_qcqp_rlt!, additional_params

end