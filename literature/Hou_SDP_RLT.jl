module HouSDPRLT

using JuMP
using LinearAlgebra
using MathOptInterface
using Parameters: @unpack
using ..RLTBigM: prepare_instance

const MOI = MathOptInterface

# Extracted unchanged from SDPRLTCompEIComparison.
# Hou SDP--RLT: lifted QCQP rows, x-side RLT, Shor PSD,
# and the sparsity cone. Bibliographic details remain to be documented.

function set_tight_sdp_tolerances!(m; tol::Float64=1e-9)
    attrs = (
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
        "MSK_DPAR_INTPNT_CO_TOL_INFEAS",
    )
    for a in attrs
        try
            set_optimizer_attribute(m, a, tol)
        catch
        end
    end
    return nothing
end

function add_shor!(m, x, X)
    @constraint(
        m,
        [1 x'; x X] in PSDCone()
    )
    return nothing
end

function add_sparsity_cone!(m, x, X, ρ::Int)
    n = length(x)

    # S = [ rho      x'
    #       x   Diag(diag X) ].
    #
    # IMPORTANT:
    # Do not build this with a ternary @expression container:
    # that creates Matrix{Any} in some JuMP versions, and
    # `Matrix{Any} in PSDCone()` is not recognized.
    #
    # Build a typed affine-expression matrix instead.
    S = Matrix{JuMP.AffExpr}(undef, n + 1, n + 1)

    S[1,1] = JuMP.AffExpr(Float64(ρ))

    for i in 1:n
        S[1, i+1] = 1.0 * x[i]
        S[i+1, 1] = 1.0 * x[i]
    end

    for i in 1:n, j in 1:n
        if i == j
            S[i+1, j+1] = 1.0 * X[i,i]
        else
            S[i+1, j+1] = JuMP.AffExpr(0.0)
        end
    end

    @constraint(m, S in PSDCone())

    return nothing
end

function build_hou_sdp_rlt(
    data;
    optimizer,
    verbose::Bool=false,
    tight::Bool=true,
    tol::Float64=1e-9,
)
    params = prepare_instance(data)

    @unpack n, ρ, Q0, q0,
            Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η = params

    m = Model(optimizer)
    verbose || set_silent(m)
    tight && set_tight_sdp_tolerances!(m; tol=tol)

    set_name(m, "Hou_SDP_RLT")

    @variable(m, x[1:n])
    @variable(m, X[1:n,1:n], Symmetric)

    @objective(
        m,
        Min,
        0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n)
        + q0' * x
    )

    # Original quadratic rows, lifted through X.
    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(
            m,
            0.5 * sum(Qmat[i,j] * X[i,j] for i=1:n, j=1:n)
            + qvec' * x + rterm <= 0
        )
    end

    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(
            m,
            0.5 * sum(Pmat[i,j] * X[i,j] for i=1:n, j=1:n)
            + pvec' * x + sterm == 0
        )
    end

    # Original linear rows + their x-side first-level RLT.
    if ℓ > 0
        @constraint(m, A * x .<= b)
        @constraint(
            m,
            A*X*A'
            .- A*(x*b')
            .- (b*x')*A'
            .+ b*b'
            .>= 0
        )
    end

    if η > 0
        @constraint(m, H * x .== h)
        @constraint(m, H * X .== h * x')
    end

    add_shor!(m, x, X)
    add_sparsity_cone!(m, x, X, ρ)

    return m
end

export set_tight_sdp_tolerances!, add_shor!, add_sparsity_cone!, build_hou_sdp_rlt

end # module
