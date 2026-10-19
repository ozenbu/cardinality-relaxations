module Gk2SSP

using LinearAlgebra
using JuMP
using MosekTools
using MathOptInterface
using Random
const MOI = MathOptInterface

export build_Gk2_SSP_model,
       solve_Gk2_SSP,
       exact_ssp_by_support_enumeration,
       make_gaussian_ssp_instance

# -----------------------------------------------------------------------------
# Paper notation:
#   G^n_{k,2} = { X in S^n_+ :
#       sum_{j != i} Y^{ {i,j} }_{ii} = (k-1) X_ii,
#       Y^{ {i,j} }_{ii} <= X_ii,
#       [ Y^{ {i,j} }_{ii}   X_ij
#         X_ij                 Y^{ {i,j} }_{jj} ] in S^2_+
#   }
#
# For implementation we store one pair p=(i,j), i<j, and two diagonal
# variables:
#   Yleft[p]  = Y^{ {i,j} }_{ii}
#   Yright[p] = Y^{ {i,j} }_{jj}.
# The off-diagonal entry is exactly X_ij, as in (3.9).
# -----------------------------------------------------------------------------

all_pairs(n::Int) = [(i, j) for i in 1:n-1 for j in i+1:n]

"""
    build_Gk2_SSP_model(A, b, k; optimizer=MosekTools.Optimizer, verbose=false)

Build the SSP relaxation obtained from formulation (3.10) by replacing
FW^n_k with G^n_{k,2} from (3.9).

The model is

    min  b'b - tr(X A' b b' A)
    s.t. tr(X A' A) = 1,
         X in G^n_{k,2}.

This is the paper's X-only conic formulation; no regression vector x is
needed in this model.
"""
function build_Gk2_SSP_model(
    A::AbstractMatrix,
    b::AbstractVector,
    k::Int;
    optimizer=MosekTools.Optimizer,
    verbose::Bool=false,
)
    m, n = size(A)
    @assert length(b) == m "length(b) must equal size(A,1)"
    @assert 2 <= k <= n "G^n_{k,2} requires 2 <= k <= n"

    model = Model(optimizer)
    if !verbose
        set_silent(model)
    end
    set_name(model, "SSP_G_$(k)_2_n$(n)")

    # Constants appearing in (3.10).
    C = Matrix(Symmetric(A' * A))
    c = A' * b
    B = Matrix(Symmetric(c * c'))  # A' b b' A
    btb = dot(b, b)

    # Main lifted matrix X.
    @variable(model, X[1:n, 1:n], Symmetric)
    @constraint(model, X in PSDCone())

    pairs = all_pairs(n)
    npairs = length(pairs)

    # For p=(i,j), i<j:
    #   Yleft[p]  = Y^{ {i,j} }_{ii}
    #   Yright[p] = Y^{ {i,j} }_{jj}.
    @variable(model, Yleft[1:npairs] >= 0)
    @variable(model, Yright[1:npairs] >= 0)

    # Pairwise PSD blocks and valid upper bounds from (3.9).
    for p in 1:npairs
        i, j = pairs[p]

        @constraint(model, Yleft[p] <= X[i, i])
        @constraint(model, Yright[p] <= X[j, j])

        @constraint(
            model,
            Symmetric([
                Yleft[p]  X[i, j];
                X[i, j]   Yright[p]
            ]) in PSDCone()
        )
    end

    # Sum_{j != i} Y^{ {i,j} }_{ii} = (k-1) X_ii.
    for i in 1:n
        expr = AffExpr()
        for p in 1:npairs
            a, cidx = pairs[p]
            if i == a
                add_to_expression!(expr, 1.0, Yleft[p])
            elseif i == cidx
                add_to_expression!(expr, 1.0, Yright[p])
            end
        end
        @constraint(model, expr == (k - 1) * X[i, i])
    end

    # SSP normalization tr(X A'A) = 1.
    @constraint(
        model,
        sum(C[i, j] * X[i, j] for i in 1:n, j in 1:n) == 1.0
    )

    # SSP lower-bound objective from (3.10).
    @objective(
        model,
        Min,
        btb - sum(B[i, j] * X[i, j] for i in 1:n, j in 1:n)
    )

    return (
        model=model,
        X=X,
        Yleft=Yleft,
        Yright=Yright,
        pairs=pairs,
        C=C,
        B=B,
        btb=btb,
    )
end

"""
    solve_Gk2_SSP(A, b, k; optimizer=MosekTools.Optimizer, verbose=false)

Build and solve the G^n_{k,2}-based SSP relaxation.
Returns status, lower bound, primal X and the auxiliary pair-diagonal values.
"""
function solve_Gk2_SSP(
    A::AbstractMatrix,
    b::AbstractVector,
    k::Int;
    optimizer=MosekTools.Optimizer,
    verbose::Bool=false,
)
    built = build_Gk2_SSP_model(
        A,
        b,
        k;
        optimizer=optimizer,
        verbose=verbose,
    )

    t = @elapsed optimize!(built.model)
    st = termination_status(built.model)

    if st in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
        return (
            status=st,
            lower_bound=objective_value(built.model),
            X=value.(built.X),
            Yleft=value.(built.Yleft),
            Yright=value.(built.Yright),
            pairs=built.pairs,
            time=t,
        )
    end

    return (
        status=st,
        lower_bound=nothing,
        X=nothing,
        Yleft=nothing,
        Yright=nothing,
        pairs=built.pairs,
        time=t,
    )
end

# -----------------------------------------------------------------------------
# Exact small-scale SSP benchmark by support enumeration.
# For least squares, an optimum with <= k selected columns can be represented
# using a support of size exactly k (adding columns cannot worsen the residual),
# so for k>=1 it is enough to enumerate supports of cardinality k.
# -----------------------------------------------------------------------------
function exact_ssp_by_support_enumeration(
    A::AbstractMatrix,
    b::AbstractVector,
    k::Int,
)
    m, n = size(A)
    @assert length(b) == m
    @assert 1 <= k <= n
    @assert n <= 62 "bit-mask enumeration uses UInt64; use n <= 62"

    best_obj = Inf
    best_support = Int[]
    best_x = zeros(Float64, n)

    total_masks = UInt64(1) << n
    for mask in UInt64(0):(total_masks - UInt64(1))
        count_ones(mask) == k || continue

        S = Int[]
        for j in 1:n
            if ((mask >> (j - 1)) & UInt64(1)) == UInt64(1)
                push!(S, j)
            end
        end

        AS = A[:, S]
        xS = AS \ b
        residual = b - AS * xS
        obj = dot(residual, residual)

        if obj < best_obj
            best_obj = obj
            best_support = copy(S)
            fill!(best_x, 0.0)
            best_x[S] = xS
        end
    end

    return (
        optimum=best_obj,
        support=best_support,
        x=best_x,
    )
end

"""
    make_gaussian_ssp_instance(n, k; m=2n, seed=1, normalize=true)

Generate a small random SSP instance. If `normalize=true`, every column of A
and b are normalized to Euclidean norm one, matching the convention used in
the numerical section of the factor-width paper.
"""
function make_gaussian_ssp_instance(
    n::Int,
    k::Int;
    m::Int=2n,
    seed::Int=1,
    normalize::Bool=true,
)
    @assert 1 <= k <= n

    rng = Random.MersenneTwister(seed)

    A = randn(rng, m, n)
    b = randn(rng, m)

    if normalize
        for j in 1:n
            nj = norm(A[:, j])
            nj > 0 && (A[:, j] ./= nj)
        end
        nb = norm(b)
        nb > 0 && (b ./= nb)
    end

    return (
        A=A,
        b=b,
        n=n,
        m=m,
        k=k,
        seed=seed,
    )
end

end # module Gk2SSP
