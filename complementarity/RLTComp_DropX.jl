# ============================================================
# RLTComp_DropX.jl
# ============================================================

module RLTCompDropX

using LinearAlgebra: diag
using JuMP
using MathOptInterface
const MOI = MathOptInterface
using Parameters: @unpack

# Single source of truth for instance parsing
using ..RLTBigM: prepare_instance

# ---------- helpers ----------
outer_xy(a, b) = a * transpose(b)

# ============================================================
# Drop-X helpers
# ============================================================

function canon_pair(i::Int, j::Int)
    return i <= j ? (i, j) : (j, i)
end

function make_drop_entries(drop_X_entries, drop_X_blocks)
    entries = Set{Tuple{Int,Int}}()

    # Scalar entries, e.g. [(1,2), (3,3)]
    for (i, j) in drop_X_entries
        push!(entries, canon_pair(Int(i), Int(j)))
    end

    # Block entries, e.g. [([1,2], [3,4])]
    for (I, J) in drop_X_blocks
        for i in I, j in J
            push!(entries, canon_pair(Int(i), Int(j)))
        end
    end

    return entries
end

function is_dropped_X(i::Int, j::Int, drop_entries)
    return canon_pair(i, j) in drop_entries
end

function row_support(row; tol::Real = 1e-10)
    return findall(k -> abs(row[k]) > tol, eachindex(row))
end

function contains_dropped_X_from_rows(a, b, drop_entries; tol::Real = 1e-10)
    isempty(drop_entries) && return false

    I = row_support(a; tol = tol)
    J = row_support(b; tol = tol)

    for i in I, j in J
        if is_dropped_X(Int(i), Int(j), drop_entries)
            return true
        end
    end

    return false
end

function contains_dropped_X_from_row_and_index(a, j::Int, drop_entries; tol::Real = 1e-10)
    isempty(drop_entries) && return false

    I = row_support(a; tol = tol)

    for i in I
        if is_dropped_X(Int(i), j, drop_entries)
            return true
        end
    end

    return false
end

function assert_aggregate_zero_X(params, drop_entries; tol::Real = 1e-10)
    isempty(drop_entries) && return nothing

    @unpack n, Q0, Qi, Pi = params

    for (i, j) in drop_entries
        @assert 1 <= i <= n && 1 <= j <= n "Drop-X entry ($i,$j) is outside 1:n."

        @assert abs(Q0[i,j]) <= tol "Q0[$i,$j] is nonzero, cannot drop X[$i,$j]."
        @assert abs(Q0[j,i]) <= tol "Q0[$j,$i] is nonzero, cannot drop X[$i,$j]."

        for (k, Qmat) in enumerate(Qi)
            @assert abs(Qmat[i,j]) <= tol "Qi[$k][$i,$j] is nonzero, cannot drop X[$i,$j]."
            @assert abs(Qmat[j,i]) <= tol "Qi[$k][$j,$i] is nonzero, cannot drop X[$i,$j]."
        end

        for (k, Pmat) in enumerate(Pi)
            @assert abs(Pmat[i,j]) <= tol "Pi[$k][$i,$j] is nonzero, cannot drop X[$i,$j]."
            @assert abs(Pmat[j,i]) <= tol "Pi[$k][$j,$i] is nonzero, cannot drop X[$i,$j]."
        end
    end

    return nothing
end

"""
    additional_params(params)

Add complementarity-side derived constants once:
- xL, xU : vectors for the general box bounds on x
- eeT    : all-ones matrix (e eᵀ)
"""
function additional_params(params)
    xL  = -diag(params.Mminus)
    xU  =  diag(params.Mplus)
    eeT = params.e * params.e'
    return merge(params, (; xL, xU, eeT))
end

# ============================================================
#  Block: FC_comp (common)
#
#  Drop-X rule:
#  - original objective and original QCQP constraints are unchanged;
#  - original linear constraints are unchanged;
#  - only RLT-generated scalar constraints are removed;
#  - an RLT scalar row is removed if it contains a dropped X-entry.
# ============================================================

function add_FC_comp!(
    m, x, v, X, V, W, params;
    drop_entries = Set{Tuple{Int,Int}}(),
    drop_tol::Real = 1e-10,
)
    @unpack n, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, xL, xU = params

    # ------------------------------------------------------------
    # Original QCQP rows, lifted through X
    # These are NOT removed by Drop-X.
    # ------------------------------------------------------------
    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(m,
            0.5 * sum(Qmat[i,j] * X[i,j] for i = 1:n, j = 1:n) +
            qvec' * x + rterm <= 0
        )
    end

    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(m,
            0.5 * sum(Pmat[i,j] * X[i,j] for i = 1:n, j = 1:n) +
            pvec' * x + sterm == 0
        )
    end

    # ------------------------------------------------------------
    # Original linear rows
    # These are NOT removed by Drop-X.
    # ------------------------------------------------------------
    if ℓ > 0
        @constraint(m, A * x .<= b)
    end

    if η > 0
        @constraint(m, H * x .== h)
    end

    # ------------------------------------------------------------
    # General x-bounds, part of the original model
    # These are NOT removed by Drop-X.
    # ------------------------------------------------------------
    @constraint(m, xL .<= x)
    @constraint(m, x .<= xU)

    # ------------------------------------------------------------
    # Relaxed complementarity in the lift
    # ------------------------------------------------------------
    @constraint(m, [i = 1:n], W[i,i] == 0)

    # ============================================================
    # First-level RLT constraints generated from original rows
    # Drop-X is applied only here.
    # ============================================================

    # ------------------------------------------------------------
    # x-box self-RLT
    # ------------------------------------------------------------

    # (x - xL)(x - xL)^T >= 0
    for i in 1:n, j in 1:n
        if !is_dropped_X(i, j, drop_entries)
            @constraint(m,
                X[i,j]
                - x[i] * xL[j]
                - xL[i] * x[j]
                + xL[i] * xL[j] >= 0
            )
        end
    end

    # (x - xL)(xU - x)^T >= 0
    for i in 1:n, j in 1:n
        if !is_dropped_X(i, j, drop_entries)
            @constraint(m,
                x[i] * xU[j]
                - X[i,j]
                - xL[i] * xU[j]
                + xL[i] * x[j] >= 0
            )
        end
    end

    # (xU - x)(xU - x)^T >= 0
    for i in 1:n, j in 1:n
        if !is_dropped_X(i, j, drop_entries)
            @constraint(m,
                X[i,j]
                - xU[i] * x[j]
                - x[i] * xU[j]
                + xU[i] * xU[j] >= 0
            )
        end
    end

    # ------------------------------------------------------------
    # RLT constraints from Ax <= b
    # ------------------------------------------------------------
    if ℓ > 0

        # (b - A x)(b - A x)^T >= 0
        for r in 1:ℓ, s in 1:ℓ
            arow = A[r, :]
            brow = A[s, :]

            if !contains_dropped_X_from_rows(arow, brow, drop_entries; tol = drop_tol)
                @constraint(m,
                    sum(A[r,i] * A[s,j] * X[i,j] for i in 1:n, j in 1:n)
                    - b[s] * sum(A[r,i] * x[i] for i in 1:n)
                    - b[r] * sum(A[s,j] * x[j] for j in 1:n)
                    + b[r] * b[s] >= 0
                )
            end
        end

        # (b - A x)(x - xL)^T >= 0
        for r in 1:ℓ, j in 1:n
            arow = A[r, :]

            if !contains_dropped_X_from_row_and_index(arow, j, drop_entries; tol = drop_tol)
                @constraint(m,
                    - sum(A[r,i] * X[i,j] for i in 1:n)
                    + b[r] * x[j]
                    + xL[j] * sum(A[r,i] * x[i] for i in 1:n)
                    - b[r] * xL[j] >= 0
                )
            end
        end

        # (b - A x)(xU - x)^T >= 0
        for r in 1:ℓ, j in 1:n
            arow = A[r, :]

            if !contains_dropped_X_from_row_and_index(arow, j, drop_entries; tol = drop_tol)
                @constraint(m,
                    sum(A[r,i] * X[i,j] for i in 1:n)
                    - xU[j] * sum(A[r,i] * x[i] for i in 1:n)
                    - b[r] * x[j]
                    + b[r] * xU[j] >= 0
                )
            end
        end
    end

    # ------------------------------------------------------------
    # Equality-products from Hx = h
    # ------------------------------------------------------------
    if η > 0

        # H X = h x^T
        # Drop scalar equality rows containing dropped X.
        for r in 1:η, j in 1:n
            hrow = H[r, :]

            if !contains_dropped_X_from_row_and_index(hrow, j, drop_entries; tol = drop_tol)
                @constraint(m,
                    sum(H[r,i] * X[i,j] for i in 1:n) == h[r] * x[j]
                )
            end
        end

        # H W = h v^T contains no X, so it is kept entirely.
        @constraint(m, H * W .== outer_xy(h, v))
    end

    return nothing
end

# ============================================================
#  Block: FE_v
#  Cardinality equality + equality × variable products
# ============================================================

function add_FE_v!(m, x, v, X, V, W, params)
    @unpack n, ρ, e = params

    @constraint(m, sum(v) == n - ρ)
    @constraint(m, (e' * V) .== (n - ρ) .* (v'))
    @constraint(m, (e' * W') .== (n - ρ) .* (x'))

    return nothing
end

# ============================================================
#  Block: FB_v
#  Binary lift surrogate: diag(V)=v
# ============================================================

function add_FB_v!(m, v, V, params)
    @unpack n = params

    @constraint(m, [i = 1:n], V[i,i] == v[i])

    return nothing
end

# ============================================================
#  Block: FU_v
#  Explicit 0 <= v <= e and its self-RLT + cross-RLT
# ============================================================

function add_FU_v!(m, x, v, X, V, W, params)
    @unpack n, A, b, ℓ, e, eeT, xL, xU = params

    # Bound rows: 0 <= v <= e
    @constraint(m, 0 .<= v)
    @constraint(m, v .<= e)

    # Self-RLT of 0 <= v <= e
    @constraint(m, V .>= 0)
    @constraint(m, v * e' .- V .>= 0)
    @constraint(m, V .- v * e' .- e * v' .+ eeT .>= 0)

    # Cross-RLT with general x-bounds
    # These contain W but no X, so Drop-X does not affect them.
    @constraint(m, W .- xL * v' .>= 0)
    @constraint(m, x * e' .- W .- xL * e' .+ xL * v' .>= 0)
    @constraint(m, xU * v' .- W .>= 0)
    @constraint(m, xU * e' .- xU * v' .- x * e' .+ W .>= 0)

    # Cross-RLT with Ax <= b
    # These contain W but no X, so Drop-X does not affect them.
    if ℓ > 0
        @constraint(m, b * v' .- A * W .>= 0)
        @constraint(m, b * e' .- (A * x) * e' .- b * v' .+ A * W .>= 0)
    end

    return nothing
end

# ============================================================
#  Build & solve
# ============================================================

"""
    build_and_solve(data; variant="RLT_CompBinBou", build_only=false, optimizer, verbose=false,
                    drop_X_entries=[], drop_X_blocks=[])

Variants:
- "EXACT_CompBin"
- "EXACT_CompBou"
- "RLT_CompBin"
- "RLT_CompBinBou"
- "RLT_CompBou"

Drop-X is applied only to RLT relaxations.
"""
function build_and_solve(data;
    variant::String = "RLT_CompBinBou",
    build_only::Bool = false,
    optimizer,
    verbose::Bool = false,
    drop_X_entries = Tuple{Int,Int}[],
    drop_X_blocks = Tuple{Vector{Int},Vector{Int}}[],
    check_aggregate_zero::Bool = true,
    drop_tol::Real = 1e-10,
)
    params = additional_params(prepare_instance(data))

    drop_entries = make_drop_entries(drop_X_entries, drop_X_blocks)

    if check_aggregate_zero
        assert_aggregate_zero_X(params, drop_entries; tol = drop_tol)
    end

    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, e, xL, xU = params

    # Aliases
    if variant == "EXACT_S2"; variant = "EXACT_CompBin"; end
    if variant == "EXACT_S3"; variant = "EXACT_CompBou"; end
    if variant == "RLT_S2";   variant = "RLT_CompBin"; end
    if variant == "RLT_S2U";  variant = "RLT_CompBinBou"; end
    if variant == "RLT_S3";   variant = "RLT_CompBou"; end

    m = Model(optimizer)

    if !verbose
        set_silent(m)
    else
        try
            set_optimizer_attribute(m, "OutputFlag", 1)
        catch
        end
    end

    try
        set_optimizer_attribute(m, "NonConvex", 2)
    catch
    end

    set_name(m, "SQCQP_Complementarity_DropX")

    # ============================================================
    # EXACT_CompBin
    # ============================================================
    if variant == "EXACT_CompBin"
        @variable(m, x[1:n])
        @variable(m, v[1:n], Bin)

        @objective(m, Min, 0.5 * x' * Q0 * x + q0' * x)

        if ℓ > 0
            @constraint(m, A * x .<= b)
        end
        if η > 0
            @constraint(m, H * x .== h)
        end

        @constraint(m, xL .<= x)
        @constraint(m, x .<= xU)

        @constraint(m, sum(v) == n - ρ)

        @constraint(m, [i = 1:n], x[i] >= xL[i] * (1 - v[i]))
        @constraint(m, [i = 1:n], x[i] <= xU[i] * (1 - v[i]))

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

        return term == MOI.OPTIMAL ?
            (:OPTIMAL, objective_value(m), value.(x), value.(v)) :
            (Symbol(term), nothing, nothing, nothing)
    end

    # ============================================================
    # EXACT_CompBou
    # ============================================================
    if variant == "EXACT_CompBou"
        @variable(m, x[1:n])
        @variable(m, 0 <= v[1:n] <= 1)

        @objective(m, Min, 0.5 * x' * Q0 * x + q0' * x)

        if ℓ > 0
            @constraint(m, A * x .<= b)
        end
        if η > 0
            @constraint(m, H * x .== h)
        end

        @constraint(m, xL .<= x)
        @constraint(m, x .<= xU)

        @constraint(m, sum(v) == n - ρ)
        @constraint(m, [i = 1:n], x[i] * v[i] == 0)

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

        return term == MOI.OPTIMAL ?
            (:OPTIMAL, objective_value(m), value.(x), value.(v)) :
            (Symbol(term), nothing, nothing, nothing)
    end

    # ============================================================
    # RLT relaxations
    # ============================================================
    @variable(m, x[1:n])
    @variable(m, v[1:n])
    @variable(m, X[1:n, 1:n], Symmetric)
    @variable(m, V[1:n, 1:n], Symmetric)
    @variable(m, W[1:n, 1:n])

    @objective(m, Min,
        0.5 * sum(Q0[i,j] * X[i,j] for i = 1:n, j = 1:n) + q0' * x
    )

    add_FC_comp!(
        m, x, v, X, V, W, params;
        drop_entries = drop_entries,
        drop_tol = drop_tol,
    )

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
        return (
            :OPTIMAL,
            objective_value(m),
            value.(x),
            value.(v),
            value.(X),
            value.(V),
            value.(W),
        )
    else
        return (Symbol(term), nothing, nothing, nothing, nothing, nothing, nothing)
    end
end

export build_and_solve,
       add_FC_comp!,
       add_FE_v!,
       add_FB_v!,
       add_FU_v!,
       additional_params,
       make_drop_entries,
       assert_aggregate_zero_X

end # module RLTCompDropX