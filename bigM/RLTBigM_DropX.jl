# ============================================================
# RLTBigM_DropX.jl
# ============================================================

module RLTBigMDropX

using LinearAlgebra
using Parameters
using JuMP
using Gurobi
using MathOptInterface
const MOI = MathOptInterface

# Single source of truth for instance parsing
using ..RLTBigM: prepare_instance

# ---------- helpers ----------
outer_xy(a, b) = a * transpose(b)

ensure_tuple(x) =
    x === nothing ? () :
    (x isa AbstractVector || x isa Tuple ? x : (x,))

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



# ============================================================
# FC block with Drop-X
#
# Drop-X rule:
# - original objective and original QCQP constraints are unchanged;
# - original linear constraints are unchanged;
# - Big-M linking constraints are unchanged;
# - diag(U)=u is unchanged;
# - only RLT-generated scalar rows are removed;
# - an RLT scalar row is removed if it contains any dropped X-entry.
# ============================================================

function add_FC!(
    m, x, u, X, R, U, params;
    drop_entries = Set{Tuple{Int,Int}}(),
    drop_tol::Real = 1e-10,
)
    @unpack n, Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η, Mminus, Mplus = params

    Mminus_diag = diag(Mminus)
    Mplus_diag  = diag(Mplus)

    # ------------------------------------------------------------
    # Original QCQP rows, lifted through X
    # These are NOT removed by Drop-X.
    # ------------------------------------------------------------
    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(m,
            0.5 * sum(Qmat[i,j] * X[i,j] for i in 1:n, j in 1:n) +
            qvec' * x + rterm <= 0
        )
    end

    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(m,
            0.5 * sum(Pmat[i,j] * X[i,j] for i in 1:n, j in 1:n) +
            pvec' * x + sterm == 0
        )
    end

    # ------------------------------------------------------------
    # Original linear constraints
    # These are NOT removed by Drop-X.
    # ------------------------------------------------------------
    if ℓ > 0
        @constraint(m, A * x .<= b)
    end

    if η > 0
        @constraint(m, H * x .== h)
    end

    # ------------------------------------------------------------
    # Original Big-M linking constraints
    # These are NOT removed by Drop-X.
    # ------------------------------------------------------------
    @constraint(m, -Mminus * u .<= x)
    @constraint(m,  x .<= Mplus * u)

    # ------------------------------------------------------------
    # Binary lift surrogate
    # This contains no X and is NOT removed.
    # ------------------------------------------------------------
    @constraint(m, [i = 1:n], U[i,i] == u[i])

    # ============================================================
    # RLT-generated rows
    # Drop-X is applied only from here onward.
    # ============================================================

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

        # H R = h u^T contains no X, so it is kept.
        @constraint(m, H * R .== outer_xy(h, u))
    end

    # ------------------------------------------------------------
    # Big-M McCormick/RLT blocks
    # These rows contain X, R, and U together.
    # If X[i,j] is dropped, the whole scalar row is skipped.
    # ------------------------------------------------------------

    # (Mplus*u - x)(Mplus*u - x)^T >= 0
    for i in 1:n, j in 1:n
        if !is_dropped_X(i, j, drop_entries)
            @constraint(m,
                Mplus_diag[i] * U[i,j] * Mplus_diag[j]
                - Mplus_diag[i] * R[j,i]
                - R[i,j] * Mplus_diag[j]
                + X[i,j] >= 0
            )
        end
    end

    # (Mminus*u + x)(Mminus*u + x)^T >= 0
    for i in 1:n, j in 1:n
        if !is_dropped_X(i, j, drop_entries)
            @constraint(m,
                Mminus_diag[i] * U[i,j] * Mminus_diag[j]
                + Mminus_diag[i] * R[j,i]
                + R[i,j] * Mminus_diag[j]
                + X[i,j] >= 0
            )
        end
    end

    # (Mplus*u - x)(Mminus*u + x)^T >= 0
    for i in 1:n, j in 1:n
        if !is_dropped_X(i, j, drop_entries)
            @constraint(m,
                Mplus_diag[i] * U[i,j] * Mminus_diag[j]
                + Mplus_diag[i] * R[j,i]
                - R[i,j] * Mminus_diag[j]
                - X[i,j] >= 0
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

        # (b - A x)_r (Mplus*u - x)_j >= 0
        for r in 1:ℓ, j in 1:n
            arow = A[r, :]

            if !contains_dropped_X_from_row_and_index(arow, j, drop_entries; tol = drop_tol)
                @constraint(m,
                    sum(A[r,i] * X[i,j] for i in 1:n)
                    - b[r] * x[j]
                    - Mplus_diag[j] * sum(A[r,i] * R[i,j] for i in 1:n)
                    + b[r] * Mplus_diag[j] * u[j] >= 0
                )
            end
        end

        # (b - A x)_r (Mminus*u + x)_j >= 0
        for r in 1:ℓ, j in 1:n
            arow = A[r, :]

            if !contains_dropped_X_from_row_and_index(arow, j, drop_entries; tol = drop_tol)
                @constraint(m,
                    - sum(A[r,i] * X[i,j] for i in 1:n)
                    + b[r] * x[j]
                    - Mminus_diag[j] * sum(A[r,i] * R[i,j] for i in 1:n)
                    + b[r] * Mminus_diag[j] * u[j] >= 0
                )
            end
        end
    end

    return nothing
end

# ============================================================
# FE block: equality cardinality
# ============================================================

function add_FE!(m, x, u, X, R, U, params)
    @unpack ρ, e = params

    @constraint(m, sum(u) == ρ)
    @constraint(m, R * e .== ρ .* x)
    @constraint(m, U * e .== ρ .* u)

    return nothing
end

# ============================================================
# FU block: explicit u <= e and its RLT consequences
# No X appears here, so Drop-X does not affect this block.
# ============================================================

function add_FU!(m, x, u, X, R, U, params)
    @unpack A, b, ℓ, Mminus, Mplus, e = params

    @constraint(m, u .<= 1)

    eeT = ones(length(u), length(u))
    @constraint(m, U .- (u * e') .- (e * u') .+ eeT .>= 0)

    if ℓ > 0
        @constraint(m, (b * e') .- (b * u') .- (A * x) * e' .+ A * R .>= 0)
    end

    # (e-u)(Mplus*u - x)^T >= 0
    @constraint(m, Mplus * (u * e') .- x * e' .- Mplus * U .+ R .>= 0)

    # (e-u)(x + Mminus*u)^T >= 0
    @constraint(m, Mminus * (u * e') .+ x * e' .- Mminus * U .- R .>= 0)

    return nothing
end

# ============================================================
# Build & solve
# ============================================================

"""
    build_and_solve(data; variant="E", optimizer=Gurobi.Optimizer, ...)

Supported Drop-X RLT variants:
- "E"  : eᵀu = rho
- "EU" : eᵀu = rho, u <= e and corresponding U-RLT rows

Drop-X is applied only to RLT-generated scalar rows containing dropped X entries.
"""
function build_and_solve(data;
    variant::String = "E",
    build_only::Bool = false,
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
    drop_X_entries = Tuple{Int,Int}[],
    drop_X_blocks = Tuple{Vector{Int},Vector{Int}}[],
    check_aggregate_zero::Bool = true,
    drop_tol::Real = 1e-10,
)
    params = prepare_instance(data)

    drop_entries = make_drop_entries(drop_X_entries, drop_X_blocks)

    if check_aggregate_zero
        assert_aggregate_zero_X(params, drop_entries; tol = drop_tol)
    end

    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η, Mminus, Mplus = params

    if !(variant in ("E", "EU"))
        error("RLTBigMDropX currently supports only variants \"E\" and \"EU\".")
    end

    m = Model(optimizer)

    if !verbose
        set_silent(m)
    else
        try
            set_optimizer_attribute(m, "OutputFlag", 1)
        catch
        end
    end

    set_name(m, "SQCQP_RLT_BigM_DropX")

    @variable(m, x[1:n])
    @variable(m, u[1:n])
    @variable(m, X[1:n, 1:n], Symmetric)
    @variable(m, R[1:n, 1:n])
    @variable(m, U[1:n, 1:n], Symmetric)

    @objective(m, Min,
        0.5 * sum(Q0[i,j] * X[i,j] for i in 1:n, j in 1:n) + q0' * x
    )

    add_FC!(
        m, x, u, X, R, U, params;
        drop_entries = drop_entries,
        drop_tol = drop_tol,
    )

    add_FE!(m, x, u, X, R, U, params)

    if variant == "EU"
        add_FU!(m, x, u, X, R, U, params)
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
            value.(u),
            value.(X),
            value.(R),
            value.(U),
        )
    elseif term == MOI.DUAL_INFEASIBLE
        return (:UNBOUNDED, nothing, nothing, nothing, nothing, nothing, nothing)
    elseif term == MOI.INFEASIBLE_OR_UNBOUNDED
        return (:INF_OR_UNBD, nothing, nothing, nothing, nothing, nothing, nothing)
    elseif term == MOI.INFEASIBLE
        return (:INFEASIBLE, nothing, nothing, nothing, nothing, nothing, nothing)
    else
        return (:OTHER, term, nothing, nothing, nothing, nothing, nothing)
    end
end

export build_and_solve,
       prepare_instance,
       add_FC!,
       add_FE!,
       add_FU!,
       make_drop_entries,
       assert_aggregate_zero_X

end # module RLTBigMDropX