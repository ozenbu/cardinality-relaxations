###############################
# Drop-X test instances
###############################
#
# Purpose:
#   Small deterministic QCQP instances for testing whether selected X-block
#   RLT constraints can be removed.
#
# Main property:
#   For every entry listed in inst["drop_X_entries"], the corresponding entry is
#   zero in Q0, in every Qi, and in every Pi.
#
# Notes:
#   - These are test instances, not random benchmark instances.
#   - Existing solver/model files do not need to be modified.
#   - The metadata fields "drop_X_entries" and "drop_X_blocks" are included so
#     the experiment script can automatically decide which X-entries/blocks to drop.
#   - xbar is kept only as optional metadata; it is not required by the solvers.

module DropXTestInstances

export make_dropX_instances, check_dropX_instances

using LinearAlgebra

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

"Return box constraints ell <= x <= u encoded as A*x <= b."
function box_constraints(ell::Vector{Float64}, u::Vector{Float64})
    n = length(ell)
    A = vcat(Matrix{Float64}(I, n, n), -Matrix{Float64}(I, n, n))
    b = vcat(u, -ell)
    return A, b
end

"Compute common and asymmetric Big-M vectors from bounds."
function bigM_from_bounds(ell::Vector{Float64}, u::Vector{Float64}; scale::Float64 = 1.0)
    M      = scale .* max.(abs.(ell), abs.(u))
    Mminus = scale .* max.(0.0, -ell)
    Mplus  = scale .* max.(0.0,  u)
    return M, Mminus, Mplus
end

"Symmetrize a square matrix."
sym(A::AbstractMatrix) = 0.5 .* (Matrix{Float64}(A) .+ Matrix{Float64}(A)')

"Set selected symmetric entries to zero."
function zero_entries!(Q::Matrix{Float64}, entries::Vector{Tuple{Int,Int}})
    for (i,j) in entries
        Q[i,j] = 0.0
        Q[j,i] = 0.0
    end
    return Q
end

"Expand block pairs into scalar matrix entries."
function entries_from_blocks(blocks::Vector{Tuple{Vector{Int},Vector{Int}}})
    out = Tuple{Int,Int}[]
    for (I,J) in blocks
        for i in I, j in J
            push!(out, (min(i,j), max(i,j)))
        end
    end
    return unique(out)
end

"Package one instance in the same style as the existing dictionaries."
function pack_instance(;
    id::String,
    n::Int,
    rho::Int,
    Q0::Matrix{Float64},
    q0::Vector{Float64},
    ell::Vector{Float64},
    u::Vector{Float64},
    xbar::Vector{Float64},
    Qi_list::Vector{Matrix{Float64}} = Matrix{Float64}[],
    qi_list::Vector{Vector{Float64}} = Vector{Vector{Float64}}(),
    ri_list::Vector{Float64} = Float64[],
    Pi_list::Vector{Matrix{Float64}} = Matrix{Float64}[],
    pi_list::Vector{Vector{Float64}} = Vector{Vector{Float64}}(),
    si_list::Vector{Float64} = Float64[],
    H::Union{Nothing,Matrix{Float64}} = nothing,
    h::Union{Nothing,Vector{Float64}} = nothing,
    extra_A::Union{Nothing,Matrix{Float64}} = nothing,
    extra_b::Union{Nothing,Vector{Float64}} = nothing,
    drop_X_entries::Vector{Tuple{Int,Int}},
    drop_X_blocks::Vector{Tuple{Vector{Int},Vector{Int}}} = Tuple{Vector{Int},Vector{Int}}[],
    convexity::String = "CVX",
    base_tag::String = "B",
    seed::Int = 0,
    bigM_scale::Float64 = 1.0,
)
    @assert size(Q0) == (n,n)
    @assert length(q0) == n
    @assert length(ell) == n && length(u) == n
    @assert length(xbar) == n
    @assert norm(Q0 - Q0', Inf) <= 1e-9

    A_box, b_box = box_constraints(ell, u)
    A = extra_A === nothing ? A_box : vcat(A_box, extra_A)
    b = extra_b === nothing ? b_box : vcat(b_box, extra_b)

    M, Mminus, Mplus = bigM_from_bounds(ell, u; scale = bigM_scale)

    inst = Dict{String,Any}()

    inst["id"] = id
    inst["n"] = n
    inst["rho"] = rho

    inst["Q0"] = Q0
    inst["q0"] = q0

    inst["Qi"] = isempty(Qi_list) ? nothing : Tuple(Qi_list)
    inst["qi"] = isempty(qi_list) ? nothing : Tuple(qi_list)
    inst["ri"] = isempty(ri_list) ? nothing : ri_list

    inst["Pi"] = isempty(Pi_list) ? nothing : Tuple(Pi_list)
    inst["pi"] = isempty(pi_list) ? nothing : Tuple(pi_list)
    inst["si"] = isempty(si_list) ? nothing : si_list

    inst["A"] = A
    inst["b"] = b
    inst["H"] = H
    inst["h"] = h

    inst["ell"] = ell
    inst["u"] = u
    inst["M"] = M
    inst["M_minus"] = Mminus
    inst["M_plus"] = Mplus

    # Metadata only. The solvers do not use xbar.
    inst["xbar"] = xbar

    inst["base_tag"] = base_tag
    inst["convexity"] = convexity
    inst["seed"] = seed
    inst["bigM_scale"] = bigM_scale

    inst["n_LI"] = extra_A === nothing ? 0 : size(extra_A, 1)
    inst["n_LE"] = H === nothing ? 0 : size(H, 1)
    inst["n_QI"] = length(Qi_list)
    inst["n_QE"] = length(Pi_list)

    # Metadata for Drop-X experiments.
    inst["drop_X_entries"] = drop_X_entries
    inst["drop_X_blocks"] = drop_X_blocks

    return inst
end

"Return true if Q[i,j] and Q[j,i] are numerically zero for all requested entries."
function entries_zero(Q::Union{Nothing,Matrix{Float64}}, entries; tol::Float64 = 1e-10)
    Q === nothing && return true
    for (i,j) in entries
        if abs(Q[i,j]) > tol || abs(Q[j,i]) > tol
            return false
        end
    end
    return true
end

"Check the aggregate-zero property for all requested Drop-X entries."
function check_aggregate_zero(inst::Dict{String,Any}; tol::Float64 = 1e-10)
    entries = inst["drop_X_entries"]

    ok = entries_zero(inst["Q0"], entries; tol = tol)

    if inst["Qi"] !== nothing
        for Q in inst["Qi"]
            ok &= entries_zero(Matrix{Float64}(Q), entries; tol = tol)
        end
    end

    if inst["Pi"] !== nothing
        for P in inst["Pi"]
            ok &= entries_zero(Matrix{Float64}(P), entries; tol = tol)
        end
    end

    return ok
end

"Optional feasibility check for the stored anchor. This is not used by default."
function check_anchor(inst::Dict{String,Any}; tol::Float64 = 1e-8)
    if !haskey(inst, "xbar") || inst["xbar"] === nothing
        return true
    end

    x = inst["xbar"]

    A = inst["A"]
    b = inst["b"]
    @assert maximum(A * x .- b) <= tol "Anchor violates A*x <= b in $(inst["id"])"

    if inst["H"] !== nothing
        H = inst["H"]
        h = inst["h"]
        @assert maximum(abs.(H * x .- h)) <= tol "Anchor violates H*x == h in $(inst["id"])"
    end

    if inst["Qi"] !== nothing
        for (Q,q,r) in zip(inst["Qi"], inst["qi"], inst["ri"])
            val = 0.5 * dot(x, Q * x) + dot(q, x) + r
            @assert val <= tol "Anchor violates QI in $(inst["id"]): value = $val"
        end
    end

    if inst["Pi"] !== nothing
        for (P,p,s) in zip(inst["Pi"], inst["pi"], inst["si"])
            val = 0.5 * dot(x, P * x) + dot(p, x) + s
            @assert abs(val) <= tol "Anchor violates QE in $(inst["id"]): value = $val"
        end
    end

    return true
end

# ------------------------------------------------------------
# Five purpose-built instances
# ------------------------------------------------------------

function make_dropX_instances()
    instances = Vector{Dict{String,Any}}()

    # ========================================================
    # 1) Clean scalar off-diagonal test, box-only, convex QP.
    # Drop X_12. No Qi/Pi. H = nothing.
    # ========================================================
    begin
        n = 4
        rho = 2
        ell = fill(-1.0, n)
        u   = fill( 1.0, n)
        xbar = [0.5, 0.0, 0.0, 0.0]

        drop_entries = [(1,2)]

        Q0 = sym([
            4.0  0.0  0.5  0.0
            0.0  3.0 -0.4  0.2
            0.5 -0.4  5.0  0.3
            0.0  0.2  0.3  2.0
        ])
        zero_entries!(Q0, drop_entries)

        q0 = [1.0, -2.0, 0.5, 1.5]

        push!(instances, pack_instance(
            id = "dropX01_box_scalar_offdiag_CVX",
            n = n, rho = rho,
            Q0 = Q0, q0 = q0,
            ell = ell, u = u, xbar = xbar,
            drop_X_entries = drop_entries,
            convexity = "CVX",
            seed = 1001,
        ))
    end

    # ========================================================
    # 2) Singleton diagonal-block test, box-only, nonconvex QP.
    # Drop X_11. This is the singleton-block version of Q_aa = 0.
    # ========================================================
    begin
        n = 4
        rho = 2
        ell = fill(-1.0, n)
        u   = fill( 1.0, n)
        xbar = [0.0, 1.0, 0.0, 0.0]

        drop_entries = [(1,1)]

        Q0 = sym([
             0.0  0.0  0.0  0.0
             0.0 -3.0  1.2  0.3
             0.0  1.2  2.0 -0.6
             0.0  0.3 -0.6  1.5
        ])
        zero_entries!(Q0, drop_entries)

        q0 = [2.0, -1.0, 1.0, 0.0]

        push!(instances, pack_instance(
            id = "dropX02_box_singleton_diag_NCVX",
            n = n, rho = rho,
            Q0 = Q0, q0 = q0,
            ell = ell, u = u, xbar = xbar,
            drop_X_entries = drop_entries,
            convexity = "NCVX",
            seed = 1002,
        ))
    end

    # ========================================================
    # 3) Scalar off-diagonal test with one convex quadratic inequality.
    # Drop X_12. Aggregate-zero holds in Q0 and Q1.
    # ========================================================
    begin
        n = 5
        rho = 3
        ell = [-2.0, -1.0, -1.0, -2.0, -1.0]
        u   = [ 2.0,  1.5,  1.0,  2.0,  1.0]
        xbar = [1.0, 0.0, 0.0, 0.0, 0.0]

        drop_entries = [(1,2)]

        Q0 = sym([
            3.0  0.0  0.4 -0.2  0.0
            0.0  2.5 -0.1  0.3  0.2
            0.4 -0.1  4.0  0.5 -0.3
           -0.2  0.3  0.5  3.0  0.1
            0.0  0.2 -0.3  0.1  2.0
        ])
        zero_entries!(Q0, drop_entries)

        q0 = [-1.0, 0.5, 1.0, -0.5, 0.0]

        Q1 = Diagonal([1.0, 2.0, 1.0, 0.5, 1.5]) |> Matrix{Float64}
        zero_entries!(Q1, drop_entries)
        q1 = [0.2, -0.1, 0.0, 0.3, -0.2]
        r1 = -4.0

        push!(instances, pack_instance(
            id = "dropX03_box_QI_scalar_offdiag_CVX",
            n = n, rho = rho,
            Q0 = Q0, q0 = q0,
            ell = ell, u = u, xbar = xbar,
            Qi_list = [Q1],
            qi_list = [q1],
            ri_list = [r1],
            drop_X_entries = drop_entries,
            convexity = "CVX",
            seed = 1003,
        ))
    end

    # ========================================================
    # 4) Scalar off-diagonal test with one quadratic equality.
    # Drop X_12. Aggregate-zero holds in Q0 and P1.
    # The equality is x3^2 - x4^2 = 0.
    # ========================================================
    begin
        n = 5
        rho = 3
        ell = fill(-1.0, n)
        u   = fill( 1.0, n)
        xbar = zeros(Float64, n)

        drop_entries = [(1,2)]

        Q0 = sym([
            1.5  0.0  0.2  0.0  0.1
            0.0 -2.0  0.3 -0.4  0.0
            0.2  0.3  1.0  0.5 -0.2
            0.0 -0.4  0.5  2.0  0.3
            0.1  0.0 -0.2  0.3  1.0
        ])
        zero_entries!(Q0, drop_entries)

        q0 = [0.0, 1.0, -1.0, 0.5, -0.5]

        P1 = zeros(Float64, n, n)
        P1[3,3] = 2.0
        P1[4,4] = -2.0
        zero_entries!(P1, drop_entries)

        p1 = zeros(Float64, n)
        s1 = 0.0

        push!(instances, pack_instance(
            id = "dropX04_box_QE_scalar_offdiag_NCVX",
            n = n, rho = rho,
            Q0 = Q0, q0 = q0,
            ell = ell, u = u, xbar = xbar,
            Pi_list = [P1],
            pi_list = [p1],
            si_list = [s1],
            drop_X_entries = drop_entries,
            convexity = "NCVX",
            seed = 1004,
        ))
    end

    # ========================================================
    # 5) Block-separable linear constraints with a cross-block X reduction.
    # Blocks: I1 = {1,2}, I2 = {3,4}, I3 = {5,6}.
    # Drop X_{I1,I2}, i.e., entries (1,3),(1,4),(2,3),(2,4).
    # Aggregate-zero holds in Q0 and Q1.
    # ========================================================
    begin
        n = 6
        rho = 3
        ell = zeros(Float64, n)
        u   = ones(Float64, n)

        # Metadata only; not used by the solvers.
        xbar = [0.25, 0.25, 0.20, 0.10, 0.30, 0.30]

        drop_blocks = [([1,2], [3,4])]
        drop_entries = entries_from_blocks(drop_blocks)

        Q0 = sym([
            4.0  0.5  0.0  0.0  0.2 -0.1
            0.5  3.0  0.0  0.0 -0.2  0.3
            0.0  0.0  2.5  0.4  0.1  0.2
            0.0  0.0  0.4  3.5 -0.3  0.1
            0.2 -0.2  0.1 -0.3  2.0  0.5
           -0.1  0.3  0.2  0.1  0.5  2.2
        ])
        zero_entries!(Q0, drop_entries)

        q0 = [1.0, -1.0, 0.5, -0.5, 0.2, -0.2]

        Q1 = Diagonal([1.0, 1.0, 0.5, 0.5, 1.5, 1.5]) |> Matrix{Float64}
        zero_entries!(Q1, drop_entries)
        q1 = zeros(Float64, n)
        r1 = -2.0

        extra_A = [
             1.0 -1.0  0.0  0.0  0.0  0.0
             0.0  0.0  1.0  1.0  0.0  0.0
             0.0  0.0  0.0  0.0  1.0  1.0
        ]
        extra_b = [0.40, 0.70, 0.80]

        H = [
            1.0  1.0  0.0  0.0  0.0  0.0
            0.0  0.0  0.0  0.0  1.0 -1.0
        ]
        h = [0.50, 0.0]

        push!(instances, pack_instance(
            id = "dropX05_blocksep_AH_crossblock_QI_CVX",
            n = n, rho = rho,
            Q0 = Q0, q0 = q0,
            ell = ell, u = u, xbar = xbar,
            Qi_list = [Q1],
            qi_list = [q1],
            ri_list = [r1],
            H = H,
            h = h,
            extra_A = extra_A,
            extra_b = extra_b,
            drop_X_entries = drop_entries,
            drop_X_blocks = drop_blocks,
            convexity = "CVX",
            seed = 1005,
        ))
    end

    return instances
end

"Run sanity checks for all generated instances."
function check_dropX_instances(; verbose::Bool = true)
    instances = make_dropX_instances()

    for inst in instances
        @assert check_aggregate_zero(inst) "Aggregate-zero check failed for $(inst["id"])"

        if verbose
            println("OK: ", inst["id"])
            println("  drop_X_entries = ", inst["drop_X_entries"])
            println("  n = ", inst["n"], ", rho = ", inst["rho"],
                    ", n_QI = ", inst["n_QI"], ", n_QE = ", inst["n_QE"],
                    ", n_LE = ", inst["n_LE"], ", n_LI = ", inst["n_LI"])
        end
    end

    return true
end

end # module DropXTestInstances