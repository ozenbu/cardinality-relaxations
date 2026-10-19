module RLTCompNoBounds

# ============================================================
# Complementarity RLT formulation WITHOUT redundant x-bounds
# and WITHOUT any RLT products generated from those x-bounds.
#
# Non-support indicator:
#     v = e-u
# so v_i = 1 means x_i is excluded from the support.
#
# Let
#     k := n-rho.
#
# Two exact cardinality reformulations are supported:
#
#   (E)  e'v = k
#   (I)  e'v >= k
#
# The E- and I-variants have different first-level RLT closures.
# In particular:
#
#   E gives the equality-products
#       W e = k x,
#       V e = k v.
#
#   I instead uses the nonnegative slack
#       s_rho(v) := e'v-k >= 0
#   and its RLT products with the other nonnegative linear
#   factors present in the formulation.
#
# IMPORTANT (NO-BOUNDS version):
#   - no explicit xL <= x <= xU rows;
#   - no x-box self-RLT;
#   - no (Ax<=b) x x-bound RLT;
#   - no v-bound x x-bound RLT;
#   - no cardinality-slack x x-bound RLT.
#
# Original Ax<=b and Hx=h remain, together with their RLT lifts.
# ============================================================

using JuMP
using MathOptInterface
const MOI = MathOptInterface
using Parameters: @unpack

# Reuse the common instance parser.
using ..RLTBigM: prepare_instance

# ---------- helpers ----------
outer_xy(a, b) = a * transpose(b)


"""
    additional_params(params)

Add constants used by the complementarity RLT blocks.
No x-bounds are constructed or used in this module.
"""
function additional_params(params)
    eeT = params.e * params.e'
    return merge(params, (; eeT))
end


# ============================================================
# Block: C_Comp  (NO x-bounds)
# ============================================================
function add_FC_comp_nobounds!(m, x, v, X, V, W, params)
    @unpack n, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η = params

    # ---------- original QCQP rows, lifted through X ----------
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

    # ---------- original linear rows ----------
    if ℓ > 0
        @constraint(m, A * x .<= b)
    end

    if η > 0
        @constraint(m, H * x .== h)
    end

    # ---------- lifted complementarity x_i v_i = 0 ----------
    @constraint(m, [i=1:n], W[i,i] == 0)

    # ========================================================
    # First-level RLT from ORIGINAL x-linear rows only.
    # ========================================================

    # (b-Ax)(b-Ax)' >= 0
    if ℓ > 0
        @constraint(
            m,
            A*X*A'
            .- A*(x*b')
            .- (b*x')*A'
            .+ b*b'
            .>= 0
        )
    end

    # Equality x variables:
    #   (Hx-h) x' = 0,
    #   (Hx-h) v' = 0.
    if η > 0
        @constraint(m, H * X .== outer_xy(h, x))
        @constraint(m, H * W .== outer_xy(h, v))
    end

    return nothing
end


# ============================================================
# Block: E_k^v
# Cardinality equality e'v = k, k=n-rho.
# ============================================================
function add_FE_v!(m, x, v, X, V, W, params)
    @unpack n, ρ, e = params
    k = n - ρ

    # e'v = k
    @constraint(m, sum(v) == k)

    # Equality x variables:
    #   (e'v-k)x = 0  ->  We = kx
    #   (e'v-k)v = 0  ->  Ve = kv
    @constraint(m, W * e .== k .* x)
    @constraint(m, V * e .== k .* v)

    return nothing
end


# ============================================================
# Block: I_k^v
# Cardinality inequality e'v >= k, k=n-rho.
#
# Let s(v) = e'v-k >= 0.
# In the NO-BOUNDS formulation, the other x-side nonnegative
# factors available to pair with s are only b-Ax >= 0.
#
# Hence this block contains:
#   s >= 0,
#   s^2 >= 0,
#   s(b-Ax) >= 0.
#
# Products s*v and s*(e-v) belong to IU^v because the factors
# v>=0 and e-v>=0 are introduced only by U^v.
# ============================================================
function add_FI_v!(m, x, v, X, V, W, params)
    @unpack n, ρ, A, b, ℓ, e = params
    k = n - ρ

    # e'v-k >= 0
    @constraint(m, sum(v) >= k)

    # (e'v-k)^2 >= 0
    # -> e'Ve - 2k e'v + k^2 >= 0
    @constraint(
        m,
        sum(V[i,j] for i=1:n, j=1:n)
        - 2 * k * sum(v)
        + k^2
        >= 0
    )

    # (e'v-k)(b-Ax) >= 0
    # -> (e'v)b - AWe - kb + kAx >= 0
    if ℓ > 0
        @constraint(
            m,
            sum(v) .* b
            .- A * W * e
            .- k .* b
            .+ k .* (A * x)
            .>= 0
        )
    end

    # No additional product with Hx=h is required:
    # (e'v-k)(Hx-h)=0 is implied by Hx=h and HW=hv'.

    return nothing
end


# ============================================================
# Block: B^v
# Binary lift v_i^2 = v_i.
# ============================================================
function add_FB_v!(m, v, V, params)
    @unpack n = params
    @constraint(m, [i=1:n], V[i,i] == v[i])
    return nothing
end


# ============================================================
# Block: U^v  (NO cross-RLT with x-bounds)
#
# Adds the explicit support bounds
#     v >= 0,
#     e-v >= 0,
# their self-RLT products, and their products with b-Ax>=0.
# ============================================================
function add_FU_v_nobounds!(m, x, v, X, V, W, params)
    @unpack A, b, ℓ, e, eeT = params

    # 0 <= v <= e
    @constraint(m, 0 .<= v)
    @constraint(m, v .<= e)

    # v v' >= 0
    @constraint(m, V .>= 0)

    # v(e-v)' >= 0
    @constraint(m, v*e' .- V .>= 0)

    # (e-v)(e-v)' >= 0
    @constraint(m, V .- v*e' .- e*v' .+ eeT .>= 0)

    # Products with b-Ax >= 0.
    if ℓ > 0
        # (b-Ax)v' >= 0
        @constraint(m, b*v' .- A*W .>= 0)

        # (b-Ax)(e-v)' >= 0
        @constraint(
            m,
            b*e'
            .- (A*x)*e'
            .- b*v'
            .+ A*W
            .>= 0
        )
    end

    return nothing
end


# ============================================================
# Block: IU^v
# Joint products between the cardinality slack
#     e'v-k >= 0
# and the support-bound factors introduced by U^v:
#     v >= 0,
#     e-v >= 0.
#
# This block must therefore be used only together with U^v.
# ============================================================
function add_FIU_v!(m, v, V, params)
    @unpack n, ρ, e = params
    k = n - ρ

    # (e'v-k)v >= 0
    # -> Ve-kv >= 0
    @constraint(
        m,
        V * e .- k .* v .>= 0
    )

    # (e'v-k)(e-v) >= 0
    # -> (e'v)e - Ve - ke + kv >= 0
    @constraint(
        m,
        sum(v) .* e
        .- V * e
        .- k .* e
        .+ k .* v
        .>= 0
    )

    return nothing
end


# ============================================================
# Build & solve
# ============================================================
"""
    build_and_solve(
        data;
        variant="RLT_CompBinBou_E",
        build_only=false,
        optimizer,
        verbose=false,
    )

Core/no-x-bounds complementarity RLT formulations.

Explicit variants
-----------------
E-cardinality:
    RLT_CompBin_E
    RLT_CompBinBou_E
    RLT_CompBou_E

I-cardinality:
    RLT_CompBin_I
    RLT_CompBinBou_I
    RLT_CompBou_I

Blocks
------
    CompBin_E     = C + B + E
    CompBinBou_E  = C + B + E + U
    CompBou_E     = C + E + U

    CompBin_I     = C + B + I
    CompBinBou_I  = C + B + I + U + IU
    CompBou_I     = C + I + U + IU

Backward compatibility
----------------------
The old names are retained as aliases for the E-cardinality models:
    RLT_CompBin     -> RLT_CompBin_E
    RLT_CompBinBou  -> RLT_CompBinBou_E
    RLT_CompBou     -> RLT_CompBou_E
    RLT_S2          -> RLT_CompBin_E
    RLT_S2U         -> RLT_CompBinBou_E
    RLT_S3          -> RLT_CompBou_E
"""
function build_and_solve(
    data;
    variant::String="RLT_CompBinBou_E",
    build_only::Bool=false,
    optimizer,
    verbose::Bool=false,
)
    params = additional_params(prepare_instance(data))
    @unpack n, Q0, q0 = params

    # --------------------------------------------------------
    # Backward-compatible aliases: old models were E-models.
    # --------------------------------------------------------
    aliases = Dict(
        "RLT_CompBin"    => "RLT_CompBin_E",
        "RLT_CompBinBou" => "RLT_CompBinBou_E",
        "RLT_CompBou"    => "RLT_CompBou_E",
        "RLT_S2"         => "RLT_CompBin_E",
        "RLT_S2U"        => "RLT_CompBinBou_E",
        "RLT_S3"         => "RLT_CompBou_E",
    )
    variant = get(aliases, variant, variant)

    allowed = (
        "RLT_CompBin_E",
        "RLT_CompBinBou_E",
        "RLT_CompBou_E",
        "RLT_CompBin_I",
        "RLT_CompBinBou_I",
        "RLT_CompBou_I",
    )

    if !(variant in allowed)
        error(
            "RLTCompNoBounds supports only the six E/I RLT variants. " *
            "Got: $variant"
        )
    end

    m = Model(optimizer)

    if !verbose
        set_silent(m)
    else
        # OutputFlag is Gurobi-specific. Ignore it for other solvers.
        try
            set_optimizer_attribute(m, "OutputFlag", 1)
        catch
        end
    end

    set_name(m, "SQCQP_Complementarity_NoBounds_$(variant)")

    @variable(m, x[1:n])
    @variable(m, v[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, V[1:n,1:n], Symmetric)
    @variable(m, W[1:n,1:n])

    @objective(
        m,
        Min,
        0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n)
        + q0' * x
    )

    # --------------------------------------------------------
    # Common core block.
    # --------------------------------------------------------
    add_FC_comp_nobounds!(m, x, v, X, V, W, params)

    # --------------------------------------------------------
    # Cardinality block: E or I.
    # --------------------------------------------------------
    if endswith(variant, "_E")
        add_FE_v!(m, x, v, X, V, W, params)
    elseif endswith(variant, "_I")
        add_FI_v!(m, x, v, X, V, W, params)
    end

    # --------------------------------------------------------
    # Binary-lift block B^v.
    # --------------------------------------------------------
    if variant in (
        "RLT_CompBin_E",
        "RLT_CompBinBou_E",
        "RLT_CompBin_I",
        "RLT_CompBinBou_I",
    )
        add_FB_v!(m, v, V, params)
    end

    # --------------------------------------------------------
    # Explicit support-bound block U^v.
    # --------------------------------------------------------
    if variant in (
        "RLT_CompBinBou_E",
        "RLT_CompBou_E",
        "RLT_CompBinBou_I",
        "RLT_CompBou_I",
    )
        add_FU_v_nobounds!(m, x, v, X, V, W, params)
    end

    # --------------------------------------------------------
    # Joint I x U products: only for cardinality-I + U models.
    # --------------------------------------------------------
    if variant in (
        "RLT_CompBinBou_I",
        "RLT_CompBou_I",
    )
        add_FIU_v!(m, v, V, params)
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
        return (
            Symbol(term),
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
        )
    end
end


export build_and_solve,
       add_FC_comp_nobounds!,
       add_FE_v!,
       add_FI_v!,
       add_FB_v!,
       add_FU_v_nobounds!,
       add_FIU_v!,
       additional_params

end # module RLTCompNoBounds
