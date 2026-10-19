module RLTCompNoBounds

# ============================================================
# RLTComp variant WITHOUT x-bounds and their RLT products.
#
# The only difference from RLTComp:
#   - We do NOT add    xL <= x <= xU
#   - We do NOT add the x-box self-RLT (three constraint blocks
#     that involve xL, xU as products with x, X)
#   - We do NOT add the cross-RLT of (Ax<=b) with x-bounds
#   - We do NOT add the cross-RLT of v with x-bounds inside F_U^v
#
# Everything else (complementarity, F_E^v, F_B^v, F_U^v except its
# cross-RLT with xL/xU, original QCQP rows lifted via X, linear
# rows Ax<=b, Hx=h, and their RLT products among themselves) is
# preserved.
#
# This lets us ask: does Comp-RLT implicitly bound x from the
# lifted quadratic + complementarity + halfspace, or does it need
# an explicit x-box to be bounded?
#
# Same public API as RLTComp:
#   build_and_solve(data; variant="RLT_CompBinBou", ...)
# ============================================================

using LinearAlgebra: diag
using JuMP
using MathOptInterface
const MOI = MathOptInterface
using Parameters: @unpack

# Reuse prepare_instance from RLTBigM.
using ..RLTBigM: prepare_instance

# ---------- helpers ----------
outer_xy(a, b) = a * transpose(b)


"""
    additional_params(params)

Add derived constants. xL, xU are still computed (some code paths in
F_E^v / F_U^v reference them symbolically) but they will NOT be enforced
as constraints on x or its RLT products in this variant.
"""
function additional_params(params)
    xL  = -diag(params.Mminus)
    xU  =  diag(params.Mplus)
    eeT = params.e * params.e'
    return merge(params, (; xL, xU, eeT))
end


# ============================================================
#  Block: FC_comp (NO x-bounds, NO x-box self-RLT, NO A x cross-RLT with box)
# ============================================================
function add_FC_comp_nobounds!(m, x, v, X, V, W, params)
    @unpack n, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η = params

    # ---------- original QCQP rows (lifted via X) ----------
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

    # ---------- original linear rows ----------
    if ℓ > 0
        @constraint(m, A * x .<= b)
    end
    if η > 0
        @constraint(m, H * x .== h)
    end

    # ---------- NO x-bounds constraints  (removed) ----------
    # (was: xL <= x <= xU)

    # ---------- relaxed complementarity in the lift ----------
    @constraint(m, [i=1:n], W[i,i] == 0)

    # ============================================================
    # First-level RLT constraints from original linear rows only
    # (x-box self-RLT and cross-RLT with A are REMOVED here)
    # ============================================================

    # --- (Ax<=b) × (Ax<=b) ---
    if ℓ > 0
        @constraint(m, A*X*A' .- A*(x*b') .- (b*x')*A' .+ (b*b') .>= 0)
    end

    # --- (Hx=h) × variables: equality-products ---
    if η > 0
        @constraint(m, H * X .== outer_xy(h, x))
        @constraint(m, H * W .== outer_xy(h, v))
    end

    return nothing
end


# ============================================================
#  Block: FE_v (unchanged)
# ============================================================
function add_FE_v!(m, x, v, X, V, W, params)
    @unpack n, ρ, e = params
    @constraint(m, sum(v) == n - ρ)
    @constraint(m, (e' * V) .== (n - ρ) .* (v'))
    @constraint(m, (e' * W') .== (n - ρ) .* (x'))
    return nothing
end


# ============================================================
#  Block: FB_v (unchanged)
# ============================================================
function add_FB_v!(m, v, V, params)
    @unpack n = params
    @constraint(m, [i=1:n], V[i,i] == v[i])
    return nothing
end


# ============================================================
#  Block: FU_v with NO cross-RLT vs. x-bounds
#   Keeps: 0<=v<=e, self-RLT of that block, cross-RLT with A x<=b
#   Drops: cross-RLT of v with xL/xU (four constraint families)
# ============================================================
function add_FU_v_nobounds!(m, x, v, X, V, W, params)
    @unpack n, A, b, ℓ, e, eeT = params

    # Bound rows: 0 <= v <= e
    @constraint(m, 0 .<= v)
    @constraint(m, v .<= e)

    # Self-RLT of 0<=v<=e (V-block)
    @constraint(m, V .>= 0)
    @constraint(m, v*e' .- V .>= 0)
    @constraint(m, V .- v*e' .- e*v' .+ eeT .>= 0)

    # --- NO cross-RLT with x-bounds (removed) ---

    # Cross-RLT with Ax<=b:
    if ℓ > 0
        @constraint(m, b*v' .- A*W .>= 0)
        @constraint(m, b*e' .- (A*x)*e' .- b*v' .+ A*W .>= 0)
    end

    return nothing
end


# ============================================================
#  Build & solve
# ============================================================
"""
    build_and_solve(data; variant="RLT_CompBinBou", build_only=false, optimizer, verbose=false)

Variants supported: same names as RLTComp but with x-bounds and their
RLT products removed. EXACT variants are NOT provided here because
without x-bounds the EXACT MIQP is meaningless (Gurobi would need
some finite range on x to enumerate).
"""
function build_and_solve(data;
    variant::String="RLT_CompBinBou",
    build_only::Bool=false,
    optimizer,
    verbose::Bool=false
)
    params = additional_params(prepare_instance(data))
    @unpack n, ρ, Q0, q0 = params

    # Aliases
    if variant == "RLT_S2";   variant = "RLT_CompBin"; end
    if variant == "RLT_S2U";  variant = "RLT_CompBinBou"; end
    if variant == "RLT_S3";   variant = "RLT_CompBou"; end

    if !(variant in ("RLT_CompBin", "RLT_CompBinBou", "RLT_CompBou"))
        error("RLTCompNoBounds only supports RLT variants. Got: $variant")
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

    set_name(m, "SQCQP_Complementarity_NoBounds")

    @variable(m, x[1:n])
    @variable(m, v[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, V[1:n,1:n], Symmetric)
    @variable(m, W[1:n,1:n])

    @objective(m, Min, 0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n) + q0' * x)

    add_FC_comp_nobounds!(m, x, v, X, V, W, params)
    add_FE_v!(m, x, v, X, V, W, params)

    if variant == "RLT_CompBin"
        add_FB_v!(m, v, V, params)
    elseif variant == "RLT_CompBinBou"
        add_FB_v!(m, v, V, params)
        add_FU_v_nobounds!(m, x, v, X, V, W, params)
    elseif variant == "RLT_CompBou"
        add_FU_v_nobounds!(m, x, v, X, V, W, params)
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

export build_and_solve, add_FC_comp_nobounds!, add_FE_v!, add_FB_v!,
       add_FU_v_nobounds!, additional_params

end # module RLTCompNoBounds