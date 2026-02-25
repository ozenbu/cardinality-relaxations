# ============================================================
# SDP_complementarity.jl
#
# Reuses: RLTBigM.prepare_instance(data)
# Adds:   add_PSD_comp! (single plug-in for PSD constraint)
#
# Builds:
#   SDP_S2   : binary v, no explicit v-box
#   SDP_S2U  : binary v + explicit 0<=v<=1
#   SDP_S3   : continous 0<=v<=1
#
# Optional (if you want): RLTSDP_* by calling your existing RLT blocks + PSD plug-in.
# ============================================================

module SDPComplementarity

using LinearAlgebra
using JuMP
using MathOptInterface
const MOI = MathOptInterface

# Reuse your existing instance prep
using ..RLTBigM: prepare_instance, outer_xy  # outer_xy already defined in RLTBigM

# ----------------------------
# PSD plug-in (call this to add SDP strength)
# ----------------------------
function add_PSD_comp!(m, x, v, X, V, W)
    Y = @expression(m, [
        1    x'   v'
        x    X    W
        v    W'   V
    ])
    @constraint(m, Y in PSDCone())
    return nothing
end

# ----------------------------
# Common constraints for SDP complementarity family (NO ST/RLT equalities)
# ----------------------------
function add_common_comp_SDP!(m, x, v, X, V, W, params)
    n   = params.n
    ρ   = params.ρ
    Q0  = params.Q0
    q0  = params.q0
    Qi  = params.Qi
    qi  = params.qi
    ri  = params.ri
    Pi  = params.Pi
    pi  = params.pi
    si  = params.si
    A   = params.A
    b   = params.b
    ℓ   = params.ℓ
    H   = params.H
    h   = params.h
    η   = params.η
    e   = params.e

    # Objective (SDP/Shor)
    @objective(m, Min, 0.5 * sum(Q0[i,j]*X[i,j] for i=1:n, j=1:n) + q0' * x)

    # Original QCQP rows (lifted via X)
    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(m,
            0.5 * sum(Qmat[i,j]*X[i,j] for i=1:n, j=1:n) + qvec' * x + rterm <= 0)
    end
    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(m,
            0.5 * sum(Pmat[i,j]*X[i,j] for i=1:n, j=1:n) + pvec' * x + sterm == 0)
    end

    # Original linear rows
    if ℓ > 0; @constraint(m, A * x .<= b); end
    if η > 0; @constraint(m, H * x .== h); end

    # Bounds (as in S2/S2+/S3)
    @constraint(m, 0 .<= x)
    @constraint(m, x .<= e)

    # Cardinality equality (common)
    @constraint(m, sum(v) == n - ρ)

    # Complementarity relaxation in lift
    @constraint(m, [i=1:n], W[i,i] == 0)     # diag(W)=0

    # PSD lift (plug-in)
    add_PSD_comp!(m, x, v, X, V, W)

    return nothing
end

# ----------------------------
# Variant add-ons
# ----------------------------
function add_binlift_v!(m, v, V, n::Int)
    @constraint(m, [i=1:n], V[i,i] == v[i])  # diag(V)=v
    return nothing
end

function add_v_box!(m, v, e)
    @constraint(m, 0 .<= v)
    @constraint(m, v .<= e)                  # e = ones(n)
    return nothing
end

# ----------------------------
# Build & solve
# ----------------------------
"""
    build_and_solve(data; variant="SDP_S2", optimizer, build_only=false, verbose=false)

Variants:
- "SDP_S2"   : diag(V)=v, no explicit v-box
- "SDP_S2U"  : diag(V)=v + explicit 0<=v<=1
- "SDP_S3"   : explicit 0<=v<=1, no diag(V)=v
"""
function build_and_solve(data;
    variant::String="SDP_S2",
    optimizer,
    build_only::Bool=false,
    verbose::Bool=false
)
    params = prepare_instance(data)
    n = params.n
    e = params.e

    m = Model(optimizer)
    verbose ? set_optimizer_attribute(m, "LOG", 1) : set_silent(m)

    # SDP variables
    @variable(m, x[1:n])
    @variable(m, v[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, V[1:n,1:n], Symmetric)
    @variable(m, W[1:n,1:n])

    # Common SDP block
    add_common_comp_SDP!(m, x, v, X, V, W, params)

    # Variant-specific
    if variant == "SDP_S2"
        add_binlift_v!(m, v, V, n)

    elseif variant == "SDP_S2U"
        add_binlift_v!(m, v, V, n)
        add_v_box!(m, v, e)

    elseif variant == "SDP_S3"
        add_v_box!(m, v, e)

    else
        error("Unknown variant: $variant. Use SDP_S2, SDP_S2U, SDP_S3.")
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

end # module SDPComplementarity

using MosekTools
using .RLTComp
using .SDPComplementarity

function demo_sdp_comp(; verbose::Bool=false)
    data = RLTComp.make_data_test4()

    for var in ("SDP_S2", "SDP_S2U", "SDP_S3")
        println("---- $var ----")
        res = SDPComplementarity.build_and_solve(
            data;
            variant=var,
            optimizer=MosekTools.Optimizer,
            verbose=verbose
        )

        println("status = ", res[1], "  obj = ", res[2])
        if res[1] == :OPTIMAL
            println("x = ", res[3])
            println("v = ", res[4])
        end
        println()
    end

    return nothing
end

demo_sdp_comp(verbose=false)