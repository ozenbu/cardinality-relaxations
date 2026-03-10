module SDPComplementarity

using LinearAlgebra
using JuMP
using MosekTools
using MathOptInterface
const MOI = MathOptInterface

# Reuse your existing instance prep (must have been loaded before this module)
using ..RLTBigM: prepare_instance

# ----------------------------
# PSD lift: Y = [1 x' v'; x X W; v W' V] ⪰ 0
# (Implemented via a symmetric variable Y with equality linking)
# ----------------------------
function add_PSD_comp!(m::Model, x, v, X, V, W; n::Int)
    @variable(m, Y[1:(2n+1), 1:(2n+1)], Symmetric)
    @constraint(m, Y .== [1.0 x' v'; x X W; v W' V])
    @constraint(m, Y in PSDCone())
    return Y
end

# ----------------------------
# Common constraints (NO ST/RLT equalities)
# ----------------------------
function add_common_comp_SDP!(m::Model, x, v, X, V, W, params)
    n  = params.n
    ρ  = params.ρ
    Q0 = params.Q0
    q0 = params.q0
    Qi = params.Qi
    qi = params.qi
    ri = params.ri
    Pi = params.Pi
    pi = params.pi
    si = params.si
    A  = params.A
    b  = params.b
    ℓ  = params.ℓ
    H  = params.H
    h  = params.h
    η  = params.η
    e  = params.e

    # Objective: 0.5⟨Q0, X⟩ + q0ᵀx
    @objective(m, Min, 0.5 * sum(Q0[i,j]*X[i,j] for i=1:n, j=1:n) + q0' * x)

    # QCQP rows lifted via X
    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(m, 0.5 * sum(Qmat[i,j]*X[i,j] for i=1:n, j=1:n) + qvec' * x + rterm <= 0)
    end
    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(m, 0.5 * sum(Pmat[i,j]*X[i,j] for i=1:n, j=1:n) + pvec' * x + sterm == 0)
    end

    # Linear rows
    if ℓ > 0; @constraint(m, A * x .<= b); end
    if η > 0; @constraint(m, H * x .== h); end

    # Bounds on x (as in your formulations)
    @constraint(m, 0 .<= x)
    @constraint(m, x .<= e)

    # Cardinality equality
    @constraint(m, sum(v) == n - ρ)

    # Complementarity relaxation in the lift
    @constraint(m, [i=1:n], W[i,i] == 0)

    # PSD lift
    add_PSD_comp!(m, x, v, X, V, W; n=n)

    return nothing
end

# Variant add-ons
add_binlift_v!(m, v, V, n::Int) = (@constraint(m, [i=1:n], V[i,i] == v[i]); nothing)
add_v_box!(m, v, e) = (@constraint(m, 0 .<= v); @constraint(m, v .<= e); nothing)

"""
    build_SDP_comp_model(data; variant="SDP_S2", build_only=false, optimizer=MosekTools.Optimizer, verbose=false)

Variants (reduced to 2, since SDP_S2U is redundant):
- "SDP_S2": diag(V)=v, no explicit v-box (PSD + diag(V)=v ⇒ 0<=v<=1 automatically)
- "SDP_S3": explicit 0<=v<=1, no diag(V)=v
"""
function build_SDP_comp_model(data::Dict{String,Any};
    variant::String="SDP_S2",
    build_only::Bool=false,
    optimizer = MosekTools.Optimizer,
    verbose::Bool=false
)
    params = prepare_instance(data)
    n = params.n
    e = params.e

    m = Model(optimizer)
    verbose ? nothing : set_silent(m)

    # Decision variables
    @variable(m, x[1:n])
    @variable(m, v[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, V[1:n,1:n], Symmetric)
    @variable(m, W[1:n,1:n])

    # Common block
    add_common_comp_SDP!(m, x, v, X, V, W, params)

    # Variant-specific
    if variant == "SDP_S2"
        add_binlift_v!(m, v, V, n)
        # no v-box needed (implied)
    elseif variant == "SDP_S3"
        add_v_box!(m, v, e)
        # no diag(V)=v
    else
        error("Unknown variant: $variant. Use SDP_S2 or SDP_S3.")
    end

    return m
end

# PrettyPrint wrapper
function solve_and_print(data::Dict{String,Any};
    variant::String="SDP_S2",
    optimizer = MosekTools.Optimizer,
    verbose::Bool=false,
    show_mats::Bool=true,
    pp_kwargs...
)
    if !isdefined(Main, :PrettyPrint)
        error("PrettyPrint is not loaded. Please include(\"instances/prettyprint.jl\") before calling SDPComplementarity.solve_and_print.")
    end

    m = build_SDP_comp_model(data; variant=variant, optimizer=optimizer, verbose=verbose)
    optimize!(m)

    Main.PrettyPrint.print_model_solution(
        m;
        variant=variant,
        relaxation=:SDP,
        show_mats=show_mats,
        pp_kwargs...
    )

    return m
end

export build_SDP_comp_model, solve_and_print

end # module SDPComplementarity

if isinteractive()
    # ---- PrettyPrint first ----
    include("instances/prettyprint.jl")
    using .PrettyPrint


    using MosekTools

    # ---- Instances from .jl ----
    include("instances/alper_stqp_instance.jl")
    using .AlperStqpInstances
    alp_inst = alper_stqp_rho3_instance()

    include("instances/diff_RLTEU_RLTIU_bigM_instance.jl")
    using .EUIUdiffinstance
    diff_inst = euiu_diff_instance()

    # =========================================================
    # Choose which instance to run (edit only this line)
    # =========================================================
    inst = alp_inst
    #inst = diff_inst

    variants = ("SDP_S2", "SDP_S3")

    println("\n==============================")
    println(" SDP Complementarity ")
    println("==============================")
    println("instance id = ", get(inst, "id", "unknown"), "\n")

    for var in variants
        println("\n------------------------------")
        println("variant = ", var)
        println("------------------------------")
        SDPComplementarity.solve_and_print(
            inst;
            variant=var,
            optimizer=MosekTools.Optimizer,
            show_mats=true
        )
    end
end