module SDPComplementarity

using LinearAlgebra
using JuMP
using MosekTools
using MathOptInterface
const MOI = MathOptInterface
using Parameters: @unpack

# Single source of truth for parsing + enrichment
using ..RLTBigM: prepare_instance
using ..RLTComp: additional_params

# ------------------------------------------------------------
# PSD lift: Y = [1 x' v'; x X W; v W' V] ⪰ 0
# ------------------------------------------------------------
function add_PSD_comp!(m::Model, x, v, X, V, W; n::Int)
    @variable(m, Y[1:(2n+1), 1:(2n+1)], Symmetric)
    @constraint(m, Y .== [1.0 x' v'; x X W; v W' V])
    @constraint(m, Y in PSDCone())
    return Y
end

# ------------------------------------------------------------
# Common constraints (SDP-only; NO ST/RLT equalities)
# params is assumed to be already enriched by `additional_params`
# ------------------------------------------------------------
function add_common_comp_SDP!(m::Model, x, v, X, V, W, params)
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, e, xL, xU = params

    # Objective: 0.5⟨Q0, X⟩ + q0ᵀx
    @objective(m, Min, 0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n) + q0' * x)

    # QCQP rows lifted via X (Qi/Pi are tuples; loops are safe even if empty)
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

    # Linear rows
    if ℓ > 0
        @constraint(m, A * x .<= b)
    end
    if η > 0
        @constraint(m, H * x .== h)
    end

    # General x-bounds (UPDATED)
    @constraint(m, xL .<= x)
    @constraint(m, x .<= xU)

    # Cardinality equality: eᵀv = n-ρ
    @constraint(m, sum(v) == n - ρ)

    # Relaxed complementarity in the lift: diag(W)=0
    @constraint(m, [i=1:n], W[i,i] == 0)

    # PSD lift
    add_PSD_comp!(m, x, v, X, V, W; n=n)

    return nothing
end

# Variant add-ons
add_binlift_v!(m, v, V, n::Int) = (@constraint(m, [i=1:n], V[i,i] == v[i]); nothing)
add_v_box!(m, v, e) = (@constraint(m, 0 .<= v); @constraint(m, v .<= e); nothing)

# ------------------------------------------------------------
# Build SDP model
# Variants:
# - SDP_CompBin: diag(V)=v (binary-lift surrogate), no explicit v-box
# - SDP_CompBou: explicit 0<=v<=e, no diag(V)=v
#
# Aliases:
# - SDP_S2 -> SDP_CompBin
# - SDP_S3 -> SDP_CompBou
# ------------------------------------------------------------
function build_SDP_comp_model(data::Dict{String,Any};
    variant::String="SDP_CompBin",
    optimizer = MosekTools.Optimizer,
    verbose::Bool=false
)
    # Aliases
    if variant == "SDP_S2"; variant = "SDP_CompBin"; end
    if variant == "SDP_S3"; variant = "SDP_CompBou"; end

    # Enrich once (adds xL, xU, eeT)
    params = additional_params(prepare_instance(data))
    n = params.n
    e = params.e

    m = Model(optimizer)
    if !verbose
        set_silent(m)
    end

    # Decision variables
    @variable(m, x[1:n])
    @variable(m, v[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, V[1:n,1:n], Symmetric)
    @variable(m, W[1:n,1:n])

    # Common block
    add_common_comp_SDP!(m, x, v, X, V, W, params)

    # Variant-specific
    if variant == "SDP_CompBin"
        add_binlift_v!(m, v, V, n)   # diag(V)=v
        # no explicit v-box needed
    elseif variant == "SDP_CompBou"
        add_v_box!(m, v, e)          # 0<=v<=e
        # no diag(V)=v
    else
        error("Unknown variant: $variant. Use SDP_CompBin or SDP_CompBou (or SDP_S2/SDP_S3 aliases).")
    end

    return m
end

# PrettyPrint wrapper
function solve_and_print(data::Dict{String,Any};
    variant::String="SDP_CompBin",
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
    include(normpath(joinpath(@__DIR__, "..", "instances", "prettyprint.jl")))
    using .PrettyPrint

    using MosekTools

    include(normpath(joinpath(@__DIR__, "..", "instances", "alper_stqp_instance.jl")))
    using .AlperStqpInstances
    alp_inst = alper_stqp_rho3_instance()

    include(normpath(joinpath(@__DIR__, "..", "instances", "diff_RLTEU_RLTIU_bigM_instance.jl")))
    using .EUIUdiffinstance
    diff_inst = euiu_diff_instance()

    inst = alp_inst
    # inst = diff_inst

    variants = ("SDP_CompBin", "SDP_CompBou")

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


if isinteractive()
    # ---- PrettyPrint first ----
    include(normpath(joinpath(@__DIR__, "..", "instances", "prettyprint.jl")))
    using .PrettyPrint

    using MosekTools

    # ---- Instances from .jl ----
    include(normpath(joinpath(@__DIR__, "..", "instances", "alper_stqp_instance.jl")))
    using .AlperStqpInstances
    alp_inst = alper_stqp_rho3_instance()

    include(normpath(joinpath(@__DIR__, "..", "instances", "diff_RLTEU_RLTIU_bigM_instance.jl")))
    using .EUIUdiffinstance
    diff_inst = euiu_diff_instance()

    include(normpath(joinpath(@__DIR__, "..", "instances", "SDP_separation_instance.jl")))
    using .SDPCompBinSeparationInstance
    sdp_sep_ins = sdp_compbin_vs_bigm_instance()
    # =========================================================
    # Choose which instance to run (edit only this line)
    # =========================================================
    inst = alp_inst
    inst = sdp_sep_ins
    #inst = diff_inst

    variants = ("SDP_CompBin", "SDP_CompBou")   

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