module SDPComplementarity

using LinearAlgebra
using JuMP
using MosekTools
using MathOptInterface
const MOI = MathOptInterface

# ----------------------------
# PSD lift: Y = [1 x' v'; x X W; v W' V] ⪰ 0
# ----------------------------
function add_PSD_comp!(m::Model, x, v, X, V, W; n::Int)
    @variable(m, Y[1:(2n+1), 1:(2n+1)], Symmetric)
    @constraint(m, Y .== [1.0 x' v'; x X W; v W' V])
    @constraint(m, Y in PSDCone())
    return Y
end

# ----------------------------
# Prepare params (self-contained; supports your instance Dict)
# ----------------------------
function prepare_instance_comp_sdp(data::Dict{String,Any})
    n   = Int(data["n"])
    ρ   = Int(data["rho"])
    Q0  = Matrix{Float64}(data["Q0"])
    q0  = Vector{Float64}(data["q0"])

    Qi  = get(data, "Qi", nothing)
    qi  = get(data, "qi", nothing)
    ri  = get(data, "ri", nothing)

    Pi  = get(data, "Pi", nothing)
    pi  = get(data, "pi", nothing)
    si  = get(data, "si", nothing)

    A   = get(data, "A", nothing)
    b   = get(data, "b", nothing)
    H   = get(data, "H", nothing)
    h   = get(data, "h", nothing)

    mminus = Vector{Float64}(data["Mminus"])
    mplus  = Vector{Float64}(data["Mplus"])
    bigM_scale = Float64(get(data, "bigM_scale", 1.0))

    xL = -bigM_scale .* mminus
    xU =  bigM_scale .* mplus

    e  = ones(Float64, n)

    hasA = (A isa AbstractMatrix) && (b isa AbstractVector) && (size(A,1) > 0)
    hasH = (H isa AbstractMatrix) && (h isa AbstractVector) && (size(H,1) > 0)

    ℓ = hasA ? 1 : 0
    η = hasH ? 1 : 0

    return (
        n=n, ρ=ρ, Q0=Q0, q0=q0,
        Qi=Qi, qi=qi, ri=ri,
        Pi=Pi, pi=pi, si=si,
        A=A, b=b, ℓ=ℓ,
        H=H, h=h, η=η,
        xL=xL, xU=xU,
        e=e
    )
end

# ----------------------------
# Common constraints (SDP-only; NO ST/RLT equalities)
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
    xL = params.xL
    xU = params.xU

    # Objective: 0.5⟨Q0, X⟩ + q0ᵀx
    @objective(m, Min, 0.5 * sum(Q0[i,j]*X[i,j] for i=1:n, j=1:n) + q0' * x)

    # QCQP rows lifted via X (skip if nothing)
    if Qi !== nothing
        for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
            @constraint(m, 0.5 * sum(Qmat[i,j]*X[i,j] for i=1:n, j=1:n) + qvec' * x + rterm <= 0)
        end
    end
    if Pi !== nothing
        for (Pmat, pvec, sterm) in zip(Pi, pi, si)
            @constraint(m, 0.5 * sum(Pmat[i,j]*X[i,j] for i=1:n, j=1:n) + pvec' * x + sterm == 0)
        end
    end

    # Linear rows
    if ℓ > 0; @constraint(m, A * x .<= b); end
    if η > 0; @constraint(m, H * x .== h); end

    # General x-bounds: xL <= x <= xU   (UPDATED)
    @constraint(m, xL .<= x)
    @constraint(m, x .<= xU)

    # Cardinality equality: e'v = n-ρ
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

"""
    build_SDP_comp_model(data; variant="SDP_CompBin", optimizer=MosekTools.Optimizer, verbose=false)

Variants:
- "SDP_CompBin": diag(V)=v (binary-lift surrogate), no explicit v-box (implied by PSD + diag(V)=v)
- "SDP_CompBou": explicit 0<=v<=e, no diag(V)=v

Backward-compatible aliases:
- "SDP_S2" -> "SDP_CompBin"
- "SDP_S3" -> "SDP_CompBou"
"""
function build_SDP_comp_model(data::Dict{String,Any};
    variant::String="SDP_CompBin",
    optimizer = MosekTools.Optimizer,
    verbose::Bool=false
)
    # aliases
    if variant == "SDP_S2"; variant = "SDP_CompBin"; end
    if variant == "SDP_S3"; variant = "SDP_CompBou"; end

    params = prepare_instance_comp_sdp(data)
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
    if variant == "SDP_CompBin"
        add_binlift_v!(m, v, V, n)   # diag(V)=v
        # no v-box needed (implied)
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

    # =========================================================
    # Choose which instance to run (edit only this line)
    # =========================================================
    inst = alp_inst
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