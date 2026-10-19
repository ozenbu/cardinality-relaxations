module SDPComplementarity

using LinearAlgebra
using JuMP
using MosekTools
using MathOptInterface
const MOI = MathOptInterface
using Parameters: @unpack

# Single source of truth for instance parsing.
# IMPORTANT: this SDP-only module does NOT construct or use
# any additional/redundant x-bounds.
using ..RLTBigM: prepare_instance

# ------------------------------------------------------------
# Full complementarity moment PSD lift:
#
#     [ 1   x'   v'
#       x   X    W
#       v   W'   V ] \succeq 0
#
# This is imposed directly as an affine PSD constraint.
# No auxiliary Y matrix or linking equalities are introduced.
# ------------------------------------------------------------
function add_PSD_comp!(m::Model, x, v, X, V, W)
    @constraint(
        m,
        Symmetric([
            1.0  x'   v';
            x    X    W;
            v    W'   V
        ]) in PSDCone()
    )
    return nothing
end

# ------------------------------------------------------------
# Common constraints (SDP-only; NO RLT/ST products)
# NO derived/redundant x-bounds.
# Cardinality E/I is added separately below.
# ------------------------------------------------------------
function add_common_comp_SDP!(m::Model, x, v, X, V, W, params)
    @unpack n, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η = params

    # Objective: 0.5<Q0,X> + q0'x
    @objective(m, Min,
        0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n) + q0' * x)

    # Original quadratic inequalities, lifted through X.
    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(m,
            0.5 * sum(Qmat[i,j] * X[i,j] for i=1:n, j=1:n)
            + qvec' * x + rterm <= 0)
    end

    # Original quadratic equalities, lifted through X.
    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(m,
            0.5 * sum(Pmat[i,j] * X[i,j] for i=1:n, j=1:n)
            + pvec' * x + sterm == 0)
    end

    # Original linear rows only.
    if ℓ > 0
        @constraint(m, A * x .<= b)
    end
    if η > 0
        @constraint(m, H * x .== h)
    end

    # Lifted complementarity x_i v_i = 0.
    @constraint(m, [i=1:n], W[i,i] == 0)

    # Full moment PSD constraint.
    add_PSD_comp!(m, x, v, X, V, W)

    return nothing
end

# ------------------------------------------------------------
# Cardinality blocks
# v_i = 1 means coordinate i is inactive; k = n-rho.
# E: e'v = k
# I: e'v >= k
# IMPORTANT: SDP-only means no RLT products such as We=kx or Ve=kv.
# ------------------------------------------------------------
function add_cardinality_E!(m::Model, v, params)
    @unpack n, ρ = params
    @constraint(m, sum(v) == n - ρ)
    return nothing
end

function add_cardinality_I!(m::Model, v, params)
    @unpack n, ρ = params
    @constraint(m, sum(v) >= n - ρ)
    return nothing
end

# ------------------------------------------------------------
# Variant blocks
# ------------------------------------------------------------
# Binary lift: diag(V)=v.
# Together with the full PSD lift this already implies 0<=v<=e.
function add_binlift_v!(m::Model, v, V, n::Int)
    @constraint(m, [i=1:n], V[i,i] == v[i])
    return nothing
end

# Explicit v-box used by CompBou.
function add_v_box!(m::Model, v, e)
    @constraint(m, 0 .<= v)
    @constraint(m, v .<= e)
    return nothing
end

# ------------------------------------------------------------
# Variant canonicalization
#
# At SDP-only level:
#   CompBinBou == CompBin
# because Y \succeq 0 and diag(V)=v imply 0<=v<=e.
# Hence CompBinBou names are aliases of CompBin; we do not add
# numerically redundant v-box constraints.
#
# CompBou is weaker: it has 0<=v<=e but no diag(V)=v.
# ------------------------------------------------------------
function canonical_variant(variant::String)
    aliases = Dict(
        # Backward-compatible old E-only names
        "SDP_CompBin"       => "SDP_CompBin_E",
        "SDP_CompBinBou"    => "SDP_CompBin_E",
        "SDP_CompBou"       => "SDP_CompBou_E",
        "SDP_S2"            => "SDP_CompBin_E",
        "SDP_S2U"           => "SDP_CompBin_E",
        "SDP_S3"            => "SDP_CompBou_E",

        # Explicit BinBou aliases
        "SDP_CompBinBou_E"  => "SDP_CompBin_E",
        "SDP_CompBinBou_I"  => "SDP_CompBin_I",
    )
    return get(aliases, variant, variant)
end

# ------------------------------------------------------------
# Build SDP model
# Canonical variants:
#   SDP_CompBin_E
#   SDP_CompBin_I
#   SDP_CompBou_E
#   SDP_CompBou_I
#
# Accepted aliases:
#   SDP_CompBinBou_E -> SDP_CompBin_E
#   SDP_CompBinBou_I -> SDP_CompBin_I
# ------------------------------------------------------------
function build_SDP_comp_model(data::Dict{String,Any};
    variant::String="SDP_CompBin_E",
    optimizer=MosekTools.Optimizer,
    verbose::Bool=false
)
    variant = canonical_variant(variant)

    allowed = (
        "SDP_CompBin_E",
        "SDP_CompBin_I",
        "SDP_CompBou_E",
        "SDP_CompBou_I",
    )

    if !(variant in allowed)
        error(
            "Unknown SDP complementarity variant: $variant. " *
            "Use SDP_CompBin_E, SDP_CompBin_I, SDP_CompBou_E, or SDP_CompBou_I. " *
            "SDP_CompBinBou_E/I are accepted aliases of SDP_CompBin_E/I."
        )
    end

    # No complementarity-side enrichment: in particular, no xL/xU.
    params = prepare_instance(data)
    n = params.n
    e = params.e

    m = Model(optimizer)
    if !verbose
        set_silent(m)
    end

    set_name(m, "SQCQP_SDP_Complementarity_NoBounds_$(variant)")

    @variable(m, x[1:n])
    @variable(m, v[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, V[1:n,1:n], Symmetric)
    @variable(m, W[1:n,1:n])

    # Common SDP-only block.
    add_common_comp_SDP!(m, x, v, X, V, W, params)

    # Cardinality: E or I.
    if endswith(variant, "_E")
        add_cardinality_E!(m, v, params)
    elseif endswith(variant, "_I")
        add_cardinality_I!(m, v, params)
    end

    # Complementarity variant.
    if startswith(variant, "SDP_CompBin")
        # diag(V)=v; explicit v-box is redundant under full PSD.
        add_binlift_v!(m, v, V, n)
    elseif startswith(variant, "SDP_CompBou")
        # Weaker SDP relaxation: explicit v-box only.
        add_v_box!(m, v, e)
    end

    return m
end

# ------------------------------------------------------------
# PrettyPrint wrapper
# ------------------------------------------------------------
function solve_and_print(data::Dict{String,Any};
    variant::String="SDP_CompBin_E",
    optimizer=MosekTools.Optimizer,
    verbose::Bool=false,
    show_mats::Bool=true,
    pp_kwargs...
)
    if !isdefined(Main, :PrettyPrint)
        error("PrettyPrint is not loaded. Please include(\"instances/prettyprint.jl\") before calling SDPComplementarity.solve_and_print.")
    end

    m = build_SDP_comp_model(
        data;
        variant=variant,
        optimizer=optimizer,
        verbose=verbose,
    )
    optimize!(m)

    Main.PrettyPrint.print_model_solution(
        m;
        variant=variant,
        relaxation=:SDP,
        show_mats=show_mats,
        pp_kwargs...,
    )

    return m
end

export build_SDP_comp_model,
       solve_and_print,
       add_PSD_comp!,
       add_common_comp_SDP!,
       add_cardinality_E!,
       add_cardinality_I!,
       add_binlift_v!,
       add_v_box!,
       canonical_variant

end # module SDPComplementarity

# ------------------------------------------------------------
# Optional interactive demo
# ------------------------------------------------------------
if isinteractive()
    include(normpath(joinpath(@__DIR__, "..", "instances", "prettyprint.jl")))
    using .PrettyPrint
    using MosekTools

    include(normpath(joinpath(@__DIR__, "..", "instances", "alper_stqp_instance.jl")))
    using .AlperStqpInstances
    inst = AlperStqpInstances.alper_stqp_rho3_instance()

    variants = (
        "SDP_CompBin_E",
        "SDP_CompBin_I",
        "SDP_CompBou_E",
        "SDP_CompBou_I",
    )

    println("\n==============================")
    println(" SDP Complementarity -- NoBounds E/I ")
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
            show_mats=true,
        )
    end
end
