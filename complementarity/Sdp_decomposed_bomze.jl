module SDPDecomposedBomze

using LinearAlgebra
using JuMP
using MosekTools
using Parameters: @unpack

using ..RLTBigM: prepare_instance
using ..RLTComp: additional_params
using ..SDPComplementarity: add_binlift_v!

# ------------------------------------------------------------
# Local 3x3 blocks only — replaces the global PSD block:
#   [1   x_i  v_i]
#   [x_i Xii  0  ] >= 0   for all i
#   [v_i 0    v_i]
# ------------------------------------------------------------
function add_local_blocks_comp!(m::Model, x, v, X; n::Int)
    for i in 1:n
        @constraint(m,
            Symmetric([1.0  x[i]   v[i];
                       x[i] X[i,i] 0.0;
                       v[i] 0.0    v[i]]) in PSDCone()
        )
    end
end

function build_decomposed_bomze_model(data::Dict{String,Any};
    optimizer = MosekTools.Optimizer,
    verbose::Bool = false
)
    params = additional_params(prepare_instance(data))
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η, xL, xU = params

    m = Model(optimizer)
    if !verbose; set_silent(m); end

    @variable(m, x[1:n])
    @variable(m, v[1:n])
    @variable(m, X[1:n, 1:n], Symmetric)
    @variable(m, V[1:n, 1:n], Symmetric)

    # Objective
    @objective(m, Min,
        0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n) + q0' * x
    )

    # QCQP inequalities
    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(m,
            0.5 * sum(Qmat[i,j] * X[i,j] for i=1:n, j=1:n) + qvec' * x + rterm <= 0
        )
    end

    # QCQP equalities
    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(m,
            0.5 * sum(Pmat[i,j] * X[i,j] for i=1:n, j=1:n) + pvec' * x + sterm == 0
        )
    end

    # Linear constraints
    if ℓ > 0; @constraint(m, A * x .<= b); end
    if η > 0; @constraint(m, H * x .== h); end

    # Box constraints
    @constraint(m, xL .<= x)
    @constraint(m, x .<= xU)

    # Cardinality: eᵀv = n - ρ
    @constraint(m, sum(v) == n - ρ)

    # diag(V) = v
    add_binlift_v!(m, v, V, n)

    # Local 3x3 blocks ONLY — no global [1 x'; x X] block
    add_local_blocks_comp!(m, x, v, X; n=n)

    return m
end

function solve_and_print(data::Dict{String,Any};
    optimizer  = MosekTools.Optimizer,
    verbose::Bool  = false,
    show_mats::Bool = true
)
    if !isdefined(Main, :PrettyPrint)
        error("PrettyPrint is not loaded.")
    end

    m = build_decomposed_bomze_model(data; optimizer=optimizer, verbose=verbose)
    optimize!(m)

    Main.PrettyPrint.print_model_solution(
        m;
        variant    = "SDP_DecomposedBomze",
        relaxation = :SDP,
        show_mats  = show_mats
    )

    return m
end

export build_decomposed_bomze_model, solve_and_print

end # module SDPDecomposedBomze


if isinteractive()
    include(normpath(joinpath(@__DIR__, "..", "instances", "prettyprint.jl")))
    using .PrettyPrint

    using MosekTools

    include(normpath(joinpath(@__DIR__, "..", "instances", "bomze_counterexample.jl")))
    using .CounterexampleInstances
    inst = bomze_counterexample(; bigM_scale = 1.0)

    println("\n==============================")
    println(" Bomze-type Decomposed SDP")
    println("==============================")
    println("instance id = ", get(inst, "id", "unknown"))
    println()
    println("Expected results:")
    println("  z_exact          =  0.0   (cardinality forces x1*x2 = 0)")
    println("  z_full_comp_SDP  = -0.25  (from SDP_CompBin)")
    println("  z_decomposed     = -0.5   (weaker: no global [1 x'; x X])")
    println()

    SDPDecomposedBomze.solve_and_print(inst; show_mats=true)
end