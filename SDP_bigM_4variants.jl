module SDPBigM

using JuMP, LinearAlgebra
using MosekTools
using Parameters
using MathOptInterface
const MOI = MathOptInterface

# Re-use instance parsing from RLTBigM
using ..RLTBigM: prepare_instance

"Add the lifted PSD block"
function add_SDP_lift!(model::Model, x, u, X, R, U; n::Int)
    # Z = [ 1   x'      u'
    #       x   X       R
    #       u   R'      U ]
    @variable(model, Z[1:(2n+1), 1:(2n+1)], Symmetric)
    @constraint(model, Z .== [1.0 x' u'; x X R; u R' U])
    @constraint(model, Z in PSDCone())
    return Z
end

function build_SDP_model(data::Dict{String, Any};
                         variant::String="E",
                         build_only::Bool=false,
                         optimizer=MosekTools.Optimizer)

    vars = prepare_instance(data)
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η, Mminus, Mplus, e = vars

    model = Model(optimizer)

    # Decision variables
    @variable(model, x[1:n])
    @variable(model, u[1:n])
    @variable(model, X[1:n, 1:n], Symmetric)
    @variable(model, U[1:n, 1:n], Symmetric)
    @variable(model, R[1:n, 1:n])

    # Objective: 0.5⟨Q0, X⟩ + q0ᵀx
    @objective(model, Min, 0.5 * dot(Q0, X) + dot(q0, x))

    # Quadratic inequalities
    for (Q, q, r) in zip(Qi, qi, ri)
        @constraint(model, 0.5 * dot(Q, X) + dot(q, x) + r <= 0)
    end

    # Quadratic equalities
    for (P, p, s) in zip(Pi, pi, si)
        @constraint(model, 0.5 * dot(P, X) + dot(p, x) + s == 0)
    end

    # Linear constraints
    if ℓ > 0
        @constraint(model, A * x .<= b)
    end
    if η > 0
        @constraint(model, H * x .== h)
    end

    # Big-M links: -M⁻ u ≤ x ≤ M⁺ u
    @constraint(model, [i=1:n],  x[i] <=  Mplus[i,i] * u[i])
    @constraint(model, [i=1:n], -x[i] <=  Mminus[i,i] * u[i])

    # Cardinality (E or I)
    if variant == "E"
        @constraint(model, sum(u) == ρ)
    elseif variant == "I"
        @constraint(model, sum(u) <= ρ)
    else
        error("Unknown variant: $variant (must be \"E\" or \"I\")")
    end

    # diag(U) = u
    @constraint(model, [i=1:n], U[i, i] == u[i])

    # SDP lifting block (modularized)
    add_SDP_lift!(model, x, u, X, R, U; n=n)

    if build_only
        return model
    end

    return model
end

"Build -> optimize -> PrettyPrint from the JuMP model (no manual unpack)."
function solve_and_print(data::Dict{String,Any};
                         variant::String="E",
                         relaxation = :SDP,
                         optimizer=MosekTools.Optimizer,
                         pp_kwargs...)

    if !isdefined(Main, :PrettyPrint)
        error("PrettyPrint is not loaded. Please include(\"instances/prettyprint.jl\") before calling SDPBigM.solve_and_print.")
    end

    model = build_SDP_model(data; variant=variant, build_only=true, optimizer=optimizer)
    optimize!(model)

    Main.PrettyPrint.print_model_solution(model;
        variant=variant,
        relaxation=relaxation,
        pp_kwargs...
    )

    return model
end

"Run both SDP variants (E and I) on a given instance Dict."
function demo_SDP(instance_data::Dict{String,Any};
                  optimizer=MosekTools.Optimizer,
                  show_mats::Bool=true,
                  show_Z::Bool=false)

    for variant in ("E", "I")
        solve_and_print(instance_data;
            variant=variant,
            relaxation=:SDP,
            optimizer=optimizer,
            show_mats=show_mats,
            show_Z=show_Z
        )
    end
    return nothing
end

end  # module SDPBigM


if isinteractive()
    include("instances/prettyprint.jl")
    include("instances/alper_stqp_instance.jl")     # module AlperStqpInstances
    include("instances/diff_RLTEU_RLTIU_bigM_instance.jl")  # module EUIUdiffinstance

    using .PrettyPrint
    using .RLTBigM
    using .SDPBigM
    using .AlperStqpInstances
    using .EUIUdiffinstance

    alp_inst  = alper_stqp_rho3_instance()
    diff_inst = euiu_diff_instance()

    # Run SDP BigM on both stored instances
    SDPBigM.demo_SDP(alp_inst;  show_mats=true, show_Z=false)
    #SDPBigM.demo_SDP(diff_inst; show_mats=true, show_Z=false)
end