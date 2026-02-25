# This file keeps RLTBigM.jl unchanged and adds a masked builder.

include("RLT_bigM_4variants.jl")

module RLTBigMConsRemoval

using ..RLTBigM
using LinearAlgebra
using Parameters
using JuMP
using Gurobi
using MathOptInterface
using Dates, Printf

const MOI = MathOptInterface

# ----------------------------
# Mask utilities
# ----------------------------

const DEFAULT_MASK = (
    FC = (
        quad_ineq = true,
        quad_eq   = true,
        Ax_le_b   = true,
        Hx_eq_h   = true,
        bigM_link = true,
        diagU_eq_u = true,
        H_lift = true,
        McCormick_pp = true,
        McCormick_mm = true,
        McCormick_pm = true,
        RLT_A_self = true,
        RLT_A_Mplus = true,
        RLT_A_Mminus = true,
    ),
    FE = (
        sumu_eq_rho = true,
        Re_eq_rho_x = true,
        Ue_eq_rho_u = true,
    ),
    FI = (
        sumu_le_rho = true,
        square_card = true,
        A_part = true,
        card_Mplus = true,
        card_Mminus = true,
    ),
    FU = (
        u_le_1 = true,
        U_box = true,
        A_eu = true,
        eu_Mplus = true,
        eu_Mminus = true,
    ),
    FIU = (
        coupling = true,
    ),
)

function deepmerge(a::NamedTuple, b::NamedTuple)
    pairs = Pair{Symbol,Any}[]
    for k in keys(a)
        if haskey(b, k)
            av = getfield(a, k)
            bv = getfield(b, k)
            if (av isa NamedTuple) && (bv isa NamedTuple)
                push!(pairs, k => deepmerge(av, bv))
            else
                push!(pairs, k => bv)
            end
        else
            push!(pairs, k => getfield(a, k))
        end
    end
    return (; pairs...)
end


normalize_mask(mask) = mask === nothing ? DEFAULT_MASK : deepmerge(DEFAULT_MASK, mask)

# ----------------------------
# Masked constraint blocks
# ----------------------------

function add_FC_masked!(m, x, u, X, R, U, params; mask=nothing)
    mask = normalize_mask(mask).FC
    @unpack n, Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η, Mminus, Mplus, e = params

    if mask.quad_ineq
        for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
            @constraint(m,
                0.5 * sum(Qmat[i,j] * X[i,j] for i=1:n, j=1:n) + qvec' * x + rterm <= 0)
        end
    end

    if mask.quad_eq
        for (Pmat, pvec, sterm) in zip(Pi, pi, si)
            @constraint(m,
                0.5 * sum(Pmat[i,j] * X[i,j] for i=1:n, j=1:n) + pvec' * x + sterm == 0)
        end
    end

    if mask.Ax_le_b && ℓ > 0
        @constraint(m, A * x .<= b)
    end
    if mask.Hx_eq_h && η > 0
        @constraint(m, H * x .== h)
    end

    if mask.bigM_link
        @constraint(m, -Mminus * u .<= x)
        @constraint(m,  x .<= Mplus * u)
    end

    if mask.diagU_eq_u
        @constraint(m, [i=1:n], U[i,i] == u[i])
    end

    if mask.H_lift && η > 0
        @constraint(m, H * X .== RLTBigM.outer_xy(h, x))
        @constraint(m, H * R .== RLTBigM.outer_xy(h, u))
    end

    if mask.McCormick_pp
        @constraint(m, Mplus*U*Mplus  .- Mplus*R' .- R*Mplus  .+ X .>= 0)
    end
    if mask.McCormick_mm
        @constraint(m, Mminus*U*Mminus .+ Mminus*R' .+ R*Mminus .+ X .>= 0)
    end
    if mask.McCormick_pm
        @constraint(m, Mplus*U*Mminus .+ Mplus*R' .- R*Mminus .- X .>= 0)
    end

    if ℓ > 0
        if mask.RLT_A_self
            @constraint(m, A*X*A' .- A*(x*b') .- (b*x')*A' .+ (b*b') .>= 0)
        end
        if mask.RLT_A_Mplus
            @constraint(m, A*X .- (b*x') .- A*R*Mplus .+ (b*u')*Mplus .>= 0)
        end
        if mask.RLT_A_Mminus
            @constraint(m, -A*X .+ (b*x') .- A*R*Mminus .+ (b*u')*Mminus .>= 0)
        end
    end

    return nothing
end

function add_FE_masked!(m, x, u, X, R, U, params; mask=nothing)
    mask = normalize_mask(mask).FE
    @unpack ρ, e = params

    if mask.sumu_eq_rho
        @constraint(m, sum(u) == ρ)
    end
    if mask.Re_eq_rho_x
        @constraint(m, R * e .== ρ .* x)
    end
    if mask.Ue_eq_rho_u
        @constraint(m, U * e .== ρ .* u)
    end
    return nothing
end

function add_FI_masked!(m, x, u, X, R, U, params; mask=nothing)
    mask = normalize_mask(mask).FI
    @unpack ρ, A, b, ℓ, Mminus, Mplus, e = params

    if mask.sumu_le_rho
        @constraint(m, sum(u) <= ρ)
    end
    if mask.square_card
        @constraint(m, ρ^2 - 2ρ*sum(u) + dot(e, U*e) >= 0)
    end

    if mask.A_part && ℓ > 0
        @constraint(m, ρ .* b .- (b*u')*e .- ρ .* (A*x) .+ A*R*e .>= 0)
    end

    if mask.card_Mplus
        @constraint(m, ρ .* (Mplus*u) .- ρ .* x .- Mplus*U*e .+ R*e .>= 0)
    end
    if mask.card_Mminus
        @constraint(m, ρ .* (Mminus*u) .+ ρ .* x .- Mminus*U*e .- R*e .>= 0)
    end
    return nothing
end

function add_FU_masked!(m, x, u, X, R, U, params; mask=nothing)
    mask = normalize_mask(mask).FU
    @unpack A, b, ℓ, Mminus, Mplus, e = params
    n = length(u)

    if mask.u_le_1
        @constraint(m, u .<= 1)
    end

    if mask.U_box
        eeT = ones(n, n)
        @constraint(m, U .- (u*e') .- (e*u') .+ eeT .>= 0)
    end

    if mask.A_eu && ℓ > 0
        @constraint(m, (b*e') .- (b*u') .- (A*x)*e' .+ A*R .>= 0)
    end

    if mask.eu_Mplus
        @constraint(m, Mplus*(u*e') .- x*e' .- Mplus*U .+ R .>= 0)
    end
    if mask.eu_Mminus
        @constraint(m, Mminus*(u*e') .+ x*e' .- Mminus*U .- R .>= 0)
    end

    return nothing
end

function add_FIU_masked!(m, u, U, params; mask=nothing)
    mask = normalize_mask(mask).FIU
    @unpack ρ, e = params

    if mask.coupling
        @constraint(m, ρ .* e .- ρ .* u .- sum(u) .* e .+ U*e .>= 0)
    end
    return nothing
end

# ----------------------------
# Build & solve (masked)
# ----------------------------

function build_and_solve_masked(data;
        variant::String="EU",
        mask=nothing,
        build_only::Bool=false,
        optimizer=Gurobi.Optimizer,
        verbose::Bool=false)

    # Reuse the same instance unpacker
    params = RLTBigM.prepare_instance(data)
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η, Mminus, Mplus, e = params

    # Keep EXACT path identical by delegating to the original module
    if variant == "EXACT"
        return RLTBigM.build_and_solve(data; variant="EXACT",
                                       build_only=build_only,
                                       optimizer=optimizer,
                                       verbose=verbose)
    end

    m = Model(optimizer)
    set_silent(m)
    verbose && set_optimizer_attribute(m, "OutputFlag", 1)

    set_name(m, "SQCQP_RLT_masked")

    @variable(m, x[1:n])
    @variable(m, u[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, R[1:n,1:n])
    @variable(m, U[1:n,1:n], Symmetric)

    @objective(m, Min, 0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n) + q0' * x)

    add_FC_masked!(m, x, u, X, R, U, params; mask=mask)

    if startswith(variant, "E")
        add_FE_masked!(m, x, u, X, R, U, params; mask=mask)
    else
        add_FI_masked!(m, x, u, X, R, U, params; mask=mask)
    end

    if endswith(variant, "U")
        add_FU_masked!(m, x, u, X, R, U, params; mask=mask)
        if startswith(variant, "I")
            add_FIU_masked!(m, u, U, params; mask=mask)
        end
    end

    if build_only
        return m
    end

    optimize!(m)
    term = termination_status(m)

    if term == MOI.OPTIMAL
        return (:OPTIMAL, objective_value(m),
                value.(x), value.(u), value.(X), value.(R), value.(U))
    elseif term == MOI.DUAL_INFEASIBLE
        return (:UNBOUNDED, nothing, nothing, nothing, nothing, nothing, nothing)
    elseif term == MOI.INFEASIBLE_OR_UNBOUNDED
        return (:INF_OR_UNBD, nothing, nothing, nothing, nothing, nothing, nothing)
    elseif term == MOI.INFEASIBLE
        return (:INFEASIBLE, nothing, nothing, nothing, nothing, nothing, nothing)
    else
        return (:OTHER, term, nothing, nothing, nothing, nothing, nothing)
    end
end

# ----------------------------
# Convenience: common FU ablations
# ----------------------------

function fu_only_u_le_1_mask()
    return (FU=(u_le_1=true, U_box=false, A_eu=false, eu_Mplus=false, eu_Mminus=false),)
end

function fu_drop_everything_but_U_box_mask()
    return (FU=(u_le_1=true, U_box=true, A_eu=false, eu_Mplus=false, eu_Mminus=false),)
end

end # module RLTBigMConsRemoval

using .RLTBigMConsRemoval
using LinearAlgebra

# small rational example
Q0 = [
    3//10000    127//1250   79//2500    867//10000;
    127//1250   1//500      1001//10000 1059//10000;
    79//2500    1001//10000 -1//2000    -703//10000;
    867//10000  1059//10000 -703//10000 -1063//10000
]
Q0d = Q0*20000

q0 = [-1973//10000, -2535//10000, -1967//10000, -973//10000]
q0d = q0*20000

A2 = [ 1.0  0.0  0.0  0.0
        0.0  1.0  0.0  0.0
        0.0  0.0  1.0  0.0
        0.0  0.0  0.0  1.0
        -1.0  0.0  0.0  0.0
        0.0 -1.0  0.0  0.0
        0.0  0.0 -1.0  0.0
        0.0  0.0  0.0 -1.0 ]

b2 = [1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0]

data_test4 = Dict(
    "n"  => 4,
    "rho"=> 3.0,
    "Q0" => Q0d,
    "q0" => q0d,
    "Qi" => nothing, "qi" => nothing, "ri" => nothing,
    "Pi" => nothing, "pi" => nothing, "si" => nothing,
    "A"  => A2, "b"  => b2,
    "H"  => nothing, "h"  => nothing,
    "M"  => I(4)
)

# Full EU
res_full = RLTBigMConsRemoval.build_and_solve_masked(data_test4; variant="EU")

# EU but in FU keep only u <= 1
mask = RLTBigMConsRemoval.fu_only_u_le_1_mask()
res_abl = RLTBigMConsRemoval.build_and_solve_masked(data_test4; variant="EU", mask=mask)
 