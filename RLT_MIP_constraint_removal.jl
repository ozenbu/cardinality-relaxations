# ============================================================
# RLT_MIP_constraint_removal.jl
#
# - Keeps RLT_bigM_4variants.jl unchanged.
# - Builds the "photo model" MILP:
#     z_b = 1  -->  (new RLT block b is enforced)
#     minimize sum(z_b)
#     keep ALL original/base constraints always ON
#     force relaxation objective ≈ v* (full model value)
#
# NOTES:
# - Matrix ">= 0" is interpreted elementwise.
# - One binary controls a whole block via loops over entries.
# ============================================================

include("RLT_bigM_4variants.jl")

module RLTBlockMIP

using ..RLTBigM
using LinearAlgebra
using Parameters          # <-- FIX: provides @unpack
using JuMP
using Gurobi
using MathOptInterface
const MOI = MathOptInterface

# -----------------------
# Small helpers
# -----------------------

_new_bin(m::Model, name::Symbol) = @variable(m, base_name=String(name), binary=true)

function _add_vec_ge0_indicator!(m::Model, z, v)
    for i in eachindex(v)
        @constraint(m, z --> { v[i] >= 0 })
    end
    return nothing
end

function _add_mat_ge0_indicator!(m::Model, z, M)
    for i in axes(M, 1), j in axes(M, 2)
        @constraint(m, z --> { M[i, j] >= 0 })
    end
    return nothing
end

function _add_vec_eq_indicator!(m::Model, z, vL, vR)
    @assert length(vL) == length(vR)
    for i in eachindex(vL)
        @constraint(m, z --> { vL[i] == vR[i] })
    end
    return nothing
end

function _add_mat_eq_indicator!(m::Model, z, ML, MR)
    @assert size(ML) == size(MR)
    for i in axes(ML, 1), j in axes(ML, 2)
        @constraint(m, z --> { ML[i, j] == MR[i, j] })
    end
    return nothing
end

# -----------------------
# Base (always ON) constraints
# -----------------------
function add_base_constraints!(m, x, u, X, R, U, params; variant::String, u_nonneg::Bool=true)
    @unpack n, ρ, Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η, Mminus, Mplus, e = params

    # Quadratic constraints lifted to X
    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(m, 0.5 * sum(Qmat[i,j] * X[i,j] for i=1:n, j=1:n) + qvec' * x + rterm <= 0)
    end
    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(m, 0.5 * sum(Pmat[i,j] * X[i,j] for i=1:n, j=1:n) + pvec' * x + sterm == 0)
    end

    # Original linear constraints
    if ℓ > 0
        @constraint(m, A * x .<= b)
    end
    if η > 0
        @constraint(m, H * x .== h)
    end

    # Big-M links (original)
    @constraint(m, -Mminus * u .<= x)
    @constraint(m,  x .<= Mplus  * u)

    # diag(U) = u (original)
    @constraint(m, [i=1:n], U[i,i] == u[i])

    # Cardinality base
    if variant == "EU"
        @constraint(m, sum(u) == ρ)
        @constraint(m, u .<= 1)     # u <= e
    elseif variant == "IU"
        @constraint(m, sum(u) <= ρ)
        @constraint(m, u .<= 1)     # u <= e
    else
        error("variant must be \"EU\" or \"IU\" in this file.")
    end

    if u_nonneg
        @constraint(m, u .>= 0)
    end

    return nothing
end

# -----------------------
# New first-level RLT blocks (switchable)
# -----------------------
function add_new_blocks_with_binaries!(m, x, u, X, R, U, z::Dict{Symbol,JuMP.VariableRef}, params; variant::String)
    @unpack n, ρ, A, b, ℓ, H, h, η, Mminus, Mplus, e = params

    # -------- FC-derived new blocks --------
    if ℓ > 0
        FC_A_self  = A*X*A' .- A*(x*b') .- (b*x')*A' .+ (b*b')
        _add_mat_ge0_indicator!(m, z[:FC_A_self], FC_A_self)

        FC_A_Mplus = A*X .- (b*x') .- A*R*Mplus .+ (b*u')*Mplus
        _add_mat_ge0_indicator!(m, z[:FC_A_Mplus], FC_A_Mplus)

        FC_A_Mminus = -A*X .+ (b*x') .- A*R*Mminus .+ (b*u')*Mminus
        _add_mat_ge0_indicator!(m, z[:FC_A_Mminus], FC_A_Mminus)
    end

    if η > 0
        HX_L = H*X
        HX_R = h*x'
        _add_mat_eq_indicator!(m, z[:FC_HX], HX_L, HX_R)

        HR_L = H*R
        HR_R = h*u'
        _add_mat_eq_indicator!(m, z[:FC_HR], HR_L, HR_R)
    end

    FC_Mc_pp = Mplus*U*Mplus  .- Mplus*R' .- R*Mplus  .+ X
    _add_mat_ge0_indicator!(m, z[:FC_McCormick_pp], FC_Mc_pp)

    FC_Mc_mm = Mminus*U*Mminus .+ Mminus*R' .+ R*Mminus .+ X
    _add_mat_ge0_indicator!(m, z[:FC_McCormick_mm], FC_Mc_mm)

    FC_Mc_pm = Mplus*U*Mminus .+ Mplus*R' .- R*Mminus .- X
    _add_mat_ge0_indicator!(m, z[:FC_McCormick_pm], FC_Mc_pm)

    # -------- FE / FI / FU / FIU (variant-dependent) --------
    if variant == "EU"
        _add_vec_eq_indicator!(m, z[:FE_Re], R*e, ρ .* x)
        _add_vec_eq_indicator!(m, z[:FE_Ue], U*e, ρ .* u)

        eeT = ones(n, n)

        FU_Ubox = U .- (u*e') .- (e*u') .+ eeT
        _add_mat_ge0_indicator!(m, z[:FU_U_box], FU_Ubox)

        if ℓ > 0
            FU_Aeu = (b*e') .- (b*u') .- (A*x)*e' .+ A*R
            _add_mat_ge0_indicator!(m, z[:FU_A_eu], FU_Aeu)
        end

        FU_eu_Mplus  = Mplus*(u*e')  .- x*e' .- Mplus*U .+ R
        _add_mat_ge0_indicator!(m, z[:FU_eu_Mplus], FU_eu_Mplus)

        FU_eu_Mminus = Mminus*(u*e') .+ x*e' .- Mminus*U .- R
        _add_mat_ge0_indicator!(m, z[:FU_eu_Mminus], FU_eu_Mminus)

    elseif variant == "IU"
        if ℓ > 0
            FI_Apart = ρ .* b .- (b*u')*e .- ρ .* (A*x) .+ A*R*e
            _add_vec_ge0_indicator!(m, z[:FI_A_part], FI_Apart)
        end

        FI_card_Mplus  = ρ .* (Mplus*u)  .- ρ .* x .- Mplus*U*e  .+ R*e
        _add_vec_ge0_indicator!(m, z[:FI_card_Mplus], FI_card_Mplus)

        FI_card_Mminus = ρ .* (Mminus*u) .+ ρ .* x .- Mminus*U*e .- R*e
        _add_vec_ge0_indicator!(m, z[:FI_card_Mminus], FI_card_Mminus)

        Ue = U*e
        FI_sq = ρ^2 - 2ρ*sum(u) + sum(Ue[i] for i=1:n)
        @constraint(m, z[:FI_square_card] --> { FI_sq >= 0 })

        eeT = ones(n, n)

        FU_Ubox = U .- (u*e') .- (e*u') .+ eeT
        _add_mat_ge0_indicator!(m, z[:FU_U_box], FU_Ubox)

        if ℓ > 0
            FU_Aeu = (b*e') .- (b*u') .- (A*x)*e' .+ A*R
            _add_mat_ge0_indicator!(m, z[:FU_A_eu], FU_Aeu)
        end

        FU_eu_Mplus  = Mplus*(u*e')  .- x*e' .- Mplus*U .+ R
        _add_mat_ge0_indicator!(m, z[:FU_eu_Mplus], FU_eu_Mplus)

        FU_eu_Mminus = Mminus*(u*e') .+ x*e' .- Mminus*U .- R
        _add_mat_ge0_indicator!(m, z[:FU_eu_Mminus], FU_eu_Mminus)

        FIU_vec = ρ .* e .- ρ .* u .- sum(u) .* e .+ U*e
        _add_vec_ge0_indicator!(m, z[:FIU_coupling], FIU_vec)
    else
        error("variant must be \"EU\" or \"IU\".")
    end

    return nothing
end

# -----------------------
# Create z keys that actually exist for this instance
# -----------------------
function _make_z_dict!(m::Model, params; variant::String)
    @unpack ℓ, η = params
    z = Dict{Symbol,JuMP.VariableRef}()

    # FC
    if ℓ > 0
        z[:FC_A_self]   = _new_bin(m, :FC_A_self)
        z[:FC_A_Mplus]  = _new_bin(m, :FC_A_Mplus)
        z[:FC_A_Mminus] = _new_bin(m, :FC_A_Mminus)
    end
    if η > 0
        z[:FC_HX] = _new_bin(m, :FC_HX)
        z[:FC_HR] = _new_bin(m, :FC_HR)
    end
    z[:FC_McCormick_pp] = _new_bin(m, :FC_McCormick_pp)
    z[:FC_McCormick_mm] = _new_bin(m, :FC_McCormick_mm)
    z[:FC_McCormick_pm] = _new_bin(m, :FC_McCormick_pm)

    if variant == "EU"
        z[:FE_Re]        = _new_bin(m, :FE_Re)
        z[:FE_Ue]        = _new_bin(m, :FE_Ue)
        z[:FU_U_box]     = _new_bin(m, :FU_U_box)
        if ℓ > 0
            z[:FU_A_eu]  = _new_bin(m, :FU_A_eu)
        end
        z[:FU_eu_Mplus]  = _new_bin(m, :FU_eu_Mplus)
        z[:FU_eu_Mminus] = _new_bin(m, :FU_eu_Mminus)

    elseif variant == "IU"
        if ℓ > 0
            z[:FI_A_part] = _new_bin(m, :FI_A_part)
        end
        z[:FI_card_Mplus]   = _new_bin(m, :FI_card_Mplus)
        z[:FI_card_Mminus]  = _new_bin(m, :FI_card_Mminus)
        z[:FI_square_card]  = _new_bin(m, :FI_square_card)

        z[:FU_U_box]     = _new_bin(m, :FU_U_box)
        if ℓ > 0
            z[:FU_A_eu]  = _new_bin(m, :FU_A_eu)
        end
        z[:FU_eu_Mplus]  = _new_bin(m, :FU_eu_Mplus)
        z[:FU_eu_Mminus] = _new_bin(m, :FU_eu_Mminus)

        z[:FIU_coupling] = _new_bin(m, :FIU_coupling)
    else
        error("variant must be \"EU\" or \"IU\".")
    end

    return z
end

# -----------------------
# Full model (all new blocks ON) to compute v*
# -----------------------
function full_relaxation_value(data; variant::String="EU",
                               optimizer=Gurobi.Optimizer,
                               verbose::Bool=false,
                               u_nonneg::Bool=true)

    params = RLTBigM.prepare_instance(data)
    @unpack n, Q0, q0 = params

    m = Model(optimizer)
    verbose && set_optimizer_attribute(m, "OutputFlag", 1)
    !verbose && set_silent(m)

    @variable(m, x[1:n])
    @variable(m, u[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, R[1:n,1:n])
    @variable(m, U[1:n,1:n], Symmetric)

    @expression(m, obj_relax, 0.5 * sum(Q0[i,j]*X[i,j] for i=1:n, j=1:n) + q0' * x)
    @objective(m, Min, obj_relax)

    add_base_constraints!(m, x, u, X, R, U, params; variant=variant, u_nonneg=u_nonneg)

    z = _make_z_dict!(m, params; variant=variant)
    for (_, zv) in z
        fix(zv, 1; force=true)
    end

    add_new_blocks_with_binaries!(m, x, u, X, R, U, z, params; variant=variant)

    optimize!(m)
    term = termination_status(m)
    if term != MOI.OPTIMAL
        return (term, nothing)
    end
    return (term, objective_value(m))
end

# -----------------------
# Photo-model MILP:
#   minimize sum(z)
#   base constraints always ON
#   z -> new RLT blocks
#   enforce relaxation objective == v* (within tol)
# -----------------------
function select_min_blocks(data; variant::String="EU",
                           tol::Float64=1e-6,
                           optimizer=Gurobi.Optimizer,
                           verbose::Bool=false,
                           u_nonneg::Bool=true)

    term_full, vstar = full_relaxation_value(data; variant=variant,
                                             optimizer=optimizer,
                                             verbose=false,
                                             u_nonneg=u_nonneg)
    if term_full != MOI.OPTIMAL
        return (term_full, vstar, nothing)
    end

    params = RLTBigM.prepare_instance(data)
    @unpack n, Q0, q0 = params

    m = Model(optimizer)
    verbose && set_optimizer_attribute(m, "OutputFlag", 1)
    !verbose && set_silent(m)

    @variable(m, x[1:n])
    @variable(m, u[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, R[1:n,1:n])
    @variable(m, U[1:n,1:n], Symmetric)

    @expression(m, obj_relax, 0.5 * sum(Q0[i,j]*X[i,j] for i=1:n, j=1:n) + q0' * x)

    add_base_constraints!(m, x, u, X, R, U, params; variant=variant, u_nonneg=u_nonneg)

    z = _make_z_dict!(m, params; variant=variant)
    add_new_blocks_with_binaries!(m, x, u, X, R, U, z, params; variant=variant)
    @constraint(m, obj_relax >= vstar)
    @objective(m, Min, sum(zv for (_, zv) in z))

    optimize!(m)
    term = termination_status(m)
    if term != MOI.OPTIMAL
        return (term, vstar, nothing)
    end

    zvals = Dict(k => round(Int, value(z[k])) for k in keys(z))
    return (term, vstar, zvals)
end

end # module RLTBlockMIP


# ============================================================
# Example usage (your test instance)
# ============================================================
using .RLTBlockMIP
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

# Run:
term, vstar, zvals = RLTBlockMIP.select_min_blocks(data_test4; variant="EU", tol=1e-6, verbose=true)
println("term = ", term)
println("v*   = ", vstar)
println("z    = ", zvals)
