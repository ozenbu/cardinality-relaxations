using JSON
using JuMP, MosekTools
using LinearAlgebra
using Main.RLTBigM
using Main.RLT_SDP_Batch: normalize_instance!

const MOI = JuMP.MOI

# rounded matrix printing
function print_matrix_rounded(name::AbstractString, M; digits::Int = 4)
    println("\n", name, " (rounded to ", digits, " digits) =")
    Mround = round.(M; digits = digits)
    show(stdout, "text/plain", Mround)
    println()
end

# ---------------------------------------------------------
# add the correct RLT blocks depending on variant
#
# EU: FC, FE, FU
# IU: FC, FI, FU, FIU
# ---------------------------------------------------------
function add_relaxation_blocks!(
    m::Model,
    relaxation_variant::String,
    x,
    u,
    X,
    R,
    U,
    params,
)
    if relaxation_variant == "EU"
        RLTBigM.add_FC!(m, x, u, X, R, U, params)
        RLTBigM.add_FE!(m, x, u, X, R, U, params)
        RLTBigM.add_FU!(m, x, u, X, R, U, params)
    elseif relaxation_variant == "IU"
        RLTBigM.add_FC!(m, x, u, X, R, U, params)
        RLTBigM.add_FI!(m, x, u, X, R, U, params)
        RLTBigM.add_FU!(m, x, u, X, R, U, params)
        RLTBigM.add_FIU!(m, x, u, X, R, U, params)
    else
        error("Unknown relaxation_variant = $relaxation_variant. Use \"EU\" or \"IU\".")
    end
end

# ---------------------------------------------------------
# U Block-SDP "closest point" model
#
# Variables: x, u, X, R, U, W, δ
#
# Constraints:
#   - Full RLT blocks (EU or IU) in (x,u,X,R,U).
#   - Block SDP:
#         W = [ 1    u'  ;
#               u    U   ]  and  W ⪰ 0.
#   - Distance to base solution:
#         |x - x*|_∞ ≤ δ,
#         |u - u*|_∞ ≤ δ,
#         |X - X*|_∞ ≤ δ.
#
# Objective: min δ.
#
# If infeasible, there is no point satisfying RLT + U block SDP.
# If feasible, δ* measures how far we must move from (x*,u*,X*) to hit the RLT ∩ blockSDP set.
# ---------------------------------------------------------
function build_blockSDP_closest_model(
    relaxation_variant::String,
    data::Dict{String,Any},
    x_star::Vector{Float64},
    u_star::Vector{Float64},
    X_star::Matrix{Float64},
)
    params = RLTBigM.prepare_instance(data)
    n = params.n

    m = Model(MosekTools.Optimizer)
    set_silent(m)

    # RLT variables
    @variable(m, x[1:n])
    @variable(m, u[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, R[1:n,1:n])
    @variable(m, U[1:n,1:n], Symmetric)


    # RLT blocks (EU or IU)
    add_relaxation_blocks!(m, relaxation_variant, x, u, X, R, U, params)
    
    """
    @variable(m, Y[1:n+1,1:n+1], Symmetric)
    @constraint(m, Y .== [1.0  u'
                          u    U] )
    @constraint(m, Y in PSDCone())
    """     
              
    @constraint(m, [1.0  u'
                    u    U] in PSDCone())
    

    # To linearize min max
    @variable(m, δ >= 0)

    # |x - x*| <= δ  (componentwise)
    @constraint(m, x .- x_star .<=  δ)
    @constraint(m, x_star .- x .<= δ)

    # |u - u*| <= δ
    #@constraint(m, u .- u_star .<=  δ)
    #@constraint(m, u_star .- u .<= δ)

    # |X - X*| <= δ  (entrywise)
    @constraint(m, X .- X_star .<=  δ)
    @constraint(m, X_star .- X .<= δ)

    @objective(m, Min, δ)

    return m
end

# ---------------------------------------------------------
# Driver: run block-SDP test on all instances
#
# For each instance:
#   1) Solve base relaxation (EU or IU) to get (x*,u*,X*,R*,U*).
#   2) Build block-SDP "closest point" model.
#   3) Solve and classify:
#        - OPTIMAL / FEASIBLE_POINT -> "feasible" (RLT ∩ blockSDP nonempty)
#        - INFEASIBLE / INFEASIBLE_OR_UNBOUNDED -> "infeasible"
#        - anything else -> "unknown / numerically problematic"
#
# Also prints δ*, and the actual max deviations in x,u,X.
# ---------------------------------------------------------
function run_blockSDP_tests(
    relaxation_variant::String;
    print_details::Bool = true,
)
    raw = JSON.parsefile("instances.json")
    insts = [normalize_instance!(deepcopy(d)) for d in raw]

    println("Total number of instances   : ", length(insts))
    println("Base relaxation variant     : ", relaxation_variant)
    println("Test                        : block-SDP closest point")

    feasible_count = 0
    infeasible_count = 0
    unknown_count = 0
    skipped_count = 0

    for (idx, data) in enumerate(insts)
        id = get(data, "id", "unknown")
        println("\n==============================")
        println("Instance index = $idx, id = $id")
        println("==============================")

        # 1) Solve base EU/IU relaxation
        res_base = RLTBigM.build_and_solve(data; variant = relaxation_variant)
        status_base = res_base[1]
        println("Base relaxation status = ", status_base)

        if status_base != :OPTIMAL
            println("Skipping test: base relaxation is not OPTIMAL.")
            skipped_count += 1
            continue
        end

        _, obj_base, x_star, u_star, X_star, R_star, U_star = res_base
        println("Base objective = ", obj_base)

        # 2) Build and solve block-SDP closest model
        m_feas = build_blockSDP_closest_model(
            relaxation_variant,
            data,
            x_star,
            u_star,
            X_star,
        )
        optimize!(m_feas)

        s = termination_status(m_feas)
        println("Block-SDP test status = ", s)

        # 3) Classification
        if s in (MOI.OPTIMAL, MOI.FEASIBLE_POINT)
            feasible_count += 1

            if print_details
                δ_star = value(m_feas[:δ])

                x_new = value.(m_feas[:x])
                u_new = value.(m_feas[:u])
                X_new = value.(m_feas[:X])
                U_new = value.(m_feas[:U])
                R_new = value.(m_feas[:R])

                diffx = maximum(abs.(x_new .- x_star))
                diffu = maximum(abs.(u_new .- u_star))
                diffX = maximum(abs.(X_new .- X_star))
                diffU = maximum(abs.(U_new .- U_star))
                diffR = maximum(abs.(R_new .- R_star))

                println("  δ* (objective)          = ", round(δ_star; digits = 6))
                println("  max |x_new - x_star|    = ", round(diffx;   digits = 6))
                println("  max |u_new - u_star|    = ", round(diffu;   digits = 6))
                println("  max |X_new - X_star|    = ", round(diffX;   digits = 6))
                println("  max |U_new - U_star|    = ", round(diffU;   digits = 6))
                println("  max |R_new - R_star|    = ", round(diffR;   digits = 6))

                # Optional: check eigenvalues of the block matrix W
                # W_val = value.(m_feas[:W])
                # lam_min_W = minimum(eigvals(Symmetric(W_val)))
                # println("  min eigenvalue(W)       = ", round(lam_min_W; digits = 8))
            end

        elseif s in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            infeasible_count += 1
        else
            unknown_count += 1
            println("  -> classified as UNKNOWN / numerically problematic.")
        end
    end

    println("\n========== block-SDP TEST SUMMARY ==========")
    println("Relaxation variant           : ", relaxation_variant)
    println("Feasible instances           : ", feasible_count)
    println("Infeasible instances         : ", infeasible_count)
    println("Unknown / numerical issues   : ", unknown_count)
    println("Skipped (base not OPTIMAL)   : ", skipped_count)
    println("Total                        : ", length(insts))
    println("============================================")
end

# Example calls:
run_blockSDP_tests("EU")
# run_blockSDP_tests("IU")






























