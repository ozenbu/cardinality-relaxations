# Run file: cwlp_lr.jl
# Required packages: JuMP, HiGHS, CSV, DataFrames, LinearAlgebra
# Data directory: set by DATA_DIR at the top of the file
# The script produces results_lr.csv and results_table.tex

using JuMP
using HiGHS
using CSV
using DataFrames
using LinearAlgebra
using Printf
import MathOptInterface as MOI
using Base.Filesystem: basename

# ============================================================
# User settings
# ============================================================
const DATA_DIR = joinpath(@__DIR__, "Data sets")
#const DATA_DIR = "/home/s2719899/Desktop/IP/Data sets"
const OUTPUT_CSV = joinpath(DATA_DIR, "results_lr.csv")
const OUTPUT_TEX = joinpath(DATA_DIR, "results_table.tex")

const MAX_ITER = 500
const EXACT_TIME_LIMIT = 300.0

# ============================================================
# Data structure
# ============================================================

struct CWLPInstance
    name::String
    m::Int                    # number of facilities
    n::Int                    # number of customers
    K::Vector{Float64}        # facility capacities
    f::Vector{Float64}        # fixed opening costs
    b::Vector{Float64}        # customer demands
    c::Matrix{Float64}        # shipping costs, size n x m
end

# ============================================================
# Read OR-Library CAP instance
# Format:
#   m n
#   K_j f_j   for j = 1,...,m
#   then for each customer i:
#       b_i
#       c_{i1} ... c_{im}
# ============================================================

function read_cap_instance(path::String)
    raw = parse.(Float64, split(read(path, String)))

    m = Int(raw[1])
    n = Int(raw[2])

    idx = 3
    K = Vector{Float64}(undef, m)
    f = Vector{Float64}(undef, m)

    for j in 1:m
        K[j] = raw[idx]
        f[j] = raw[idx + 1]
        idx += 2
    end

    b = Vector{Float64}(undef, n)
    c = Matrix{Float64}(undef, n, m)

    for i in 1:n
        b[i] = raw[idx]
        idx += 1
        for j in 1:m
            c[i, j] = raw[idx]
            idx += 1
        end
    end

    return CWLPInstance(basename(path), m, n, K, f, b, c)
end

# ============================================================
# Optional exact solve of the original CWLP
# ============================================================

function solve_cwlp_exact(inst::CWLPInstance; time_limit::Float64 = 300.0, silent::Bool = true)
    model = Model(HiGHS.Optimizer)
    set_optimizer_attribute(model, "time_limit", time_limit)
    if silent
        set_silent(model)
    end

    @variable(model, x[1:inst.n, 1:inst.m] >= 0)
    @variable(model, y[1:inst.m], Bin)

    @objective(model, Min,
        sum(inst.c[i, j] * x[i, j] for i in 1:inst.n, j in 1:inst.m) +
        sum(inst.f[j] * y[j] for j in 1:inst.m)
    )

    @constraint(model, [i in 1:inst.n], sum(x[i, j] for j in 1:inst.m) == 1)
    @constraint(model, [j in 1:inst.m], sum(inst.b[i] * x[i, j] for i in 1:inst.n) <= inst.K[j] * y[j])

    optimize!(model)

    status = termination_status(model)
    has_primal = JuMP.has_values(model)

    obj = has_primal ? objective_value(model) : Inf
    bound = try
        objective_bound(model)
    catch
        NaN
    end

    return (; status, obj, bound)
end

# ============================================================
# Facility subproblem
#
# For fixed lambda and facility j:
#
#   min f_j y_j + sum_i (c_ij - lambda_i) x_ij
#   s.t. sum_i b_i x_ij <= K_j y_j
#        0 <= x_ij <= 1
#        y_j in {0,1}
#
# If y_j = 1, this is a continuous knapsack:
#   maximize sum_i (lambda_i - c_ij) x_ij
#   s.t. sum_i b_i x_ij <= K_j, 0 <= x_ij <= 1
#
# We solve it greedily by profit/weight ratio:
#   (lambda_i - c_ij) / b_i
# ============================================================

function solve_facility_subproblem(inst::CWLPInstance, λ::Vector{Float64}, j::Int)
    n = inst.n

    profit = λ .- inst.c[:, j]
    ratio = [profit[i] / inst.b[i] for i in 1:n]
    order = sortperm(1:n, by = i -> ratio[i], rev = true)

    xj = zeros(n)
    remaining_cap = inst.K[j]
    saving = 0.0

    for i in order
        if remaining_cap <= 1e-9
            break
        end
        if profit[i] <= 1e-12
            break
        end

        frac = min(1.0, remaining_cap / inst.b[i])
        xj[i] = frac
        saving += profit[i] * frac
        remaining_cap -= inst.b[i] * frac
    end

    # Compare:
    # closed facility -> value 0
    # open facility   -> f_j - saving
    open_value = inst.f[j] - saving

    if open_value < -1e-9
        return open_value, xj, 1.0
    else
        return 0.0, zeros(n), 0.0
    end
end

# ============================================================
# Lagrangian subproblem
# ============================================================

function lagrangian_subproblem(inst::CWLPInstance, λ::Vector{Float64})
    X = zeros(inst.n, inst.m)
    Y = zeros(inst.m)

    lb = sum(λ)

    for j in 1:inst.m
        φj, xj, yj = solve_facility_subproblem(inst, λ, j)
        lb += φj
        X[:, j] = xj
        Y[j] = yj
    end

    g = 1 .- vec(sum(X, dims = 2))   # subgradient
    return lb, X, Y, g
end

# ============================================================
# Feasible solution with fixed open facilities
# Solve transportation LP:
#
#   min sum c_ij x_ij + sum_{j open} f_j
#   s.t. sum_j x_ij = 1
#        sum_i b_i x_ij <= K_j   for open j
#        x_ij >= 0
# ============================================================

function solve_transport_given_open_set(inst::CWLPInstance, open::BitVector)
    open_idx = findall(open)

    if isempty(open_idx)
        return Inf, nothing
    end

    if sum(inst.K[open_idx]) < sum(inst.b) - 1e-9
        return Inf, nothing
    end

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[1:inst.n, open_idx] >= 0)

    @objective(model, Min,
        sum(inst.c[i, j] * x[i, j] for i in 1:inst.n, j in open_idx) +
        sum(inst.f[j] for j in open_idx)
    )

    @constraint(model, [i in 1:inst.n], sum(x[i, j] for j in open_idx) == 1)
    @constraint(model, [j in open_idx], sum(inst.b[i] * x[i, j] for i in 1:inst.n) <= inst.K[j])

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        return Inf, nothing
    end

    Xfull = zeros(inst.n, inst.m)
    for i in 1:inst.n, j in open_idx
        Xfull[i, j] = value(x[i, j])
    end

    return objective_value(model), Xfull
end

# ============================================================
# Initial feasible solution
# Open facilities by cheapest fixed-cost-per-capacity
# until enough total capacity is available, then solve LP.
# ============================================================

function initial_feasible_solution(inst::CWLPInstance)
    score = inst.f ./ inst.K
    order = sortperm(1:inst.m, by = j -> score[j])

    open = falses(inst.m)
    total_demand = sum(inst.b)
    total_cap = 0.0

    for j in order
        open[j] = true
        total_cap += inst.K[j]
        if total_cap >= total_demand - 1e-9
            break
        end
    end

    ub, X = solve_transport_given_open_set(inst, open)

    if !isfinite(ub)
        error("Could not build an initial feasible solution for instance $(inst.name).")
    end

    return ub, open, X
end

# ============================================================
# Repair heuristic
# Start from LR open facilities.
# If capacity is insufficient, open more facilities using the
# same fixed-cost-per-capacity rule, then solve the LP.
# ============================================================

function repair_solution(inst::CWLPInstance, Ylr::Vector{Float64})
    open = BitVector(Ylr .> 0.5)

    total_demand = sum(inst.b)
    current_cap = sum(inst.K[findall(open)])

    if current_cap < total_demand - 1e-9
        closed = findall(.!open)
        order = sort(closed, by = j -> inst.f[j] / inst.K[j])

        for j in order
            open[j] = true
            current_cap += inst.K[j]
            if current_cap >= total_demand - 1e-9
                break
            end
        end
    end

    ub, X = solve_transport_given_open_set(inst, open)
    return ub, open, X
end

# ============================================================
# Subgradient method
# ============================================================

function subgradient_lr(
    inst::CWLPInstance;
    max_iter::Int = 500,
    alpha0::Float64 = 2.0,
    stall_limit::Int = 20,
    tol_subgrad::Float64 = 1e-6
)
    λ = zeros(inst.n)

    ub_best, best_open, best_X_feas = initial_feasible_solution(inst)
    lb_best = -Inf

    alpha = alpha0
    stall = 0

    history = DataFrame(
        iter = Int[],
        lb = Float64[],
        best_lb = Float64[],
        ub = Float64[],
        best_ub = Float64[],
        step = Float64[],
        normg = Float64[]
    )

    for k in 1:max_iter
        lb, Xlr, Ylr, g = lagrangian_subproblem(inst, λ)

        if lb > lb_best + 1e-9
            lb_best = lb
            stall = 0
        else
            stall += 1
        end

        ub, open, Xfeas = repair_solution(inst, Ylr)
        if ub < ub_best
            ub_best = ub
            best_open = copy(open)
            best_X_feas = Xfeas
        end

        normg = norm(g)

        if normg <= tol_subgrad
            push!(history, (
                iter = k,
                lb = lb,
                best_lb = lb_best,
                ub = ub,
                best_ub = ub_best,
                step = 0.0,
                normg = normg
            ))
            break
        end

        step = alpha * max(ub_best - lb, 1e-6) / (normg^2)

        push!(history, (
            iter = k,
            lb = lb,
            best_lb = lb_best,
            ub = ub,
            best_ub = ub_best,
            step = step,
            normg = normg
        ))

        λ .+= step .* g

        if stall >= stall_limit
            alpha /= 2
            stall = 0
            if alpha < 1e-3
                break
            end
        end

        rel_gap = (ub_best - lb_best) / max(abs(ub_best), 1.0)
        if rel_gap <= 1e-4
            break
        end
    end

    gap_pct = 100.0 * (ub_best - lb_best) / max(abs(ub_best), 1.0)

    return (
        lb_best = lb_best,
        ub_best = ub_best,
        gap_pct = gap_pct,
        λ = λ,
        history = history,
        best_open = best_open,
        best_X_feas = best_X_feas
    )
end

# ============================================================
# Run one instance
# ============================================================

function run_one_instance(path::String; max_iter::Int = 500, exact_time_limit::Float64 = 300.0)
    inst = read_cap_instance(path)

    println("--------------------------------------------------")
    println("Running instance: ", inst.name)
    println("Facilities = ", inst.m, ", Customers = ", inst.n)

    t0 = time()
    lr = subgradient_lr(inst; max_iter = max_iter)
    lr_time = time() - t0

    t1 = time()
    exact = solve_cwlp_exact(inst; time_limit = exact_time_limit)
    exact_time = time() - t1

    exact_obj = isfinite(exact.obj) ? exact.obj : NaN
    ub_vs_exact_gap = isfinite(exact_obj) ? 100.0 * (lr.ub_best - exact_obj) / max(abs(exact_obj), 1.0) : NaN
    exact_minus_lb = isfinite(exact_obj) ? 100.0 * (exact_obj - lr.lb_best) / max(abs(exact_obj), 1.0) : NaN

    println(@sprintf("LR best LB   = %.4f", lr.lb_best))
    println(@sprintf("LR best UB   = %.4f", lr.ub_best))
    println(@sprintf("LR gap (%%)   = %.4f", lr.gap_pct))
    println(@sprintf("LR time (s)  = %.4f", lr_time))
    println(@sprintf("Exact obj    = %.4f", exact_obj))
    println(@sprintf("Exact time   = %.4f", exact_time))

    row = DataFrame(
        instance = [inst.name],
        m = [inst.m],
        n = [inst.n],
        lr_lb = [lr.lb_best],
        lr_ub = [lr.ub_best],
        lr_gap_pct = [lr.gap_pct],
        lr_time_sec = [lr_time],
        exact_obj = [exact_obj],
        exact_bound = [exact.bound],
        exact_time_sec = [exact_time],
        ub_vs_exact_pct = [ub_vs_exact_gap],
        exact_minus_lb_pct = [exact_minus_lb]
    )

    return row, lr
end

# ============================================================
# Run all instances in the folder whose names start with "cap"
# ============================================================

function run_all_instances(folder::String; max_iter::Int = 500, exact_time_limit::Float64 = 300.0)
    files = sort(filter(f -> startswith(f, "cap") && endswith(f, ".txt"), readdir(folder)))

    results = DataFrame()
    histories = Dict{String, DataFrame}()

    for file in files
        path = joinpath(folder, file)
        row, lr = run_one_instance(path; max_iter = max_iter, exact_time_limit = exact_time_limit)
        append!(results, row)
        histories[file] = lr.history
    end

    CSV.write(OUTPUT_CSV, results)

    for (name, hist) in histories
        histfile = replace(name, ".txt" => "_history.csv")
        CSV.write(joinpath(folder, histfile), hist)
    end

    return results, histories
end

# ============================================================
# Write LaTeX table
# ============================================================

function write_latex_table(df::DataFrame, path::String)
    open(path, "w") do io
        println(io, "\\begin{tabular}{lrrrrrr}")
        println(io, "\\toprule")
        println(io, "Instance & LB & UB & Gap (\\%) & LR time & Exact & Exact time\\\\")
        println(io, "\\midrule")
        for r in eachrow(df)
            @printf(
                io,
                "%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f\\\\\n",
                r.instance,
                r.lr_lb,
                r.lr_ub,
                r.lr_gap_pct,
                r.lr_time_sec,
                r.exact_obj,
                r.exact_time_sec
            )
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
end

# ============================================================
# Main
# ============================================================

function main()
    results, histories = run_all_instances(DATA_DIR; max_iter = MAX_ITER, exact_time_limit = EXACT_TIME_LIMIT)

    println("\n================ FINAL RESULTS ================\n")
    println(results)

    write_latex_table(results, OUTPUT_TEX)

    println("\nCSV written to: ", OUTPUT_CSV)
    println("LaTeX table written to: ", OUTPUT_TEX)

    return results, histories
end

main()