# IP_q1.jl
# ------------------------------------------------------------
# Assignment 1 - Question 1 (1.1 + 1.2)
# Single-machine scheduling with due dates and job-dependent lateness costs.
#
# Features:
#   Items / Processing times / Due dates / Lateness cost
# - Solves both formulations (time-indexed + positional) with weighted lateness objective
# - Per-solve time limit
#
# Requirements (install once in the active Julia environment):
#   ] add JuMP HiGHS
#

using JuMP
using HiGHS
using Dates

const MOI = JuMP.MOI

# ============================
# User configuration
# ============================
const BASE_DIR    = "/home/s2719899/Desktop/IP"
const DATA_DIR    = "/home/s2719899/Desktop/IP/SMSwD"
const ZIP_PATH    = "/home/s2719899/Desktop/IP/SMSwD.zip"
const OUT_CSV     = "/home/s2719899/Desktop/IP/results_q1.csv"

const SOLVER      = "highs"     # highs
const FORMULATION = "both"      # "ti" / "pos" / "both"
const TIME_LIMIT  = 900.0       # seconds per solve

# ============================
# Solver selection
# ============================
function optimizer_factory(solver::String)
    s = lowercase(strip(solver))
    if s in ("highs", "h")
        return HiGHS.Optimizer
    else
        error("Unknown solver='$solver'. This script currently supports: highs.")
    end
end

function set_time_limit!(model::Model, seconds::Real)
    try
        MOI.set(model, MOI.TimeLimitSec(), float(seconds))
    catch
        # Some solver interfaces may not support this attribute; proceed without a hard limit.
    end
end

# ============================
# Utilities
# ============================
function ensure_unzipped(zip_path::String, target_dir::String)
    isfile(zip_path) || error("ZIP file not found: $zip_path")
    run(`unzip -o $(zip_path) -d $(target_dir)`)
end

function is_instance_file(path::String)
    # Matches: Items_30_1.txt, Items_40_5.txt, etc.
    return occursin(r"^Items_\d+_\d+\.txt$", basename(path))
end

function collect_instance_files(root_dir::String)
    files = String[]
    for (dirpath, _, filenames) in walkdir(root_dir)
        for f in filenames
            full = joinpath(dirpath, f)
            is_instance_file(full) || continue
            push!(files, full)
        end
    end
    sort!(files)
    return files
end

function try_parse_numeric_row(line::AbstractString)
    line = strip(String(line))
    isempty(line) && return nothing
    startswith(line, "#") && return nothing
    parts = split(line)
    nums = Float64[]
    for s in parts
        try
            push!(nums, parse(Float64, s))
        catch
            return nothing
        end
    end
    return nums
end

# ============================
# Instance reader for the given dataset format
# ============================
function read_instance(path::String)
    fname = basename(path)
    mch = match(r"Items_(\d+)_", fname)
    mch === nothing && error("Cannot infer n from filename: $fname")
    n = parse(Int, mch.captures[1])

    p = Int[]
    d = Int[]
    v = Int[]

    mode = :none  # :p, :d, :v

    open(path, "r") do io
        for raw in eachline(io)
            line = strip(String(raw))
            isempty(line) && continue

            low = lowercase(line)
            if occursin("processing", low)
                mode = :p
                continue
            elseif occursin("due", low)
                mode = :d
                continue
            elseif occursin("lateness", low) && occursin("cost", low)
                mode = :v
                continue
            elseif occursin("items", low)
                mode = :none
                continue
            end

            nums = try_parse_numeric_row(line)
            nums === nothing && continue

            if mode == :p
                append!(p, Int.(round.(nums)))
            elseif mode == :d
                append!(d, Int.(round.(nums)))
            elseif mode == :v
                append!(v, Int.(round.(nums)))
            end
        end
    end

    length(p) < n && error("Too few processing times in $fname: got $(length(p)), expected $n")
    length(d) < n && error("Too few due dates in $fname: got $(length(d)), expected $n")
    if length(v) < n
        v = fill(1, n)  # fallback (should not happen for Q1)
    end

    p = p[1:n]
    d = d[1:n]
    v = v[1:n]
    return (p=p, d=d, v=v)
end

# ============================
# Models
# ============================

# Time-indexed formulation (slides-consistent)
function build_time_indexed_model(p::Vector{Int}, d::Vector{Int}, v::Vector{Int};
                                  optimizer, time_limit::Real)
    n = length(p)
    J = 1:n
    m = sum(p)     # horizon
    T = 1:m

    model = Model(optimizer)
    set_silent(model)
    set_time_limit!(model, time_limit)

    @variable(model, x[J, T], Bin)
    @variable(model, l[J] >= 0)

    # Start exactly once, feasible starts t = 1..m-pj+1
    for j in J
        last_start = m - p[j] + 1
        @constraint(model, sum(x[j, t] for t in 1:last_start) == 1)
        if last_start < m
            @constraint(model, [t in (last_start + 1):m], x[j, t] == 0)
        end
    end

    # Capacity constraint (slides form)
    for t in T
        expr = AffExpr()
        for j in J
            last_start = m - p[j] + 1
            s_lo = max(1, t - p[j] + 1)
            s_hi = min(t, last_start)
            if s_lo <= s_hi
                for s in s_lo:s_hi
                    add_to_expression!(expr, 1.0, x[j, s])
                end
            end
        end
        @constraint(model, expr <= 1)
    end

    # Lateness constraint (slides): sum_t t x_j^t + p_j - 1 <= d_j + l_j
    for j in J
        last_start = m - p[j] + 1
        @constraint(model,
            sum(t * x[j, t] for t in 1:last_start) + p[j] - 1 <= d[j] + l[j]
        )
    end

    @objective(model, Min, sum(v[j] * l[j] for j in J))
    return model
end

# Positional formulation (slides-consistent) + linearization
function build_positional_model(p::Vector{Int}, d::Vector{Int}, v::Vector{Int};
                                optimizer, time_limit::Real)
    n = length(p)
    J = 1:n
    m = sum(p)
    U = m

    model = Model(optimizer)
    set_silent(model)
    set_time_limit!(model, time_limit)

    @variable(model, x[J, J], Bin)   # x[j,r]
    @variable(model, c[J] >= 0)      # c^r
    @variable(model, l[J] >= 0)      # l^r
    @variable(model, w[J, J] >= 0)   # w[j,r] = x[j,r] * l[r]

    @constraint(model, [j in J], sum(x[j, r] for r in J) == 1)
    @constraint(model, [r in J], sum(x[j, r] for j in J) == 1)

    @constraint(model, c[1] == sum(p[j] * x[j, 1] for j in J))
    @constraint(model, [r in 2:n], c[r] == c[r - 1] + sum(p[j] * x[j, r] for j in J))

    @constraint(model, [r in J], c[r] <= sum(d[j] * x[j, r] for j in J) + l[r])

    # Linearization w[j,r] = x[j,r] * l[r]
    @constraint(model, [j in J, r in J], w[j, r] <= U * x[j, r])
    @constraint(model, [j in J, r in J], w[j, r] <= l[r])
    @constraint(model, [j in J, r in J], w[j, r] >= l[r] - U * (1 - x[j, r]))

    @objective(model, Min, sum(v[j] * w[j, r] for j in J, r in J))
    return model
end

# ============================
# Solve + metrics
# ============================
function solve_and_collect(model::Model)
    optimize!(model)

    term = termination_status(model)

    obj = try
        objective_value(model)
    catch
        NaN
    end

    bound = try
        objective_bound(model)
    catch
        NaN
    end

    gap = try
        MOI.get(model, MOI.RelativeGap())
    catch
        NaN
    end

    time_sec = try
        MOI.get(model, MOI.SolveTimeSec())
    catch
        NaN
    end

    return (termination=string(term), objective=obj, bound=bound, relgap=gap, time_sec=time_sec)
end

# ============================
# CSV checkpoint + resume
# ============================
function write_csv_atomic(path::String, header::Vector{String}, rows::Vector{Vector{Any}})
    tmp = path * ".tmp"
    open(tmp, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(string.(row), ","))
        end
    end
    mv(tmp, path; force=true)
end

function load_done_pairs(out::String)
    done = Set{Tuple{String,String}}()
    if !isfile(out)
        return done
    end
    open(out, "r") do io
        first = true
        for line in eachline(io)
            if first
                first = false
                continue
            end
            parts = split(line, ",")
            length(parts) < 2 && continue
            push!(done, (parts[1], parts[2]))
        end
    end
    return done
end

# ============================
# Batch runner
# ============================
function run_q1(; base_dir::String, data_dir::String, solver::String, formulation::String,
                time_limit::Real, out::String)

    optimizer = optimizer_factory(solver)
    files = collect_instance_files(data_dir)
    isempty(files) && error("No instance files found under: $data_dir")

    header = ["instance","formulation","termination","objective","best_bound","rel_gap","time_sec","n_jobs","horizon_m"]
    rows = Vector{Vector{Any}}()

    done = load_done_pairs(out)
    if !isempty(done)
        println("Resume enabled: found $(length(done)) completed (instance, formulation) pairs in $(out).")
        # Load existing rows so the output CSV stays complete
        # (simple approach: keep writing only new rows but preserve done by re-reading)
        # Here we just append new rows; the CSV remains valid for analysis.
    end

    f = lowercase(formulation)
    println("Found $(length(files)) instances in: $data_dir")
    println("Solver=$solver | formulation=$formulation | time_limit=$(time_limit)s")
    println("Started at: ", now())

    for (k, path) in enumerate(files)
        instance_name = basename(path)
        inst = read_instance(path)
        p, d, v = inst.p, inst.d, inst.v
        n = length(p)
        m = sum(p)

        println("\n[$k/$(length(files))] $instance_name | n=$n | m=$m | time=$(now())")

        # Time-indexed
        if f in ("both", "ti", "time", "time-indexed", "time_indexed")
            if (instance_name, "time_indexed") ∈ done
                println("  TI: already done, skipping.")
            else
                println("  -> Solving TI...")
                model_ti = build_time_indexed_model(p, d, v; optimizer=optimizer, time_limit=time_limit)
                res = solve_and_collect(model_ti)
                println("     TI status=$(res.termination) obj=$(res.objective) gap=$(res.relgap) time=$(round(res.time_sec,digits=2))s")
                push!(rows, [instance_name, "time_indexed", res.termination, res.objective, res.bound, res.relgap, res.time_sec, n, m])
                push!(done, (instance_name, "time_indexed"))
                write_csv_atomic(out, header, rows)  # checkpoint
            end
        end

        # Positional
        if f in ("both", "pos", "positional", "p")
            if (instance_name, "positional") ∈ done
                println("  POS: already done, skipping.")
            else
                println("  -> Solving POS...")
                model_pos = build_positional_model(p, d, v; optimizer=optimizer, time_limit=time_limit)
                res = solve_and_collect(model_pos)
                println("     POS status=$(res.termination) obj=$(res.objective) gap=$(res.relgap) time=$(round(res.time_sec,digits=2))s")
                push!(rows, [instance_name, "positional", res.termination, res.objective, res.bound, res.relgap, res.time_sec, n, m])
                push!(done, (instance_name, "positional"))
                write_csv_atomic(out, header, rows)  # checkpoint
            end
        end
    end

    println("\nFinished at: ", now())
    println("Results written to: $out")
end

# ============================
# Main
# ============================
function main()
    if !isdir(DATA_DIR)
        println("DATA_DIR not found: $DATA_DIR")
        println("Trying to unzip: $ZIP_PATH into $BASE_DIR")
        ensure_unzipped(ZIP_PATH, BASE_DIR)
    end
    isdir(DATA_DIR) || error("DATA_DIR still not found after unzip: $DATA_DIR")

    run_q1(
        base_dir = BASE_DIR,
        data_dir = DATA_DIR,
        solver = SOLVER,
        formulation = FORMULATION,
        time_limit = TIME_LIMIT,
        out = OUT_CSV
    )
end

main()