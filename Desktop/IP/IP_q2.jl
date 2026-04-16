# IP_q2.jl
# ------------------------------------------------------------
# Assignment 1 - Question 2.1
# Single-machine scheduling with due dates and sequence-dependent setup times.
#
# Study:
#   Compare four positional-formulation variants:
#     M0: baseline (z binary, no VI2)
#     M1: + VI2 (sum_i z[i,j,r] = x[j,r]) with z[j,j,r]=0
#     M2: preprocessing (relax z integrality: z continuous in [0,1])
#     M3: VI2 + preprocessing
#
# Objective for Q2.1:
#   All lateness costs are 1 => minimize total lateness sum_r l[r].
#
# Requirements:
#   ] add JuMP HiGHS
#

using JuMP
using HiGHS
using Dates

const MOI = JuMP.MOI

# ============================
# User configuration
# ============================
const BASE_DIR   = "/home/s2719899/Desktop/IP"
const DATA_DIR   = "/home/s2719899/Desktop/IP/SMSwDS"
const ZIP_PATH   = "/home/s2719899/Desktop/IP/SMSwDS.zip"
const OUT_CSV    = "/home/s2719899/Desktop/IP/results_q2_21.csv"

const SOLVER     = "highs"
const TIME_LIMIT = 900.0  # seconds per solve (15 minutes)

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
        # If unsupported, continue without a hard time limit.
    end
end

# ============================
# Utilities
# ============================
function ensure_unzipped(zip_path::String, target_dir::String)
    isfile(zip_path) || error("ZIP file not found: $zip_path")
    run(`unzip -o $(zip_path) -d $(target_dir)`)
end

function is_instance_file_setup(path::String)
    # Matches: ItemsSetup_10_1.txt, ItemsSetup_15_5.txt, etc.
    return occursin(r"^ItemsSetup_\d+_\d+\.txt$", basename(path))
end

function collect_instance_files(root_dir::String)
    files = String[]
    for (dirpath, _, filenames) in walkdir(root_dir)
        for f in filenames
            full = joinpath(dirpath, f)
            is_instance_file_setup(full) || continue
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
# CSV: load existing rows for resume
# ============================
function load_existing_csv(out::String)
    header = String[]
    rows = Vector{Vector{Any}}()
    done = Set{Tuple{String,String}}()

    if !isfile(out)
        return header, rows, done
    end

    open(out, "r") do io
        first = true
        for line in eachline(io)
            if first
                header = split(chomp(line), ",")
                first = false
                continue
            end
            parts = split(chomp(line), ",")
            isempty(parts) && continue
            push!(rows, Any[parts...])
            if length(parts) >= 2
                push!(done, (parts[1], parts[2]))
            end
        end
    end

    return header, rows, done
end

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

# ============================
# Instance reader: ItemsSetup format
# ============================
function read_instance_setup(path::String)
    fname = basename(path)
    mch = match(r"ItemsSetup_(\d+)_", fname)
    mch === nothing && error("Cannot infer n from filename: $fname")
    n = parse(Int, mch.captures[1])
    J = 1:n

    p = Float64[]
    d = Float64[]
    mode = :none  # :p, :d, :setup

    setup_rows = Vector{Vector{Float64}}()

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
            elseif occursin("setup", low) && occursin("times", low)
                mode = :setup
                continue
            elseif occursin("lateness", low) && occursin("cost", low)
                mode = :none
                continue
            elseif occursin("items", low)
                mode = :none
                continue
            end

            nums = try_parse_numeric_row(line)
            nums === nothing && continue

            if mode == :p
                append!(p, nums)
            elseif mode == :d
                append!(d, nums)
            elseif mode == :setup
                push!(setup_rows, nums)
            end
        end
    end

    length(p) < n && error("Too few processing times in $fname: got $(length(p)), expected $n")
    length(d) < n && error("Too few due dates in $fname: got $(length(d)), expected $n")
    p = p[1:n]
    d = d[1:n]

    length(setup_rows) < n && error("Too few setup rows in $fname: got $(length(setup_rows)), expected $n")

    s = zeros(Float64, n, n)
    for i in J
        row = setup_rows[i]
        length(row) < n && error("Setup row $i too short in $fname: got $(length(row)), expected $n")
        for j in J
            s[i, j] = row[j]
        end
    end

    return (p=p, d=d, s=s)
end

# ============================
# Positional model with setup times
# ============================
# IMPORTANT FIX:
# Only ONE semicolon is allowed in a function argument list to start keyword arguments.
function build_positional_setup_model(
    p::Vector{Float64},
    d::Vector{Float64},
    s::Matrix{Float64};
    optimizer,
    time_limit::Real,
    addVI2::Bool=false,
    relaxZ::Bool=false
)
    n = length(p)
    J = 1:n

    model = Model(optimizer)
    set_silent(model)
    set_time_limit!(model, time_limit)

    # Decision variables
    @variable(model, x[J, J], Bin)      # x[j,r]
    @variable(model, c[J] >= 0)         # completion time of position r
    @variable(model, l[J] >= 0)         # lateness at position r

    # z[i,j,r] only for r = 2..n
    if relaxZ
        @variable(model, 0 <= z[J, J, 2:n] <= 1)
    else
        @variable(model, z[J, J, 2:n], Bin)
    end

    # Assignment
    @constraint(model, [j in J], sum(x[j, r] for r in J) == 1)
    @constraint(model, [r in J], sum(x[j, r] for j in J) == 1)

    # Disallow self-predecessor arcs explicitly
    @constraint(model, [j in J, r in 2:n], z[j, j, r] == 0)

    # Linking constraints (linearization of z = x_prev * x_curr)
    for r in 2:n, i in J, j in J
        @constraint(model, z[i, j, r] <= x[i, r-1])
        @constraint(model, z[i, j, r] <= x[j, r])
        @constraint(model, z[i, j, r] >= x[i, r-1] + x[j, r] - 1)
    end

    # Completion times (assume zero initial setup before the first job)
    @constraint(model, c[1] == sum(p[j] * x[j, 1] for j in J))
    @constraint(model, [r in 2:n],
        c[r] == c[r-1] +
                sum(p[j] * x[j, r] for j in J) +
                sum(s[i, j] * z[i, j, r] for i in J, j in J)
    )

    # Lateness constraint: c[r] <= due(job at r) + l[r]
    @constraint(model, [r in J],
        c[r] <= sum(d[j] * x[j, r] for j in J) + l[r]
    )

    # Second valid inequality (VI2):
    # Using sum over all i is safe because z[j,j,r]=0 is enforced.
    if addVI2
        @constraint(model, [r in 2:n, j in J],
            sum(z[i, j, r] for i in J) == x[j, r]
        )
    end

    # Objective for Q2.1: total lateness (unit costs)
    @objective(model, Min, sum(l[r] for r in J))

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
# Batch runner for Q2.1
# ============================
function run_q2_21(; data_dir::String, solver::String, time_limit::Real, out::String)
    optimizer = optimizer_factory(solver)
    files = collect_instance_files(data_dir)
    isempty(files) && error("No instance files found under: $data_dir")

    default_header = [
        "instance","variant","addVI2","relaxZ",
        "termination","objective","best_bound","rel_gap","time_sec",
        "n_jobs","sum_p"
    ]

    header, rows, done = load_existing_csv(out)
    if isempty(header)
        header = default_header
    end

    println("Found $(length(files)) instances in: $data_dir")
    println("Solver=$solver | time_limit=$(time_limit)s")
    println("Started at: ", now())
    if !isempty(done)
        println("Resume enabled: found $(length(done)) completed (instance, variant) pairs in $(out).")
    end

    variants = [
        ("M0_base", false, false),
        ("M1_VI2",  true,  false),
        ("M2_relaxZ", false, true),
        ("M3_VI2_relaxZ", true, true),
    ]

    for (k, path) in enumerate(files)
        instance_name = basename(path)
        inst = read_instance_setup(path)
        p, d, s = inst.p, inst.d, inst.s
        n = length(p)
        sump = sum(p)

        println("\n[$k/$(length(files))] $instance_name | n=$n | sum(p)=$(round(sump, digits=2)) | time=$(now())")

        for (vname, addVI2, relaxZ) in variants
            if (instance_name, vname) ∈ done
                println("  $vname: already done, skipping.")
                continue
            end

            println("  -> Solving $vname (addVI2=$addVI2, relaxZ=$relaxZ)...")
            model = build_positional_setup_model(
                p, d, s;
                optimizer=optimizer,
                time_limit=time_limit,
                addVI2=addVI2,
                relaxZ=relaxZ
            )

            res = solve_and_collect(model)
            println("     status=$(res.termination) obj=$(res.objective) gap=$(res.relgap) time=$(round(res.time_sec,digits=2))s")

            push!(rows, Any[
                instance_name, vname, addVI2, relaxZ,
                res.termination, res.objective, res.bound, res.relgap, res.time_sec,
                n, sump
            ])

            push!(done, (instance_name, vname))
            write_csv_atomic(out, header, rows)  # checkpoint
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

    run_q2_21(
        data_dir = DATA_DIR,
        solver = SOLVER,
        time_limit = TIME_LIMIT,
        out = OUT_CSV
    )
end

main()