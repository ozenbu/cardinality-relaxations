# ============================================================
# search_comp_dropX_gap.jl
# ============================================================

using Gurobi
using Random
using LinearAlgebra
using Printf
using Dates
using Serialization

const ROOT = @__DIR__

# Assumption: these modules are already included/loaded in the session.
using .RLTCompDropX
using .DropXTestInstances

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

const COMP_VARIANTS = ["RLT_CompBin", "RLT_CompBinBou", "RLT_CompBou"]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

function canon_pair(i::Int, j::Int)
    return i <= j ? (i, j) : (j, i)
end

function normalize_entries(v)
    v === nothing && return Tuple{Int,Int}[]

    entries = Tuple{Int,Int}[]
    for (i, j) in v
        push!(entries, canon_pair(Int(i), Int(j)))
    end
    return unique(entries)
end

function normalize_blocks(v)
    v === nothing && return Tuple{Vector{Int},Vector{Int}}[]

    blocks = Tuple{Vector{Int},Vector{Int}}[]
    for (I, J) in v
        push!(blocks, (Int.(collect(I)), Int.(collect(J))))
    end
    return blocks
end

function entries_from_blocks(blocks)
    entries = Tuple{Int,Int}[]
    for (I, J) in blocks
        for i in I, j in J
            push!(entries, canon_pair(Int(i), Int(j)))
        end
    end
    return unique(entries)
end

function all_drop_entries(inst)
    entries = normalize_entries(get(inst, "drop_X_entries", Tuple{Int,Int}[]))
    blocks  = normalize_blocks(get(inst, "drop_X_blocks", Tuple{Vector{Int},Vector{Int}}[]))
    append!(entries, entries_from_blocks(blocks))
    return unique(entries)
end

function zero_drop_entries!(Q::Matrix{Float64}, entries)
    for (i, j) in entries
        Q[i, j] = 0.0
        Q[j, i] = 0.0
    end
    return Q
end

function symrand(rng, n)
    A = randn(rng, n, n)
    return 0.5 .* (A .+ A')
end

function symmetrize(A)
    return 0.5 .* (A .+ A')
end

function round_mat(A; digits::Int = 4)
    return round.(Matrix{Float64}(A); digits = digits)
end

function round_vec(v; digits::Int = 4)
    return round.(Vector{Float64}(v); digits = digits)
end

function fmt(x)
    if x === missing
        return "missing"
    elseif x isa Real
        return @sprintf("%.9g", x)
    else
        return string(x)
    end
end

function tuple_mats_to_vec(x)
    x === nothing && return Matrix{Float64}[]
    return [Matrix{Float64}(Q) for Q in x]
end

function tuple_vecs_to_vec(x)
    x === nothing && return Vector{Float64}[]
    return [Vector{Float64}(q) for q in x]
end

function scalar_list_to_vec(x)
    x === nothing && return Float64[]
    return Float64.(collect(x))
end

# Feasible certificate only for keeping perturbed quadratic constraints feasible.
function certificate_point(inst)
    id = get(inst, "id", "")
    n = inst["n"]

    if occursin("dropX05", id)
        return [0.25, 0.25, 0.0, 0.0, 0.0, 0.0]
    else
        return zeros(Float64, n)
    end
end

# ------------------------------------------------------------
# Solve one complementarity variant: full vs drop
# ------------------------------------------------------------

function solve_comp_pair(inst, variant::String;
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
)
    entries = normalize_entries(get(inst, "drop_X_entries", Tuple{Int,Int}[]))
    blocks  = normalize_blocks(get(inst, "drop_X_blocks", Tuple{Vector{Int},Vector{Int}}[]))

    try
        full = RLTCompDropX.build_and_solve(
            inst;
            variant = variant,
            optimizer = optimizer,
            verbose = verbose,
            drop_X_entries = Tuple{Int,Int}[],
            drop_X_blocks  = Tuple{Vector{Int},Vector{Int}}[],
            check_aggregate_zero = true,
        )

        drop = RLTCompDropX.build_and_solve(
            inst;
            variant = variant,
            optimizer = optimizer,
            verbose = verbose,
            drop_X_entries = entries,
            drop_X_blocks  = blocks,
            check_aggregate_zero = true,
        )

        status_full = full[1]
        status_drop = drop[1]

        obj_full = status_full == :OPTIMAL ? Float64(full[2]) : missing
        obj_drop = status_drop == :OPTIMAL ? Float64(drop[2]) : missing

        diff = (obj_full === missing || obj_drop === missing) ? missing : obj_full - obj_drop

        return (; status_full, status_drop, obj_full, obj_drop, diff, error_msg = "")
    catch err
        return (;
            status_full = :ERROR,
            status_drop = :ERROR,
            obj_full = missing,
            obj_drop = missing,
            diff = missing,
            error_msg = sprint(showerror, err),
        )
    end
end

function solve_all_comp_pairs(inst;
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
)
    results = Dict{String,Any}()

    for var in COMP_VARIANTS
        results[var] = solve_comp_pair(
            inst,
            var;
            optimizer = optimizer,
            verbose = verbose,
        )
    end

    return results
end

# ------------------------------------------------------------
# Perturbation generator
# ------------------------------------------------------------

function perturb_instance(
    base::Dict{String,Any},
    rng::AbstractRNG;
    scale_Q0::Float64 = 0.50,
    scale_q0::Float64 = 0.50,
    scale_Qcon::Float64 = 0.25,
    scale_qcon::Float64 = 0.25,
    slack_qi::Float64 = 1.0,
    perturb_constraints::Bool = true,
    round_data::Bool = true,
    round_digits::Int = 4,
)
    inst = deepcopy(base)
    n = inst["n"]
    D = all_drop_entries(inst)

    xcert = certificate_point(inst)

    # Objective perturbation
    Q0 = Matrix{Float64}(inst["Q0"])
    q0 = Vector{Float64}(inst["q0"])

    Q0_new = symmetrize(Q0 .+ scale_Q0 .* symrand(rng, n))
    q0_new = q0 .+ scale_q0 .* randn(rng, n)

    if round_data
        Q0_new = round_mat(Q0_new; digits = round_digits)
        q0_new = round_vec(q0_new; digits = round_digits)
    end

    zero_drop_entries!(Q0_new, D)

    inst["Q0"] = Q0_new
    inst["q0"] = q0_new

    # Constraint perturbation
    if perturb_constraints
        Qi_list = tuple_mats_to_vec(get(inst, "Qi", nothing))
        qi_list = tuple_vecs_to_vec(get(inst, "qi", nothing))
        ri_list = scalar_list_to_vec(get(inst, "ri", nothing))

        for k in eachindex(Qi_list)
            Q = symmetrize(Qi_list[k] .+ scale_Qcon .* symrand(rng, n))
            q = qi_list[k] .+ scale_qcon .* randn(rng, n)

            if round_data
                Q = round_mat(Q; digits = round_digits)
                q = round_vec(q; digits = round_digits)
            end

            zero_drop_entries!(Q, D)

            r = -0.5 * dot(xcert, Q * xcert) - dot(q, xcert) - slack_qi
            if round_data
                r = round(Float64(r); digits = round_digits)
            end

            Qi_list[k] = Q
            qi_list[k] = q
            ri_list[k] = Float64(r)
        end

        inst["Qi"] = isempty(Qi_list) ? nothing : Tuple(Qi_list)
        inst["qi"] = isempty(qi_list) ? nothing : Tuple(qi_list)
        inst["ri"] = isempty(ri_list) ? nothing : ri_list

        Pi_list = tuple_mats_to_vec(get(inst, "Pi", nothing))
        pi_list = tuple_vecs_to_vec(get(inst, "pi", nothing))
        si_list = scalar_list_to_vec(get(inst, "si", nothing))

        for k in eachindex(Pi_list)
            P = symmetrize(Pi_list[k] .+ scale_Qcon .* symrand(rng, n))
            p = pi_list[k] .+ scale_qcon .* randn(rng, n)

            if round_data
                P = round_mat(P; digits = round_digits)
                p = round_vec(p; digits = round_digits)
            end

            zero_drop_entries!(P, D)

            s = -0.5 * dot(xcert, P * xcert) - dot(p, xcert)
            if round_data
                s = round(Float64(s); digits = round_digits)
            end

            Pi_list[k] = P
            pi_list[k] = p
            si_list[k] = Float64(s)
        end

        inst["Pi"] = isempty(Pi_list) ? nothing : Tuple(Pi_list)
        inst["pi"] = isempty(pi_list) ? nothing : Tuple(pi_list)
        inst["si"] = isempty(si_list) ? nothing : si_list
    end

    @assert DropXTestInstances.check_aggregate_zero(inst) "Aggregate-zero violated after perturbation."

    return inst
end

# ------------------------------------------------------------
# Candidate printer
# ------------------------------------------------------------

function print_candidate_summary(inst, results)
    println("\n================ Complementarity candidate ================")
    println("id = ", get(inst, "id", "unknown"))
    println("n = ", inst["n"])
    println("rho = ", inst["rho"])
    println("drop_X_entries = ", get(inst, "drop_X_entries", nothing))
    println("drop_X_blocks  = ", get(inst, "drop_X_blocks", nothing))

    println("\nResults:")
    for var in COMP_VARIANTS
        r = results[var]
        println("  ", var,
                " | full=", fmt(r.obj_full),
                " drop=", fmt(r.obj_drop),
                " diff=", fmt(r.diff),
                " status=(", string(r.status_full), ",", string(r.status_drop), ")")
    end

    println("\nQ0 = ")
    show(stdout, "text/plain", inst["Q0"])
    println()

    println("\nq0 = ")
    show(stdout, "text/plain", inst["q0"])
    println()

    if inst["Pi"] !== nothing
        for (k, P) in enumerate(inst["Pi"])
            println("\nPi[$k] = ")
            show(stdout, "text/plain", P)
            println()
        end
        for (k, p) in enumerate(inst["pi"])
            println("\npi[$k] = ")
            show(stdout, "text/plain", p)
            println()
        end
        println("\nsi = ", inst["si"])
    end

    if inst["Qi"] !== nothing
        for (k, Q) in enumerate(inst["Qi"])
            println("\nQi[$k] = ")
            show(stdout, "text/plain", Q)
            println()
        end
        for (k, q) in enumerate(inst["qi"])
            println("\nqi[$k] = ")
            show(stdout, "text/plain", q)
            println()
        end
        println("\nri = ", inst["ri"])
    end

    println("============================================================\n")
end

# ------------------------------------------------------------
# Search driver
# ------------------------------------------------------------

function search_comp_dropX_gap(;
    seed::Int = 4321,
    trials_per_instance::Int = 2000,
    max_seconds::Float64 = 900.0,   # 15 minutes
    tol_gap::Float64 = 1e-6,
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
    perturb_constraints::Bool = true,
    scale_Q0::Float64 = 0.50,
    scale_q0::Float64 = 0.50,
    scale_Qcon::Float64 = 0.25,
    scale_qcon::Float64 = 0.25,
    round_data::Bool = true,
    round_digits::Int = 4,
)
    rng = MersenneTwister(seed)
    t0 = time()

    println("\n==============================================")
    println(" Search for complementarity Drop-X strict loss")
    println("==============================================")
    println("time limit = ", max_seconds, " seconds")
    println("tol_gap    = ", tol_gap)
    println("variants   = ", COMP_VARIANTS)
    println("==============================================\n")

    DropXTestInstances.check_dropX_instances()
    bases = DropXTestInstances.make_dropX_instances()

    best = nothing
    best_diff = -Inf
    total_trials = 0

    for base in bases
        base_id = get(base, "id", "unknown")

        println("\n------------------------------------------------------------")
        println("Base instance: ", base_id)
        println("------------------------------------------------------------")

        for t in 1:trials_per_instance
            elapsed = time() - t0
            if elapsed >= max_seconds
                println("\nTime limit reached.")
                if best !== nothing
                    println("Best observed complementarity diff:")
                    println("  base    = ", best.base_id)
                    println("  trial   = ", best.trial)
                    println("  variant = ", best.variant)
                    println("  diff    = ", fmt(best.diff))
                    println("  full    = ", fmt(best.result.obj_full))
                    println("  drop    = ", fmt(best.result.obj_drop))
                end
                return best
            end

            total_trials += 1

            cand = perturb_instance(
                base,
                rng;
                scale_Q0 = scale_Q0,
                scale_q0 = scale_q0,
                scale_Qcon = scale_Qcon,
                scale_qcon = scale_qcon,
                perturb_constraints = perturb_constraints,
                round_data = round_data,
                round_digits = round_digits,
            )

            cand["id"] = "$(base_id)_compstress$(t)_seed$(seed)_round$(round_digits)"

            results = solve_all_comp_pairs(
                cand;
                optimizer = optimizer,
                verbose = verbose,
            )

            for var in COMP_VARIANTS
                r = results[var]

                if r.diff !== missing
                    if r.diff > best_diff
                        best_diff = r.diff
                        best = (
                            cand = deepcopy(cand),
                            results = deepcopy(results),
                            result = r,
                            base_id = base_id,
                            trial = t,
                            variant = var,
                            diff = r.diff,
                        )
                    end

                    if r.diff > tol_gap
                        println("\nFOUND complementarity strict loss!")
                        println("  base    = ", base_id)
                        println("  trial   = ", t)
                        println("  variant = ", var)
                        println("  full    = ", fmt(r.obj_full))
                        println("  drop    = ", fmt(r.obj_drop))
                        println("  diff    = ", fmt(r.diff))

                        print_candidate_summary(cand, results)

                        ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
                        outfile = joinpath(ROOT, "found_comp_dropX_gap_$(ts).jls")
                        serialize(outfile, cand)
                        println("  saved candidate to: ", outfile)

                        return cand, results
                    end
                end
            end

            if t % 25 == 0
                println("  tried $t / $trials_per_instance for this base",
                        " | total=", total_trials,
                        " | best diff=", fmt(best_diff),
                        " | elapsed=", @sprintf("%.1f", time() - t0), "s")
            end
        end
    end

    println("\nNo complementarity strict loss found.")
    println("total trials = ", total_trials)

    if best !== nothing
        println("\nBest observed complementarity diff:")
        println("  base    = ", best.base_id)
        println("  trial   = ", best.trial)
        println("  variant = ", best.variant)
        println("  diff    = ", fmt(best.diff))
        println("  full    = ", fmt(best.result.obj_full))
        println("  drop    = ", fmt(best.result.obj_drop))

        ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        outfile = joinpath(ROOT, "best_comp_dropX_candidate_$(ts).jls")
        serialize(outfile, best.cand)
        println("  saved best candidate to: ", outfile)
    end

    return best
end

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

best_comp = search_comp_dropX_gap(
    seed = 4321,
    trials_per_instance = 2000,
    max_seconds = 900.0,  # 15 minutes
    tol_gap = 1e-6,
    perturb_constraints = true,
    scale_Q0 = 0.50,
    scale_q0 = 0.50,
    scale_Qcon = 0.25,
    scale_qcon = 0.25,
    round_data = true,
    round_digits = 4,
)