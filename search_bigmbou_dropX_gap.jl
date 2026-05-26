# ============================================================
# search_bigmbou_dropX_gap.jl
# ============================================================

using Gurobi
using Random
using LinearAlgebra
using Printf
using Dates
using Serialization

# ------------------------------------------------------------
# Root
# ------------------------------------------------------------

const ROOT = @__DIR__

# ------------------------------------------------------------
# Modules
# Assumption: RLTBigM, RLTBigMDropX, RLTCompDropX,
# and DropXTestInstances have already been included/loaded.
# ------------------------------------------------------------

using .RLTBigM
using .RLTBigMDropX
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
    blocks = normalize_blocks(get(inst, "drop_X_blocks", Tuple{Vector{Int},Vector{Int}}[]))
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

# Sparse feasible certificate used only to keep perturbed constraints feasible.
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
# Solve helper: BigMBou = EU internally
# ------------------------------------------------------------

function solve_bigmbou_pair(inst;
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
)
    entries = normalize_entries(get(inst, "drop_X_entries", Tuple{Int,Int}[]))
    blocks  = normalize_blocks(get(inst, "drop_X_blocks", Tuple{Vector{Int},Vector{Int}}[]))

    try
        full = RLTBigMDropX.build_and_solve(
            inst;
            variant = "EU",  # RLT_BigMBou
            optimizer = optimizer,
            verbose = verbose,
            drop_X_entries = Tuple{Int,Int}[],
            drop_X_blocks = Tuple{Vector{Int},Vector{Int}}[],
            check_aggregate_zero = true,
        )

        drop = RLTBigMDropX.build_and_solve(
            inst;
            variant = "EU",  # RLT_BigMBou Drop-X
            optimizer = optimizer,
            verbose = verbose,
            drop_X_entries = entries,
            drop_X_blocks = blocks,
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

function solve_bigm_pair(inst;
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
)
    entries = normalize_entries(get(inst, "drop_X_entries", Tuple{Int,Int}[]))
    blocks  = normalize_blocks(get(inst, "drop_X_blocks", Tuple{Vector{Int},Vector{Int}}[]))

    try
        full = RLTBigMDropX.build_and_solve(
            inst;
            variant = "E",  # RLT_BigM
            optimizer = optimizer,
            verbose = verbose,
            drop_X_entries = Tuple{Int,Int}[],
            drop_X_blocks = Tuple{Vector{Int},Vector{Int}}[],
            check_aggregate_zero = true,
        )

        drop = RLTBigMDropX.build_and_solve(
            inst;
            variant = "E",  # RLT_BigM Drop-X
            optimizer = optimizer,
            verbose = verbose,
            drop_X_entries = entries,
            drop_X_blocks = blocks,
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

function solve_comp_check(inst;
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
    variant::String,
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
            drop_X_blocks = Tuple{Vector{Int},Vector{Int}}[],
            check_aggregate_zero = true,
        )

        drop = RLTCompDropX.build_and_solve(
            inst;
            variant = variant,
            optimizer = optimizer,
            verbose = verbose,
            drop_X_entries = entries,
            drop_X_blocks = blocks,
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

function solve_all_comp_checks(inst;
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
)
    results = Dict{String,Any}()

    for var in COMP_VARIANTS
        results[var] = solve_comp_check(
            inst;
            optimizer = optimizer,
            verbose = verbose,
            variant = var,
        )
    end

    return results
end

# ------------------------------------------------------------
# Optional exact BigM solve
# ------------------------------------------------------------

function solve_exact_bigM(inst;
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
)
    try
        res = RLTBigM.build_and_solve(
            inst;
            variant = "EXACT",
            optimizer = optimizer,
            verbose = verbose,
        )

        status = res[1]
        obj = status == :OPTIMAL ? Float64(res[2]) : missing

        return (; status, obj, error_msg = "")
    catch err
        return (; status = :ERROR, obj = missing, error_msg = sprint(showerror, err))
    end
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

    # --------------------------------------------------------
    # Perturb objective Q0, q0 while keeping aggregate-zero.
    # --------------------------------------------------------
    Q0 = Matrix{Float64}(inst["Q0"])
    q0 = Vector{Float64}(inst["q0"])

    Q0_new = symmetrize(Q0 .+ scale_Q0 .* symrand(rng, n))
    q0_new = q0 .+ scale_q0 .* randn(rng, n)

    if round_data
        Q0_new = round_mat(Q0_new; digits = round_digits)
        q0_new = round_vec(q0_new; digits = round_digits)
    end

    # Enforce aggregate-zero exactly after rounding.
    zero_drop_entries!(Q0_new, D)

    inst["Q0"] = Q0_new
    inst["q0"] = q0_new

    # --------------------------------------------------------
    # Optionally perturb quadratic inequalities/equalities,
    # while keeping the certificate point feasible.
    # --------------------------------------------------------
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

            # Enforce aggregate-zero exactly after rounding.
            zero_drop_entries!(Q, D)

            # Set r after rounding Q and q so that g(xcert) = -slack_qi.
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

            # Enforce aggregate-zero exactly after rounding.
            zero_drop_entries!(P, D)

            # Set s after rounding P and p so that h(xcert) = 0.
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

    # Sanity check: aggregate-zero must still hold exactly.
    @assert DropXTestInstances.check_aggregate_zero(inst) "Aggregate-zero violated after perturbation."

    return inst
end

# ------------------------------------------------------------
# Print candidate data
# ------------------------------------------------------------

function print_candidate_data(inst)
    println("\n================ Candidate data ================")
    println("id = ", get(inst, "id", "unknown"))
    println("n = ", inst["n"])
    println("rho = ", inst["rho"])
    println("drop_X_entries = ", get(inst, "drop_X_entries", nothing))
    println("drop_X_blocks  = ", get(inst, "drop_X_blocks", nothing))

    println("\nQ0 = ")
    show(stdout, "text/plain", inst["Q0"])
    println()

    println("\nq0 = ")
    show(stdout, "text/plain", inst["q0"])
    println()

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

    println("\nA = ")
    show(stdout, "text/plain", inst["A"])
    println()

    println("\nb = ")
    show(stdout, "text/plain", inst["b"])
    println()

    if inst["H"] !== nothing
        println("\nH = ")
        show(stdout, "text/plain", inst["H"])
        println()

        println("\nh = ")
        show(stdout, "text/plain", inst["h"])
        println()
    end

    println("\nell = ", inst["ell"])
    println("u   = ", inst["u"])
    println("M   = ", inst["M"])
    println("M_minus = ", inst["M_minus"])
    println("M_plus  = ", inst["M_plus"])
    println("================================================\n")
end

# ------------------------------------------------------------
# Search driver
# ------------------------------------------------------------

function search_bigmbou_gap(;
    seed::Int = 1234,
    trials_per_instance::Int = 200,
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
    check_exact::Bool = true,
)
    rng = MersenneTwister(seed)

    println("\n==============================================")
    println(" Search for RLT-BigMBou Drop-X strict loss")
    println("==============================================\n")

    DropXTestInstances.check_dropX_instances()
    bases = DropXTestInstances.make_dropX_instances()

    best = nothing
    best_absdiff = -Inf

    for base in bases
        base_id = get(base, "id", "unknown")

        println("\n------------------------------------------------------------")
        println("Base instance: ", base_id)
        println("------------------------------------------------------------")

        for t in 1:trials_per_instance
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

            cand["id"] = "$(base_id)_pert$(t)_seed$(seed)_round$(round_digits)"

            res = solve_bigmbou_pair(
                cand;
                optimizer = optimizer,
                verbose = verbose,
            )

            if res.diff !== missing
                if abs(res.diff) > best_absdiff
                    best_absdiff = abs(res.diff)
                    best = (cand = deepcopy(cand), res = res, base_id = base_id, trial = t)
                end

                if res.diff > tol_gap
                    println("\nFOUND BigMBou strict loss!")
                    println("  base  = ", base_id)
                    println("  trial = ", t)
                    println("  full  = ", fmt(res.obj_full))
                    println("  drop  = ", fmt(res.obj_drop))
                    println("  diff  = ", fmt(res.diff))

                    # Also check RLT-BigM on the same candidate.
                    res_bigm = solve_bigm_pair(cand; optimizer = optimizer, verbose = verbose)

                    println("  RLT_BigM diff = ", fmt(res_bigm.diff),
                            "   full=", fmt(res_bigm.obj_full),
                            " drop=", fmt(res_bigm.obj_drop))

                    # Check all complementarity variants on the same candidate.
                    res_comps = solve_all_comp_checks(cand; optimizer = optimizer, verbose = verbose)

                    for var in COMP_VARIANTS
                        rc = res_comps[var]
                        println("  ", var, " diff = ", fmt(rc.diff),
                                "   full=", fmt(rc.obj_full),
                                " drop=", fmt(rc.obj_drop))
                    end

                    if check_exact
                        exact = solve_exact_bigM(cand; optimizer = optimizer, verbose = verbose)
                        println("  Exact BigM obj = ", fmt(exact.obj),
                                "   status=", string(exact.status))
                    end

                    print_candidate_data(cand)

                    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
                    outfile = joinpath(ROOT, "found_bigmbou_dropX_gap_$(ts).jls")
                    serialize(outfile, cand)
                    println("  saved candidate to: ", outfile)

                    return cand, res
                end
            end

            if t % 25 == 0
                println("  tried $t / $trials_per_instance",
                        "   best abs diff so far = ", fmt(best_absdiff))
            end
        end
    end

    println("\nNo strict BigMBou loss found.")
    if best !== nothing
        println("Best candidate:")
        println("  base  = ", best.base_id)
        println("  trial = ", best.trial)
        println("  full  = ", fmt(best.res.obj_full))
        println("  drop  = ", fmt(best.res.obj_drop))
        println("  diff  = ", fmt(best.res.diff))

        res_bigm = solve_bigm_pair(best.cand; optimizer = optimizer, verbose = verbose)
        println("  RLT_BigM diff = ", fmt(res_bigm.diff),
                "   full=", fmt(res_bigm.obj_full),
                " drop=", fmt(res_bigm.obj_drop))

        res_comps = solve_all_comp_checks(best.cand; optimizer = optimizer, verbose = verbose)

        for var in COMP_VARIANTS
            rc = res_comps[var]
            println("  ", var, " diff = ", fmt(rc.diff),
                    "   full=", fmt(rc.obj_full),
                    " drop=", fmt(rc.obj_drop))
        end

        print_candidate_data(best.cand)

        ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        outfile = joinpath(ROOT, "best_bigmbou_dropX_candidate_$(ts).jls")
        serialize(outfile, best.cand)
        println("  saved best candidate to: ", outfile)
    end

    return best
end

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

best = search_bigmbou_gap(
    trials_per_instance = 200,
    tol_gap = 1e-6,
    perturb_constraints = true,
    scale_Q0 = 0.50,
    scale_q0 = 0.50,
    scale_Qcon = 0.25,
    scale_qcon = 0.25,
    round_data = true,
    round_digits = 4,
    check_exact = true,
)