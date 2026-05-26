# ============================================================
# run_dropX_tests.jl
# ============================================================

using Gurobi
using Printf
using Dates


using .RLTBigM
using .RLTBigMDropX
using .RLTCompDropX
using .DropXTestInstances

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

function normalize_entries(v)
    v === nothing && return Tuple{Int,Int}[]

    entries = Tuple{Int,Int}[]
    for (i, j) in v
        push!(entries, (Int(i), Int(j)))
    end
    return entries
end

function normalize_blocks(v)
    v === nothing && return Tuple{Vector{Int},Vector{Int}}[]

    blocks = Tuple{Vector{Int},Vector{Int}}[]
    for (I, J) in v
        push!(blocks, (Int.(collect(I)), Int.(collect(J))))
    end
    return blocks
end

function fmt(x)
    if x === missing
        return "missing"
    elseif x isa Real
        return @sprintf("%.8g", x)
    else
        return string(x)
    end
end

function csv_escape(x)
    s = string(x)
    return "\"" * replace(s, "\"" => "\"\"") * "\""
end

# ------------------------------------------------------------
# Variant label mapping
# ------------------------------------------------------------

function bigm_internal_variant(label::String)
    if label in ["RLT_BigM", "BigM", "E"]
        return "E"
    elseif label in ["RLT_BigMBou", "BigMBou", "EU"]
        return "EU"
    else
        error("Unknown BigM label: $label")
    end
end

function bigm_display_label(label::String)
    if label in ["E", "BigM", "RLT_BigM"]
        return "RLT_BigM"
    elseif label in ["EU", "BigMBou", "RLT_BigMBou"]
        return "RLT_BigMBou"
    else
        return label
    end
end

# ------------------------------------------------------------
# Solve helpers
# ------------------------------------------------------------

function solve_exact_bigM_safe(inst;
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

function solve_obj_safe(model_type::Symbol, inst;
    variant::String,
    drop::Bool,
    drop_X_entries = Tuple{Int,Int}[],
    drop_X_blocks = Tuple{Vector{Int},Vector{Int}}[],
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
)
    try
        if model_type == :BigM
            internal_variant = bigm_internal_variant(variant)

            res = RLTBigMDropX.build_and_solve(
                inst;
                variant = internal_variant,
                optimizer = optimizer,
                verbose = verbose,
                drop_X_entries = drop ? drop_X_entries : Tuple{Int,Int}[],
                drop_X_blocks  = drop ? drop_X_blocks  : Tuple{Vector{Int},Vector{Int}}[],
                check_aggregate_zero = true,
            )

        elseif model_type == :Comp
            res = RLTCompDropX.build_and_solve(
                inst;
                variant = variant,
                optimizer = optimizer,
                verbose = verbose,
                drop_X_entries = drop ? drop_X_entries : Tuple{Int,Int}[],
                drop_X_blocks  = drop ? drop_X_blocks  : Tuple{Vector{Int},Vector{Int}}[],
                check_aggregate_zero = true,
            )

        else
            error("Unknown model_type = $model_type")
        end

        status = res[1]
        obj = status == :OPTIMAL ? Float64(res[2]) : missing

        return (; status, obj, error_msg = "")
    catch err
        return (; status = :ERROR, obj = missing, error_msg = sprint(showerror, err))
    end
end

# ------------------------------------------------------------
# CSV writer
# ------------------------------------------------------------

function write_results_csv(rows, filename)
    headers = [
        "instance",
        "model_type",
        "variant",
        "drop_entries",
        "drop_blocks",
        "status_full",
        "status_drop",
        "obj_full",
        "obj_drop",
        "diff_full_minus_drop",
        "abs_diff",
        "error_full",
        "error_drop",
    ]

    open(filename, "w") do io
        println(io, join(headers, ","))

        for r in rows
            vals = [
                r.instance,
                r.model_type,
                r.variant,
                r.drop_entries,
                r.drop_blocks,
                r.status_full,
                r.status_drop,
                r.obj_full,
                r.obj_drop,
                r.diff_full_minus_drop,
                r.abs_diff,
                r.error_full,
                r.error_drop,
            ]
            println(io, join(csv_escape.(vals), ","))
        end
    end
end

# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

function run_dropX_tests(;
    bigm_variants = ["RLT_BigM", "RLT_BigMBou"],
    comp_variants = ["RLT_CompBin", "RLT_CompBinBou", "RLT_CompBou"],
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
    include_exact::Bool = true,
)
    println("\n==============================================")
    println(" Drop-X computational test")
    println("==============================================\n")

    println("Checking purpose-built Drop-X instances...")
    DropXTestInstances.check_dropX_instances()
    println("Instance checks passed.\n")

    instances = DropXTestInstances.make_dropX_instances()

    rows = NamedTuple[]

    for inst in instances
        id = get(inst, "id", "unknown_instance")

        drop_entries = normalize_entries(get(inst, "drop_X_entries", Tuple{Int,Int}[]))
        drop_blocks  = normalize_blocks(get(inst, "drop_X_blocks", Tuple{Vector{Int},Vector{Int}}[]))

        println("\n------------------------------------------------------------")
        println("Instance: ", id)
        println("drop_X_entries = ", drop_entries)
        println("drop_X_blocks  = ", drop_blocks)
        println("------------------------------------------------------------")

        # ---------------- Exact BigM ----------------
        if include_exact
            exact = solve_exact_bigM_safe(
                inst;
                optimizer = optimizer,
                verbose = verbose,
            )

            @printf("Exact BigM      | obj=%-10s status=%s\n",
                    fmt(exact.obj), string(exact.status))

            if exact.status == :ERROR
                println("  exact error: ", exact.error_msg)
            end

            push!(rows, (;
                instance = id,
                model_type = "Exact",
                variant = "EXACT_BigM",
                drop_entries = repr(drop_entries),
                drop_blocks = repr(drop_blocks),
                status_full = string(exact.status),
                status_drop = "",
                obj_full = fmt(exact.obj),
                obj_drop = "",
                diff_full_minus_drop = "",
                abs_diff = "",
                error_full = exact.error_msg,
                error_drop = "",
            ))
        end

        # ---------------- BigM RLT variants ----------------
        for var in bigm_variants
            display_var = bigm_display_label(var)

            full = solve_obj_safe(
                :BigM, inst;
                variant = var,
                drop = false,
                drop_X_entries = drop_entries,
                drop_X_blocks = drop_blocks,
                optimizer = optimizer,
                verbose = verbose,
            )

            drp = solve_obj_safe(
                :BigM, inst;
                variant = var,
                drop = true,
                drop_X_entries = drop_entries,
                drop_X_blocks = drop_blocks,
                optimizer = optimizer,
                verbose = verbose,
            )

            diff = (full.obj === missing || drp.obj === missing) ? missing : full.obj - drp.obj
            absdiff = diff === missing ? missing : abs(diff)

            @printf("BigM %-12s | full=%-10s drop=%-10s diff(full-drop)=%s | status=(%s,%s)\n",
                    display_var, fmt(full.obj), fmt(drp.obj), fmt(diff),
                    string(full.status), string(drp.status))

            if full.status == :ERROR
                println("  full error: ", full.error_msg)
            end
            if drp.status == :ERROR
                println("  drop error: ", drp.error_msg)
            end

            push!(rows, (;
                instance = id,
                model_type = "BigM",
                variant = display_var,
                drop_entries = repr(drop_entries),
                drop_blocks = repr(drop_blocks),
                status_full = string(full.status),
                status_drop = string(drp.status),
                obj_full = fmt(full.obj),
                obj_drop = fmt(drp.obj),
                diff_full_minus_drop = fmt(diff),
                abs_diff = fmt(absdiff),
                error_full = full.error_msg,
                error_drop = drp.error_msg,
            ))
        end

        # ---------------- Complementarity variants ----------------
        for var in comp_variants
            full = solve_obj_safe(
                :Comp, inst;
                variant = var,
                drop = false,
                drop_X_entries = drop_entries,
                drop_X_blocks = drop_blocks,
                optimizer = optimizer,
                verbose = verbose,
            )

            drp = solve_obj_safe(
                :Comp, inst;
                variant = var,
                drop = true,
                drop_X_entries = drop_entries,
                drop_X_blocks = drop_blocks,
                optimizer = optimizer,
                verbose = verbose,
            )

            diff = (full.obj === missing || drp.obj === missing) ? missing : full.obj - drp.obj
            absdiff = diff === missing ? missing : abs(diff)

            @printf("Comp %-14s | full=%-10s drop=%-10s absdiff=%s | status=(%s,%s)\n",
                    var, fmt(full.obj), fmt(drp.obj), fmt(absdiff),
                    string(full.status), string(drp.status))

            if full.status == :ERROR
                println("  full error: ", full.error_msg)
            end
            if drp.status == :ERROR
                println("  drop error: ", drp.error_msg)
            end

            push!(rows, (;
                instance = id,
                model_type = "Comp",
                variant = var,
                drop_entries = repr(drop_entries),
                drop_blocks = repr(drop_blocks),
                status_full = string(full.status),
                status_drop = string(drp.status),
                obj_full = fmt(full.obj),
                obj_drop = fmt(drp.obj),
                diff_full_minus_drop = fmt(diff),
                abs_diff = fmt(absdiff),
                error_full = full.error_msg,
                error_drop = drp.error_msg,
            ))
        end
    end

    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    outfile = joinpath(ROOT, "dropX_results_$ts.csv")
    write_results_csv(rows, outfile)

    println("\n==============================================")
    println("Done.")
    println("Results saved to:")
    println(outfile)
    println("==============================================\n")

    return rows
end

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

rows = run_dropX_tests()