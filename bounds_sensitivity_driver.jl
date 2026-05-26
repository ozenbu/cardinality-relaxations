# ============================================================
# bounds_sensitivity_driver.jl
#
# Driver for the bounds-sensitivity experiment.
#
# For each instance in instances.json, solves:
#   - EXACT_CompBin (reference optimum)
#   - RLT_CompBin    × {nobounds, M_scale = 1, 10, 100}
#   - RLT_CompBinBou × {nobounds, M_scale = 1, 10, 100}
#   - RLT_CompBou    × {nobounds, M_scale = 1, 10, 100}
#
# Total per instance: 1 + 3*4 = 13 models.
#
# Output:
#   - bounds_sensitivity_results.csv  (wide table)
#   - sanity_checks.txt               (proposition + monotonicity tests)
# ============================================================

using JSON
using JuMP
using MathOptInterface
const MOI = MathOptInterface
using Gurobi
using LinearAlgebra
using Printf



# ============================================================
# JSON loader -- convert JSON arrays to Julia matrices/vectors
# ============================================================
"""
    to_matrix(x)

JSON 2D arrays come in as Vector{Vector{Any}}. Convert to Matrix{Float64}.
JSON 1D arrays come in as Vector{Any}. Convert to Vector{Float64}.
"""
to_matrix(x::Nothing) = nothing
function to_matrix(x)
    if x isa AbstractVector
        if isempty(x)
            return Float64[]
        elseif x[1] isa AbstractVector
            # 2D
            return Float64.(reduce(hcat, x)')  # rows = inner vectors
        else
            # 1D
            return Float64.(collect(x))
        end
    end
    return x
end

"""
    instance_from_json(rec::Dict)

Turn one JSON record into a Dict ready for `prepare_instance`.

Notes:
- The driver passes `M_minus` and `M_plus` (vectors). prepare_instance
  converts them to Diagonal matrices.
- Optional fields (Qi, qi, ri, Pi, pi, si, H, h, A, b) may be `nothing`
  in the JSON; we pass them through.
- For lists of matrices (Qi, Pi) -> convert each inner element.
- For lists of vectors (qi, pi) -> convert each inner element.
- For lists of scalars (ri, si) -> Float64 cast each element.
"""
function instance_from_json(rec::Dict)
    d = Dict{String,Any}()

    d["n"]   = Int(rec["n"])
    d["rho"] = rec["rho"]

    d["Q0"] = to_matrix(get(rec, "Q0", nothing))
    d["q0"] = to_matrix(get(rec, "q0", nothing))

    # Quadratic inequality lists
    function convert_list_of_matrices(x)
        x === nothing && return nothing
        return [to_matrix(elt) for elt in x]
    end
    function convert_list_of_vectors(x)
        x === nothing && return nothing
        return [to_matrix(elt) for elt in x]
    end
    function convert_list_of_scalars(x)
        x === nothing && return nothing
        return [Float64(elt) for elt in x]
    end

    d["Qi"] = convert_list_of_matrices(get(rec, "Qi", nothing))
    d["qi"] = convert_list_of_vectors(get(rec, "qi", nothing))
    d["ri"] = convert_list_of_scalars(get(rec, "ri", nothing))

    d["Pi"] = convert_list_of_matrices(get(rec, "Pi", nothing))
    d["pi"] = convert_list_of_vectors(get(rec, "pi", nothing))
    d["si"] = convert_list_of_scalars(get(rec, "si", nothing))

    d["A"] = to_matrix(get(rec, "A", nothing))
    d["b"] = to_matrix(get(rec, "b", nothing))
    d["H"] = to_matrix(get(rec, "H", nothing))
    d["h"] = to_matrix(get(rec, "h", nothing))

    # Big-M: prepare_instance accepts M_minus/M_plus (vectors)
    d["M_minus"] = to_matrix(get(rec, "M_minus", nothing))
    d["M_plus"]  = to_matrix(get(rec, "M_plus",  nothing))

    # carry metadata for the result table
    d["id"]         = get(rec, "id", "unknown")
    d["convexity"]  = get(rec, "convexity", "")
    d["base_tag"]   = get(rec, "base_tag", "")
    d["bigM_scale"] = get(rec, "bigM_scale", 1.0)

    return d
end

# ============================================================
# Solve one configuration safely (returns (status, obj_or_nothing))
# ============================================================
function solve_one(data, variant::String;
                   bounds_mode::Symbol=:full,
                   M_scale::Real=1.0)
    try
        if startswith(variant, "EXACT")
            res = RLTCompBoundsSens.build_and_solve(
                data; variant=variant,
                bounds_mode=:full, M_scale=M_scale,
                optimizer=Gurobi.Optimizer, verbose=false
            )
            status = res[1]; obj = res[2]
            return (status, obj)
        else
            res = RLTCompBoundsSens.build_and_solve(
                data; variant=variant,
                bounds_mode=bounds_mode, M_scale=M_scale,
                optimizer=Gurobi.Optimizer, verbose=false
            )
            status = res[1]; obj = res[2]
            return (status, obj)
        end
    catch err
        return (:ERROR, nothing)
    end
end

# ============================================================
# Main driver
# ============================================================
function run_experiment(json_path::String;
                        out_csv::String="bounds_sensitivity_results.csv",
                        out_sanity::String="sanity_checks.txt",
                        instance_indices=nothing)

    raw = JSON.parsefile(json_path)
    @assert raw isa AbstractVector "Expected a JSON array of instances."

    if instance_indices === nothing
        indices = 1:length(raw)
    else
        indices = instance_indices
    end

    println("Loaded ", length(raw), " instances; running ",
            length(indices), " of them.")

    # Column layout
    rlt_variants = ["RLT_CompBin", "RLT_CompBinBou", "RLT_CompBou"]
    configs      = [(:none, 1.0), (:full, 1.0), (:full, 10.0), (:full, 100.0)]
    config_tags  = ["nobounds", "M1", "M10", "M100"]

    # Build header
    header = String["id", "n", "rho", "convexity", "base_tag",
                    "EXACT", "EXACT_status"]
    for var in rlt_variants
        short = replace(var, "RLT_" => "")  # CompBin / CompBinBou / CompBou
        for tag in config_tags
            push!(header, "$(short)_$(tag)")
            push!(header, "$(short)_$(tag)_status")
        end
    end

    rows = Vector{Vector{Any}}()
    sanity_lines = String[]

    for (count, idx) in enumerate(indices)
        rec  = raw[idx]
        data = instance_from_json(rec)

        id        = data["id"]
        n         = data["n"]
        rho       = data["rho"]
        convexity = data["convexity"]
        base_tag  = data["base_tag"]

        println("\n[", count, "/", length(indices), "] ", id,
                "   n=", n, "  rho=", rho, "  ", convexity)

        # EXACT
        (ex_status, ex_obj) = solve_one(data, "EXACT_CompBin")
        ex_str = ex_obj === nothing ? "NA" : @sprintf("%.6e", ex_obj)
        println("  EXACT_CompBin    -> ", ex_status, "  obj=", ex_str)

        row = Any[id, n, rho, convexity, base_tag,
                  ex_obj === nothing ? "NA" : ex_obj,
                  String(ex_status)]

        # Per-variant per-config table for this instance (for sanity checks)
        per_inst = Dict{Tuple{String,String}, Union{Float64,Nothing}}()

        for var in rlt_variants
            short = replace(var, "RLT_" => "")
            for (cfg, tag) in zip(configs, config_tags)
                (bm, ms) = cfg
                (st, ob) = solve_one(data, var; bounds_mode=bm, M_scale=ms)
                ob_str = ob === nothing ? "NA" : @sprintf("%.6e", ob)
                println("  ", rpad(var, 16), " ", rpad(tag, 9),
                        " -> ", st, "  obj=", ob_str)
                push!(row, ob === nothing ? "NA" : ob)
                push!(row, String(st))
                per_inst[(short, tag)] = ob
            end
        end

        push!(rows, row)

        # --- per-instance sanity checks ---
        tol = 1e-6
        push!(sanity_lines, "==== $(id) ====")

        # 1. CompBinBou ≈ CompBou (proposition)
        for tag in config_tags
            a = per_inst[("CompBinBou", tag)]
            b = per_inst[("CompBou", tag)]
            if a !== nothing && b !== nothing
                d = abs(a - b)
                ok = d <= tol * max(1.0, abs(a), abs(b))
                push!(sanity_lines,
                    @sprintf("  Proposition (CompBinBou=CompBou) @%-8s : diff=%.3e  %s",
                            tag, d, ok ? "OK" : "FAIL"))
            end
        end

        # 2. CompBin <= CompBinBou (CompBin weaker)
        for tag in config_tags
            cbin = per_inst[("CompBin",     tag)]
            cbb  = per_inst[("CompBinBou",  tag)]
            if cbin !== nothing && cbb !== nothing
                ok = cbin <= cbb + tol
                push!(sanity_lines,
                    @sprintf("  CompBin<=CompBinBou      @%-8s : %.6e <= %.6e  %s",
                            tag, cbin, cbb, ok ? "OK" : "FAIL"))
            end
        end

        # 3. RLT <= EXACT
        if ex_obj !== nothing
            for var_short in ("CompBin", "CompBinBou", "CompBou"), tag in config_tags
                rv = per_inst[(var_short, tag)]
                if rv !== nothing
                    ok = rv <= ex_obj + tol
                    push!(sanity_lines,
                        @sprintf("  %s_%s<=EXACT          : %.6e <= %.6e  %s",
                                var_short, tag, rv, ex_obj, ok ? "OK" : "FAIL"))
                end
            end
        end

        # 4. M-monotonicity: nobounds <= M100 <= M10 <= M1
        for var_short in ("CompBin", "CompBinBou", "CompBou")
            vals = [per_inst[(var_short, t)] for t in ("nobounds","M100","M10","M1")]
            if all(v -> v !== nothing, vals)
                labels = ["nobounds","M100","M10","M1"]
                ok_chain = true
                for k in 1:length(vals)-1
                    if !(vals[k] <= vals[k+1] + tol)
                        ok_chain = false; break
                    end
                end
                push!(sanity_lines,
                    @sprintf("  M-monotonicity %-11s : %.6e <= %.6e <= %.6e <= %.6e  %s",
                            var_short, vals[1], vals[2], vals[3], vals[4],
                            ok_chain ? "OK" : "FAIL"))
            end
        end
        push!(sanity_lines, "")
    end

    # ---- write CSV ----
    open(out_csv, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join(map(string, r), ","))
        end
    end

    # ---- write sanity checks ----
    open(out_sanity, "w") do io
        for line in sanity_lines
            println(io, line)
        end
    end

    println("\nWrote ", out_csv)
    println("Wrote ", out_sanity)
    return nothing
end

# ============================================================
# Entry point
# ============================================================
if isinteractive()
    json_path = normpath(joinpath(@__DIR__, "instances.json"))
    run_experiment(json_path)
end