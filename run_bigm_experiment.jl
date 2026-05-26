###############################
# run_bigm_experiment.jl
#
# Bounds-sensitivity driver for the Big-M RLT family.
# Parallel to run_bounds_experiment.jl on the Comp side.
#
# For each instance in instances.json, solve:
#   - EXACT (reference optimum)  via Main.RLTBigM.build_and_solve(data; variant="EXACT", ...)
#   - RLT_E   x {M_scale = 1, 10, 100}
#   - RLT_EU  x {M_scale = 1, 10, 100}
#
# Total per instance: 1 + 2*3 = 7 models.
#
# Note: "nobounds" is NOT a meaningful configuration here, because
# the Big-M RLT formulation structurally depends on the bounds
# -M^- u <= x <= M^+ u. Removing them is not the same as in the
# Comp family.
#
# M_scale is applied at the DATA level (deepcopy + rescale M_minus,
# M_plus, M) before calling Main.RLTBigM.build_and_solve. This way
# we do not modify the RLTBigM module itself.
#
# Usage (in REPL, after the standard RLTBigM workflow is loaded):
#   julia> include("/home/s2719899/BAM/run_bigm_experiment.jl")
#   julia> run_bigm_experiment("/home/s2719899/BAM/instances.json")
#
# Outputs:
#   - bigm_sensitivity_results.csv
#   - sanity_checks_bigm.txt
#   - errors_bigm.log
###############################

using LinearAlgebra
using Printf

# Shared instance loader (must be in the same directory as this file).
include(joinpath(@__DIR__, "InstanceIO.jl"))

# Main.RLTBigM is expected to be loaded already.
const _bigm_build_and_solve = Main.RLTBigM.build_and_solve


# ============================================================
# Apply M_scale at the data level (does not touch RLTBigM module)
# ============================================================
function _scale_M(data::Dict{String,Any}, M_scale::Real)
    out = Dict{String,Any}()
    for (k, v) in data
        out[k] = v
    end
    # Preserve original arrays; copy and rescale only Big-M fields.
    for key in ("M_minus", "M_plus", "Mminus", "Mplus", "M")
        if haskey(out, key) && out[key] !== nothing
            out[key] = M_scale .* collect(out[key])
        end
    end
    return out
end


# ============================================================
# Solve one configuration. On exception, log to errors_bigm.log
# and return (:ERROR, nothing). On non-OPTIMAL termination, the
# tuple coming out of build_and_solve is propagated as-is.
# ============================================================
const _ERROR_LOG_BIGM = Ref{String}("errors_bigm.log")

function _solve_one_bigm(data, variant::String; M_scale::Real=1.0)
    try
        scaled = M_scale == 1.0 ? data : _scale_M(data, M_scale)
        res = _bigm_build_and_solve(
            scaled;
            variant = variant,
            optimizer = Main.Gurobi.Optimizer,
            verbose = false,
        )
        # res can be:
        #   (:OPTIMAL, obj, x, u)
        #   (:UNBOUNDED, nothing, nothing, nothing)
        #   (:INFEASIBLE, ...) etc.
        status = res[1]
        obj    = length(res) >= 2 ? res[2] : nothing
        return (status, obj)
    catch err
        open(_ERROR_LOG_BIGM[], "a") do io
            id = get(data, "id", "unknown")
            println(io, "=" ^ 70)
            println(io, "instance_id = $id")
            println(io, "variant     = $variant")
            println(io, "M_scale     = $M_scale")
            println(io, "-" ^ 70)
            showerror(io, err); println(io)
            for frame in stacktrace(catch_backtrace())
                println(io, "  ", frame)
            end
            println(io)
        end
        return (:ERROR, nothing)
    end
end


# ============================================================
# Main driver
# ============================================================
"""
    run_bigm_experiment(json_path; ...)

Solve 7 models (1 EXACT + 2 RLT variants x 3 M_scale values) for
each instance in `json_path`. Writes three output files:
CSV with all objective values, sanity_checks_bigm.txt, errors_bigm.log.
"""
function run_bigm_experiment(json_path::AbstractString;
                              out_csv::AbstractString    = "bigm_sensitivity_results.csv",
                              out_sanity::AbstractString = "sanity_checks_bigm.txt",
                              out_errors::AbstractString = "errors_bigm.log",
                              instance_indices           = nothing)

    _ERROR_LOG_BIGM[] = out_errors
    open(out_errors, "w") do io
        println(io, "Big-M bounds-sensitivity driver -- error log")
        println(io)
    end

    all_instances = load_instances(json_path)
    indices = instance_indices === nothing ? (1:length(all_instances)) : instance_indices
    println("Loaded ", length(all_instances), " instances; running ",
            length(indices), " of them.")

    rlt_variants = ["E", "EU"]
    configs      = [1.0, 10.0, 100.0]
    config_tags  = ["M1", "M10", "M100"]

    header = String["id", "n", "rho", "convexity", "base_tag",
                    "EXACT", "EXACT_status"]
    for var in rlt_variants
        for tag in config_tags
            push!(header, "$(var)_$(tag)")
            push!(header, "$(var)_$(tag)_status")
        end
    end

    rows         = Vector{Vector{Any}}()
    sanity_lines = String[]

    for (count, idx) in enumerate(indices)
        data      = all_instances[idx]
        id        = get(data, "id", "unknown")
        n         = data["n"]; rho = data["rho"]
        convexity = get(data, "convexity", "")
        base_tag  = get(data, "base_tag", "")

        println("\n[$count/$(length(indices))] $id  n=$n  rho=$rho  $convexity")

        (ex_status, ex_obj) = _solve_one_bigm(data, "EXACT")
        ex_str = ex_obj === nothing ? "NA" : @sprintf("%.6e", ex_obj)
        println("  EXACT            -> $ex_status  obj=$ex_str")

        row = Any[id, n, rho, convexity, base_tag,
                  ex_obj === nothing ? "NA" : ex_obj, String(ex_status)]

        per_inst = Dict{Tuple{String,String}, Union{Float64,Nothing}}()
        for var in rlt_variants
            for (ms, tag) in zip(configs, config_tags)
                (st, ob) = _solve_one_bigm(data, var; M_scale=ms)
                ob_str = ob === nothing ? "NA" : @sprintf("%.6e", ob)
                println("  $(rpad("RLT_"*var,16)) $(rpad(tag,9)) -> $st  obj=$ob_str")
                push!(row, ob === nothing ? "NA" : ob)
                push!(row, String(st))
                per_inst[(var, tag)] = ob
            end
        end
        push!(rows, row)

        # ---- per-instance sanity checks ----
        tol = 1e-6
        push!(sanity_lines, "==== $id ====")

        # 1. E <= EU (E weaker, since EU adds the F_U block)
        for tag in config_tags
            e_v  = per_inst[("E",  tag)]
            eu_v = per_inst[("EU", tag)]
            if e_v !== nothing && eu_v !== nothing
                ok = e_v <= eu_v + tol
                push!(sanity_lines,
                    @sprintf("  E<=EU                    @%-8s : %.6e <= %.6e  %s",
                            tag, e_v, eu_v, ok ? "OK" : "FAIL"))
            end
        end

        # 2. RLT <= EXACT
        if ex_obj !== nothing
            for var in rlt_variants, tag in config_tags
                rv = per_inst[(var, tag)]
                if rv !== nothing
                    ok = rv <= ex_obj + tol
                    push!(sanity_lines,
                        @sprintf("  %s_%s<=EXACT          : %.6e <= %.6e  %s",
                                var, tag, rv, ex_obj, ok ? "OK" : "FAIL"))
                end
            end
        end

        # 3. M-monotonicity: M100 <= M10 <= M1
        for var in rlt_variants
            vals = [per_inst[(var, t)] for t in ("M100","M10","M1")]
            if all(v -> v !== nothing, vals)
                ok_chain = true
                for k in 1:length(vals)-1
                    if !(vals[k] <= vals[k+1] + tol); ok_chain = false; break; end
                end
                push!(sanity_lines,
                    @sprintf("  M-monotonicity %-3s       : %.6e <= %.6e <= %.6e  %s",
                            var, vals[1], vals[2], vals[3],
                            ok_chain ? "OK" : "FAIL"))
            end
        end
        push!(sanity_lines, "")
    end

    open(out_csv, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join(map(string, r), ","))
        end
    end
    open(out_sanity, "w") do io
        for line in sanity_lines; println(io, line); end
    end

    println("\nWrote $out_csv")
    println("Wrote $out_sanity")
    println("Wrote $out_errors  (inspect this file if any ERROR statuses appeared)")
    return nothing
end
