###############################
# Pairwise relation report (τ = 1e-3 by default)
# Reads: level_table.csv OR level_table.json
# Prints, for each level, pairwise dominance/equality/mixed relations on the given instance set.
###############################

using CSV
using DataFrames
using JSON

# -----------------------------
# Config: statuses
# -----------------------------
const GOOD_STATUS      = Set(["OPTIMAL", "ALMOST_OPTIMAL", "LOCALLY_SOLVED"])
const UNBOUNDED_STATUS = Set(["DUAL_INFEASIBLE", "UNBOUNDED"])

# -----------------------------
# Read level table (CSV or JSON)
# -----------------------------
function read_level_table(path::AbstractString)::DataFrame
    if endswith(lowercase(path), ".csv")
        return CSV.read(path, DataFrame)
    elseif endswith(lowercase(path), ".json")
        raw = JSON.parsefile(path)
        # expected schema: Vector of Dicts, each has a "results" vector
        rows = Vector{Dict{String,Any}}()
        for inst in raw
            inst_id    = get(inst, "inst_id", missing)
            inst_label = get(inst, "inst_label", missing)
            for r in inst["results"]
                push!(rows, Dict(
                    "inst_id"    => inst_id,
                    "inst_label" => inst_label,
                    "level"      => r["level"],
                    "model"      => r["model"],
                    "status"     => r["status"],
                    "obj"        => r["obj"],
                    "time_sec"   => r["time_sec"],
                ))
            end
        end
        return DataFrame(rows)
    else
        error("Unsupported file extension for: $path (use .csv or .json)")
    end
end

# -----------------------------
# Safe helpers
# -----------------------------
_status_str(x) = x === missing ? "" : String(x)
_obj_float(x)  = (x === missing || x === nothing) ? NaN : Float64(x)

"""
compare_pair(subdf, X, Y; tau)

Returns:
  (relation::String, nX::Int, nY::Int, neq::Int, nmiss::Int)

Where:
  nX   = #instances where X is tighter than Y by at least tau
  nY   = #instances where Y is tighter than X by at least tau
  neq  = #instances where |zX - zY| <= tau
  nmiss= #instances where comparison is not decidable (unknown status / missing)
Notes:
  - Minimization: tighter = larger objective value.
  - If one side is UNBOUNDED and the other is GOOD, the GOOD one is counted as tighter.
"""
function compare_pair(subdf::DataFrame, X::String, Y::String; tau::Float64=1e-3)
    # map (inst_label, model) -> (status, obj)
    inst_map = Dict{String, Dict{String, Tuple{String, Float64}}}()
    for row in eachrow(subdf)
        inst = String(row.inst_label)
        mdl  = String(row.model)
        st   = _status_str(row.status)
        obj  = _obj_float(row.obj)
        get!(inst_map, inst, Dict{String, Tuple{String, Float64}}())[mdl] = (st, obj)
    end

    nX = 0; nY = 0; neq = 0; nmiss = 0

    for (inst, mdict) in inst_map
        if !(haskey(mdict, X) && haskey(mdict, Y))
            nmiss += 1
            continue
        end
        stX, zX = mdict[X]
        stY, zY = mdict[Y]

        if (stX in GOOD_STATUS) && (stY in GOOD_STATUS) && isfinite(zX) && isfinite(zY)
            d = zX - zY
            if abs(d) <= tau
                neq += 1
            elseif d > tau
                nX += 1
            else
                nY += 1
            end
        elseif (stX in UNBOUNDED_STATUS) && (stY in GOOD_STATUS)
            nY += 1
        elseif (stY in UNBOUNDED_STATUS) && (stX in GOOD_STATUS)
            nX += 1
        elseif (stX in UNBOUNDED_STATUS) && (stY in UNBOUNDED_STATUS)
            # both unbounded: treat as equal for pairwise purposes
            neq += 1
        else
            # SLOW_PROGRESS / OTHER / missing status etc.
            nmiss += 1
        end
    end

    # classify
    relation =
        (nX == 0 && nY == 0) ? "Always equal" :
        (nX > 0  && nY == 0) ? "$X dominates $Y" :
        (nY > 0  && nX == 0) ? "$Y dominates $X" :
                               "Incomparable / mixed"

    return relation, nX, nY, neq, nmiss
end

# -----------------------------
# Reporting
# -----------------------------
function report_pairwise_relations(path::AbstractString; tau::Float64=1e-3)
    df = read_level_table(path)

    # ensure string columns exist
    df.level = String.(df.level)
    df.model = String.(df.model)

    levels = sort(unique(df.level))

    println("Pairwise relations on table: $path")
    println("Tolerance τ = $(tau)")
    println("Interpretation (minimization): larger z* = tighter bound.\n")

    for lvl in levels
        sub = df[df.level .== lvl, :]
        models = sort(unique(sub.model))

        println("============================================================")
        println("LEVEL: $lvl")
        println("models: ", join(models, ", "))
        println("------------------------------------------------------------")

        # all unordered pairs (X < Y)
        for i in 1:length(models)-1
            for j in i+1:length(models)
                X = models[i]
                Y = models[j]
                rel, nX, nY, neq, nmiss = compare_pair(sub, X, Y; tau=tau)

                # requested format
                println("$X vs $Y: $rel (#X>τ=$nX, #Y>τ=$nY, #eq=$neq, #miss=$nmiss)")
            end
        end
        println()
    end

    return nothing
end

# -----------------------------
# Example:
#   report_pairwise_relations("level_table.csv"; tau=1e-3)
report_pairwise_relations("level_table.json"; tau=1e-3)
# -----------------------------
