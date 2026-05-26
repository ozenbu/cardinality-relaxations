###############################
# combine_results.jl
#
# Combine the Comp-side and BigM-side results into a single master
# CSV for side-by-side comparison.
#
# Inputs:
#   - bounds_sensitivity_results.csv   (Comp family: 3 variants x 4 configs)
#   - bigm_sensitivity_results.csv     (BigM family: 2 variants x 3 configs)
#
# Output:
#   - combined_results.csv
#
# Layout (per instance, one row):
#   id, n, rho, convexity, base_tag, EXACT,
#   then Comp columns (CompBin / CompBinBou / CompBou) x (nobounds, M1, M10, M100),
#   then BigM columns (E / EU) x (M1, M10, M100),
#   then gap-to-EXACT for each RLT cell.
#
# Usage:
#   julia> include("/home/s2719899/BAM/combine_results.jl")
#   julia> combine_results()
###############################

using Printf

# ------------------------------------------------------------
# Minimal CSV reader (no DataFrames dependency)
# ------------------------------------------------------------
function _read_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("$path is empty")
    header = strip.(split(lines[1], ","))
    rows = Vector{Dict{String,String}}()
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        cells = strip.(split(ln, ","))
        @assert length(cells) == length(header) "Row length mismatch in $path: got $(length(cells)) cells, header has $(length(header))"
        d = Dict{String,String}()
        for (h, c) in zip(header, cells)
            d[h] = c
        end
        push!(rows, d)
    end
    return header, rows
end

# ------------------------------------------------------------
# Parse a CSV cell into Union{Float64,Nothing}
# ------------------------------------------------------------
function _to_float(s::AbstractString)
    s = strip(s)
    (s == "NA" || s == "" || s == "nothing") && return nothing
    try
        return parse(Float64, s)
    catch
        return nothing
    end
end

# ------------------------------------------------------------
# Format Float64 or nothing as a CSV cell
# ------------------------------------------------------------
_fmt(::Nothing) = "NA"
_fmt(x::Float64) = isnan(x) ? "NA" : @sprintf("%.6e", x)
_fmt(x::Real)    = _fmt(Float64(x))

# ------------------------------------------------------------
# Compute gap (RLT - EXACT) for minimization sense.
# Positive means EXACT > RLT (RLT is a lower bound).
# Relative gap = (EXACT - RLT) / max(1, |EXACT|).
# ------------------------------------------------------------
function _relgap(rlt::Union{Float64,Nothing}, ex::Union{Float64,Nothing})
    (rlt === nothing || ex === nothing) && return nothing
    return (ex - rlt) / max(1.0, abs(ex))
end

# ------------------------------------------------------------
# Main combiner
# ------------------------------------------------------------
function combine_results(;
    comp_csv::AbstractString = "bounds_sensitivity_results.csv",
    bigm_csv::AbstractString = "bigm_sensitivity_results.csv",
    out_csv::AbstractString  = "combined_results.csv",
)
    println("Reading $comp_csv ...")
    comp_header, comp_rows = _read_csv(comp_csv)
    println("  -> $(length(comp_rows)) instances")

    println("Reading $bigm_csv ...")
    bigm_header, bigm_rows = _read_csv(bigm_csv)
    println("  -> $(length(bigm_rows)) instances")

    # Index BigM rows by instance id
    bigm_by_id = Dict{String, Dict{String,String}}()
    for r in bigm_rows
        bigm_by_id[r["id"]] = r
    end

    # Column layout for the master CSV
    comp_cells = [
        ("CompBin",    "nobounds"), ("CompBin",    "M1"), ("CompBin",    "M10"), ("CompBin",    "M100"),
        ("CompBinBou", "nobounds"), ("CompBinBou", "M1"), ("CompBinBou", "M10"), ("CompBinBou", "M100"),
        ("CompBou",    "nobounds"), ("CompBou",    "M1"), ("CompBou",    "M10"), ("CompBou",    "M100"),
    ]
    bigm_cells = [
        ("E",  "M1"), ("E",  "M10"), ("E",  "M100"),
        ("EU", "M1"), ("EU", "M10"), ("EU", "M100"),
    ]

    header = String["id", "n", "rho", "convexity", "base_tag", "EXACT"]
    for (v, t) in comp_cells
        push!(header, "$(v)_$(t)")
    end
    for (v, t) in bigm_cells
        push!(header, "$(v)_$(t)")
    end
    # Gap columns
    for (v, t) in comp_cells
        push!(header, "gap_$(v)_$(t)")
    end
    for (v, t) in bigm_cells
        push!(header, "gap_$(v)_$(t)")
    end

    open(out_csv, "w") do io
        println(io, join(header, ","))
        for (i, c) in enumerate(comp_rows)
            id = c["id"]
            n = c["n"]; rho = c["rho"]; cvx = c["convexity"]; btag = c["base_tag"]

            ex_comp = _to_float(c["EXACT"])
            b = get(bigm_by_id, id, nothing)
            ex_bigm = b === nothing ? nothing : _to_float(b["EXACT"])

            # Prefer Comp-side EXACT (CompBin solves the same combinatorial program).
            # If they disagree by more than 1e-4 absolute, warn.
            ex_master = ex_comp !== nothing ? ex_comp : ex_bigm
            if ex_comp !== nothing && ex_bigm !== nothing
                if abs(ex_comp - ex_bigm) > 1e-4 * max(1.0, abs(ex_comp))
                    @warn "EXACT mismatch for $id: Comp=$ex_comp, BigM=$ex_bigm"
                end
            end

            cells = String[]
            push!(cells, id, string(n), string(rho), cvx, btag, _fmt(ex_master))

            # Comp obj values
            comp_vals = Vector{Union{Float64,Nothing}}()
            for (v, t) in comp_cells
                val = _to_float(c["$(v)_$(t)"])
                push!(comp_vals, val)
                push!(cells, _fmt(val))
            end

            # BigM obj values (may be missing if BigM row not found)
            bigm_vals = Vector{Union{Float64,Nothing}}()
            for (v, t) in bigm_cells
                val = b === nothing ? nothing : _to_float(b["$(v)_$(t)"])
                push!(bigm_vals, val)
                push!(cells, _fmt(val))
            end

            # Gap columns
            for val in comp_vals
                push!(cells, _fmt(_relgap(val, ex_master)))
            end
            for val in bigm_vals
                push!(cells, _fmt(_relgap(val, ex_master)))
            end

            println(io, join(cells, ","))
        end
    end

    println("Wrote $out_csv")
    println("  Columns: $(length(header))")
    println("  Rows:    $(length(comp_rows))")
    return nothing
end


# ------------------------------------------------------------
# Convenience: print a per-instance side-by-side summary
# ------------------------------------------------------------
"""
    summarize_combined(; combined_csv="combined_results.csv")

Print a human-readable per-instance summary: EXACT, the best Comp
relaxation (CompBinBou_M1), and the BigM E/EU values across M scales.
"""
function summarize_combined(; combined_csv::AbstractString="combined_results.csv")
    header, rows = _read_csv(combined_csv)

    println()
    println(@sprintf("%-40s %3s %3s %5s %14s | %14s %14s %14s %14s | %14s %14s %14s %14s",
                     "id", "n", "rho", "cvx",
                     "EXACT",
                     "E_M1", "E_M10", "E_M100", "EU_M1",
                     "EU_M10", "EU_M100", "Comp_nob", "Comp_M1"))
    println("-" ^ 240)

    for r in rows
        id  = r["id"]; n = r["n"]; rho = r["rho"]; cvx = r["convexity"]
        ex  = _to_float(r["EXACT"])
        e1  = _to_float(r["E_M1"]);    e10  = _to_float(r["E_M10"]);    e100  = _to_float(r["E_M100"])
        eu1 = _to_float(r["EU_M1"]);   eu10 = _to_float(r["EU_M10"]);   eu100 = _to_float(r["EU_M100"])
        nob = _to_float(r["CompBinBou_nobounds"])
        cm1 = _to_float(r["CompBinBou_M1"])

        fm(x) = x === nothing ? "         NA" : @sprintf("%14.6e", x)

        println(@sprintf("%-40s %3s %3s %5s %s | %s %s %s %s | %s %s %s %s",
                         id, n, rho, cvx,
                         fm(ex),
                         fm(e1), fm(e10), fm(e100), fm(eu1),
                         fm(eu10), fm(eu100), fm(nob), fm(cm1)))
    end
    return nothing
end
