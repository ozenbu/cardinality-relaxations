

using CSV, DataFrames, Statistics, JSON

# ------------------------------------------------------------
# Label helper (same as before)
# ------------------------------------------------------------
function attach_labels_from_json!(df::DataFrame, json_path::AbstractString)
    raw = JSON.parsefile(json_path)

    base = Dict{Int,String}()
    for r in raw
        base[Int(r["inst_id"])] = string(r["inst_label"])
    end

    dict_same  = base
    dict_minus = Dict(k - 1 => v for (k, v) in base)
    dict_plus  = Dict(k + 1 => v for (k, v) in base)

    ids = Int.(df.inst_id)

    labels_same  = [get(dict_same,  id, missing) for id in ids]
    labels_minus = [get(dict_minus, id, missing) for id in ids]
    labels_plus  = [get(dict_plus,  id, missing) for id in ids]

    c_same  = count(!ismissing, labels_same)
    c_minus = count(!ismissing, labels_minus)
    c_plus  = count(!ismissing, labels_plus)

    labels =
        (c_minus >= c_same && c_minus >= c_plus) ? labels_minus :
        (c_plus  >= c_same && c_plus  >= c_minus) ? labels_plus  :
        labels_same

    if :inst_label ∉ names(df)
        df[!, :inst_label] = Vector{Union{Missing,String}}(labels)
    else
        df[!, :inst_label] = coalesce.(df.inst_label, labels)
    end

    return df
end

"""
Per-relaxation EU vs IU objective-difference summary.

Difference metric (only when both runs are OPTIMAL):
    d = |EU_obj - IU_obj| / max(1, |EU_obj|)

Difference rule:
- Only comparable rows (EU and IU both OPTIMAL and obj present) are used for diff/max.
- diff if d > threshold (e.g., threshold=0.01 means "1% relative difference").

Output columns (short):
rel, N, missEU, missIU, diffN, maxD, maxID, maxLbl
"""

function summarize_eu_iu_objdiff(csv_path::AbstractString;
                                 threshold::Float64 = 0.01,
                                 label_json::Union{Nothing,String} = nothing,
                                 out_csv::AbstractString = "EU_IU_objdiff_summary.csv",
                                 out_details_csv::AbstractString = "EU_IU_objdiff_details.csv")

    df = CSV.read(csv_path, DataFrame)
    df[!, :rel] = String.(df.mode)

    if label_json !== nothing
        attach_labels_from_json!(df, label_json)
    end

    # d per row (missing if not comparable)
    n = nrow(df)
    d = Vector{Union{Missing,Float64}}(undef, n)

    for i in 1:n
        eu_ok = (df[i, :EU_status] == "OPTIMAL") && !ismissing(df[i, :EU_obj])
        iu_ok = (df[i, :IU_status] == "OPTIMAL") && !ismissing(df[i, :IU_obj])

        if eu_ok && iu_ok
            eu = Float64(df[i, :EU_obj])
            iu = Float64(df[i, :IU_obj])
            denom = max(1.0, abs(eu))
            d[i] = abs(eu - iu) / denom
        else
            d[i] = missing
        end
    end
    df[!, :d] = d

    # Details: only rows with d > threshold
    details = df[(.!ismissing.(df.d)) .& (df.d .> threshold),
        [:inst_id, :inst_label, :rel, :EU_obj, :IU_obj, :d,
         :EU_status, :IU_status, :EU_gap, :IU_gap, :EU_time_sec, :IU_time_sec]
    ]
    CSV.write(out_details_csv, details)

    summary = combine(groupby(df, :rel)) do sdf
        N = nrow(sdf)

        # Count rows that are NOT comparable because EU/IU is not OPTIMAL or obj missing
        eu_comp = (sdf.EU_status .== "OPTIMAL") .& .!ismissing.(sdf.EU_obj)
        iu_comp = (sdf.IU_status .== "OPTIMAL") .& .!ismissing.(sdf.IU_obj)

        missEU = count(.!eu_comp)
        missIU = count(.!iu_comp)

        ok = .!ismissing.(sdf.d)
        diffind = findall(ok .& (sdf.d .> threshold))
        diffN = length(diffind)

        if diffN == 0
            maxD   = 0.0
            maxID  = missing
            maxLbl = missing
        else
            diffs = sdf.d[diffind]
            maxD, pos = findmax(diffs)
            rowpos = diffind[pos]
            maxID  = sdf.inst_id[rowpos]
            maxLbl = sdf.inst_label[rowpos]
        end

        return (N=N, missEU=missEU, missIU=missIU, diffN=diffN, maxD=maxD, maxID=maxID, maxLbl=maxLbl)
    end

    sort!(summary, [:diffN, :maxD], rev=true)
    CSV.write(out_csv, summary)

    return summary, details
end

# -------------------------
# Run
# -------------------------
results_csv  = "gradual_sdp_results.csv"
results_json = "gradual_sdp_results.json"

threshold = 0.01  # 1% relative objective discrepancy

summary, details = summarize_eu_iu_objdiff(results_csv;
    threshold = threshold,
    label_json = results_json
)

println("\n=== EU vs IU objdiff summary (|EU_obj-IU_obj|/max(1,|EU_obj|) > $(threshold)) ===")
show(summary; allrows=true, allcols=true)
println()

println("Wrote: EU_IU_objdiff_summary.csv")
println("Wrote: EU_IU_objdiff_details.csv")


using CSV, DataFrames
using Plots

gr()

const DELTA_RM = 0.05

# -----------------------------
# Helpers
# -----------------------------
is_valid_time(status, t) = (status == "OPTIMAL") && !ismissing(t) && isfinite(Float64(t))
is_valid_obj(status, v)  = (status == "OPTIMAL") && !ismissing(v) && isfinite(Float64(v))

"""
Compute per-row finite penalty rM from observed valid ratios.
If no valid ratios exist, returns default_rM.
"""
function compute_rM(ratios::AbstractVector{<:Real}; default_rM::Float64 = 10.0, delta::Float64 = DELTA_RM)
    vals = [Float64(x) for x in ratios if isfinite(Float64(x))]
    isempty(vals) && return default_rM
    return (1.0 + delta) * maximum(vals)
end

"""
Build a tau grid from observed ratios (finite) plus endpoints.
"""
function build_tau_grid(r::Vector{Float64}; tau_max::Float64)
    vals = sort(unique(r))
    vals = vals[vals .>= 1.0]
    vals = vals[vals .<= tau_max]
    if isempty(vals) || vals[end] < tau_max
        push!(vals, tau_max)
    end
    return vals
end

# ============================================================
# TIME RATIOS (finite rM, no Inf)
# ============================================================

"""
Compute time ratios r_{p,s} = t_{p,s} / min_{s'} t_{p,s'} for a given variant.

Rules:
- Valid if status == OPTIMAL and time exists.
- Instances with no valid runs are excluded.
- Invalid rows get ratio = rM (finite penalty), where
  rM = (1+delta) * max observed valid ratio.
"""
function time_ratios(df::DataFrame; variant::Symbol = :EU, delta::Float64 = DELTA_RM)
    status_col = Symbol(string(variant), "_status")
    time_col   = Symbol(string(variant), "_time_sec")

    d = copy(df)
    d[!, :rel] = String.(d.mode)

    n = nrow(d)
    valid = falses(n)
    tval  = Vector{Union{Missing,Float64}}(undef, n)

    for i in 1:n
        st = d[i, status_col]
        ti = d[i, time_col]
        if is_valid_time(st, ti)
            valid[i] = true
            tval[i]  = Float64(ti)
        else
            valid[i] = false
            tval[i]  = missing
        end
    end
    d[!, :valid] = valid
    d[!, :t]     = tval

    tmin_df = combine(groupby(d, :inst_id)) do sdf
        ok = sdf.valid
        if any(ok)
            return (tmin = minimum(skipmissing(sdf.t[ok])),)
        else
            return (tmin = missing,)
        end
    end
    d = leftjoin(d, tmin_df, on = :inst_id)

    keep_inst = .!ismissing.(d.tmin)
    d = d[keep_inst, :]

    n2 = nrow(d)
    r = Vector{Float64}(undef, n2)
    valid_ratio_vals = Float64[]

    for i in 1:n2
        if d[i, :valid]
            ri = Float64(d[i, :t]) / Float64(d[i, :tmin])
            r[i] = ri
            push!(valid_ratio_vals, ri)
        else
            r[i] = NaN
        end
    end

    rM = compute_rM(valid_ratio_vals; default_rM = 10.0, delta = delta)

    for i in 1:n2
        if !d[i, :valid]
            r[i] = rM
        end
    end

    d[!, :r]  = r
    d[!, :rM] = fill(rM, n2)

    return d
end

"""
Plot the time performance profile rho_s(tau) for a given variant.
"""
function plot_time_profile(df::DataFrame;
                           variant::Symbol = :EU,
                           tau_max::Float64 = 10.0,
                           outpath::AbstractString = "time_profile.png",
                           title_str::AbstractString = "")

    d = time_ratios(df; variant = variant)
    inst_ids = sort(unique(Int.(d.inst_id)))
    np = length(inst_ids)
    np == 0 && error("No eligible instances for variant = $(variant).")

    rM = d.rM[1]
    tau_max_eff = max(1.0, min(tau_max, rM))
    tau_vals = build_tau_grid(d.r; tau_max = tau_max_eff)

    modes = sort(unique(d.rel))

    p = plot(
        xlabel = "τ (time ratio)",
        ylabel = "ρ(τ)",
        xlims  = (1.0, tau_max_eff),
        ylims  = (0.0, 1.0),
        legend = :bottomright,
        size   = (1400, 750),
    )

    for m in modes
        sub = d[d.rel .== m, :]
        rmap = Dict{Int,Float64}()
        for row in eachrow(sub)
            rmap[Int(row.inst_id)] = Float64(row.r)
        end
        rvec = [get(rmap, pid, rM) for pid in inst_ids]

        rho = [count(x -> x <= τ, rvec) / np for τ in tau_vals]
        plot!(p, tau_vals, rho; seriestype = :steppost, label = m)
    end

    title!(p, title_str == "" ? "Time performance profile ($(variant))" : title_str)
    display(p)
    savefig(p, outpath)
    println("Saved: $(outpath)   (instances used: $(np)),  rM=$(round(rM; digits=4))")
    return p
end

# ============================================================
# QUALITY RATIOS (exact-referenced objective error, epsilon-floor, finite rM)
# ============================================================

"""
Compute quality ratios using EXACT reference objective.

Given exact objective z*_p, and relaxation objective z_{p,s}:

q_{p,s} = |z*_p - z_{p,s}| / max(1, |z*_p|)
qtilde  = max(q, eps)
r_{p,s} = qtilde / min_{s'} qtilde

Rules:
- Valid if exact_obj is present AND (status == OPTIMAL and obj present).
- Instances with no valid runs are excluded (for that variant).
- Invalid rows get ratio = rM (finite penalty), where
  rM = (1+delta) * max observed valid ratio.
"""
function quality_ratios(df::DataFrame;
                        variant::Symbol = :EU,
                        eps::Float64 = 1e-6,
                        delta::Float64 = DELTA_RM)

    status_col = Symbol(string(variant), "_status")
    obj_col    = Symbol(string(variant), "_obj")

    d = copy(df)
    d[!, :rel] = String.(d.mode)

    n = nrow(d)
    valid = falses(n)
    qtil  = Vector{Union{Missing,Float64}}(undef, n)

    for i in 1:n
        zstar = d[i, :exact_obj]
        st    = d[i, status_col]
        z     = d[i, obj_col]

        if !ismissing(zstar) && isfinite(Float64(zstar)) && is_valid_obj(st, z)
            valid[i] = true
            zstar_f = Float64(zstar)
            z_f     = Float64(z)
            denom   = max(1.0, abs(zstar_f))
            q       = abs(zstar_f - z_f) / denom
            qtil[i] = max(q, eps)
        else
            valid[i] = false
            qtil[i]  = missing
        end
    end

    d[!, :valid] = valid
    d[!, :qtil]  = qtil

    qmin_df = combine(groupby(d, :inst_id)) do sdf
        ok = sdf.valid
        if any(ok)
            return (qmin = minimum(skipmissing(sdf.qtil[ok])),)
        else
            return (qmin = missing,)
        end
    end
    d = leftjoin(d, qmin_df, on = :inst_id)

    keep_inst = .!ismissing.(d.qmin)
    d = d[keep_inst, :]

    n2 = nrow(d)
    r = Vector{Float64}(undef, n2)
    valid_ratio_vals = Float64[]

    for i in 1:n2
        if d[i, :valid]
            ri = Float64(d[i, :qtil]) / Float64(d[i, :qmin])
            r[i] = ri
            push!(valid_ratio_vals, ri)
        else
            r[i] = NaN
        end
    end

    rM = compute_rM(valid_ratio_vals; default_rM = 10.0, delta = delta)

    for i in 1:n2
        if !d[i, :valid]
            r[i] = rM
        end
    end

    d[!, :r]  = r
    d[!, :rM] = fill(rM, n2)

    return d
end

"""
Plot the quality performance profile rho_s(tau) for a given variant.
"""
function plot_quality_profile(df::DataFrame;
                              variant::Symbol = :EU,
                              tau_max::Float64 = 100.0,
                              eps::Float64 = 1e-6,
                              outpath::AbstractString = "quality_profile.png",
                              title_str::AbstractString = "")

    d = quality_ratios(df; variant = variant, eps = eps)
    inst_ids = sort(unique(Int.(d.inst_id)))
    np = length(inst_ids)
    np == 0 && error("No eligible instances for variant = $(variant) (missing exact or no OPTIMAL).")

    rM = d.rM[1]
    tau_max_eff = max(1.0, min(tau_max, rM))
    tau_vals = build_tau_grid(d.r; tau_max = tau_max_eff)

    modes = sort(unique(d.rel))

    p = plot(
        xlabel = "τ (quality ratio)",
        ylabel = "ρ(τ)",
        xlims  = (1.0, tau_max_eff),
        ylims  = (0.0, 1.0),
        legend = :bottomright,
        size   = (1400, 750),
    )

    for m in modes
        sub = d[d.rel .== m, :]
        rmap = Dict{Int,Float64}()
        for row in eachrow(sub)
            rmap[Int(row.inst_id)] = Float64(row.r)
        end
        rvec = [get(rmap, pid, rM) for pid in inst_ids]

        rho = [count(x -> x <= τ, rvec) / np for τ in tau_vals]
        plot!(p, tau_vals, rho; seriestype = :steppost, label = m)
    end

    title!(p, title_str == "" ? "Quality performance profile w.r.t. EXACT ($(variant))" : title_str)
    display(p)
    savefig(p, outpath)
    println("Saved: $(outpath)   (instances used: $(np)),  rM=$(round(rM; digits=4)),  eps=$(eps)")
    return p
end

# -------------------------
# Run
# -------------------------
df = CSV.read("gradual_sdp_results.csv", DataFrame)

# Time profiles
plot_time_profile(df; variant = :EU, tau_max = 10.0, outpath = "time_profile_EU.png")
plot_time_profile(df; variant = :IU, tau_max = 10.0, outpath = "time_profile_IU.png")

# Quality profiles (EXACT-referenced objective error)
plot_quality_profile(df; variant = :EU, tau_max = 100.0, eps = 1e-6, outpath = "quality_profile_EU.png")
plot_quality_profile(df; variant = :IU, tau_max = 100.0, eps = 1e-6, outpath = "quality_profile_IU.png")
