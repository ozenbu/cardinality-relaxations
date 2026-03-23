module RLT_SDP_Batch

using JSON, CSV
using JuMP, MosekTools, LinearAlgebra
using OrderedCollections: OrderedDict
using MathOptInterface
const MOI = MathOptInterface

using Main.RLT_SDP_Combo: solve_RLT_SDP, RelaxationModes
using Main.RLT_SDP_Comp_Combo: build_RLT_SDP_comp_model, RelaxationModes as RelaxationModesComp

# ------------------------
# Basic coercions
# ------------------------
_as_float(x) = x isa Number ? Float64(x) : throw(ArgumentError("Expected number, got $(typeof(x))"))
_as_vec(x)   = x === nothing ? nothing : Float64.(collect(x))

function gap_pct(obj_relax, exact_obj; digits::Int = 4)
    if ismissing(exact_obj) || obj_relax === nothing || ismissing(obj_relax)
        return missing
    end
    raw = 100.0 * abs(Float64(obj_relax) - Float64(exact_obj)) / max(1.0, abs(Float64(exact_obj)))
    return round(raw; digits = digits)
end

function _as_mat_with_n(x, n::Int)
    x === nothing && return nothing
    if x isa AbstractMatrix
        M = Float64.(x)
        return size(M,2) == n ? M : (size(M,1) == n ? M' : error("bad shape $(size(M)) for n=$n"))
    elseif x isa AbstractVector
        # JSON often gives matrices as Vector{Vector}
        @assert !isempty(x) && x[1] isa AbstractVector
        rows = [permutedims(Float64.(v)) for v in x]
        Mrow = reduce(vcat, rows)
        return size(Mrow, 2) == n ? Mrow :
               size(Mrow, 1) == n ? Mrow' : error("cannot coerce matrix with n=$n")
    else
        error("unsupported type $(typeof(x))")
    end
end

_as_tuple_vec(x) = x === nothing ? nothing :
    (x isa Tuple ? x : Tuple(x)) .|> v -> Float64.(collect(v))

function _as_tuple_mat(x, n::Int)
    x === nothing && return nothing
    xs = x isa Tuple ? x : Tuple(x)
    return Tuple(_as_mat_with_n(M, n) for M in xs)
end

_as_tuple_num(x) = x === nothing ? nothing :
    (x isa Tuple ? x : Tuple(x)) .|> _as_float

# ------------------------
# Big-M helpers: accept vector or diagonal matrix; return diagonal vector
# ------------------------
function _as_diagvec(x, n::Int; name::String="M")
    x === nothing && return nothing

    if x isa AbstractVector
        v = Float64.(collect(x))
        length(v) == n || error("$name must have length n=$n, got length=$(length(v))")
        return v

    elseif x isa Diagonal
        v = Float64.(diag(x))
        length(v) == n || error("$name size mismatch, need n=$n")
        return v

    elseif x isa AbstractMatrix
        M = Float64.(x)
        size(M,1) == n && size(M,2) == n || error("$name must be n×n, got $(size(M))")

        # Require diagonal (avoid silently ignoring off-diagonals)
        off = M .- Diagonal(diag(M))
        maximum(abs.(off)) ≤ 1e-12 || error("$name must be diagonal.")
        return Float64.(diag(M))

    else
        error("Unsupported type for $name: $(typeof(x))")
    end
end

# Pull the first existing key among candidates
function _get_any(d::Dict{String,Any}, keys::Tuple{Vararg{String}})
    for k in keys
        if haskey(d, k) && d[k] !== nothing
            return d[k]
        end
    end
    return nothing
end

# ------------------------
# Normalizer: makes instances compatible with BOTH BigM + Comp pipelines
# - Supports either (M_minus,M_plus) OR single M
# - Accepts vector or diagonal matrix
# - Writes BOTH naming styles:
#     underscore: M_minus, M_plus
#     camel:      Mminus, Mplus
# - Also keeps/creates M (legacy symmetric)
# ------------------------
function normalize_instance!(d::Dict{String,Any})
    @assert haskey(d,"n")
    d["n"] = Int(d["n"])
    n = d["n"]

    # rho as Int (cardinality)
    if haskey(d, "rho") && d["rho"] !== nothing
        d["rho"] = Int(round(_as_float(d["rho"])))
    end

    # bigM_scale default
    d["bigM_scale"] = Float64(get(d, "bigM_scale", 1.0))

    # Matrices (n×n)
    for k in ("Q0","A","H")
        if haskey(d,k) && d[k] !== nothing
            d[k] = _as_mat_with_n(d[k], n)
        end
    end

    # Vectors
    for k in ("q0","b","h")
        if haskey(d,k) && d[k] !== nothing
            d[k] = _as_vec(d[k])
        end
    end

    # Quadratic constraints bundles (may be nothing)
    d["Qi"] = _as_tuple_mat(get(d,"Qi", nothing), n)
    d["Pi"] = _as_tuple_mat(get(d,"Pi", nothing), n)
    d["qi"] = _as_tuple_vec(get(d,"qi", nothing))
    d["pi"] = _as_tuple_vec(get(d,"pi", nothing))
    d["ri"] = _as_tuple_num(get(d,"ri", nothing))
    d["si"] = _as_tuple_num(get(d,"si", nothing))

    # ------------------------
    # Unified Big-M handling (NO DEFAULTS)
    # ------------------------
    mminus_raw = _get_any(d, ("M_minus", "Mminus"))
    mplus_raw  = _get_any(d, ("M_plus",  "Mplus"))
    m_raw      = _get_any(d, ("M",))

    has_pair   = (mminus_raw !== nothing) || (mplus_raw !== nothing)
    has_single = (m_raw !== nothing)

    if has_pair
        (mminus_raw !== nothing && mplus_raw !== nothing) ||
            error("Instance $(get(d, "id", "(no id)")): provide both M_minus and M_plus (or Mminus and Mplus), or provide only M.")
            
        mminus = _as_diagvec(mminus_raw, n; name="M_minus")
        mplus  = _as_diagvec(mplus_raw,  n; name="M_plus")

        # canonical storage (both styles)
        d["M_minus"] = mminus
        d["M_plus"]  = mplus
        d["Mminus"]  = mminus
        d["Mplus"]   = mplus

        # optional legacy symmetric M (for older code paths)
        d["M"] = max.(mminus, mplus)

    elseif has_single
        m = _as_diagvec(m_raw, n; name="M")

        d["M"]       = m
        d["M_minus"] = copy(m)
        d["M_plus"]  = copy(m)
        d["Mminus"]  = copy(m)
        d["Mplus"]   = copy(m)

    else
        error("Instance $inst_id: missing Big-M data. Provide either (M_minus & M_plus) or a single M.")
    end

    return d
end

# ------------------------
# Exact callback wrapper
# ------------------------
function _get_exact(data::Dict{String,Any}; exact_cb=nothing)
    if exact_cb === nothing
        return ("MISSING", missing, nothing, missing)
    end
    try
        t0 = time_ns()
        res = exact_cb(data)
        t  = round((time_ns() - t0) / 1e9; digits = 4)
        st = string(res[1])
        obj = res[2] === nothing ? missing : Float64(res[2])
        x = (length(res) >= 3 && res[3] !== nothing) ? _as_vec(res[3]) : nothing
        return (st, obj, x, t)
    catch err
        @warn "exact_cb error" err
        return ("ERROR", missing, nothing, missing)
    end
end

_is_good_status(st::MOI.TerminationStatusCode) =
    st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED)

# ------------------------
# Solve Comp model (one mode)
# ------------------------
function _solve_comp(data::Dict{String,Any};
                     variant::String,
                     optimizer = MosekTools.Optimizer,
                     relaxation::Symbol = :RLT)

    m = build_RLT_SDP_comp_model(data;
        variant    = variant,
        optimizer  = optimizer,
        relaxation = relaxation,
    )

    t0 = time_ns()
    optimize!(m)
    t  = round((time_ns() - t0) / 1e9; digits = 4)

    st  = termination_status(m)
    obj = _is_good_status(st) ? objective_value(m) : nothing
    return st, obj, t
end

# ------------------------
# Analyze one instance: BigM family (EU/IU)
# ------------------------
function analyze_instance(data::Dict{String,Any};
                          modes = RelaxationModes,
                          optimizer = MosekTools.Optimizer,
                          exact_cb = nothing,
                          inst_label::AbstractString = "")
    exact_status, exact_obj, exact_x, exact_time = _get_exact(data; exact_cb=exact_cb)

    rows = NamedTuple[]
    for md in modes
        println(inst_label == "" ? ">>> solving relaxation = $(md)"
                                 : ">>> [$(inst_label)] solving relaxation = $(md)")

        stE, objE, _, _, _, _, _, tE = solve_RLT_SDP(
            data; variant="EU", optimizer=optimizer, relaxation=md,
        )
        stI, objI, _, _, _, _, _, tI = solve_RLT_SDP(
            data; variant="IU", optimizer=optimizer, relaxation=md,
        )

        push!(rows, (
            mode        = String(md),
            EU_status   = string(stE),
            EU_obj      = (objE === nothing ? missing : objE),
            EU_gap      = gap_pct(objE, exact_obj),
            EU_time_sec = tE,
            IU_status   = string(stI),
            IU_obj      = (objI === nothing ? missing : objI),
            IU_gap      = gap_pct(objI, exact_obj),
            IU_time_sec = tI,
        ))
    end

    exact_block = OrderedDict(
        "exact_status"   => exact_status,
        "exact_obj"      => exact_obj,
        "exact_x"        => (exact_x === nothing ? nothing : collect(exact_x)),
        "exact_time_sec" => exact_time
    )
    return rows, exact_block
end

# ------------------------
# Analyze one instance: Complementarity family (CompBin / CompBinBou / CompBou)
# ------------------------
function analyze_instance_comp(data::Dict{String,Any};
                               modes = RelaxationModesComp,
                               optimizer = MosekTools.Optimizer,
                               exact_cb = nothing,
                               inst_label::AbstractString = "")

    exact_status, exact_obj, exact_x, exact_time = _get_exact(data; exact_cb=exact_cb)

    rows = NamedTuple[]
    for md in modes
        println(inst_label == "" ? ">>> solving relaxation = $(md)"
                                 : ">>> [$(inst_label)] solving relaxation = $(md)")

        stB,  objB,  tB  = _solve_comp(data; variant="RLT_CompBin",     optimizer=optimizer, relaxation=md)
        stBU, objBU, tBU = _solve_comp(data; variant="RLT_CompBinBou",  optimizer=optimizer, relaxation=md)
        stBo, objBo, tBo = _solve_comp(data; variant="RLT_CompBou",     optimizer=optimizer, relaxation=md)

        push!(rows, (
            mode               = String(md),

            CompBin_status     = string(stB),
            CompBin_obj        = (objB === nothing ? missing : objB),
            CompBin_gap        = gap_pct(objB, exact_obj),
            CompBin_time_sec   = tB,

            CompBinBou_status  = string(stBU),
            CompBinBou_obj     = (objBU === nothing ? missing : objBU),
            CompBinBou_gap     = gap_pct(objBU, exact_obj),
            CompBinBou_time_sec= tBU,

            CompBou_status     = string(stBo),
            CompBou_obj        = (objBo === nothing ? missing : objBo),
            CompBou_gap        = gap_pct(objBo, exact_obj),
            CompBou_time_sec   = tBo,
        ))
    end

    exact_block = OrderedDict(
        "exact_status"   => exact_status,
        "exact_obj"      => exact_obj,
        "exact_x"        => (exact_x === nothing ? nothing : collect(exact_x)),
        "exact_time_sec" => exact_time
    )
    return rows, exact_block
end

# ------------------------
# Batch: BigM family
# ------------------------
function run_batch(instances_json::AbstractString;
                   out_csv::AbstractString="batch_results.csv",
                   out_json::AbstractString="batch_results.json",
                   modes = RelaxationModes,
                   optimizer = MosekTools.Optimizer,
                   exact_cb = nothing)

    raw = JSON.parsefile(instances_json)
    insts = [normalize_instance!(deepcopy(d)) for d in raw]

    all_rows = NamedTuple[]
    json_out = OrderedDict[]

    for (k, data) in enumerate(insts)
        inst_label = get(data, "id", "inst_$(k)")

        println("\n==============================")
        println("instance $k : $inst_label")
        println("==============================")

        rows, exact_block = analyze_instance(
            data; modes=modes, optimizer=optimizer, exact_cb=exact_cb, inst_label=inst_label,
        )

        for r in rows
            push!(all_rows, (
                inst_id        = k,
                inst_label     = inst_label,
                n              = get(data,"n",missing),
                rho            = get(data,"rho",missing),

                mode           = r.mode,

                EU_status      = r.EU_status,
                EU_obj         = r.EU_obj,
                EU_gap         = r.EU_gap,
                EU_time_sec    = r.EU_time_sec,

                IU_status      = r.IU_status,
                IU_obj         = r.IU_obj,
                IU_gap         = r.IU_gap,
                IU_time_sec    = r.IU_time_sec,

                exact_status   = exact_block["exact_status"],
                exact_obj      = exact_block["exact_obj"],
                exact_time_sec = exact_block["exact_time_sec"]
            ))
        end

        push!(json_out, OrderedDict(
            "inst_id"        => k,
            "inst_label"     => inst_label,
            "n"              => get(data,"n",nothing),
            "rho"            => get(data,"rho",nothing),
            "exact_status"   => exact_block["exact_status"],
            "exact_obj"      => exact_block["exact_obj"],
            "exact_x"        => exact_block["exact_x"],
            "exact_time_sec" => exact_block["exact_time_sec"],
            "relaxations"    => [OrderedDict(
                "mode"        => r.mode,
                "EU_status"   => r.EU_status,
                "EU_obj"      => r.EU_obj,
                "EU_gap (%)"  => r.EU_gap,
                "EU_time_sec" => r.EU_time_sec,
                "IU_status"   => r.IU_status,
                "IU_obj"      => r.IU_obj,
                "IU_gap (%)"  => r.IU_gap,
                "IU_time_sec" => r.IU_time_sec
            ) for r in rows]
        ))
    end

    CSV.write(out_csv, all_rows; transform=(col,val)->(val===nothing ? missing : val))
    open(out_json, "w") do io
        JSON.print(io, json_out, 2)
    end

    return (csv = out_csv, json = out_json, count = length(insts))
end

# ------------------------
# Batch: Complementarity family
# ------------------------
function run_batch_comp(instances_json::AbstractString;
                        out_csv::AbstractString="batch_comp_results.csv",
                        out_json::AbstractString="batch_comp_results.json",
                        modes = RelaxationModesComp,
                        optimizer = MosekTools.Optimizer,
                        exact_cb = nothing)

    raw = JSON.parsefile(instances_json)
    insts = [normalize_instance!(deepcopy(d)) for d in raw]

    all_rows = NamedTuple[]
    json_out = OrderedDict[]

    for (k, data) in enumerate(insts)
        inst_label = get(data, "id", "inst_$(k)")

        println("\n==============================")
        println("instance $k : $inst_label")
        println("==============================")

        rows, exact_block = analyze_instance_comp(
            data; modes=modes, optimizer=optimizer, exact_cb=exact_cb, inst_label=inst_label,
        )

        for r in rows
            push!(all_rows, (
                inst_id            = k,
                inst_label         = inst_label,
                n                  = get(data,"n",missing),
                rho                = get(data,"rho",missing),

                mode               = r.mode,

                CompBin_status     = r.CompBin_status,
                CompBin_obj        = r.CompBin_obj,
                CompBin_gap        = r.CompBin_gap,
                CompBin_time_sec   = r.CompBin_time_sec,

                CompBinBou_status  = r.CompBinBou_status,
                CompBinBou_obj     = r.CompBinBou_obj,
                CompBinBou_gap     = r.CompBinBou_gap,
                CompBinBou_time_sec= r.CompBinBou_time_sec,

                CompBou_status     = r.CompBou_status,
                CompBou_obj        = r.CompBou_obj,
                CompBou_gap        = r.CompBou_gap,
                CompBou_time_sec   = r.CompBou_time_sec,

                exact_status       = exact_block["exact_status"],
                exact_obj          = exact_block["exact_obj"],
                exact_time_sec     = exact_block["exact_time_sec"]
            ))
        end

        push!(json_out, OrderedDict(
            "inst_id"        => k,
            "inst_label"     => inst_label,
            "n"              => get(data,"n",nothing),
            "rho"            => get(data,"rho",nothing),
            "exact_status"   => exact_block["exact_status"],
            "exact_obj"      => exact_block["exact_obj"],
            "exact_x"        => exact_block["exact_x"],
            "exact_time_sec" => exact_block["exact_time_sec"],
            "relaxations"    => [OrderedDict(
                "mode"                  => r.mode,

                "CompBin_status"        => r.CompBin_status,
                "CompBin_obj"           => r.CompBin_obj,
                "CompBin_gap (%)"       => r.CompBin_gap,
                "CompBin_time_sec"      => r.CompBin_time_sec,

                "CompBinBou_status"     => r.CompBinBou_status,
                "CompBinBou_obj"        => r.CompBinBou_obj,
                "CompBinBou_gap (%)"    => r.CompBinBou_gap,
                "CompBinBou_time_sec"   => r.CompBinBou_time_sec,

                "CompBou_status"        => r.CompBou_status,
                "CompBou_obj"           => r.CompBou_obj,
                "CompBou_gap (%)"       => r.CompBou_gap,
                "CompBou_time_sec"      => r.CompBou_time_sec,
            ) for r in rows]
        ))
    end

    CSV.write(out_csv, all_rows; transform=(col,val)->(val===nothing ? missing : val))
    open(out_json, "w") do io
        JSON.print(io, json_out, 2)
    end

    return (csv = out_csv, json = out_json, count = length(insts))
end

export normalize_instance!, run_batch, run_batch_comp, gap_pct

end # module RLT_SDP_Batch


# ============================================================
# Plotting utilities (updated to CompBin naming)
# ============================================================
module PlotGradualSDP

using CSV, DataFrames, Plots
using Measures: mm
import Main.RLT_SDP_Batch: gap_pct

const GOOD_STATUS = Set(["OPTIMAL", "ALMOST_OPTIMAL", "LOCALLY_SOLVED"])

const RelaxationOrder = [
    "RLT",
    "RLT_SOC2x2_diag_X", "RLT_SOC2x2_diag_U", "RLT_SOC2x2_diag_XU",
    "RLT_SOC2x2_full_X", "RLT_SOC2x2_full_U", "RLT_SOC2x2_full_XU",
    "RLT_SOC_directional_X", "RLT_SOC_directional_U", "RLT_SOC_directional_XU",
    "RLT_SOC_hybrid_X", "RLT_SOC_hybrid_U", "RLT_SOC_hybrid_XU",
    "RLT_PSD3x3_X", "RLT_PSD3x3_U", "RLT_PSD3x3_XU",
    "RLT_blockSDP_X", "RLT_blockSDP_U", "RLT_blockSDP_XU",
    "RLT_full_SDP",
]

const RelaxationOrderComp = [
    "RLT",
    "RLT_SOC2x2_diag_X", "RLT_SOC2x2_diag_V", "RLT_SOC2x2_diag_XV",
    "RLT_SOC2x2_full_X", "RLT_SOC2x2_full_V", "RLT_SOC2x2_full_XV",
    "RLT_SOC_directional_X", "RLT_SOC_directional_V", "RLT_SOC_directional_XV",
    "RLT_SOC_hybrid_X", "RLT_SOC_hybrid_V", "RLT_SOC_hybrid_XV",
    "RLT_PSD3x3_X", "RLT_PSD3x3_V", "RLT_PSD3x3_XV",
    "RLT_blockSDP_X", "RLT_blockSDP_V", "RLT_blockSDP_XV",
    "RLT_full_SDP",
]

load_results(path::AbstractString) = CSV.read(path, DataFrame)

_gap_to_nan(obj, ex) = begin
    g = gap_pct(obj, ex)
    ismissing(g) ? NaN : Float64(g)
end

function plot_instance(df::DataFrame, inst_id::Int; outdir::AbstractString = "plots")
    mkpath(outdir)

    sub = df[df.inst_id .== inst_id, :]
    nrow(sub) == 0 && (@warn "No rows for inst_id = $inst_id"; return nothing)

    sub[!, :mode_str] = String.(sub.mode)
    RelaxationSet = Set(RelaxationOrder)
    sub = sub[[m in RelaxationSet for m in sub.mode_str], :]
    nrow(sub) == 0 && (@warn "No known modes for inst_id = $inst_id"; return nothing)

    k     = nrow(sub)
    modes = copy(sub.mode_str)

    EU_gap = Vector{Float64}(undef, k)
    IU_gap = Vector{Float64}(undef, k)
    times  = Vector{Float64}(undef, k)

    for (idx, row) in enumerate(eachrow(sub))
        ex = row.exact_obj
        EU_gap[idx] = (row.EU_status in GOOD_STATUS) ? _gap_to_nan(row.EU_obj, ex) : NaN
        IU_gap[idx] = (row.IU_status in GOOD_STATUS) ? _gap_to_nan(row.IU_obj, ex) : NaN

        t = NaN
        if row.EU_status in GOOD_STATUS && !ismissing(row.EU_time_sec)
            t = row.EU_time_sec
        end
        if row.IU_status in GOOD_STATUS && !ismissing(row.IU_time_sec)
            t = isnan(t) ? row.IU_time_sec : max(t, row.IU_time_sec)
        end
        times[idx] = t
    end

    perm = sortperm([isnan(g) ? -Inf : g for g in EU_gap]; rev=true)
    EU_gap, IU_gap, times, modes = EU_gap[perm], IU_gap[perm], times[perm], modes[perm]
    x = collect(1:k)

    n   = sub.n[1]
    rho = sub.rho[1]
    label = hasproperty(sub, :inst_label) ? sub.inst_label[1] : "inst_$(inst_id)"

    p = plot(
        x, EU_gap;
        xlabel    = "Relaxation mode",
        ylabel    = "Gap (%)",
        xticks    = (x, modes),
        xrotation = 60,
        legend    = :topleft,
        marker    = :circle,
        linestyle = :solid,
        label     = "EU gap",
        size      = (1200, 600),
        margin    = 20mm,
    )
    plot!(p, x, IU_gap; marker=:diamond, linestyle=:solid, label="IU gap")

    finite_times = filter(!isnan, times)
    tmax = isempty(finite_times) ? 1.0 : maximum(finite_times)
    ymax = 1.05 * tmax

    plot!(twinx(), x, times;
        ylabel    = "Time (s)",
        marker    = :square,
        linestyle = :dash,
        label     = "time",
        legend    = :topright,
        ylims     = (0, ymax),
        color     = :deeppink,
    )

    title!(p, "instance $label (n=$n, ρ=$rho)")
    outpath = joinpath(outdir, "inst_$(label).png")
    savefig(p, outpath)
    display(p)
    return p
end

function plot_all_instances(path::AbstractString; outdir::AbstractString = "plots")
    df = load_results(path)
    for id in sort(unique(df.inst_id))
        @info "Plotting inst_id $id"
        plot_instance(df, id; outdir=outdir)
    end
    return nothing
end

function plot_instance_comp(df::DataFrame, inst_id::Int; outdir::AbstractString = "plots_comp")
    mkpath(outdir)

    sub = df[df.inst_id .== inst_id, :]
    nrow(sub) == 0 && (@warn "No rows for inst_id = $inst_id"; return nothing)

    sub[!, :mode_str] = String.(sub.mode)
    RelaxationSet = Set(RelaxationOrderComp)
    sub = sub[[m in RelaxationSet for m in sub.mode_str], :]
    nrow(sub) == 0 && (@warn "No known modes for inst_id = $inst_id"; return nothing)

    k     = nrow(sub)
    modes = copy(sub.mode_str)

    B_gap   = Vector{Float64}(undef, k)   # CompBin
    BU_gap  = Vector{Float64}(undef, k)   # CompBinBou
    Bo_gap  = Vector{Float64}(undef, k)   # CompBou
    times   = Vector{Float64}(undef, k)

    for (idx, row) in enumerate(eachrow(sub))
        ex = row.exact_obj

        B_gap[idx]  = (row.CompBin_status in GOOD_STATUS) ? _gap_to_nan(row.CompBin_obj, ex) : NaN
        BU_gap[idx] = (row.CompBinBou_status in GOOD_STATUS) ? _gap_to_nan(row.CompBinBou_obj, ex) : NaN
        Bo_gap[idx] = (row.CompBou_status in GOOD_STATUS) ? _gap_to_nan(row.CompBou_obj, ex) : NaN

        t = NaN
        if row.CompBin_status in GOOD_STATUS && !ismissing(row.CompBin_time_sec)
            t = row.CompBin_time_sec
        end
        if row.CompBinBou_status in GOOD_STATUS && !ismissing(row.CompBinBou_time_sec)
            t = isnan(t) ? row.CompBinBou_time_sec : max(t, row.CompBinBou_time_sec)
        end
        if row.CompBou_status in GOOD_STATUS && !ismissing(row.CompBou_time_sec)
            t = isnan(t) ? row.CompBou_time_sec : max(t, row.CompBou_time_sec)
        end
        times[idx] = t
    end

    perm = sortperm([isnan(g) ? -Inf : g for g in BU_gap]; rev=true)
    B_gap, BU_gap, Bo_gap, times, modes = B_gap[perm], BU_gap[perm], Bo_gap[perm], times[perm], modes[perm]
    x = collect(1:k)

    n   = sub.n[1]
    rho = sub.rho[1]
    label = hasproperty(sub, :inst_label) ? sub.inst_label[1] : "inst_$(inst_id)"

    p = plot(
        x, BU_gap;
        xlabel    = "Relaxation mode",
        ylabel    = "Gap (%)",
        xticks    = (x, modes),
        xrotation = 60,
        legend    = :topleft,
        marker    = :circle,
        linestyle = :solid,
        label     = "CompBinBou gap",
        size      = (1200, 600),
        margin    = 20mm,
    )
    plot!(p, x, B_gap;  marker=:diamond,   linestyle=:solid, label="CompBin gap")
    plot!(p, x, Bo_gap; marker=:utriangle, linestyle=:solid, label="CompBou gap")

    finite_times = filter(!isnan, times)
    tmax = isempty(finite_times) ? 1.0 : maximum(finite_times)
    ymax = 1.05 * tmax

    plot!(twinx(), x, times;
        ylabel    = "Time (s)",
        marker    = :square,
        linestyle = :dash,
        label     = "time",
        legend    = :topright,
        ylims     = (0, ymax),
    )

    title!(p, "instance $label (n=$n, ρ=$rho)")
    outpath = joinpath(outdir, "inst_$(label).png")
    savefig(p, outpath)
    display(p)
    return p
end

function plot_all_instances_comp(path::AbstractString; outdir::AbstractString = "plots_comp")
    df = load_results(path)
    for id in sort(unique(df.inst_id))
        @info "Plotting comp inst_id $id"
        plot_instance_comp(df, id; outdir=outdir)
    end
    return nothing
end

end # module PlotGradualSDP


# ============================================================
# Interactive runner
# ============================================================
if isinteractive()
    using .RLTBigM
    using .RLT_SDP_Batch
    using .PlotGradualSDP
    using MosekTools
    using Gurobi

    batch = RLT_SDP_Batch.run_batch(
        "instances.json";
        out_csv   = "gradual_sdp_results.csv",
        out_json  = "gradual_sdp_results.json",
        optimizer = MosekTools.Optimizer,
        exact_cb  = data -> RLTBigM.build_and_solve(data; variant="EXACT", optimizer=Gurobi.Optimizer),
    )
    println(batch)

    batch_comp = RLT_SDP_Batch.run_batch_comp(
        "instances.json";
        out_csv   = "gradual_comp_results.csv",
        out_json  = "gradual_comp_results.json",
        optimizer = MosekTools.Optimizer,
        exact_cb  = data -> RLTBigM.build_and_solve(data; variant="EXACT", optimizer=Gurobi.Optimizer),
    )
    println(batch_comp)

    PlotGradualSDP.plot_all_instances("gradual_sdp_results.csv"; outdir="plots")
    PlotGradualSDP.plot_all_instances_comp("gradual_comp_results.csv"; outdir="plots_comp")
end