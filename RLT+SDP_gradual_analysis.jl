###############################
# RLT+SDP gradual analysis driver (BigM + Comp)
# - read_instance! : JSON Dict -> Julia arrays + validates Big-M presence
# - uses solve_RLT_SDP (BigM family) and build_RLT_SDP_comp_model (Comp family)
# - writes CSV/JSON
###############################

module RLT_SDP_Batch

using JSON, CSV
using JuMP, MosekTools, LinearAlgebra
using OrderedCollections: OrderedDict
import MathOptInterface as MOI

# BigM / Comp solvers (defined elsewhere)
using Main.RLT_SDP_Combo: solve_RLT_SDP, RelaxationModes
using Main.RLT_SDP_Comp_Combo: build_RLT_SDP_comp_model, RelaxationModes as RelaxationModesComp

# Single source of truth for instance validity (Big-M handling etc.)
using Main.RLTBigM: prepare_instance

# ------------------------
# basic coercions
# ------------------------
_as_float(x) = x isa Number ? Float64(x) : throw(ArgumentError("Expected number, got $(typeof(x))"))
_as_vec(x)   = x === nothing ? nothing : Float64.(collect(x))

# JSON matrix -> Float64 matrix (no shape assumption)
function _as_mat(x)
    x === nothing && return nothing
    if x isa AbstractMatrix
        return Float64.(x)
    elseif x isa AbstractVector
        @assert !isempty(x) && x[1] isa AbstractVector
        rows = [permutedims(Float64.(v)) for v in x]
        return reduce(vcat, rows)
    else
        error("Unsupported matrix type: $(typeof(x))")
    end
end

# n×n required (Q0/Qi/Pi)
function _as_mat_square(x, n::Int; name::String="Q0")
    M = _as_mat(x)
    (size(M,1) == n && size(M,2) == n) || error("$name must be n×n with n=$n, got size=$(size(M))")
    return M
end

# m×n required (A/H). If given as n×m, transpose.
function _as_mat_cols_n(x, n::Int; name::String="A")
    M = _as_mat(x)
    if size(M,2) == n
        return M
    elseif size(M,1) == n
        return M'  # transpose into (m×n)
    else
        error("$name must have n=$n columns (or be transposed with n rows). Got size=$(size(M)).")
    end
end

_as_tuple_vec(x) = x === nothing ? nothing :
    (x isa Tuple ? x : Tuple(x)) .|> v -> Float64.(collect(v))

function _as_tuple_mat_square(x, n::Int; name::String="Qi")
    x === nothing && return nothing
    xs = x isa Tuple ? x : Tuple(x)
    return Tuple(_as_mat_square(M, n; name=name) for M in xs)
end

_as_tuple_num(x) = x === nothing ? nothing :
    (x isa Tuple ? x : Tuple(x)) .|> _as_float

# Big-M: accept vector OR diagonal matrix; return diagonal vector
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

    elseif x isa AbstractMatrix || x isa AbstractVector
        M = _as_mat(x)
        (size(M,1) == n && size(M,2) == n) || error("$name must be n×n (diagonal), got $(size(M))")
        off = M .- Diagonal(diag(M))
        maximum(abs.(off)) <= 1e-12 || error("$name must be diagonal (nonzero off-diagonals found).")
        return Float64.(diag(M))

    else
        error("Unsupported type for $name: $(typeof(x))")
    end
end

# pull first existing key among candidates
function _get_any(d::Dict{String,Any}, keys::Tuple{Vararg{String}})
    for k in keys
        if haskey(d, k) && d[k] !== nothing
            return d[k]
        end
    end
    return nothing
end

# ------------------------
# gap helper
# ------------------------
function gap_pct(obj_relax, exact_obj; digits::Int = 4)
    if ismissing(exact_obj) || obj_relax === nothing || ismissing(obj_relax)
        return missing
    end
    raw = 100.0 * abs(Float64(obj_relax) - Float64(exact_obj)) / max(1.0, abs(Float64(exact_obj)))
    return round(raw; digits = digits)
end

_is_good_status(st::MOI.TerminationStatusCode) =
    st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED)

# ============================================================
# read_instance!  (JSON Dict -> Julia arrays + validates Big-M)
# ============================================================
function read_instance!(d::Dict{String,Any})
    @assert haskey(d, "n") "Missing key `n`."
    d["n"] = Int(d["n"])
    n = d["n"]

    inst_label = get(d, "id", "(no id)")

    if haskey(d, "rho") && d["rho"] !== nothing
        d["rho"] = Int(round(_as_float(d["rho"])))
    else
        error("Instance $inst_label: missing key `rho`.")
    end

    # n×n matrices
    if haskey(d, "Q0") && d["Q0"] !== nothing
        d["Q0"] = _as_mat_square(d["Q0"], n; name="Q0")
    end

    # A/H are (m×n) (transpose allowed)
    if haskey(d, "A") && d["A"] !== nothing
        d["A"] = _as_mat_cols_n(d["A"], n; name="A")
        (haskey(d, "b") && d["b"] !== nothing) || error("Instance $inst_label: A provided but b missing.")
    end
    if haskey(d, "H") && d["H"] !== nothing
        d["H"] = _as_mat_cols_n(d["H"], n; name="H")
        (haskey(d, "h") && d["h"] !== nothing) || error("Instance $inst_label: H provided but h missing.")
    end

    # vectors
    for k in ("q0", "b", "h")
        if haskey(d, k) && d[k] !== nothing
            d[k] = _as_vec(d[k])
        end
    end

    # dimension checks for (A,b) and (H,h)
    if haskey(d, "A") && d["A"] !== nothing && haskey(d, "b") && d["b"] !== nothing
        length(d["b"]) == size(d["A"], 1) || error("Instance $inst_label: length(b)=$(length(d["b"])) must match size(A,1)=$(size(d["A"],1)).")
    end
    if haskey(d, "H") && d["H"] !== nothing && haskey(d, "h") && d["h"] !== nothing
        length(d["h"]) == size(d["H"], 1) || error("Instance $inst_label: length(h)=$(length(d["h"])) must match size(H,1)=$(size(d["H"],1)).")
    end

    # quadratic bundles (may be nothing)
    d["Qi"] = _as_tuple_mat_square(get(d, "Qi", nothing), n; name="Qi")
    d["Pi"] = _as_tuple_mat_square(get(d, "Pi", nothing), n; name="Pi")
    d["qi"] = _as_tuple_vec(get(d, "qi", nothing))
    d["pi"] = _as_tuple_vec(get(d, "pi", nothing))
    d["ri"] = _as_tuple_num(get(d, "ri", nothing))
    d["si"] = _as_tuple_num(get(d, "si", nothing))

    # ------------------------
    # Big-M: NO DEFAULTS
    # Accept:
    #   - pair: (M_minus,M_plus) or (Mminus,Mplus)
    #   - single: M
    # ------------------------
    mminus_raw = _get_any(d, ("M_minus", "Mminus"))
    mplus_raw  = _get_any(d, ("M_plus",  "Mplus"))
    m_raw      = _get_any(d, ("M",))

    has_pair   = (mminus_raw !== nothing) || (mplus_raw !== nothing)
    has_single = (m_raw !== nothing)

    if has_pair
        (mminus_raw !== nothing && mplus_raw !== nothing) ||
            error("Instance $inst_label: provide BOTH Mminus & Mplus (or M_minus & M_plus), or provide only M.")

        mminus = _as_diagvec(mminus_raw, n; name="Mminus")
        mplus  = _as_diagvec(mplus_raw,  n; name="Mplus")

        d["Mminus"]  = mminus
        d["Mplus"]   = mplus
        d["M_minus"] = mminus
        d["M_plus"]  = mplus

        # optional symmetric M (legacy)
        d["M"] = max.(mminus, mplus)

    elseif has_single
        m = _as_diagvec(m_raw, n; name="M")
        d["M"]      = m
        d["Mminus"] = copy(m)
        d["Mplus"]  = copy(m)
        d["M_minus"] = copy(m)
        d["M_plus"]  = copy(m)

    else
        error("Instance $inst_label: missing Big-M data. Provide either (Mminus & Mplus) or a single M.")
    end

    # final validity check via single source of truth
    prepare_instance(d)  # throws if inconsistent

    return d
end

# ============================================================
# Exact callback wrapper
# ============================================================
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

# ============================================================
# Solve Comp model (one mode)
# ============================================================
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

# ============================================================
# Analyze one instance: BigM family (EU / IU)
# ============================================================
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

        stE, objE, _, _, _, _, _, tE = solve_RLT_SDP(data; variant="EU", optimizer=optimizer, relaxation=md)
        stI, objI, _, _, _, _, _, tI = solve_RLT_SDP(data; variant="IU", optimizer=optimizer, relaxation=md)

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

# ============================================================
# Analyze one instance: Complementarity family
# (CompBin / CompBinBou / CompBou)
# ============================================================
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

        stB,  objB,  tB  = _solve_comp(data; variant="RLT_CompBin",    optimizer=optimizer, relaxation=md)
        stBU, objBU, tBU = _solve_comp(data; variant="RLT_CompBinBou", optimizer=optimizer, relaxation=md)
        stBo, objBo, tBo = _solve_comp(data; variant="RLT_CompBou",    optimizer=optimizer, relaxation=md)

        push!(rows, (
            mode                 = String(md),

            CompBin_status       = string(stB),
            CompBin_obj          = (objB === nothing ? missing : objB),
            CompBin_gap          = gap_pct(objB, exact_obj),
            CompBin_time_sec     = tB,

            CompBinBou_status    = string(stBU),
            CompBinBou_obj       = (objBU === nothing ? missing : objBU),
            CompBinBou_gap       = gap_pct(objBU, exact_obj),
            CompBinBou_time_sec  = tBU,

            CompBou_status       = string(stBo),
            CompBou_obj          = (objBo === nothing ? missing : objBo),
            CompBou_gap          = gap_pct(objBo, exact_obj),
            CompBou_time_sec     = tBo,
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

# ============================================================
# Batch: BigM family
# ============================================================
function run_batch(instances_json::AbstractString;
                   out_csv::AbstractString="gradual_sdp_results.csv",
                   out_json::AbstractString="gradual_sdp_results.json",
                   modes = RelaxationModes,
                   optimizer = MosekTools.Optimizer,
                   exact_cb = nothing)

    raw = JSON.parsefile(instances_json)
    insts = [read_instance!(deepcopy(d)) for d in raw]

    all_rows = NamedTuple[]
    json_out = OrderedDict[]

    for (k, data) in enumerate(insts)
        inst_label = get(data, "id", "inst_$(k)")

        println("\n==============================")
        println("instance $k : $inst_label")
        println("==============================")

        rows, exact_block = analyze_instance(data; modes=modes, optimizer=optimizer, exact_cb=exact_cb, inst_label=inst_label)

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

# ============================================================
# Batch: Complementarity family
# ============================================================
function run_batch_comp(instances_json::AbstractString;
                        out_csv::AbstractString="gradual_comp_results.csv",
                        out_json::AbstractString="gradual_comp_results.json",
                        modes = RelaxationModesComp,
                        optimizer = MosekTools.Optimizer,
                        exact_cb = nothing)

    raw = JSON.parsefile(instances_json)
    insts = [read_instance!(deepcopy(d)) for d in raw]

    all_rows = NamedTuple[]
    json_out = OrderedDict[]

    for (k, data) in enumerate(insts)
        inst_label = get(data, "id", "inst_$(k)")

        println("\n==============================")
        println("instance $k : $inst_label")
        println("==============================")

        rows, exact_block = analyze_instance_comp(data; modes=modes, optimizer=optimizer, exact_cb=exact_cb, inst_label=inst_label)

        for r in rows
            push!(all_rows, (
                inst_id              = k,
                inst_label           = inst_label,
                n                    = get(data,"n",missing),
                rho                  = get(data,"rho",missing),

                mode                 = r.mode,

                CompBin_status       = r.CompBin_status,
                CompBin_obj          = r.CompBin_obj,
                CompBin_gap          = r.CompBin_gap,
                CompBin_time_sec     = r.CompBin_time_sec,

                CompBinBou_status    = r.CompBinBou_status,
                CompBinBou_obj       = r.CompBinBou_obj,
                CompBinBou_gap       = r.CompBinBou_gap,
                CompBinBou_time_sec  = r.CompBinBou_time_sec,

                CompBou_status       = r.CompBou_status,
                CompBou_obj          = r.CompBou_obj,
                CompBou_gap          = r.CompBou_gap,
                CompBou_time_sec     = r.CompBou_time_sec,

                exact_status         = exact_block["exact_status"],
                exact_obj            = exact_block["exact_obj"],
                exact_time_sec       = exact_block["exact_time_sec"]
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

export read_instance!, run_batch, run_batch_comp, gap_pct

end # module RLT_SDP_Batch


# ============================================================
# Plotting utilities (time series in pink)
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
    "RLT_PSD3x3_R", "RLT_PSD3x3_XUR",
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
    sub = sub[[m in Set(RelaxationOrder) for m in sub.mode_str], :]
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

    title!(p, "instance $label (n=$n, rho=$rho)")
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
    sub = sub[[m in Set(RelaxationOrderComp) for m in sub.mode_str], :]
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
        color     = :deeppink,
    )

    title!(p, "instance $label (n=$n, rho=$rho)")
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
    """
    batch = RLT_SDP_Batch.run_batch(
        "instances.json";
        out_csv   = "gradual_sdp_results.csv",
        out_json  = "gradual_sdp_results.json",
        optimizer = MosekTools.Optimizer,
        exact_cb  = data -> RLTBigM.build_and_solve(data; variant="EXACT", optimizer=Gurobi.Optimizer),
    )
    println(batch)
    """
    """
    batch_comp = RLT_SDP_Batch.run_batch_comp(
        "instances.json";
        out_csv   = "gradual_comp_results.csv",
        out_json  = "gradual_comp_results.json",
        optimizer = MosekTools.Optimizer,
        exact_cb  = data -> RLTBigM.build_and_solve(data; variant="EXACT", optimizer=Gurobi.Optimizer),
    )
    println(batch_comp)
    """
    PlotGradualSDP.plot_all_instances("gradual_sdp_results.csv"; outdir="plots")
    PlotGradualSDP.plot_all_instances_comp("gradual_comp_results.csv"; outdir="plots_comp")
end