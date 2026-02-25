module RLT_SDP_Batch

using JSON, CSV
using JuMP, MosekTools, LinearAlgebra
using OrderedCollections: OrderedDict
using Main.RLT_SDP_Combo: solve_RLT_SDP, RelaxationModes

const MOI = JuMP.MOI

_as_float(x) = x isa Number ? Float64(x) : throw(ArgumentError("Expected number, got $(typeof(x))"))
_as_vec(x)   = x === nothing ? nothing : Float64.(collect(x))

function gap_pct(obj_relax, exact_obj; digits::Int = 4)
    if ismissing(exact_obj) || obj_relax === nothing
        return missing
    end
    raw = 100.0 * abs(obj_relax - exact_obj) / max(1.0, abs(exact_obj))
    return round(raw; digits = digits)
end

function _as_mat_with_n(x, n::Int)
    x === nothing && return nothing
    if x isa AbstractMatrix
        M = Float64.(x)
        return size(M,2) == n ? M : (size(M,1) == n ? M' : error("bad shape $(size(M)) for n=$n"))
    elseif x isa AbstractVector
        @assert !isempty(x) && x[1] isa AbstractVector
        rows = [permutedims(Float64.(v)) for v in x]
        Mrow = reduce(vcat, rows)
        return size(Mrow, 2) == n ? Mrow :
               size(Mrow, 1) == n ? Mrow' : error("cannot coerce matrix with n=$n")
    else
        error("unsupported type $(typeof(x))")
    end
end

_as_tuple_vec(x) = x === nothing ? nothing : (x isa Tuple ? x : Tuple(x)) .|> v -> Float64.(collect(v))

function _as_tuple_mat(x, n::Int)
    x === nothing && return nothing
    xs = x isa Tuple ? x : Tuple(x)
    return Tuple(_as_mat_with_n(M, n) for M in xs)
end

_as_tuple_num(x) = x === nothing ? nothing : (x isa Tuple ? x : Tuple(x)) .|> _as_float

function normalize_instance!(d::Dict{String,Any})
    @assert haskey(d,"n")
    d["n"] = Int(d["n"])
    n = d["n"]

    haskey(d,"rho") && (d["rho"] = _as_float(d["rho"]))

    for k in ("Q0","A","H")
        if haskey(d,k) && d[k] !== nothing
            d[k] = _as_mat_with_n(d[k], n)
        end
    end

    for k in ("q0","b","h")
        if haskey(d,k) && d[k] !== nothing
            d[k] = _as_vec(d[k])
        end
    end

    d["Qi"] = _as_tuple_mat(get(d,"Qi", nothing), n)
    d["Pi"] = _as_tuple_mat(get(d,"Pi", nothing), n)
    d["qi"] = _as_tuple_vec(get(d,"qi", nothing))
    d["pi"] = _as_tuple_vec(get(d,"pi", nothing))
    d["ri"] = _as_tuple_num(get(d,"ri", nothing))
    d["si"] = _as_tuple_num(get(d,"si", nothing))

    if haskey(d,"M") && d["M"] !== nothing
        d["M"] = _as_vec(d["M"])
    else
        d["M"] = ones(Float64, n)
    end

    return d
end

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

function analyze_instance(data::Dict{String,Any};
                          modes = RelaxationModes,
                          optimizer = MosekTools.Optimizer,
                          exact_cb = nothing,
                          inst_label::AbstractString = "")
    exact_status, exact_obj, exact_x, exact_time = _get_exact(data; exact_cb=exact_cb)

    rows = NamedTuple[]

    for md in modes
        if inst_label == ""
            println(">>> solving relaxation = $(md)")
        else
            println(">>> [$(inst_label)] solving relaxation = $(md)")
        end

        stE, objE, _, _, _, _, _, tE = solve_RLT_SDP(
            data;
            variant    = "EU",
            optimizer  = optimizer,
            relaxation = md,
        )

        stI, objI, _, _, _, _, _, tI = solve_RLT_SDP(
            data;
            variant    = "IU",
            optimizer  = optimizer,
            relaxation = md,
        )

        EU_gap = gap_pct(objE, exact_obj)
        IU_gap = gap_pct(objI, exact_obj)

        push!(rows, (
            mode        = String(md),
            EU_status   = string(stE),
            EU_obj      = (objE === nothing ? missing : objE),
            EU_gap      = EU_gap,
            EU_time_sec = tE,
            IU_status   = string(stI),
            IU_obj      = (objI === nothing ? missing : objI),
            IU_gap      = IU_gap,
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
            data;
            modes     = modes,
            optimizer = optimizer,
            exact_cb  = exact_cb,
            inst_label = inst_label,
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

end # module


module PlotGradualSDP

using CSV, DataFrames, Plots
using Measures: mm
import Main.RLT_SDP_Batch: gap_pct  

const RelaxationOrder = [
    "RLT",

    "RLT_SOC2x2_diag_X",
    "RLT_SOC2x2_diag_U",
    "RLT_SOC2x2_diag_XU",

    "RLT_SOC2x2_full_X",
    "RLT_SOC2x2_full_U",
    "RLT_SOC2x2_full_XU",

    "RLT_SOC_directional_X",
    "RLT_SOC_directional_U",
    "RLT_SOC_directional_XU",

    "RLT_SOC_hybrid_X",
    "RLT_SOC_hybrid_U",
    "RLT_SOC_hybrid_XU",

    "RLT_PSD3x3_X",
    "RLT_PSD3x3_U",
    "RLT_PSD3x3_XU",

    "RLT_blockSDP_X",
    "RLT_blockSDP_U",
    "RLT_blockSDP_XU",

    "RLT_full_SDP",
]



function load_results(path::AbstractString)
    CSV.read(path, DataFrame)
end

function plot_instance(df::DataFrame, inst_id::Int; outdir::AbstractString = "plots")
    mkpath(outdir)

    sub = df[df.inst_id .== inst_id, :]
    if nrow(sub) == 0
        @warn "No rows for inst_id = $inst_id"
        return nothing
    end

    # mode'u String yap ve sadece bildiğimiz modları tut
    sub[!, :mode_str] = String.(sub.mode)
    RelaxationSet = Set(RelaxationOrder)
    mask = [m in RelaxationSet for m in sub.mode_str]
    sub = sub[mask, :]
    if nrow(sub) == 0
        @warn "No known modes for inst_id = $inst_id"
        return nothing
    end

    
    # Sorting w.r.t. EU gap

    k     = nrow(sub)
    modes = copy(sub.mode_str)

    EU_gap = Vector{Float64}(undef, k)
    IU_gap = Vector{Float64}(undef, k)
    times  = Vector{Float64}(undef, k)

    for (idx, row) in enumerate(eachrow(sub))
        eu = row.EU_obj
        iu = row.IU_obj
        ex = row.exact_obj

        EU_gap[idx] = (row.EU_status == "OPTIMAL") ? gap_pct(eu, ex) : NaN
        IU_gap[idx] = (row.IU_status == "OPTIMAL") ? gap_pct(iu, ex) : NaN

        t = NaN
        if row.EU_status == "OPTIMAL" && !ismissing(row.EU_time_sec)
            t = row.EU_time_sec
        end
        if row.IU_status == "OPTIMAL" && !ismissing(row.IU_time_sec)
            t = isnan(t) ? row.IU_time_sec : max(t, row.IU_time_sec)
        end
        times[idx] = t
    end

    # Sorting from largest to smallest EU gap
    sort_key = similar(EU_gap)
    for i in 1:k
        g = EU_gap[i]
        sort_key[i] = isnan(g) ? -Inf : g  # NaN olanlar en sona
    end

    perm   = sortperm(sort_key; rev = true)  # büyük EU gap önce
    EU_gap = EU_gap[perm]
    IU_gap = IU_gap[perm]
    times  = times[perm]
    modes  = modes[perm]

    x = collect(1:k)

    n   = sub.n[1]
    rho = sub.rho[1]

    # string id for the title (e.g.: n7_rho4_S_LI_LE_CVX_seed1)
    label = hasproperty(sub, :inst_label) ? sub.inst_label[1] : "inst_$(inst_id)"

    # 1. y-axis: gap (%)
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
        color     = :olivedrab,
    )

    plot!(p, x, IU_gap;
        marker    = :diamond,
        linestyle = :solid,
        label     = "IU gap",
        color     = :maroon,
    )

    # 2. y-axis: time (s) – dynamic y-limits
    finite_times = filter(!isnan, times)
    tmax = isempty(finite_times) ? 1.0 : maximum(finite_times)
    ymax = 1.05 * tmax

    plot!(twinx(), x, times;
        ylabel    = "Time (s)",
        marker    = :square,
        color     = :aquamarine2,
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

function plot_all_instances(path::AbstractString; outdir::AbstractString = "plots")
    df = load_results(path)
    ids = sort(unique(df.inst_id))
    for id in ids
        @info "Plotting inst_id $id"
        plot_instance(df, id; outdir = outdir)
    end
    return nothing
end

end # module


if isinteractive()

    using .RLTBigM
    using .RLT_SDP_Batch
    using .PlotGradualSDP
    using MosekTools
    using Gurobi
    
    
    # 2) Run all relaxations on all instances
    batch = RLT_SDP_Batch.run_batch(
        "instances.json";
        out_csv   = "gradual_sdp_results.csv",
        out_json  = "gradual_sdp_results.json",
        optimizer = MosekTools.Optimizer,
        exact_cb  = data -> RLTBigM.build_and_solve(
            data; variant="EXACT", optimizer=Gurobi.Optimizer
        )
    )

    println(batch)
    

    # 3) Plot all instances (gap vs mode, time on right axis)
    PlotGradualSDP.plot_all_instances("gradual_sdp_results.csv"; outdir = "plots")
end 



