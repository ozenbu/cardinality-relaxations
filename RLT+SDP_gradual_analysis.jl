module RLT_SDP_Batch

using JSON, CSV
using JuMP, MosekTools, LinearAlgebra
using OrderedCollections: OrderedDict

using Main.RLT_SDP_Combo: solve_RLT_SDP, RelaxationModes
using Main.RLT_SDP_Comp_Combo: build_RLT_SDP_comp_model, RelaxationModes as RelaxationModesComp

const MOI = JuMP.MOI

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

_is_good_status(st::MOI.TerminationStatusCode) = st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED)

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

        st2,  obj2,  t2  = _solve_comp(data; variant="RLT_S2",  optimizer=optimizer, relaxation=md)
        st2u, obj2u, t2u = _solve_comp(data; variant="RLT_S2U", optimizer=optimizer, relaxation=md)
        st3,  obj3,  t3  = _solve_comp(data; variant="RLT_S3",  optimizer=optimizer, relaxation=md)

        push!(rows, (
            mode         = String(md),

            S2_status    = string(st2),
            S2_obj       = (obj2 === nothing ? missing : obj2),
            S2_gap       = gap_pct(obj2, exact_obj),
            S2_time_sec  = t2,

            S2U_status   = string(st2u),
            S2U_obj      = (obj2u === nothing ? missing : obj2u),
            S2U_gap      = gap_pct(obj2u, exact_obj),
            S2U_time_sec = t2u,

            S3_status    = string(st3),
            S3_obj       = (obj3 === nothing ? missing : obj3),
            S3_gap       = gap_pct(obj3, exact_obj),
            S3_time_sec  = t3,
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
            modes      = modes,
            optimizer  = optimizer,
            exact_cb   = exact_cb,
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
            data;
            modes      = modes,
            optimizer  = optimizer,
            exact_cb   = exact_cb,
            inst_label = inst_label,
        )

        for r in rows
            push!(all_rows, (
                inst_id        = k,
                inst_label     = inst_label,
                n              = get(data,"n",missing),
                rho            = get(data,"rho",missing),

                mode           = r.mode,

                S2_status      = r.S2_status,
                S2_obj         = r.S2_obj,
                S2_gap         = r.S2_gap,
                S2_time_sec    = r.S2_time_sec,

                S2U_status     = r.S2U_status,
                S2U_obj        = r.S2U_obj,
                S2U_gap        = r.S2U_gap,
                S2U_time_sec   = r.S2U_time_sec,

                S3_status      = r.S3_status,
                S3_obj         = r.S3_obj,
                S3_gap         = r.S3_gap,
                S3_time_sec    = r.S3_time_sec,

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

                "S2_status"   => r.S2_status,
                "S2_obj"      => r.S2_obj,
                "S2_gap (%)"  => r.S2_gap,
                "S2_time_sec" => r.S2_time_sec,

                "S2U_status"   => r.S2U_status,
                "S2U_obj"      => r.S2U_obj,
                "S2U_gap (%)"  => r.S2U_gap,
                "S2U_time_sec" => r.S2U_time_sec,

                "S3_status"   => r.S3_status,
                "S3_obj"      => r.S3_obj,
                "S3_gap (%)"  => r.S3_gap,
                "S3_time_sec" => r.S3_time_sec,
            ) for r in rows]
        ))
    end

    CSV.write(out_csv, all_rows; transform=(col,val)->(val===nothing ? missing : val))
    open(out_json, "w") do io
        JSON.print(io, json_out, 2)
    end

    return (csv = out_csv, json = out_json, count = length(insts))
end

end # module RLT_SDP_Batch


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
    if nrow(sub) == 0
        @warn "No rows for inst_id = $inst_id"
        return nothing
    end

    sub[!, :mode_str] = String.(sub.mode)
    RelaxationSet = Set(RelaxationOrder)
    sub = sub[[m in RelaxationSet for m in sub.mode_str], :]
    if nrow(sub) == 0
        @warn "No known modes for inst_id = $inst_id"
        return nothing
    end

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

    sort_key = similar(EU_gap)
    for i in 1:k
        g = EU_gap[i]
        sort_key[i] = isnan(g) ? -Inf : g
    end

    perm   = sortperm(sort_key; rev = true)
    EU_gap = EU_gap[perm]
    IU_gap = IU_gap[perm]
    times  = times[perm]
    modes  = modes[perm]

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
        plot_instance(df, id; outdir=outdir)
    end
    return nothing
end

function plot_instance_comp(df::DataFrame, inst_id::Int; outdir::AbstractString = "plots_comp")
    mkpath(outdir)

    sub = df[df.inst_id .== inst_id, :]
    if nrow(sub) == 0
        @warn "No rows for inst_id = $inst_id"
        return nothing
    end

    sub[!, :mode_str] = String.(sub.mode)
    RelaxationSet = Set(RelaxationOrderComp)
    sub = sub[[m in RelaxationSet for m in sub.mode_str], :]
    if nrow(sub) == 0
        @warn "No known modes for inst_id = $inst_id"
        return nothing
    end

    k     = nrow(sub)
    modes = copy(sub.mode_str)

    S2_gap  = Vector{Float64}(undef, k)
    S2U_gap = Vector{Float64}(undef, k)
    S3_gap  = Vector{Float64}(undef, k)
    times   = Vector{Float64}(undef, k)

    for (idx, row) in enumerate(eachrow(sub))
        ex = row.exact_obj

        S2_gap[idx]  = (row.S2_status  in GOOD_STATUS) ? _gap_to_nan(row.S2_obj,  ex) : NaN
        S2U_gap[idx] = (row.S2U_status in GOOD_STATUS) ? _gap_to_nan(row.S2U_obj, ex) : NaN
        S3_gap[idx]  = (row.S3_status  in GOOD_STATUS) ? _gap_to_nan(row.S3_obj,  ex) : NaN

        t = NaN
        if row.S2_status in GOOD_STATUS && !ismissing(row.S2_time_sec)
            t = row.S2_time_sec
        end
        if row.S2U_status in GOOD_STATUS && !ismissing(row.S2U_time_sec)
            t = isnan(t) ? row.S2U_time_sec : max(t, row.S2U_time_sec)
        end
        if row.S3_status in GOOD_STATUS && !ismissing(row.S3_time_sec)
            t = isnan(t) ? row.S3_time_sec : max(t, row.S3_time_sec)
        end
        times[idx] = t
    end

    sort_key = similar(S2U_gap)
    for i in 1:k
        g = S2U_gap[i]
        sort_key[i] = isnan(g) ? -Inf : g
    end

    perm    = sortperm(sort_key; rev=true)
    S2_gap  = S2_gap[perm]
    S2U_gap = S2U_gap[perm]
    S3_gap  = S3_gap[perm]
    times   = times[perm]
    modes   = modes[perm]

    x = collect(1:k)

    n   = sub.n[1]
    rho = sub.rho[1]
    label = hasproperty(sub, :inst_label) ? sub.inst_label[1] : "inst_$(inst_id)"

    p = plot(
        x, S2U_gap;
        xlabel    = "Relaxation mode",
        ylabel    = "Gap (%)",
        xticks    = (x, modes),
        xrotation = 60,
        legend    = :topleft,
        marker    = :circle,
        linestyle = :solid,
        label     = "S2U gap",
        size      = (1200, 600),
        margin    = 20mm,
    )

    plot!(p, x, S2_gap; marker=:diamond, linestyle=:solid, label="S2 gap")
    plot!(p, x, S3_gap; marker=:utriangle, linestyle=:solid, label="S3 gap")

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
    ids = sort(unique(df.inst_id))
    for id in ids
        @info "Plotting comp inst_id $id"
        plot_instance_comp(df, id; outdir=outdir)
    end
    return nothing
end

end # module PlotGradualSDP


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