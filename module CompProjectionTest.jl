module CompProjectionTest

using JuMP, LinearAlgebra, MosekTools
const MOI = JuMP.MOI
using Parameters: @unpack

using Main.SDPComplementarity
using Main.RLTBigM: prepare_instance
using Main.RLTComp: additional_params

# ------------------------------------------------------------
# Common x,X-only block copied conceptually, but without touching
# SDPComplementarity internals
# ------------------------------------------------------------
function add_common_xX_only!(m::Model, x, X, params)
    @unpack n, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, xL, xU = params

    @objective(m, Min,
        0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n) + q0' * x
    )

    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(m,
            0.5 * sum(Qmat[i,j] * X[i,j] for i=1:n, j=1:n) + qvec' * x + rterm <= 0
        )
    end
    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(m,
            0.5 * sum(Pmat[i,j] * X[i,j] for i=1:n, j=1:n) + pvec' * x + sterm == 0
        )
    end

    if ℓ > 0
        @constraint(m, A * x .<= b)
    end
    if η > 0
        @constraint(m, H * x .== h)
    end

    @constraint(m, xL .<= x)
    @constraint(m, x .<= xU)

    return nothing
end

# ------------------------------------------------------------
# Natural reduced candidate for CompBin
# ------------------------------------------------------------
function build_reduced_CompBin_model(data::Dict{String,Any};
    optimizer = MosekTools.Optimizer,
    verbose::Bool = false
)
    params = additional_params(prepare_instance(data))
    @unpack n, ρ, e = params

    m = Model(optimizer)
    !verbose && set_silent(m)

    @variable(m, x[1:n])
    @variable(m, v[1:n])
    @variable(m, X[1:n,1:n], Symmetric)

    add_common_xX_only!(m, x, X, params)

    @constraint(m, 0 .<= v)
    @constraint(m, v .<= e)
    @constraint(m, sum(v) == n - ρ)

    @constraint(m,
        Symmetric([
            1.0  x';
            x    X
        ]) in PSDCone()
    )

    for i in 1:n
        @constraint(m,
            Symmetric([
                1.0   x[i]    v[i];
                x[i]  X[i,i]  0.0;
                v[i]  0.0     v[i]
            ]) in PSDCone()
        )
    end

    return m
end

# ------------------------------------------------------------
# Completion test for fixed (xbar, Xbar) in full CompBin model
# ------------------------------------------------------------
function build_completion_model(
    xbar::AbstractVector,
    Xbar::AbstractMatrix,
    data::Dict{String,Any};
    optimizer = MosekTools.Optimizer,
    verbose::Bool = false
)
    params = additional_params(prepare_instance(data))
    @unpack n, ρ, e = params

    m = Model(optimizer)
    !verbose && set_silent(m)

    @variable(m, v[1:n])
    @variable(m, V[1:n,1:n], Symmetric)
    @variable(m, W[1:n,1:n])

    @objective(m, Min, 0.0)

    @constraint(m, 0 .<= v)
    @constraint(m, v .<= e)
    @constraint(m, sum(v) == n - ρ)

    @constraint(m, [i=1:n], W[i,i] == 0.0)
    @constraint(m, [i=1:n], V[i,i] == v[i])

    Xfix = 0.5 .* (Matrix(Xbar) + Matrix(Xbar)')

    @constraint(m,
        Symmetric([
            1.0   xbar'   v';
            xbar  Xfix    W;
            v     W'      V
        ]) in PSDCone()
    )

    return m
end

# ------------------------------------------------------------
# Solve reduced candidate, then test completion
# ------------------------------------------------------------
function solve_reduced_then_check_completion(data::Dict{String,Any};
    optimizer = MosekTools.Optimizer,
    verbose::Bool = false
)
    m_red = build_reduced_CompBin_model(data; optimizer=optimizer, verbose=verbose)
    optimize!(m_red)

    st_red = termination_status(m_red)
    good = st_red in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED)

    if !good
        return (
            reduced_status = st_red,
            reduced_obj = nothing,
            completion_status = nothing,
            completion_feasible = false,
            xbar = nothing,
            Xbar = nothing,
            reduced_model = m_red,
            completion_model = nothing,
        )
    end

    xbar = value.(m_red[:x])
    Xbar = value.(m_red[:X])
    obj_red = objective_value(m_red)

    m_cmp = build_completion_model(xbar, Xbar, data; optimizer=optimizer, verbose=verbose)
    optimize!(m_cmp)

    st_cmp = termination_status(m_cmp)
    feasible = st_cmp in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED)

    return (
        reduced_status = st_red,
        reduced_obj = obj_red,
        completion_status = st_cmp,
        completion_feasible = feasible,
        xbar = xbar,
        Xbar = Xbar,
        reduced_model = m_red,
        completion_model = m_cmp,
    )
end

export build_reduced_CompBin_model,
       build_completion_model,
       solve_reduced_then_check_completion

end

if isinteractive()
    # ---- PrettyPrint first ----
    include(normpath(joinpath(@__DIR__, "instances", "prettyprint.jl")))
    using .PrettyPrint

    using MosekTools

    # ---- Instances from .jl ----
    include(normpath(joinpath(@__DIR__, "instances", "alper_stqp_instance.jl")))
    using .AlperStqpInstances
    alp_inst = alper_stqp_rho3_instance()

    include(normpath(joinpath(@__DIR__, "instances", "diff_RLTEU_RLTIU_bigM_instance.jl")))
    using .EUIUdiffinstance
    diff_inst = euiu_diff_instance()

    include(normpath(joinpath(@__DIR__, "instances", "SDP_separation_instance.jl")))
    using .SDPCompBinSeparationInstance
    sdp_sep_ins = sdp_compbin_vs_bigm_instance()

    instances = [
        ("alp_inst", alp_inst),
        ("diff_inst", diff_inst),
        ("sdp_sep_ins", sdp_sep_ins),
    ]

    println("\n==========================================")
    println(" CompBin reduced-candidate / completion test ")
    println("==========================================\n")

    for (name, inst) in instances
        println("\n------------------------------------------")
        println("instance = ", name)
        println("id       = ", get(inst, "id", "unknown"))
        println("------------------------------------------")

        try
            res = CompProjectionTest.solve_reduced_then_check_completion(
                inst;
                optimizer = MosekTools.Optimizer,
                verbose = false
            )

            println("reduced status      = ", res.reduced_status)
            println("reduced obj         = ", res.reduced_obj)
            println("completion status   = ", res.completion_status)
            println("completion feasible = ", res.completion_feasible)

            if res.completion_feasible
                println("=> reduced candidate point CAN be completed to full CompBin SDP.")
            else
                println("=> reduced candidate point CANNOT be completed to full CompBin SDP.")
            end

        catch err
            println("ERROR on instance ", name)
            showerror(stdout, err)
            println()
        end
    end
end

module CompProjectionBatch

using JSON, CSV
using JuMP, MosekTools
using OrderedCollections: OrderedDict
import MathOptInterface as MOI

# Reuse existing parsing / validation
using Main.RLT_SDP_Batch: read_instance!

# Reuse the two models we already built
using Main.CompProjectionTest
using Main.SDPComplementarity

const GOOD_STATUS = Set([
    MOI.OPTIMAL,
    MOI.ALMOST_OPTIMAL,
    MOI.LOCALLY_SOLVED,
])

_good(st) = st in GOOD_STATUS

# ------------------------------------------------------------
# Solve full CompBin SDP-only model and return status/objective
# ------------------------------------------------------------
function _solve_full_compbin(data::Dict{String,Any};
                             optimizer = MosekTools.Optimizer,
                             verbose::Bool = false)
    m = SDPComplementarity.build_SDP_comp_model(
        data;
        variant   = "SDP_CompBin",
        optimizer = optimizer,
        verbose   = verbose,
    )
    optimize!(m)
    st = termination_status(m)
    obj = _good(st) ? objective_value(m) : missing
    return st, obj, m
end

# ------------------------------------------------------------
# Analyze one instance
# - reduced CompBin candidate
# - completion test from reduced (x,X)
# - full CompBin SDP-only
# ------------------------------------------------------------
function analyze_instance(data::Dict{String,Any};
                          optimizer = MosekTools.Optimizer,
                          verbose::Bool = false,
                          inst_label::AbstractString = "")
    println(inst_label == "" ? ">>> solving reduced+completion"
                             : ">>> [$inst_label] solving reduced+completion")

    red = CompProjectionTest.solve_reduced_then_check_completion(
        data;
        optimizer = optimizer,
        verbose   = verbose,
    )

    println(inst_label == "" ? ">>> solving full CompBin SDP"
                             : ">>> [$inst_label] solving full CompBin SDP")

    full_status, full_obj, _ = _solve_full_compbin(
        data;
        optimizer = optimizer,
        verbose   = verbose,
    )

    reduced_status = string(red.reduced_status)
    reduced_obj    = red.reduced_obj === nothing ? missing : Float64(red.reduced_obj)

    completion_status = red.completion_status === nothing ? "MISSING" : string(red.completion_status)
    completion_feasible = red.completion_feasible

    full_status_str = string(full_status)
    full_obj_val    = full_obj === nothing ? missing : full_obj

    obj_diff = if !ismissing(reduced_obj) && !ismissing(full_obj_val)
        reduced_obj - full_obj_val
    else
        missing
    end

    return (
        reduced_status      = reduced_status,
        reduced_obj         = reduced_obj,
        completion_status   = completion_status,
        completion_feasible = completion_feasible,
        full_status         = full_status_str,
        full_obj            = full_obj_val,
        obj_diff            = obj_diff,
    )
end

# ------------------------------------------------------------
# Batch over instances.json
# ------------------------------------------------------------
function run_batch(instances_json::AbstractString;
                   out_csv::AbstractString  = "comp_projection_results.csv",
                   out_json::AbstractString = "comp_projection_results.json",
                   optimizer = MosekTools.Optimizer,
                   verbose::Bool = false)

    raw   = JSON.parsefile(instances_json)
    insts = [read_instance!(deepcopy(d)) for d in raw]

    all_rows = NamedTuple[]
    json_out = OrderedDict[]

    for (k, data) in enumerate(insts)
        inst_label = get(data, "id", "inst_$(k)")

        println("\n==============================")
        println("instance $k : $inst_label")
        println("==============================")

        res = analyze_instance(
            data;
            optimizer  = optimizer,
            verbose    = verbose,
            inst_label = inst_label,
        )

        push!(all_rows, (
            inst_id              = k,
            inst_label           = inst_label,
            n                    = get(data, "n", missing),
            rho                  = get(data, "rho", missing),

            reduced_status       = res.reduced_status,
            reduced_obj          = res.reduced_obj,

            completion_status    = res.completion_status,
            completion_feasible  = res.completion_feasible,

            full_status          = res.full_status,
            full_obj             = res.full_obj,

            reduced_minus_full   = res.obj_diff,
        ))

        push!(json_out, OrderedDict(
            "inst_id"             => k,
            "inst_label"          => inst_label,
            "n"                   => get(data, "n", nothing),
            "rho"                 => get(data, "rho", nothing),

            "reduced_status"      => res.reduced_status,
            "reduced_obj"         => res.reduced_obj,

            "completion_status"   => res.completion_status,
            "completion_feasible" => res.completion_feasible,

            "full_status"         => res.full_status,
            "full_obj"            => res.full_obj,

            "reduced_minus_full"  => res.obj_diff,
        ))
    end

    CSV.write(out_csv, all_rows; transform=(col,val)->(val===nothing ? missing : val))
    open(out_json, "w") do io
        JSON.print(io, json_out, 2)
    end

    return (csv = out_csv, json = out_json, count = length(insts))
end

export run_batch, analyze_instance

end # module

if isinteractive()

    using MosekTools

    batch = CompProjectionBatch.run_batch(
        "instances.json";
        out_csv   = "comp_projection_results.csv",
        out_json  = "comp_projection_results.json",
        optimizer = MosekTools.Optimizer,
        verbose   = false,
    )

    println(batch)
end

using CSV, DataFrames

df = CSV.read("comp_projection_results.csv", DataFrame)

println(df)
println()
println("total instances = ", nrow(df))
println("completion feasible count = ", sum(df.completion_feasible .== true))
println("completion error count = ", sum(df.completion_feasible .== false))
println("reduced optimal count = ", sum(df.reduced_status .== "OPTIMAL"))
println("full optimal count = ", sum(df.full_status .== "OPTIMAL"))