###############################
# Per-instance model table:
#   RLT:      BigM, BigMBou, CompBin, CompBinBou, CompBou
#   SDP:      BigM (=BigMBou), CompBin (=CompBinBou), CompBou
#   RLT+SDP:  BigM, BigMBou, CompBin, CompBinBou, CompBou
###############################

module SQCQP_LevelTable

using JSON, CSV
using JuMP, MosekTools, LinearAlgebra
using OrderedCollections: OrderedDict
using MathOptInterface
const MOI = MathOptInterface

# Reuse read_instance! (no duplication)
using Main.RLT_SDP_Batch: read_instance!

# BigM / Comp model builders
using Main.RLTBigM
using Main.RLTComp
using Main.SDPComplementarity
using Main.RLT_SDP_Combo
using Main.RLT_SDP_Comp_Combo

# ------------------------
# small formatting helpers
# ------------------------
_is_good_status(st::MOI.TerminationStatusCode) =
    st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED)

_vec_json(x) = x === nothing ? missing : JSON.json(x)

function _get_val(m::Model, sym::Symbol)
    haskey(m, sym) || return nothing
    return value.(m[sym])
end

# ------------------------
# SDP-only BigM model (single model = BigM (=BigMBou))
# ------------------------
function build_SDP_BigM_model(data::Dict{String,Any};
                              optimizer = MosekTools.Optimizer,
                              verbose::Bool=false)

    params = Main.RLTBigM.prepare_instance(data)
    n   = params.n
    ρ   = params.ρ
    Q0  = params.Q0
    q0  = params.q0
    Qi  = params.Qi
    qi  = params.qi
    ri  = params.ri
    Pi  = params.Pi
    pi  = params.pi
    si  = params.si
    A   = params.A
    b   = params.b
    ℓ   = params.ℓ
    H   = params.H
    h   = params.h
    η   = params.η
    Mminus = params.Mminus
    Mplus  = params.Mplus

    m = Model(optimizer)
    verbose ? nothing : set_silent(m)

    @variable(m, x[1:n])
    @variable(m, 0 <= u[1:n] <= 1)              # u<=1 is redundant under SDP, but safe
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, R[1:n,1:n])
    @variable(m, U[1:n,1:n], Symmetric)

    # objective
    @objective(m, Min, 0.5 * sum(Q0[i,j]*X[i,j] for i=1:n, j=1:n) + q0' * x)

    # lifted QCQP rows
    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(m, 0.5 * sum(Qmat[i,j]*X[i,j] for i=1:n, j=1:n) + qvec' * x + rterm <= 0)
    end
    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(m, 0.5 * sum(Pmat[i,j]*X[i,j] for i=1:n, j=1:n) + pvec' * x + sterm == 0)
    end

    # original linear rows
    if ℓ > 0; @constraint(m, A * x .<= b); end
    if η > 0; @constraint(m, H * x .== h); end

    # Big-M links
    @constraint(m, -Mminus * u .<= x)
    @constraint(m,  x .<= Mplus  * u)

    # cardinality (E-level): sum u = ρ
    @constraint(m, sum(u) == ρ)

    # diag(U)=u
    @constraint(m, [i=1:n], U[i,i] == u[i])

    # SDP block
    @constraint(m, Symmetric([1.0 x' u'; x X R; u R' U]) in PSDCone())

    return m
end

# ------------------------
# generic “build+solve+extract x/u/v”
# ------------------------
function _solve_model_build_only(build_model_fn;
                                optimizer,
                                verbose::Bool=false,
                                want_u::Bool=false,
                                want_v::Bool=false)

    m = build_model_fn(optimizer=optimizer, verbose=verbose)
    t0 = time_ns()
    optimize!(m)
    t  = round((time_ns() - t0)/1e9; digits=4)

    st = termination_status(m)
    if !_is_good_status(st)
        return (string(st), missing, missing, missing, missing, t)
    end

    obj = objective_value(m)
    x   = _get_val(m, :x)
    u   = want_u ? _get_val(m, :u) : nothing
    v   = want_v ? _get_val(m, :v) : nothing
    return (string(st), obj, x, u, v, t)
end

# ------------------------
# Solve one instance: all 13 models
# ------------------------
function solve_instance_table(data::Dict{String,Any};
                              optimizer_lp,
                              optimizer_sdp = MosekTools.Optimizer,
                              verbose::Bool=false)

    inst_label = get(data, "id", "(no id)")

    rows = NamedTuple[]
    json_rows = OrderedDict[]

    # ---------- RLT (5) ----------
    # BigM  -> RLTBigM variant "E"
    # BigMBou -> RLTBigM variant "EU"
    for (name, var) in (("BigM","E"), ("BigMBou","EU"))
        buildfn = ( ; optimizer, verbose=false) -> Main.RLTBigM.build_and_solve(
            data; variant=var, build_only=true, optimizer=optimizer, verbose=verbose
        )
        st,obj,x,u,_,t = _solve_model_build_only(buildfn;
            optimizer=optimizer_lp, verbose=verbose, want_u=true, want_v=false)

        push!(rows, (inst_label=inst_label, level="RLT", model=name, status=st, obj=obj, time_sec=t,
                     x=_vec_json(x), u=_vec_json(u), v=missing))
        push!(json_rows, OrderedDict("inst_label"=>inst_label, "level"=>"RLT", "model"=>name,
                                     "status"=>st, "obj"=>obj, "time_sec"=>t,
                                     "x"=>x, "u"=>u, "v"=>nothing))
    end

    # CompBin / CompBinBou / CompBou under RLT
    for (name, var) in (("CompBin","RLT_CompBin"),
                        ("CompBinBou","RLT_CompBinBou"),
                        ("CompBou","RLT_CompBou"))

        buildfn = ( ; optimizer, verbose=false) -> Main.RLTComp.build_and_solve(
            data; variant=var, build_only=true, optimizer=optimizer, verbose=verbose
        )
        st,obj,x,_,v,t = _solve_model_build_only(buildfn;
            optimizer=optimizer_lp, verbose=verbose, want_u=false, want_v=true)

        push!(rows, (inst_label=inst_label, level="RLT", model=name, status=st, obj=obj, time_sec=t,
                     x=_vec_json(x), u=missing, v=_vec_json(v)))
        push!(json_rows, OrderedDict("inst_label"=>inst_label, "level"=>"RLT", "model"=>name,
                                     "status"=>st, "obj"=>obj, "time_sec"=>t,
                                     "x"=>x, "u"=>nothing, "v"=>v))
    end

    # ---------- SDP-only (3) ----------
    # BigM (=BigMBou)
    begin
        buildfn = ( ; optimizer, verbose=false) -> build_SDP_BigM_model(
            data; optimizer=optimizer, verbose=verbose
        )
        st,obj,x,u,_,t = _solve_model_build_only(buildfn;
            optimizer=optimizer_sdp, verbose=verbose, want_u=true, want_v=false)

        name = "BigM (=BigMBou)"
        push!(rows, (inst_label=inst_label, level="SDP", model=name, status=st, obj=obj, time_sec=t,
                     x=_vec_json(x), u=_vec_json(u), v=missing))
        push!(json_rows, OrderedDict("inst_label"=>inst_label, "level"=>"SDP", "model"=>name,
                                     "status"=>st, "obj"=>obj, "time_sec"=>t,
                                     "x"=>x, "u"=>u, "v"=>nothing))
    end

    # CompBin (=CompBinBou) and CompBou under SDP
    for (name, var) in (("CompBin (=CompBinBou)","SDP_CompBin"),
                        ("CompBou","SDP_CompBou"))

        buildfn = ( ; optimizer, verbose=false) -> Main.SDPComplementarity.build_SDP_comp_model(
            data; variant=var, optimizer=optimizer, verbose=verbose
        )
        st,obj,x,_,v,t = _solve_model_build_only(buildfn;
            optimizer=optimizer_sdp, verbose=verbose, want_u=false, want_v=true)

        push!(rows, (inst_label=inst_label, level="SDP", model=name, status=st, obj=obj, time_sec=t,
                     x=_vec_json(x), u=missing, v=_vec_json(v)))
        push!(json_rows, OrderedDict("inst_label"=>inst_label, "level"=>"SDP", "model"=>name,
                                     "status"=>st, "obj"=>obj, "time_sec"=>t,
                                     "x"=>x, "u"=>nothing, "v"=>v))
    end

    # ---------- RLT+SDP (5) ----------
    # BigM / BigMBou with relaxation=:RLT_full_SDP
    for (name, var) in (("BigM","E"), ("BigMBou","EU"))
        t0 = time_ns()
        st,obj,x,u,_,_,_,t = Main.RLT_SDP_Combo.solve_RLT_SDP(
            data; variant=var, optimizer=optimizer_sdp, relaxation=:RLT_full_SDP
        )
        # solve_RLT_SDP already returns t, but keep consistent:
        # (we trust returned t)
        st_str = string(st)
        x_val  = x
        u_val  = u
        obj_val = (obj === nothing ? missing : obj)

        push!(rows, (inst_label=inst_label, level="RLT+SDP", model=name, status=st_str, obj=obj_val, time_sec=t,
                     x=_vec_json(x_val), u=_vec_json(u_val), v=missing))
        push!(json_rows, OrderedDict("inst_label"=>inst_label, "level"=>"RLT+SDP", "model"=>name,
                                     "status"=>st_str, "obj"=>obj_val, "time_sec"=>t,
                                     "x"=>x_val, "u"=>u_val, "v"=>nothing))
    end

    # CompBin / CompBinBou / CompBou with relaxation=:RLT_full_SDP
    for (name, var) in (("CompBin","RLT_CompBin"),
                        ("CompBinBou","RLT_CompBinBou"),
                        ("CompBou","RLT_CompBou"))

        st,obj,x,v,_,_,_,t = Main.RLT_SDP_Comp_Combo.solve_RLT_SDP_comp(
            data; variant=var, optimizer=optimizer_sdp, relaxation=:RLT_full_SDP
        )
        st_str  = string(st)
        obj_val = (obj === nothing ? missing : obj)

        push!(rows, (inst_label=inst_label, level="RLT+SDP", model=name, status=st_str, obj=obj_val, time_sec=t,
                     x=_vec_json(x), u=missing, v=_vec_json(v)))
        push!(json_rows, OrderedDict("inst_label"=>inst_label, "level"=>"RLT+SDP", "model"=>name,
                                     "status"=>st_str, "obj"=>obj_val, "time_sec"=>t,
                                     "x"=>x, "u"=>nothing, "v"=>v))
    end

    return rows, json_rows
end

# ------------------------
# Run on instances.json
# ------------------------
function run_level_table(instances_json::AbstractString;
                         out_csv::AbstractString="level_table.csv",
                         out_json::AbstractString="level_table.json",
                         optimizer_lp,
                         optimizer_sdp = MosekTools.Optimizer,
                         verbose::Bool=false)

    raw = JSON.parsefile(instances_json)
    insts = [read_instance!(deepcopy(d)) for d in raw]

    all_rows = NamedTuple[]
    all_json = OrderedDict[]

    for (k, data) in enumerate(insts)
        inst_label = get(data, "id", "inst_$(k)")

        println("\n==============================")
        println("instance $k : $inst_label")
        println("==============================")

        rows, json_rows = solve_instance_table(
            data;
            optimizer_lp=optimizer_lp,
            optimizer_sdp=optimizer_sdp,
            verbose=verbose
        )

        append!(all_rows, rows)
        push!(all_json, OrderedDict(
            "inst_id"    => k,
            "inst_label" => inst_label,
            "results"    => json_rows
        ))
    end

    CSV.write(out_csv, all_rows)
    open(out_json, "w") do io
        JSON.print(io, all_json, 2)
    end

    return (csv=out_csv, json=out_json, count=length(insts))
end

export run_level_table, solve_instance_table, build_SDP_BigM_model

end # module SQCQP_LevelTable

using MosekTools
using Gurobi
using .SQCQP_LevelTable

res = SQCQP_LevelTable.run_level_table(
    "instances.json";
    out_csv="level_table.csv",
    out_json="level_table.json",
    optimizer_lp=Gurobi.Optimizer,
    optimizer_sdp=MosekTools.Optimizer,
)
println(res)