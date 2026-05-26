###############################
# run_bounds_experiment.jl
#
# Experiment-specific driver: model building + grid runner.
# Instance loading is delegated to InstanceIO.jl (shared file).
#
# Usage (in REPL, after the standard RLTBigM workflow is loaded):
#   julia> include("/home/s2719899/BAM/run_bounds_experiment.jl")
#   julia> run_bounds_experiment("/home/s2719899/BAM/instances.json")
#
# Outputs:
#   - bounds_sensitivity_results.csv
#   - sanity_checks.txt
#   - errors.log
###############################

using JuMP
using MathOptInterface
const MOI = MathOptInterface
using Gurobi
using LinearAlgebra: diag, Diagonal
using Parameters: @unpack
using Printf

# ---- Shared instance loader ----
# Assumes InstanceIO.jl lives in /home/s2719899/BAM/.
# Adjust the path if you placed it elsewhere.
include(joinpath(@__DIR__, "InstanceIO.jl"))

# RLTBigM is expected to be already loaded in the REPL.
const prepare_instance = Main.RLTBigM.prepare_instance


# ============================================================
# Model building helpers
# ============================================================

outer_xy_bs(a, b) = a * transpose(b)

function additional_params_bs(params; M_scale::Real=1.0)
    Mminus_scaled = Diagonal(Float64(M_scale) .* diag(params.Mminus))
    Mplus_scaled  = Diagonal(Float64(M_scale) .* diag(params.Mplus))
    xL  = -diag(Mminus_scaled)
    xU  =  diag(Mplus_scaled)
    eeT = params.e * params.e'
    return merge(params, (; Mminus=Mminus_scaled, Mplus=Mplus_scaled,
                            xL, xU, eeT))
end

function add_FC_comp_bs!(m, x, v, X, V, W, params; bounds_mode::Symbol=:full)
    @unpack n, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, xL, xU = params

    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(m, 0.5 * sum(Qmat[i,j] * X[i,j] for i=1:n, j=1:n) + qvec' * x + rterm <= 0)
    end
    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(m, 0.5 * sum(Pmat[i,j] * X[i,j] for i=1:n, j=1:n) + pvec' * x + sterm == 0)
    end

    if ℓ > 0; @constraint(m, A * x .<= b); end
    if η > 0; @constraint(m, H * x .== h); end

    if bounds_mode == :full
        @constraint(m, xL .<= x)
        @constraint(m, x .<= xU)
    end

    @constraint(m, [i=1:n], W[i,i] == 0)

    if bounds_mode == :full
        @constraint(m, X .- x*xL' .- xL*x' .+ xL*xL' .>= 0)
        @constraint(m, x*xU' .- X .- xL*xU' .+ xL*x' .>= 0)
        @constraint(m, X .- xU*x' .- x*xU' .+ xU*xU' .>= 0)
    end

    if ℓ > 0
        @constraint(m, A*X*A' .- A*(x*b') .- (b*x')*A' .+ (b*b') .>= 0)
    end

    if ℓ > 0 && bounds_mode == :full
        @constraint(m, -A*X .+ (b*x') .+ A*(x*xL') .- (b*xL') .>= 0)
        @constraint(m, A*X .- A*(x*xU') .- (b*x') .+ (b*xU') .>= 0)
    end

    if η > 0
        @constraint(m, H * X .== outer_xy_bs(h, x))
        @constraint(m, H * W .== outer_xy_bs(h, v))
    end
    return nothing
end

function add_FE_v_bs!(m, x, v, X, V, W, params)
    @unpack n, ρ, e = params
    @constraint(m, sum(v) == n - ρ)
    @constraint(m, (e' * V) .== (n - ρ) .* (v'))
    @constraint(m, (e' * W') .== (n - ρ) .* (x'))
    return nothing
end

function add_FB_v_bs!(m, v, V, params)
    @unpack n = params
    @constraint(m, [i=1:n], V[i,i] == v[i])
    return nothing
end

function add_FU_v_bs!(m, x, v, X, V, W, params; bounds_mode::Symbol=:full)
    @unpack n, A, b, ℓ, e, eeT, xL, xU = params

    @constraint(m, 0 .<= v)
    @constraint(m, v .<= e)
    @constraint(m, V .>= 0)
    @constraint(m, v*e' .- V .>= 0)
    @constraint(m, V .- v*e' .- e*v' .+ eeT .>= 0)

    if bounds_mode == :full
        @constraint(m, W .- xL*v' .>= 0)
        @constraint(m, x*e' .- W .- xL*e' .+ xL*v' .>= 0)
        @constraint(m, xU*v' .- W .>= 0)
        @constraint(m, xU*e' .- xU*v' .- x*e' .+ W .>= 0)
    end

    if ℓ > 0
        @constraint(m, b*v' .- A*W .>= 0)
        @constraint(m, b*e' .- (A*x)*e' .- b*v' .+ A*W .>= 0)
    end
    return nothing
end

function build_and_solve_bounds(data; variant::String="RLT_CompBinBou",
                          bounds_mode::Symbol=:full, M_scale::Real=1.0,
                          optimizer=Gurobi.Optimizer, verbose::Bool=false)
    @assert bounds_mode in (:full, :none)

    base_params = prepare_instance(data)
    params      = additional_params_bs(base_params; M_scale=M_scale)
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, e, xL, xU = params

    m = Model(optimizer)
    verbose || set_silent(m)
    try; set_optimizer_attribute(m, "NonConvex", 2); catch; end

    # ---------- EXACT_CompBin ----------
    if variant == "EXACT_CompBin"
        @variable(m, x[1:n])
        @variable(m, v[1:n], Bin)
        @objective(m, Min, 0.5 * x' * Q0 * x + q0' * x)
        if ℓ > 0; @constraint(m, A * x .<= b); end
        if η > 0; @constraint(m, H * x .== h); end
        @constraint(m, xL .<= x)
        @constraint(m, x .<= xU)
        @constraint(m, sum(v) == n - ρ)
        @constraint(m, [i=1:n], x[i] >= xL[i] * (1 - v[i]))
        @constraint(m, [i=1:n], x[i] <= xU[i] * (1 - v[i]))
        for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
            @constraint(m, 0.5 * x' * Qmat * x + qvec' * x + rterm <= 0)
        end
        for (Pmat, pvec, sterm) in zip(Pi, pi, si)
            @constraint(m, 0.5 * x' * Pmat * x + pvec' * x + sterm == 0)
        end
        optimize!(m)
        term = termination_status(m)
        return term == MOI.OPTIMAL ? (:OPTIMAL, objective_value(m)) :
                                     (Symbol(term), nothing)
    end

    # ---------- RLT relaxations ----------
    @variable(m, x[1:n])
    @variable(m, v[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, V[1:n,1:n], Symmetric)
    @variable(m, W[1:n,1:n])

    @objective(m, Min, 0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n) + q0' * x)

    add_FC_comp_bs!(m, x, v, X, V, W, params; bounds_mode=bounds_mode)
    add_FE_v_bs!(m, x, v, X, V, W, params)

    if variant == "RLT_CompBin"
        add_FB_v_bs!(m, v, V, params)
    elseif variant == "RLT_CompBinBou"
        add_FB_v_bs!(m, v, V, params)
        add_FU_v_bs!(m, x, v, X, V, W, params; bounds_mode=bounds_mode)
    elseif variant == "RLT_CompBou"
        add_FU_v_bs!(m, x, v, X, V, W, params; bounds_mode=bounds_mode)
    else
        error("Unknown variant: $variant.")
    end

    optimize!(m)
    term = termination_status(m)
    return term == MOI.OPTIMAL ? (:OPTIMAL, objective_value(m)) :
                                 (Symbol(term), nothing)
end


# ============================================================
# SECTION 3 -- Experiment driver
# ============================================================

const _ERROR_LOG = Ref{String}("errors.log")

function _solve_one(data, variant::String; bounds_mode::Symbol=:full, M_scale::Real=1.0)
    try
        bm = startswith(variant, "EXACT") ? :full : bounds_mode
        return build_and_solve_bounds(data; variant=variant, bounds_mode=bm, M_scale=M_scale)
    catch err
        open(_ERROR_LOG[], "a") do io
            id = get(data, "id", "unknown")
            println(io, "=" ^ 70)
            println(io, "instance_id = $id")
            println(io, "variant     = $variant")
            println(io, "bounds_mode = $bounds_mode")
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

"""
    run_bounds_experiment(json_path; ...)

Solve 13 models (1 EXACT + 3 RLT variants x 4 bounds configurations)
for each instance in `json_path`. Writes three output files:
CSV with all objective values, sanity_checks.txt, errors.log.
"""
function run_bounds_experiment(json_path::AbstractString;
                                out_csv::AbstractString="bounds_sensitivity_results.csv",
                                out_sanity::AbstractString="sanity_checks.txt",
                                out_errors::AbstractString="errors.log",
                                instance_indices=nothing)

    _ERROR_LOG[] = out_errors
    open(out_errors, "w") do io
        println(io, "Bounds sensitivity driver -- error log")
        println(io)
    end

    all_instances = load_instances(json_path)
    indices = instance_indices === nothing ? (1:length(all_instances)) : instance_indices
    println("Loaded ", length(all_instances), " instances; running ",
            length(indices), " of them.")

    rlt_variants = ["RLT_CompBin", "RLT_CompBinBou", "RLT_CompBou"]
    configs      = [(:none, 1.0), (:full, 1.0), (:full, 10.0), (:full, 100.0)]
    config_tags  = ["nobounds", "M1", "M10", "M100"]

    header = String["id", "n", "rho", "convexity", "base_tag", "EXACT", "EXACT_status"]
    for var in rlt_variants
        short = replace(var, "RLT_" => "")
        for tag in config_tags
            push!(header, "$(short)_$(tag)")
            push!(header, "$(short)_$(tag)_status")
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

        (ex_status, ex_obj) = _solve_one(data, "EXACT_CompBin")
        ex_str = ex_obj === nothing ? "NA" : @sprintf("%.6e", ex_obj)
        println("  EXACT_CompBin    -> $ex_status  obj=$ex_str")

        row = Any[id, n, rho, convexity, base_tag,
                  ex_obj === nothing ? "NA" : ex_obj, String(ex_status)]

        per_inst = Dict{Tuple{String,String}, Union{Float64,Nothing}}()
        for var in rlt_variants
            short = replace(var, "RLT_" => "")
            for (cfg, tag) in zip(configs, config_tags)
                (bm, ms) = cfg
                (st, ob) = _solve_one(data, var; bounds_mode=bm, M_scale=ms)
                ob_str = ob === nothing ? "NA" : @sprintf("%.6e", ob)
                println("  $(rpad(var,16)) $(rpad(tag,9)) -> $st  obj=$ob_str")
                push!(row, ob === nothing ? "NA" : ob)
                push!(row, String(st))
                per_inst[(short, tag)] = ob
            end
        end
        push!(rows, row)

        # sanity checks
        tol = 1e-6
        push!(sanity_lines, "==== $id ====")
        for tag in config_tags
            a = per_inst[("CompBinBou", tag)]; b = per_inst[("CompBou", tag)]
            if a !== nothing && b !== nothing
                d = abs(a - b); ok = d <= tol * max(1.0, abs(a), abs(b))
                push!(sanity_lines,
                    @sprintf("  Proposition (CompBinBou=CompBou) @%-8s : diff=%.3e  %s",
                            tag, d, ok ? "OK" : "FAIL"))
            end
        end
        for tag in config_tags
            cbin = per_inst[("CompBin", tag)]; cbb = per_inst[("CompBinBou", tag)]
            if cbin !== nothing && cbb !== nothing
                ok = cbin <= cbb + tol
                push!(sanity_lines,
                    @sprintf("  CompBin<=CompBinBou      @%-8s : %.6e <= %.6e  %s",
                            tag, cbin, cbb, ok ? "OK" : "FAIL"))
            end
        end
        if ex_obj !== nothing
            for var_short in ("CompBin", "CompBinBou", "CompBou"), tag in config_tags
                rv = per_inst[(var_short, tag)]
                if rv !== nothing
                    ok = rv <= ex_obj + tol
                    push!(sanity_lines,
                        @sprintf("  %s_%s<=EXACT          : %.6e <= %.6e  %s",
                                var_short, tag, rv, ex_obj, ok ? "OK" : "FAIL"))
                end
            end
        end
        for var_short in ("CompBin", "CompBinBou", "CompBou")
            vals = [per_inst[(var_short, t)] for t in ("nobounds","M100","M10","M1")]
            if all(v -> v !== nothing, vals)
                ok_chain = true
                for k in 1:length(vals)-1
                    if !(vals[k] <= vals[k+1] + tol); ok_chain = false; break; end
                end
                push!(sanity_lines,
                    @sprintf("  M-monotonicity %-11s : %.6e <= %.6e <= %.6e <= %.6e  %s",
                            var_short, vals[1], vals[2], vals[3], vals[4],
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