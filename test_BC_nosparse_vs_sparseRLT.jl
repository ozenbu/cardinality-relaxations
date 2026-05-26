# ============================================================
# test_BC_nosparse_vs_sparseRLT.jl
# ============================================================
#
# Test B:
#   Objective structure:
#       original
#       linear
#       diagquad
#       allsumquad
#
# Test C:
#   Bound structure:
#       explicit_nonnegative
#       two_sided_or_symmetric
#       mixed_zero_allowed
#       invalid_for_sparsity_benchmark
#
# Main comparisons:
#   z_CompBinBou - z_NoSparseRLT
#   z_BigMBou    - z_NoSparseRLT
#   z_BigMBou    - z_CompBinBou
#
# Minimization convention:
#   Larger relaxation objective value = tighter lower bound.
#
# ============================================================

using Gurobi
using JuMP
using JSON
using CSV
using DataFrames
using LinearAlgebra
using Printf
using Statistics
using Dates

const MOI = JuMP.MOI
const ROOT = @__DIR__

include(normpath(joinpath(ROOT, "bigM", "RLT_BigM.jl")))
using .RLTBigM

include(normpath(joinpath(ROOT, "complementarity", "RLT_complementarity.jl")))
using .RLTComp

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

const OBJECTIVE_MODES = [
    "original",
    "linear",
    "diagquad",
    "allsumquad",
]

const COMP_VARIANTS = [
    "RLT_CompBin",
    "RLT_CompBinBou",
    "RLT_CompBou",
]

const GOOD_STATUS = Set([
    :OPTIMAL,
    :ALMOST_OPTIMAL,
    :LOCALLY_SOLVED,
])

# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------

function fmt(x)
    if x === missing
        return "missing"
    elseif x === nothing
        return "nothing"
    elseif x isa Real
        return @sprintf("%.9g", Float64(x))
    else
        return string(x)
    end
end

function status_good(status)
    return Symbol(status) in GOOD_STATUS
end

function safe_obj(status, obj)
    if status_good(status) && obj !== missing && obj !== nothing
        return Float64(obj)
    else
        return missing
    end
end

function gap(a, b)
    return (a === missing || b === missing) ? missing : Float64(a) - Float64(b)
end

function is_good_number(x)
    return !(x === missing) && !(x === nothing) && isfinite(Float64(x))
end

function json_matrix(x)
    x === nothing && return nothing

    nr = length(x)
    nr == 0 && return zeros(Float64, 0, 0)

    nc = length(x[1])
    M = zeros(Float64, nr, nc)

    for i in 1:nr
        @assert length(x[i]) == nc "Inconsistent JSON matrix row length."
        for j in 1:nc
            M[i, j] = Float64(x[i][j])
        end
    end

    return M
end

function json_vector(x)
    x === nothing && return nothing
    return Float64.(collect(x))
end

function orient_matrix_with_ncols(x, n::Int; name::String = "matrix")
    x === nothing && return nothing

    M = json_matrix(x)

    if size(M, 2) == n
        return M
    elseif size(M, 1) == n
        return Matrix(M')
    else
        error("$name has size $(size(M)), cannot orient it with n = $n.")
    end
end

function orient_square_matrix(x, n::Int; name::String = "square matrix")
    x === nothing && return nothing

    M = json_matrix(x)

    if size(M, 1) == n && size(M, 2) == n
        return 0.5 .* (M .+ M')
    else
        error("$name has size $(size(M)), expected $n × $n.")
    end
end

function orient_matrix_list(x, n::Int; name::String = "matrix list")
    x === nothing && return nothing

    mats = Matrix{Float64}[]
    for (k, item) in enumerate(x)
        push!(mats, orient_square_matrix(item, n; name = "$name[$k]"))
    end

    return isempty(mats) ? nothing : Tuple(mats)
end

function orient_vector_list(x, n::Int; name::String = "vector list")
    x === nothing && return nothing

    vecs = Vector{Float64}[]
    for (k, item) in enumerate(x)
        v = Float64.(collect(item))
        @assert length(v) == n "$name[$k] has length $(length(v)), expected $n."
        push!(vecs, v)
    end

    return isempty(vecs) ? nothing : Tuple(vecs)
end

function fix_json_instance(raw::Dict{String,Any})
    inst = deepcopy(raw)
    n = Int(inst["n"])

    inst["rho"] = Float64(inst["rho"])

    inst["Q0"] = orient_square_matrix(inst["Q0"], n; name = "Q0")
    inst["q0"] = json_vector(inst["q0"])

    inst["Qi"] = orient_matrix_list(get(inst, "Qi", nothing), n; name = "Qi")
    inst["qi"] = orient_vector_list(get(inst, "qi", nothing), n; name = "qi")
    inst["ri"] = json_vector(get(inst, "ri", nothing))

    inst["Pi"] = orient_matrix_list(get(inst, "Pi", nothing), n; name = "Pi")
    inst["pi"] = orient_vector_list(get(inst, "pi", nothing), n; name = "pi")
    inst["si"] = json_vector(get(inst, "si", nothing))

    inst["A"] = orient_matrix_with_ncols(get(inst, "A", nothing), n; name = "A")
    inst["b"] = json_vector(get(inst, "b", nothing))

    inst["H"] = orient_matrix_with_ncols(get(inst, "H", nothing), n; name = "H")
    inst["h"] = json_vector(get(inst, "h", nothing))

    inst["ell"] = json_vector(get(inst, "ell", nothing))
    inst["u"] = json_vector(get(inst, "u", nothing))
    inst["M"] = json_vector(get(inst, "M", nothing))
    inst["M_minus"] = json_vector(get(inst, "M_minus", nothing))
    inst["M_plus"] = json_vector(get(inst, "M_plus", nothing))

    return inst
end

function read_instances(path::AbstractString)
    raw = JSON.parsefile(path)
    return [fix_json_instance(Dict{String,Any}(item)) for item in raw]
end

# ------------------------------------------------------------
# Bound category for Test C
# ------------------------------------------------------------

function bound_category(inst; tol::Float64 = 1e-8)
    ell = Vector{Float64}(inst["ell"])
    u = Vector{Float64}(inst["u"])
    Mminus = Vector{Float64}(inst["M_minus"])

    if any(ell .> tol) || any(u .< -tol)
        return "invalid_for_sparsity_benchmark"
    end

    if all(abs.(ell) .<= tol) && all(abs.(Mminus) .<= tol) && all(u .>= -tol)
        return "explicit_nonnegative"
    end

    if all(ell .< -tol) && all(u .> tol)
        return "two_sided_or_symmetric"
    end

    return "mixed_zero_allowed"
end

function validate_zero_allowed_instances(instances; tol::Float64 = 1e-8)
    bad = []

    for inst in instances
        ell = Vector{Float64}(inst["ell"])
        u = Vector{Float64}(inst["u"])
        bad_idx = findall(i -> ell[i] > tol || u[i] < -tol, eachindex(ell))

        if !isempty(bad_idx)
            push!(bad, (id = get(inst, "id", "unknown"), bad_idx = bad_idx, ell = ell, u = u))
        end
    end

    if !isempty(bad)
        println("\nWARNING: Some instances are invalid for sparsity benchmark.")
        for item in bad
            println("\nInstance: ", item.id)
            for i in item.bad_idx
                println("  x[$i] in [", item.ell[i], ", ", item.u[i], "]")
            end
        end
    else
        println("All instances satisfy ell_i <= 0 <= u_i.")
    end

    return bad
end

# ------------------------------------------------------------
# Objective variants for Test B
# ------------------------------------------------------------

function objective_variant(inst::Dict{String,Any}, mode::String)
    cand = deepcopy(inst)
    n = Int(cand["n"])

    if mode == "original"
        cand["objective_mode"] = "original"

    elseif mode == "linear"
        # min -e^T x
        cand["Q0"] = zeros(Float64, n, n)
        cand["q0"] = -ones(Float64, n)
        cand["objective_mode"] = "linear"

    elseif mode == "diagquad"
        # min -sum_i X_ii
        # Since objective is 0.5 <Q0, X>, use Q0 = -2I.
        cand["Q0"] = -2.0 .* Matrix{Float64}(I, n, n)
        cand["q0"] = zeros(Float64, n)
        cand["objective_mode"] = "diagquad"

    elseif mode == "allsumquad"
        # min -sum_{i,j} X_ij
        # Since objective is 0.5 <Q0, X>, use Q0 = -2 ee^T.
        cand["Q0"] = -2.0 .* ones(Float64, n, n)
        cand["q0"] = zeros(Float64, n)
        cand["objective_mode"] = "allsumquad"

    else
        error("Unknown objective mode: $mode")
    end

    cand["id"] = string(get(inst, "id", "unknown"), "__", mode)

    return cand
end

# ------------------------------------------------------------
# No-sparsity RLT solver
# ------------------------------------------------------------

function affine_AX_expr(A, x, row::Int)
    n = length(x)
    return sum(A[row, i] * x[i] for i in 1:n)
end

function AX_col_expr(A, X, row::Int, col::Int)
    n = size(X, 1)
    return sum(A[row, i] * X[i, col] for i in 1:n)
end

function AXAT_expr(A, X, row1::Int, row2::Int)
    n = size(X, 1)
    return sum(A[row1, i] * A[row2, j] * X[i, j] for i in 1:n, j in 1:n)
end

function HX_col_expr(H, X, row::Int, col::Int)
    n = size(X, 1)
    return sum(H[row, i] * X[i, col] for i in 1:n)
end

function solve_no_sparse_rlt(inst;
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
)
    try
        n = Int(inst["n"])

        Q0 = Matrix{Float64}(inst["Q0"])
        q0 = Vector{Float64}(inst["q0"])

        Qi = get(inst, "Qi", nothing)
        qi = get(inst, "qi", nothing)
        ri = get(inst, "ri", nothing)

        Pi = get(inst, "Pi", nothing)
        pi = get(inst, "pi", nothing)
        si = get(inst, "si", nothing)

        A = get(inst, "A", nothing)
        b = get(inst, "b", nothing)

        H = get(inst, "H", nothing)
        h = get(inst, "h", nothing)

        ell = Vector{Float64}(inst["ell"])
        u = Vector{Float64}(inst["u"])

        A = A === nothing ? zeros(Float64, 0, n) : Matrix{Float64}(A)
        b = b === nothing ? Float64[] : Vector{Float64}(b)

        H = H === nothing ? zeros(Float64, 0, n) : Matrix{Float64}(H)
        h = h === nothing ? Float64[] : Vector{Float64}(h)

        mA = size(A, 1)
        mH = size(H, 1)

        model = Model(optimizer)

        if !verbose
            set_silent(model)
        else
            try
                set_optimizer_attribute(model, "OutputFlag", 1)
            catch
            end
        end

        set_name(model, "QCQP_RLT_NoSparsity")

        @variable(model, x[1:n])
        @variable(model, X[1:n, 1:n], Symmetric)

        @objective(
            model,
            Min,
            0.5 * sum(Q0[i, j] * X[i, j] for i in 1:n, j in 1:n) +
            sum(q0[i] * x[i] for i in 1:n)
        )

        # Original quadratic inequalities, lifted by X
        if Qi !== nothing
            for k in eachindex(Qi)
                Q = Matrix{Float64}(Qi[k])
                q = Vector{Float64}(qi[k])
                r = Float64(ri[k])

                @constraint(
                    model,
                    0.5 * sum(Q[i, j] * X[i, j] for i in 1:n, j in 1:n) +
                    sum(q[i] * x[i] for i in 1:n) + r <= 0.0
                )
            end
        end

        # Original quadratic equalities, lifted by X
        if Pi !== nothing
            for k in eachindex(Pi)
                P = Matrix{Float64}(Pi[k])
                p = Vector{Float64}(pi[k])
                s = Float64(si[k])

                @constraint(
                    model,
                    0.5 * sum(P[i, j] * X[i, j] for i in 1:n, j in 1:n) +
                    sum(p[i] * x[i] for i in 1:n) + s == 0.0
                )
            end
        end

        # Original linear rows
        if mA > 0
            @constraint(model, A * x .<= b)
        end

        if mH > 0
            @constraint(model, H * x .== h)
        end

        # Variable bounds
        @constraint(model, x .>= ell)
        @constraint(model, x .<= u)

        # ----------------------------------------------------
        # RLT constraints from linear inequalities Ax <= b
        # ----------------------------------------------------

        # (b - Ax)(b - Ax)^T >= 0
        for r in 1:mA, s in 1:mA
            @constraint(
                model,
                AXAT_expr(A, X, r, s)
                - affine_AX_expr(A, x, r) * b[s]
                - b[r] * affine_AX_expr(A, x, s)
                + b[r] * b[s] >= 0.0
            )
        end

        # ----------------------------------------------------
        # RLT constraints from variable bounds
        # ----------------------------------------------------

        # (x - ell)(x - ell)^T >= 0
        for i in 1:n, j in 1:n
            @constraint(
                model,
                X[i, j] - x[i] * ell[j] - ell[i] * x[j] + ell[i] * ell[j] >= 0.0
            )
        end

        # (u - x)(u - x)^T >= 0
        for i in 1:n, j in 1:n
            @constraint(
                model,
                X[i, j] - x[i] * u[j] - u[i] * x[j] + u[i] * u[j] >= 0.0
            )
        end

        # (x - ell)(u - x)^T >= 0
        for i in 1:n, j in 1:n
            @constraint(
                model,
                -X[i, j] + x[i] * u[j] + ell[i] * x[j] - ell[i] * u[j] >= 0.0
            )
        end

        # ----------------------------------------------------
        # Cross RLT: (b - Ax)(x - ell)^T >= 0
        # ----------------------------------------------------
        for r in 1:mA, j in 1:n
            @constraint(
                model,
                -AX_col_expr(A, X, r, j)
                + b[r] * x[j]
                + affine_AX_expr(A, x, r) * ell[j]
                - b[r] * ell[j] >= 0.0
            )
        end

        # ----------------------------------------------------
        # Cross RLT: (b - Ax)(u - x)^T >= 0
        # ----------------------------------------------------
        for r in 1:mA, j in 1:n
            @constraint(
                model,
                AX_col_expr(A, X, r, j)
                - affine_AX_expr(A, x, r) * u[j]
                - b[r] * x[j]
                + b[r] * u[j] >= 0.0
            )
        end

        # Equality-products: H X = h x^T
        for r in 1:mH, j in 1:n
            @constraint(model, HX_col_expr(H, X, r, j) == h[r] * x[j])
        end

        optimize!(model)

        term = termination_status(model)

        if term == MOI.OPTIMAL
            return (;
                status = :OPTIMAL,
                obj = objective_value(model),
                error_msg = "",
            )
        else
            return (;
                status = Symbol(term),
                obj = missing,
                error_msg = "",
            )
        end
    catch err
        return (;
            status = :ERROR,
            obj = missing,
            error_msg = sprint(showerror, err),
        )
    end
end

# ------------------------------------------------------------
# Existing sparse RLT solvers
# ------------------------------------------------------------
function solve_bigm_full(inst;
    variant::String,
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
)
    try
        res = RLTBigM.build_and_solve(
            inst;
            variant = variant,
            optimizer = optimizer,
            verbose = verbose,
        )

        status = res[1]
        obj = status == :OPTIMAL ? Float64(res[2]) : missing

        return (;
            status,
            obj,
            error_msg = "",
        )
    catch err
        return (;
            status = :ERROR,
            obj = missing,
            error_msg = sprint(showerror, err),
        )
    end
end

function solve_comp_full(inst, variant::String;
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
)
    try
        res = RLTComp.build_and_solve(
            inst;
            variant = variant,
            optimizer = optimizer,
            verbose = verbose,
        )

        status = res[1]
        obj = status == :OPTIMAL ? Float64(res[2]) : missing

        return (;
            status,
            obj,
            error_msg = "",
        )
    catch err
        return (;
            status = :ERROR,
            obj = missing,
            error_msg = sprint(showerror, err),
        )
    end
end
# ------------------------------------------------------------
# One-case runner
# ------------------------------------------------------------

function solve_one_case(inst;
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
)
    nosparse = solve_no_sparse_rlt(
        inst;
        optimizer = optimizer,
        verbose = verbose,
    )

    bigm = solve_bigm_full(
        inst;
        variant = "E",      # RLT_BigM
        optimizer = optimizer,
        verbose = verbose,
    )

    bigmbou = solve_bigm_full(
        inst;
        variant = "EU",     # RLT_BigMBou
        optimizer = optimizer,
        verbose = verbose,
    )

    comp_results = Dict{String,Any}()

    for var in COMP_VARIANTS
        comp_results[var] = solve_comp_full(
            inst,
            var;
            optimizer = optimizer,
            verbose = verbose,
        )
    end

    return nosparse, bigm, bigmbou, comp_results
end

function result_row(base_inst, test_inst, nosparse, bigm, bigmbou, comp_results)
    compbin = comp_results["RLT_CompBin"]
    compbinbou = comp_results["RLT_CompBinBou"]
    compbou = comp_results["RLT_CompBou"]

    z_nosparse = nosparse.obj
    z_bigm = bigm.obj
    z_bigmbou = bigmbou.obj
    z_compbin = compbin.obj
    z_compbinbou = compbinbou.obj
    z_compbou = compbou.obj

    return Dict(
        "inst_id" => get(base_inst, "id", "unknown"),
        "test_id" => get(test_inst, "id", "unknown"),
        "objective_mode" => get(test_inst, "objective_mode", "unknown"),
        "bound_category" => bound_category(base_inst),
        "n" => Int(base_inst["n"]),
        "rho" => Float64(base_inst["rho"]),
        "base_tag" => get(base_inst, "base_tag", "unknown"),
        "convexity" => get(base_inst, "convexity", "unknown"),

        "status_NoSparseRLT" => string(nosparse.status),
        "status_RLT_BigM" => string(bigm.status),
        "status_RLT_BigMBou" => string(bigmbou.status),
        "status_RLT_CompBin" => string(compbin.status),
        "status_RLT_CompBinBou" => string(compbinbou.status),
        "status_RLT_CompBou" => string(compbou.status),

        "z_NoSparseRLT" => z_nosparse,
        "z_RLT_BigM" => z_bigm,
        "z_RLT_BigMBou" => z_bigmbou,
        "z_RLT_CompBin" => z_compbin,
        "z_RLT_CompBinBou" => z_compbinbou,
        "z_RLT_CompBou" => z_compbou,

        "gap_CompBinBou_minus_NoSparse" => gap(z_compbinbou, z_nosparse),
        "gap_CompBou_minus_NoSparse" => gap(z_compbou, z_nosparse),
        "gap_BigMBou_minus_NoSparse" => gap(z_bigmbou, z_nosparse),
        "gap_BigM_minus_NoSparse" => gap(z_bigm, z_nosparse),

        "gap_BigMBou_minus_CompBinBou" => gap(z_bigmbou, z_compbinbou),
        "gap_CompBinBou_minus_BigMBou" => gap(z_compbinbou, z_bigmbou),

        "gap_CompBinBou_minus_CompBin" => gap(z_compbinbou, z_compbin),
        "gap_CompBou_minus_CompBin" => gap(z_compbou, z_compbin),
        "gap_CompBinBou_minus_CompBou" => gap(z_compbinbou, z_compbou),

        "err_NoSparseRLT" => nosparse.error_msg,
        "err_RLT_BigM" => bigm.error_msg,
        "err_RLT_BigMBou" => bigmbou.error_msg,
        "err_RLT_CompBin" => compbin.error_msg,
        "err_RLT_CompBinBou" => compbinbou.error_msg,
        "err_RLT_CompBou" => compbou.error_msg,
    )
end

# ------------------------------------------------------------
# Summary helpers
# ------------------------------------------------------------

function values_skip_missing(v)
    vals = Float64[]
    for x in v
        if is_good_number(x)
            push!(vals, Float64(x))
        end
    end
    return vals
end

function mean_skip_missing(v)
    vals = values_skip_missing(v)
    return isempty(vals) ? missing : mean(vals)
end

function min_skip_missing(v)
    vals = values_skip_missing(v)
    return isempty(vals) ? missing : minimum(vals)
end

function max_skip_missing(v)
    vals = values_skip_missing(v)
    return isempty(vals) ? missing : maximum(vals)
end

function count_gt(v, tau)
    c = 0
    for x in v
        if is_good_number(x) && Float64(x) > tau
            c += 1
        end
    end
    return c
end

function count_abs_le(v, tau)
    c = 0
    for x in v
        if is_good_number(x) && abs(Float64(x)) <= tau
            c += 1
        end
    end
    return c
end

function count_ge_minus_tol(v, tau)
    c = 0
    for x in v
        if is_good_number(x) && Float64(x) >= -tau
            c += 1
        end
    end
    return c
end

function mask_gt(col, tau)
    return [is_good_number(x) && Float64(x) > tau for x in col]
end

function print_summary(df::DataFrame; tau::Float64 = 1e-6)
    println()
    println("============================================================")
    println("Summary by objective mode and bound category")
    println("Tolerance tau = ", tau)
    println("============================================================")

    gdf = groupby(df, [:objective_mode, :bound_category])

    for sub in gdf
        mode = sub.objective_mode[1]
        cat = sub.bound_category[1]
        ncases = nrow(sub)

        gap_comp = sub.gap_CompBinBou_minus_NoSparse
        gap_bigm = sub.gap_BigMBou_minus_NoSparse
        gap_bigmbou_comp = sub.gap_BigMBou_minus_CompBinBou

        println()
        println("objective_mode = ", mode)
        println("bound_category = ", cat)
        println("cases          = ", ncases)

        println("CompBinBou - NoSparse:")
        println("  # positive = ", count_gt(gap_comp, tau))
        println("  # equal    = ", count_abs_le(gap_comp, tau))
        println("  min        = ", fmt(min_skip_missing(gap_comp)))
        println("  mean       = ", fmt(mean_skip_missing(gap_comp)))
        println("  max        = ", fmt(max_skip_missing(gap_comp)))

        println("BigMBou - NoSparse:")
        println("  # positive = ", count_gt(gap_bigm, tau))
        println("  # equal    = ", count_abs_le(gap_bigm, tau))
        println("  min        = ", fmt(min_skip_missing(gap_bigm)))
        println("  mean       = ", fmt(mean_skip_missing(gap_bigm)))
        println("  max        = ", fmt(max_skip_missing(gap_bigm)))

        println("BigMBou - CompBinBou:")
        println("  # BigMBou >= CompBinBou = ", count_ge_minus_tol(gap_bigmbou_comp, tau), " / ", ncases)
        println("  min                    = ", fmt(min_skip_missing(gap_bigmbou_comp)))
        println("  mean                   = ", fmt(mean_skip_missing(gap_bigmbou_comp)))
        println("  max                    = ", fmt(max_skip_missing(gap_bigmbou_comp)))
    end

    println()
    println("============================================================")
    println("Rows where CompBinBou improves over NoSparseRLT")
    println("============================================================")

    improved_mask = mask_gt(df.gap_CompBinBou_minus_NoSparse, tau)
    improved = df[improved_mask, :]

    if nrow(improved) == 0
        println("No strict CompBinBou improvement over NoSparseRLT found.")
    else
        sort!(improved, [:objective_mode, :bound_category, :gap_CompBinBou_minus_NoSparse])
        for row in eachrow(improved)
            println(
                row.inst_id,
                " | mode=", row.objective_mode,
                " | category=", row.bound_category,
                " | NoSparse=", fmt(row.z_NoSparseRLT),
                " | CompBinBou=", fmt(row.z_RLT_CompBinBou),
                " | gap=", fmt(row.gap_CompBinBou_minus_NoSparse),
            )
        end
    end

    println()
    println("============================================================")
    println("Rows where CompBinBou is tighter than BigMBou")
    println("============================================================")

    comp_beats_mask = mask_gt(df.gap_CompBinBou_minus_BigMBou, tau)
    comp_beats_bigmbou = df[comp_beats_mask, :]

    if nrow(comp_beats_bigmbou) == 0
        println("No CompBinBou > BigMBou row found.")
    else
        sort!(comp_beats_bigmbou, [:objective_mode, :bound_category, :gap_CompBinBou_minus_BigMBou])
        for row in eachrow(comp_beats_bigmbou)
            println(
                row.inst_id,
                " | mode=", row.objective_mode,
                " | category=", row.bound_category,
                " | BigMBou=", fmt(row.z_RLT_BigMBou),
                " | CompBinBou=", fmt(row.z_RLT_CompBinBou),
                " | gap=", fmt(row.gap_CompBinBou_minus_BigMBou),
            )
        end
    end

    return nothing
end

# ------------------------------------------------------------
# Main driver
# ------------------------------------------------------------

function run_test_BC(;
    instances_path::AbstractString = joinpath(ROOT, "instances_clean.json"),
    out_csv::AbstractString = joinpath(ROOT, "test_BC_results.csv"),
    out_json::AbstractString = joinpath(ROOT, "test_BC_results.json"),
    optimizer = Gurobi.Optimizer,
    verbose::Bool = false,
    tau::Float64 = 1e-6,
)
    println()
    println("============================================================")
    println("Running Test B/C")
    println("instances_path = ", instances_path)
    println("objective modes = ", OBJECTIVE_MODES)
    println("============================================================")

    instances = read_instances(instances_path)
    println("Loaded instances: ", length(instances))

    validate_zero_allowed_instances(instances)

    rows = Vector{Dict{String,Any}}()
    total_cases = length(instances) * length(OBJECTIVE_MODES)
    case_counter = 0

    t0 = time()

    for inst in instances
        base_id = get(inst, "id", "unknown")
        cat = bound_category(inst)

        println()
        println("------------------------------------------------------------")
        println("Base instance: ", base_id)
        println("category     : ", cat)
        println("------------------------------------------------------------")

        for mode in OBJECTIVE_MODES
            case_counter += 1
            test_inst = objective_variant(inst, mode)

            println()
            println("Case ", case_counter, " / ", total_cases)
            println("objective_mode = ", mode)

            nosparse, bigm, bigmbou, comp_results =
                solve_one_case(test_inst; optimizer = optimizer, verbose = verbose)

            row = result_row(inst, test_inst, nosparse, bigm, bigmbou, comp_results)
            push!(rows, row)

            println("  NoSparseRLT  = ", fmt(row["z_NoSparseRLT"]), " status=", row["status_NoSparseRLT"])
            println("  RLT_CompBin  = ", fmt(row["z_RLT_CompBin"]), " status=", row["status_RLT_CompBin"])
            println("  CompBinBou   = ", fmt(row["z_RLT_CompBinBou"]), " status=", row["status_RLT_CompBinBou"])
            println("  RLT_CompBou  = ", fmt(row["z_RLT_CompBou"]), " status=", row["status_RLT_CompBou"])
            println("  RLT_BigM     = ", fmt(row["z_RLT_BigM"]), " status=", row["status_RLT_BigM"])
            println("  BigMBou      = ", fmt(row["z_RLT_BigMBou"]), " status=", row["status_RLT_BigMBou"])
            println("  gap C-NoSp   = ", fmt(row["gap_CompBinBou_minus_NoSparse"]))
            println("  gap B-NoSp   = ", fmt(row["gap_BigMBou_minus_NoSparse"]))
            println("  gap B-C      = ", fmt(row["gap_BigMBou_minus_CompBinBou"]))
            println("  elapsed      = ", @sprintf("%.1f", time() - t0), "s")
        end
    end

    df = DataFrame(rows)

    CSV.write(out_csv, df)

    open(out_json, "w") do io
        JSON.print(io, rows, 2)
    end

    println()
    println("Saved CSV  to: ", out_csv)
    println("Saved JSON to: ", out_json)

    print_summary(df; tau = tau)

    return df
end

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

df_BC = run_test_BC(
    instances_path = joinpath(ROOT, "instances_clean.json"),
    out_csv = joinpath(ROOT, "test_BC_results.csv"),
    out_json = joinpath(ROOT, "test_BC_results.json"),
    optimizer = Gurobi.Optimizer,
    verbose = false,
    tau = 1e-6,
)



# ------------------------------------------------------------
# Simple compact mean-gap table
# ------------------------------------------------------------

function mean_or_missing(col)
    vals = Float64[]
    for x in col
        if x !== missing && x !== nothing && isfinite(Float64(x))
            push!(vals, Float64(x))
        end
    end
    return isempty(vals) ? missing : mean(vals)
end

function count_positive(col; tau::Float64 = 1e-6)
    vals = Float64[]
    for x in col
        if x !== missing && x !== nothing && isfinite(Float64(x))
            push!(vals, Float64(x))
        end
    end
    return count(x -> x > tau, vals)
end

function print_simple_mean_gap_table(
    df::DataFrame;
    category = nothing,
    tau::Float64 = 1e-6,
)
    sub = category === nothing ? df : df[df.bound_category .== category, :]

    title = category === nothing ? "all instances" : string(category)

    println()
    println("============================================================")
    println("Mean gap table: $title")
    println("============================================================")

    @printf("%-12s %6s %12s %14s %16s %16s\n",
        "objective", "cases", "Comp>NS", "BigMBou>NS", "mean Comp-NS", "mean BigM-NS")
    println("-" ^ 84)

    for mode in OBJECTIVE_MODES
        ss = sub[sub.objective_mode .== mode, :]

        cases = nrow(ss)

        comp_gap = ss.gap_CompBinBou_minus_NoSparse
        bigm_gap = ss.gap_BigMBou_minus_NoSparse

        n_comp = count_positive(comp_gap; tau = tau)
        n_bigm = count_positive(bigm_gap; tau = tau)

        mean_comp = mean_or_missing(comp_gap)
        mean_bigm = mean_or_missing(bigm_gap)

        @printf("%-12s %6d %6d/%-5d %8d/%-5d %16.4f %16.4f\n",
            mode,
            cases,
            n_comp, cases,
            n_bigm, cases,
            mean_comp === missing ? NaN : mean_comp,
            mean_bigm === missing ? NaN : mean_bigm,
        )
    end
end

print_simple_mean_gap_table(
    df_BC;
    category = nothing,
    tau = 1e-6,
)

print_simple_mean_gap_table(
    df_BC;
    category = "explicit_nonnegative",
    tau = 1e-6,
)

print_simple_mean_gap_table(
    df_BC;
    category = "two_sided_or_symmetric",
    tau = 1e-6,
)

# ------------------------------------------------------------
# Print original-objective cases where CompBinBou improves NoSparseRLT
# ------------------------------------------------------------

function is_positive_gap(x; tau::Float64 = 1e-6)
    return x !== missing && x !== nothing && isfinite(Float64(x)) && Float64(x) > tau
end

function print_original_comp_improvements(
    df::DataFrame;
    category = nothing,
    tau::Float64 = 1e-6,
)
    sub = df[df.objective_mode .== "original", :]

    if category !== nothing
        sub = sub[sub.bound_category .== category, :]
    end

    mask = [is_positive_gap(x; tau = tau) for x in sub.gap_CompBinBou_minus_NoSparse]
    hits = sub[mask, :]

    sort!(hits, :gap_CompBinBou_minus_NoSparse, rev = true)

    title = category === nothing ? "all categories" : string(category)

    println()
    println("============================================================")
    println("Original objective: CompBinBou > NoSparseRLT")
    println("Category: ", title)
    println("============================================================")

    if nrow(hits) == 0
        println("No cases found.")
        return hits
    end

    @printf("%-55s %-24s %8s %8s %14s %14s %14s %14s\n",
        "instance", "category", "n", "rho",
        "NoSparse", "CompBinBou", "BigMBou", "gap C-NS")
    println("-" ^ 160)

    for row in eachrow(hits)
        @printf("%-55s %-24s %8d %8.0f %14.6f %14.6f %14.6f %14.6f\n",
            row.inst_id,
            row.bound_category,
            row.n,
            row.rho,
            row.z_NoSparseRLT,
            row.z_RLT_CompBinBou,
            row.z_RLT_BigMBou,
            row.gap_CompBinBou_minus_NoSparse,
        )
    end

    return hits
end

original_comp_hits_all = print_original_comp_improvements(
    df_BC;
    category = nothing,
    tau = 1e-6,
)

