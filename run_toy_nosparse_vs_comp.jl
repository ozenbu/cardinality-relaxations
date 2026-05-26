# ============================================================
# run_toy_nosparse_vs_comp.jl
# ============================================================

using Gurobi
using LinearAlgebra
using Printf

const ROOT = @__DIR__

include(normpath(joinpath(ROOT, "bigM", "RLT_BigM.jl")))
using .RLTBigM

include(normpath(joinpath(ROOT, "complementarity", "RLT_complementarity.jl")))
using .RLTComp

include(normpath(joinpath(ROOT, "RLTNoSparsity.jl")))
using .RLTNoSparsity

function fmt(x)
    x === nothing && return "nothing"
    x === missing && return "missing"
    return @sprintf("%.8g", Float64(x))
end

function obj_or_missing(res)
    return res[1] == :OPTIMAL ? Float64(res[2]) : missing
end

function toy1_linear_nonnegative()
    n = 2
    return Dict{String,Any}(
        "id" => "toy1_linear_nonnegative",
        "n" => n,
        "rho" => 1,
        "Q0" => zeros(n, n),
        "q0" => [-1.0, -1.0],
        "Qi" => nothing,
        "qi" => nothing,
        "ri" => nothing,
        "Pi" => nothing,
        "pi" => nothing,
        "si" => nothing,
        "A" => zeros(0, n),
        "b" => zeros(0),
        "H" => zeros(0, n),
        "h" => zeros(0),
        "ell" => [0.0, 0.0],
        "u" => [1.0, 1.0],
        "M" => [1.0, 1.0],
        "Mminus" => [0.0, 0.0],
        "Mplus" => [1.0, 1.0],
        "M_minus" => [0.0, 0.0],
        "M_plus" => [1.0, 1.0],
    )
end

function toy2_quadratic_symmetric()
    n = 2
    return Dict{String,Any}(
        "id" => "toy2_quadratic_symmetric",
        "n" => n,
        "rho" => 1,
        "Q0" => -2.0 * Matrix{Float64}(I, n, n),
        "q0" => zeros(n),
        "Qi" => nothing,
        "qi" => nothing,
        "ri" => nothing,
        "Pi" => nothing,
        "pi" => nothing,
        "si" => nothing,
        "A" => zeros(0, n),
        "b" => zeros(0),
        "H" => zeros(0, n),
        "h" => zeros(0),
        "ell" => [-1.0, -1.0],
        "u" => [1.0, 1.0],
        "M" => [1.0, 1.0],
        "Mminus" => [1.0, 1.0],
        "Mplus" => [1.0, 1.0],
        "M_minus" => [1.0, 1.0],
        "M_plus" => [1.0, 1.0],
    )
end

function solve_one(inst; optimizer=Gurobi.Optimizer)
    id = get(inst, "id", "unknown")

    println("\n============================================================")
    println("Instance: ", id)
    println("============================================================")

    no_sparse = RLTNoSparsity.build_and_solve(
        inst;
        optimizer = optimizer,
        verbose = false,
    )

    comp_binbou = RLTComp.build_and_solve(
        inst;
        variant = "RLT_CompBinBou",
        optimizer = optimizer,
        verbose = false,
    )

    comp_bou = RLTComp.build_and_solve(
        inst;
        variant = "RLT_CompBou",
        optimizer = optimizer,
        verbose = false,
    )

    bigmbou = RLTBigM.build_and_solve(
        inst;
        variant = "EU",
        optimizer = optimizer,
        verbose = false,
    )

    z_no_sparse = obj_or_missing(no_sparse)
    z_comp_binbou = obj_or_missing(comp_binbou)
    z_comp_bou = obj_or_missing(comp_bou)
    z_bigmbou = obj_or_missing(bigmbou)

    gap_comp = (z_comp_binbou === missing || z_no_sparse === missing) ?
        missing : z_comp_binbou - z_no_sparse

    gap_bigmbou = (z_bigmbou === missing || z_no_sparse === missing) ?
        missing : z_bigmbou - z_no_sparse

    println("NoSparse RLT       status=", no_sparse[1], " obj=", fmt(z_no_sparse))
    println("RLT_CompBinBou     status=", comp_binbou[1], " obj=", fmt(z_comp_binbou),
            " gap_vs_NoSparse=", fmt(gap_comp))
    println("RLT_CompBou        status=", comp_bou[1], " obj=", fmt(z_comp_bou))
    println("RLT_BigMBou        status=", bigmbou[1], " obj=", fmt(z_bigmbou),
            " gap_vs_NoSparse=", fmt(gap_bigmbou))

    return (; id, z_no_sparse, z_comp_binbou, z_comp_bou, z_bigmbou, gap_comp, gap_bigmbou)
end

function main()
    instances = [
        toy1_linear_nonnegative(),
        toy2_quadratic_symmetric(),
    ]

    results = [solve_one(inst) for inst in instances]

    println("\n============================================================")
    println("Summary")
    println("============================================================")
    for r in results
        println(r.id,
            " | NoSparse=", fmt(r.z_no_sparse),
            " | CompBinBou=", fmt(r.z_comp_binbou),
            " | gap=", fmt(r.gap_comp),
            " | BigMBou=", fmt(r.z_bigmbou))
    end

    return results
end

results = main()