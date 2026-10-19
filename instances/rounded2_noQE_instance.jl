module Rounded2NoQEInstance

using LinearAlgebra

const _INSTANCE_IO_PATH =
    normpath(joinpath(@__DIR__, "..", "InstanceIO.jl"))

const _INSTANCES_JSON_PATH =
    normpath(joinpath(@__DIR__, "..", "instances_clean.json"))

isfile(_INSTANCE_IO_PATH) || error(
    "Could not find InstanceIO.jl at $_INSTANCE_IO_PATH"
)

isfile(_INSTANCES_JSON_PATH) || error(
    "Could not find instances_clean.json at $_INSTANCES_JSON_PATH"
)

include(_INSTANCE_IO_PATH)

export rounded2_noQE_instance


const _BASE_INSTANCE_ID =
    "n4_rho3_P_LI1_LE1_QI2_QE1_NCVX_vertexLP_seq"


# ============================================================
# Recursive 2-digit rounding
# ============================================================

round2(x::Nothing) = nothing

round2(x::Number) =
    round(Float64(x); digits=2)

round2(x::AbstractArray) =
    map(round2, x)

round2(x) = x


# ============================================================
# Instance
# ============================================================

function rounded2_noQE_instance()

    all_instances =
        load_instances(
            _INSTANCES_JSON_PATH
        )

    idx =
        findfirst(
            d ->
                string(
                    get(d, "id", "")
                ) == _BASE_INSTANCE_ID,
            all_instances,
        )

    idx === nothing && error(
        "Base instance $_BASE_INSTANCE_ID was not found."
    )

    inst =
        deepcopy(
            all_instances[idx]
        )


    # ========================================================
    # 1. objective, rounded to 2 digits
    # ========================================================

    inst["Q0"] = [
     0.62  -0.82   0.26   0.28
    -0.82   0.45   0.26  -0.97
     0.26   0.26   0.50   0.34
     0.28  -0.97   0.34  -0.38
    ]

    inst["q0"] = [
        0.32,
        -0.96,
        -0.43,
        -0.99,
    ]


    # ========================================================
    # 2. Round feasible-region coefficients to 2 digits
    # ========================================================

    keys_to_round = [
        "A",
        "b",
        "H",
        "h",
        "Qi",
        "qi",
        "ri",
    ]

    for key in keys_to_round
        if haskey(inst, key)
            inst[key] =
                round2(
                    inst[key]
                )
        end
    end


    # ========================================================
    # 3. Delete quadratic constraints completely
    # ========================================================

    inst["Pi"] = nothing
    inst["pi"] = nothing
    inst["si"] = nothing

    inst["n_QE"] = 0

    inst["Qi"] = nothing
    inst["qi"] = nothing
    inst["ri"] = nothing

    inst["n_QI"] = 0


    # ========================================================
    # 4. New ID
    # ========================================================

    inst["id"] = "n4_rho3_round2_LI6_all4_test"


    println()
    println("="^70)
    println("2-DIGIT INSTANCE WITH QUADRATIC EQUALITY REMOVED")
    println("="^70)

    println("base = $_BASE_INSTANCE_ID")
    println("objective rounded to 2 digits")
    println("linear constraints rounded to 2 digits")
    println("quadratic inequalities rounded to 2 digits")
    println("quadratic equality removed")
    println("="^70)
    println()


    return inst
end


end # module Rounded2NoQEInstance