module SDPRLTEIDifferenceInstances

using LinearAlgebra

# Load the existing instance reader at MODULE LOAD TIME.
# This avoids Julia 1.12 world-age issues caused by defining
# load_instances dynamically from inside a function.
const _INSTANCE_IO_PATH = normpath(joinpath(@__DIR__, "..", "InstanceIO.jl"))

isfile(_INSTANCE_IO_PATH) || error(
    "Could not find InstanceIO.jl at $_INSTANCE_IO_PATH"
)

include(_INSTANCE_IO_PATH)

export sdp_rlt_EI_difference_instance

const _BASE_INSTANCE_ID =
    "n4_rho3_P_LI1_LE1_QI2_QE1_NCVX_vertexLP_seq"

const _INSTANCES_JSON_PATH =
    normpath(joinpath(@__DIR__, "..", "instances_clean.json"))

isfile(_INSTANCES_JSON_PATH) || error(
    "Could not find instances_clean.json at $_INSTANCES_JSON_PATH"
)


"""
    sdp_rlt_EI_difference_instance(;
        bigM_scale = 1.0,
        objective_scale = 1.0,
    )

Dedicated n=4, rho=3 instance exposing a strong E-vs-I difference
for the Big-M full SDP--RLT relaxation.

The feasible-region data are copied exactly from

    n4_rho3_P_LI1_LE1_QI2_QE1_NCVX_vertexLP_seq

in `instances_clean.json`.

Only the objective is replaced by the clean four-decimal objective

    min 0.5 <Q0,X> + q0'x

with

    Q0 =
    [  0.2248  -0.3120  -0.5142   0.0153
      -0.3120  -0.9980  -0.1252  -0.7526
      -0.5142  -0.1252  -0.1370  -0.1031
       0.0153  -0.7526  -0.1031  -0.4396 ]

    q0 = [0.1590, 0.3744, 0.8348, 0.4451].

For `bigM_scale=1` and `objective_scale=1`, the observed Big-M
full SDP--RLT values are approximately

    EU = -12.74158909366
    IU = -13.47741399548

with directed relative gap

    (EU-IU)/max(1,|EU|,|IU|) = 5.459689%.
"""
function sdp_rlt_EI_difference_instance(
    ;
    bigM_scale::Real = 1.0,
    objective_scale::Real = 1.0,
)
    @assert bigM_scale >= 1.0
    @assert objective_scale > 0.0

    all_instances = load_instances(_INSTANCES_JSON_PATH)

    idx = findfirst(
        d -> string(get(d, "id", "")) == _BASE_INSTANCE_ID,
        all_instances,
    )

    idx === nothing && error(
        "Base instance $_BASE_INSTANCE_ID was not found in instances_clean.json."
    )

    inst = deepcopy(all_instances[idx])

    # --------------------------------------------------------
    # Clean objective found by the random-objective search
    # --------------------------------------------------------
    Q0 = [
         0.2248  -0.3120  -0.5142   0.0153
        -0.3120  -0.9980  -0.1252  -0.7526
        -0.5142  -0.1252  -0.1370  -0.1031
         0.0153  -0.7526  -0.1031  -0.4396
    ]

    q0 = [
        0.1590,
        0.3744,
        0.8348,
        0.4451,
    ]

    α = Float64(objective_scale)
    inst["Q0"] = α .* Q0
    inst["q0"] = α .* q0

    # --------------------------------------------------------
    # Optional Big-M rescaling
    # --------------------------------------------------------
    β = Float64(bigM_scale)

    if haskey(inst, "M_minus")
        inst["M_minus"] = β .* Float64.(inst["M_minus"])
    end

    if haskey(inst, "M_plus")
        inst["M_plus"] = β .* Float64.(inst["M_plus"])
    end

    if haskey(inst, "M")
        inst["M"] = β .* Float64.(inst["M"])
    end

    # Compatibility aliases used by some older files.
    if haskey(inst, "M_minus")
        inst["Mminus"] = copy(inst["M_minus"])
    elseif haskey(inst, "Mminus")
        inst["Mminus"] = β .* Float64.(inst["Mminus"])
    end

    if haskey(inst, "M_plus")
        inst["Mplus"] = copy(inst["M_plus"])
    elseif haskey(inst, "Mplus")
        inst["Mplus"] = β .* Float64.(inst["Mplus"])
    end

    inst["bigM_scale"] = β

    inst["id"] =
        "n4_rho3_P_LI1_LE1_QI2_QE1_NCVX_SDP_RLT_EI_difference"

    return inst
end

end # module SDPRLTEIDifferenceInstances
