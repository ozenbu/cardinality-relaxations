module SDPCompBinSeparationInstance

using LinearAlgebra

export sdp_compbin_vs_bigm_instance, compbin_feasible_witness

function sdp_compbin_vs_bigm_instance(; bigM_scale::Real = 1.0)
    n   = 2
    rho = 1

    # Separating objective:
    # min -e'x
    Q0 = zeros(Float64, n, n)
    q0 = -ones(Float64, n)

    inst = Dict{String,Any}()

    inst["n"]   = n
    inst["rho"] = rho

    inst["Q0"] = Q0
    inst["q0"] = q0

    inst["Qi"] = nothing
    inst["qi"] = nothing
    inst["ri"] = nothing

    inst["Pi"] = nothing
    inst["pi"] = nothing
    inst["si"] = nothing

    # No common linear constraints in this SDP-only counterexample
    inst["A"] = nothing
    inst["b"] = nothing
    inst["H"] = nothing
    inst["h"] = nothing

    # Big-M bounds used by the BigM formulation
    inst["Mminus"] = zeros(Float64, n)
    inst["Mplus"]  = ones(Float64, n)

    inst["base_type"] = "counterexample"
    inst["want_convex"] = true

    inst["neg_eig_counts_input"] = [0]
    inst["neg_eig_counts_used"]  = [0]

    inst["seed"] = 0

    inst["base_tag"]  = "SEP"
    inst["convexity"] = "CVX"

    inst["n_LI"] = 0
    inst["n_LE"] = 0
    inst["n_QI"] = 0
    inst["n_QE"] = 0
    inst["bigM_scale"] = bigM_scale

    inst["id"] = "n2_rho1_SDP_CompBin_vs_BigM_sep"

    return inst
end

function compbin_feasible_witness()
    x = [1.0, 1.0]

    v = [0.5, 0.5]

    X = [
        5.0  1.0;
        1.0  5.0
    ]

    W = [
        0.0   0.75;
        0.75  0.0
    ]

    V = [
        0.5   0.25;
        0.25  0.5
    ]

    Y = [
        1.0   x[1]  x[2]  v[1]  v[2];
        x[1]  X[1,1] X[1,2] W[1,1] W[1,2];
        x[2]  X[2,1] X[2,2] W[2,1] W[2,2];
        v[1]  W[1,1] W[2,1] V[1,1] V[1,2];
        v[2]  W[1,2] W[2,2] V[2,1] V[2,2]
    ]

    return Dict(
        "x" => x,
        "v" => v,
        "X" => X,
        "W" => W,
        "V" => V,
        "Y" => Y,
    )
end

end # module