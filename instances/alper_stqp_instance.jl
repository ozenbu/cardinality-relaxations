module AlperStqpInstances

using LinearAlgebra 

export alper_stqp_rho3_instance

function alper_stqp_rho3_instance(; bigM_scale::Real = 1.0)
    n   = 6
    rho = 3

    Q = [
        2.6947  -0.2028  -1.1144  -2.4230  -2.1633   0.7710
       -0.2028   5.1998   0.5005  -2.0941   0.7828  -3.3611
       -1.1144   0.5005   5.2918   1.4119  -1.1526  -1.9723
       -2.4230  -2.0941   1.4119   4.1140  -0.2025   0.3132
       -2.1633   0.7828  -1.1526  -0.2025   7.6645  -0.0170
        0.7710  -3.3611  -1.9723   0.3132  -0.0170   3.3040
    ]

    Q0 = 2.0 .* Q
    q0 = zeros(Float64, n)

    A = -Matrix{Float64}(I, n, n)
    b = zeros(Float64, n)
    H = ones(Float64, 1, n)
    h = [1.0]

    Mminus = zeros(n)
    Mplus  = ones(n)

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

    inst["A"] = A
    inst["b"] = b
    inst["H"] = H
    inst["h"] = h

    inst["Mminus"] = Mminus
    inst["Mplus"]  = Mplus

    inst["base_type"] = "simplex"
    inst["want_convex"] = true

    inst["neg_eig_counts_input"] = [0]
    inst["neg_eig_counts_used"]  = [0]

    inst["seed"] = 0

    inst["base_tag"]  = "S"
    inst["convexity"] = "CVX"

    inst["n_LI"] = 0
    inst["n_LE"] = 0
    inst["n_QI"] = 0
    inst["n_QE"] = 0
    inst["bigM_scale"] = bigM_scale

    inst["id"] = "n6_rho3_S_StQP_rho3_CVX_fv"

    return inst
end

end # module