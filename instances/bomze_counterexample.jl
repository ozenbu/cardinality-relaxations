module CounterexampleInstances

using LinearAlgebra

export bomze_counterexample

function bomze_counterexample(; bigM_scale::Real = 1.0)
    n   = 2
    rho = 1

    #
    # Objective:
    #   min -x1*x2
    #
    # Since the modeling convention is
    #   1/2 * x'Q0*x + q0'x,
    # choose Q0 so that
    #   1/2 * x'Q0*x = -x1*x2.
    #
    Q0 = [
         0.0  -1.0
        -1.0   0.0
    ]
    q0 = zeros(Float64, n)

    #
    # Quadratic inequalities:
    #
    # 1) (x1 + x2)^2 <= 1
    #    <=> x1^2 + 2x1x2 + x2^2 - 1 <= 0
    #    <=> 1/2 * x'Q1*x + r1 <= 0
    #
    Q1 = [
        2.0  2.0
        2.0  2.0
    ]
    q1 = zeros(Float64, n)
    r1 = -1.0

    #
    # 2) x1^2 + x2^2 <= 4
    #    <=> 1/2 * x'Q2*x + r2 <= 0
    #
    Q2 = [
        2.0  0.0
        0.0  2.0
    ]
    q2 = zeros(Float64, n)
    r2 = -4.0

    Qi = [Q1, Q2]
    qi = [q1, q2]
    ri = [r1, r2]

    #
    # No quadratic equalities
    #
    Pi = nothing
    pi = nothing
    si = nothing

    #
    # No extra linear inequalities/equalities
    #
    A = zeros(Float64, 0, n)
    b = zeros(Float64, 0)

    H = zeros(Float64, 0, n)
    h = zeros(Float64, 0)

    #
    # Tight valid Big-M under the sparse feasible set is 1.
    # We keep a scale parameter in case you want to test looser choices.
    #
    Mminus = bigM_scale .* ones(Float64, n)
    Mplus  = bigM_scale .* ones(Float64, n)

    inst = Dict{String,Any}()

    inst["n"]   = n
    inst["rho"] = rho

    inst["Q0"] = Q0
    inst["q0"] = q0

    inst["Qi"] = Qi
    inst["qi"] = qi
    inst["ri"] = ri

    inst["Pi"] = Pi
    inst["pi"] = pi
    inst["si"] = si

    inst["A"] = A
    inst["b"] = b

    inst["H"] = H
    inst["h"] = h

    inst["Mminus"] = Mminus
    inst["Mplus"]  = Mplus

    inst["base_type"]   = "box"
    inst["want_convex"] = false

    #
    # Q0 has eigenvalues {1, -1}, so one negative eigenvalue
    #
    inst["neg_eig_counts_input"] = [1]
    inst["neg_eig_counts_used"]  = [1]

    inst["seed"] = 0

    inst["base_tag"]  = "B"
    inst["convexity"] = "NCVX"

    inst["n_LI"] = 0
    inst["n_LE"] = 0
    inst["n_QI"] = 2
    inst["n_QE"] = 0

    inst["bigM_scale"] = bigM_scale

    inst["id"] = "n2_rho1_box_decomp_not_full_gap"

    return inst
end

end # module