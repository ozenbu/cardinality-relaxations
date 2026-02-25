@info "RLTBigM loaded from" @__FILE__

module RLTBigM
println(">>> Script loaded")

using LinearAlgebra
using Parameters
using JuMP
using Gurobi
using MathOptInterface
using Dates, Printf
const MOI = MathOptInterface

# ---------- helpers ----------
outer_xy(a, b) = a * transpose(b)

# --- local helper for prepare_instance ---
ensure_tuple(x) =
    x === nothing ? () :
    (x isa AbstractVector || x isa Tuple ? x : (x,))

# ---------- unified instance unpack ----------
"""
    prepare_instance(data::Dict)

Takes a raw QCQP instance dictionary `data` and returns all fields
in a consistent, ready-to-use format:

- Always returns tuples for possibly-missing entries (Qi, qi, ri, Pi, pi, si).
- If A or H is missing, defaults to empty arrays.
- Big-M is handled in a unified way:
    * If `Mminus` and `Mplus` are given, they are used.
    * Otherwise a single `M` is used and we set Mminus = Mplus = M.
  All Big-M matrices are converted to Diagonal matrices.
- Provides the all-ones vector e.
- Q0 and q0 default to zeros if not provided.
"""
function prepare_instance(data::Dict)
    # Core scalars
    n   = data["n"]
    ρ   = data["rho"]

    # Objective (defaults if absent)
    Q0  = get(data, "Q0", zeros(n,n)) :: AbstractMatrix
    q0  = get(data, "q0", zeros(n))   :: AbstractVector

    # Quadratic inequality constraints
    Qi  = ensure_tuple(get(data, "Qi", ()))
    qi  = ensure_tuple(get(data, "qi", ()))
    ri  = ensure_tuple(get(data, "ri", ()))

    # Quadratic equality constraints
    Pi  = ensure_tuple(get(data, "Pi", ()))
    pi  = ensure_tuple(get(data, "pi", ()))
    si  = ensure_tuple(get(data, "si", ()))

    # Linear inequalities
    A = get(data, "A", nothing)
    b = get(data, "b", nothing)
    if A === nothing
        A = zeros(0, n); b = zeros(0)
    end
    ℓ = size(A, 1)

    # Linear equalities
    H = get(data, "H", nothing)
    h = get(data, "h", nothing)
    if H === nothing
        H = zeros(0, n); h = zeros(0)
    end
    η = size(H, 1)

    # Big-M:
    # - If user gives Mminus & Mplus, use them.
    # - Else fall back to single M and set Mminus = Mplus = M.
    if haskey(data, "Mminus") || haskey(data, "Mplus")
        if !(haskey(data, "Mminus") && haskey(data, "Mplus"))
            error("If you provide `Mminus` or `Mplus`, you must provide both.")
        end
        Mminus_raw = data["Mminus"]
        Mplus_raw  = data["Mplus"]

        Mminus_diag = ndims(Mminus_raw) == 1 ? Mminus_raw : diag(Mminus_raw)
        Mplus_diag  = ndims(Mplus_raw)  == 1 ? Mplus_raw  : diag(Mplus_raw)

        Mminus = Diagonal(Mminus_diag)
        Mplus  = Diagonal(Mplus_diag)
    else
        # Backward compatible: single symmetric M
        Mraw  = data["M"]
        Mdiag = ndims(Mraw) == 1 ? Mraw : diag(Mraw)
        Mminus = Diagonal(Mdiag)
        Mplus  = Diagonal(Mdiag)
    end

    # Ones vector
    e = ones(n)

    return (; n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η, Mminus, Mplus, e)
end

# ---------- constraint blocks ----------
function add_FC!(m, x, u, X, R, U, params)
    @unpack n, Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η, Mminus, Mplus, e = params

    # quadratic ≤ 0
    for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
        @constraint(m,
            0.5 * sum(Qmat[i,j]*X[i,j] for i=1:n, j=1:n) + qvec' * x + rterm <= 0)
    end
    # quadratic == 0
    for (Pmat, pvec, sterm) in zip(Pi, pi, si)
        @constraint(m,
            0.5 * sum(Pmat[i,j]*X[i,j] for i=1:n, j=1:n) + pvec' * x + sterm == 0)
    end

    # Original linear constraints
    if ℓ > 0; @constraint(m, A * x .<= b); end
    if η > 0; @constraint(m, H * x .== h); end

    # Big-M constraint: -M⁻ u ≤ x ≤ M⁺ u
    @constraint(m, -Mminus * u .<= x)
    @constraint(m,  x .<= Mplus * u)

    # diag(U) = u
    @constraint(m, [i=1:n], U[i,i] == u[i])

    # H-block
    if η > 0
        @constraint(m, H * X .== outer_xy(h, x))
        @constraint(m, H * R .== outer_xy(h, u))
    end

    # Big-M McCormick blocks
    @constraint(m, Mplus*U*Mplus  .- Mplus*R' .- R*Mplus  .+ X .>= 0)
    @constraint(m, Mminus*U*Mminus .+ Mminus*R' .+ R*Mminus .+ X .>= 0)
    @constraint(m, Mplus*U*Mminus .+ Mplus*R' .- R*Mminus .- X .>= 0)

    # RLT from A x ≤ b
    if ℓ > 0
        # self-RLT of A x ≤ b
        @constraint(m, A*X*A' .- A*(x*b') .- (b*x')*A' .+ (b*b') .>= 0)

        # from (b - Ax)_i (M⁺u - x)_j ≥ 0
        @constraint(m, A*X .- (b*x') .- A*R*Mplus .+ (b*u')*Mplus .>= 0)

        # from (b - Ax)_i (M⁻u + x)_j ≥ 0
        @constraint(m, -A*X .+ (b*x') .- A*R*Mminus .+ (b*u')*Mminus .>= 0)
    end
end

function add_FE!(m, x, u, X, R, U, params)
    @unpack ρ, e = params
    @constraint(m, sum(u) == ρ)
    @constraint(m, R * e .== ρ .* x)
    @constraint(m, U * e .== ρ .* u)
end

function add_FI!(m, x, u, X, R, U, params)
    @unpack ρ, A, b, ℓ, Mminus, Mplus, e = params

    @constraint(m, sum(u) <= ρ)
    @constraint(m, ρ^2 - 2ρ*sum(u) + dot(e, U*e) >= 0)

    if ℓ > 0
        @constraint(m, ρ .* b .- (b*u')*e .- ρ .* (A*x) .+ A*R*e .>= 0)
    end

    # (ρ - eᵀu)(M⁺u - x) ≥ 0
    @constraint(m, ρ .* (Mplus*u) .- ρ .* x .- Mplus*U*e .+ R*e .>= 0)

    # (ρ - eᵀu)(x + M⁻u) ≥ 0
    @constraint(m, ρ .* (Mminus*u) .+ ρ .* x .- Mminus*U*e .- R*e .>= 0)
end

function add_FU!(m, x, u, X, R, U, params)
    @unpack A, b, ℓ, Mminus, Mplus, e = params

    @constraint(m, u .<= 1)

    eeT = ones(length(u), length(u))
    @constraint(m, U .- (u*e') .- (e*u') .+ eeT .>= 0)

    if ℓ > 0
        @constraint(m, (b*e') .- (b*u') .- (A*x)*e' .+ A*R .>= 0)
    end

    # (e-u)(M⁺u - x)ᵀ ≥ 0
    @constraint(m, Mplus*(u*e') .- x*e' .- Mplus*U .+ R .>= 0)

    # (e-u)(x + M⁻u)ᵀ ≥ 0
    @constraint(m, Mminus*(u*e') .+ x*e' .- Mminus*U .- R .>= 0)
end

function add_FIU!(m, u, U, params)
    @unpack ρ, e = params
    @constraint(m, ρ .* e .- ρ .* u .- sum(u) .* e .+ U*e .>= 0)
end

# ---------- build & solve ----------
function build_and_solve(data; variant::String="EXACT",
                         build_only::Bool=false,
                         optimizer=Gurobi.Optimizer,
                         verbose=false)
    params = prepare_instance(data)
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η, Mminus, Mplus, e = params

    m = Model(optimizer)
    set_silent(m); verbose && set_optimizer_attribute(m,"OutputFlag",1)

    set_name(m, "SQCQP_RLT_or_EXACT")

    if variant == "EXACT"
        # ---------- MIQCQP (no lifting) ----------
        @variable(m, x[1:n])
        @variable(m, u[1:n], Bin)

        # Objective: 0.5 x' Q0 x + q0' x
        @objective(m, Min, 0.5 * x' * Q0 * x + q0' * x)

        # (allow nonconvex quadratic)
        set_optimizer_attribute(m, "NonConvex", 2)

        # Original constraints
        if ℓ > 0; @constraint(m, A * x .<= b); end
        if η > 0; @constraint(m, H * x .== h); end

        # Big-M links: -M⁻ u ≤ x ≤ M⁺ u
        @constraint(m, -Mminus * u .<= x)
        @constraint(m,  x .<= Mplus * u)

        # Cardinality: sum u ≤ ρ
        @constraint(m, sum(u) <= ρ)

        # Quadratic inequality rows:  0.5 x' Qi x + qi' x + ri ≤ 0
        for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
            @constraint(m, 0.5 * x' * Qmat * x + qvec' * x + rterm <= 0)
        end

        # Quadratic equality rows:  0.5 x' Pj x + pj' x + sj = 0
        for (Pmat, pvec, sterm) in zip(Pi, pi, si)
            @constraint(m, 0.5 * x' * Pmat * x + pvec' * x + sterm == 0)
        end

        if build_only
            return m
        end

        optimize!(m)
        term = termination_status(m)

        if term == MOI.OPTIMAL
            return (:OPTIMAL, objective_value(m), value.(x), value.(u))
        elseif term == MOI.DUAL_INFEASIBLE
            return (:UNBOUNDED, nothing, nothing, nothing)
        elseif term == MOI.INFEASIBLE_OR_UNBOUNDED
            return (:INF_OR_UNBD, nothing, nothing, nothing)
        elseif term == MOI.INFEASIBLE
            return (:INFEASIBLE, nothing, nothing, nothing)
        else
            return (:OTHER, term, nothing, nothing)
        end

    else
        # ---------- First-level RLT variants (lifted) ----------
        @variable(m, x[1:n])
        @variable(m, u[1:n])                 # continuous here
        @variable(m, X[1:n,1:n], Symmetric)
        @variable(m, R[1:n,1:n])
        @variable(m, U[1:n,1:n], Symmetric)

        # RLT objective
        @objective(m, Min,
            0.5 * sum(Q0[i,j]*X[i,j] for i=1:n, j=1:n) + q0' * x)

        # Common lifted/RLT block
        add_FC!(m, x, u, X, R, U, params)

        # E vs I
        if startswith(variant, "E")
            add_FE!(m, x, u, X, R, U, params)
        else
            add_FI!(m, x, u, X, R, U, params)
        end

        # U extras (+ I×U coupling)
        if endswith(variant, "U")
            add_FU!(m, x, u, X, R, U, params)
            if startswith(variant, "I")
                add_FIU!(m, u, U, params)
            end
        end

        if build_only
            return m
        end

        optimize!(m)
        term = termination_status(m)

        if term == MOI.OPTIMAL
            #  dump primal to txt: x, u, X, R, U 
            begin
                ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
                fname = "primal_sol_$(variant)_$(ts).txt"
                open(fname, "w") do io
                    println(io, "# variant=", variant,
                                "  status=OPTIMAL  obj=", objective_value(m))
                    println(io, "x = ", value.(x))
                    println(io, "u = ", value.(u))
                    print(io,   "X = "); show(io, "text/plain", value.(X)); println(io)
                    print(io,   "R = "); show(io, "text/plain", value.(R)); println(io)
                    print(io,   "U = "); show(io, "text/plain", value.(U)); println(io)
                end
                println("saved primal solution to $fname")
            end
            # dump dual to txt
            if JuMP.has_duals(m)
                ts    = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
                fname = "primal_duals_$(variant)_$(ts).txt"
                open(fname, "w") do io
                    @printf(io, "# variant=%s  status=%s  obj=%.12g\n\n",
                            variant, string(term), objective_value(m))
                    for con in JuMP.all_constraints(m;
                            include_variable_in_set_constraints=true)
                        try
                            d = dual(con)
                            println(io, string(con), " = ", @sprintf("%.9g", d))
                        catch
                            # dual raporlanmayan kısıt/solver olursa atla
                        end
                    end
                end
                println("saved primal-side dual multipliers to $fname")
            end
            return (:OPTIMAL, objective_value(m),
                    value.(x), value.(u), value.(X), value.(R), value.(U))

        elseif term == MOI.DUAL_INFEASIBLE
            return (:UNBOUNDED, nothing, nothing, nothing, nothing, nothing, nothing)
        elseif term == MOI.INFEASIBLE_OR_UNBOUNDED
            return (:INF_OR_UNBD, nothing, nothing, nothing, nothing, nothing, nothing)
        elseif term == MOI.INFEASIBLE
            return (:INFEASIBLE, nothing, nothing, nothing, nothing, nothing, nothing)
        else
            return (:OTHER, term, nothing, nothing, nothing, nothing, nothing)
        end

    end
end

# ---------- inspect ----------
function inspect_model(data, variant::String; optimizer=Gurobi.Optimizer)
    m = build_and_solve(data; variant=variant, build_only=true,
                        optimizer=optimizer, verbose=false)
    fname = "$(variant)_model.lp"
    write_to_file(m, fname)
    println("\n$fname written.\n")
    println("Number of constraints: ",
            num_constraints(m; count_variable_in_set_constraints=true))
    return nothing
end

# ---------- demo ----------
function demo()
    data1 = Dict(
        "n"  => 4,
        "rho"=> 3.0,
        "Q0" => zeros(4,4),
        "q0" => [-1.0, 0.0, 0.0, 0.0],
        "Qi" => nothing, "qi"=>nothing, "ri"=>nothing,
        "Pi" => nothing, "pi"=>nothing, "si"=>nothing,
        "A"  => nothing, "b"=>nothing,
        "H"  => nothing, "h"=>nothing,
        "M"  => I(4)
    )

    n   = 4
    ρ   = 3.0
    ℓb  = [-1.0, -0.5, -2.0, -0.3]   # lower bounds
    ū   = [ 1.0,  0.8,  0.7,  2.5]   # upper bounds
    Mvec = max.(abs.(ℓb), abs.(ū))   # Big-M not tighter than box
    Mmat = Diagonal(Mvec)

    A = [I(n); -I(n)]
    b = vcat(ū, -ℓb)
    H = nothing
    h = nothing
    
    data3 = Dict(
        "n"  => n,
        "rho"=> ρ,
        "Q0" => zeros(n,n),
        "q0" => [-29.0, 43.0, -30.0, 70.0],
        "Qi" => nothing, "qi"=>nothing, "ri"=>nothing,
        "Pi" => nothing, "pi"=>nothing, "si"=>nothing,
        "A"  => A, "b"=> b,
        "H"  => H, "h"=> h,
        "M"  => Mmat
    )

    H = [ 1.0  -2.0   0.0   0.0;
          0.0   0.0   1.0   1.0 ]
    h = [0.0, 0.0]

    dataA = Dict(
        "n"  => n, "rho"=> ρ,
        "Q0" => zeros(n,n),
        "q0" => [-29.0, 43.0, -30.0, 70.0],
        "Qi" => nothing, "qi"=>nothing, "ri"=>nothing,
        "Pi" => nothing, "pi"=>nothing, "si"=>nothing,
        "A"  => A, "b"=>b,
        "H"  => H, "h"=>h,
        "M"  => Mmat
    )

    Q1 = diagm(0 => [2.0, 2.0, 0.0, 0.0])     # 0.5*Q1 gives x1^2 + x2^2
    q1 = [-0.6, 0.4, 0.0, 0.0]
    r1 = -0.87

    dataB = Dict(
        "n"  => n, "rho"=> ρ,
        "Q0" => zeros(n,n),
        "q0" => [-12.0, 8.0, 5.0, -3.0],
        "Qi" => (Q1,), "qi" => (q1,), "ri" => (r1,),
        "Pi" => nothing, "pi"=>nothing, "si"=>nothing,
        "A"  => A, "b"=>b,
        "H"  => nothing, "h"=>nothing,
        "M"  => Mmat
    )

    P1 = diagm(0 => [0.0, 0.0, 2.0, 2.0])   # 0.5*P1 gives x3^2 + x4^2
    p1 = zeros(n)
    s1 = -2.25

    dataC = Dict(
        "n"  => n, "rho"=> ρ,
        "Q0" => zeros(n,n),
        "q0" => [5.0, -4.0, 1.0, 2.0],
        "Qi" => nothing, "qi"=>nothing, "ri"=>nothing,
        "Pi" => (P1,), "pi" => (p1,), "si" => (s1,),
        "A"  => A, "b"=>b,
        "H"  => [1.0 -2.0 0.0 0.0], "h" => [0.0],
        "M"  => Mmat
    )

    # Inequality 1: (x1 - x2)^2 ≤ 0.5
    Q2 = zeros(n,n);  Q2[1,1]=2.0; Q2[2,2]=2.0; Q2[1,2]=-2.0; Q2[2,1]=-2.0
    q2 = zeros(n)
    r2 = -0.5

    # Inequality 2: reuse the ball from dataB
    Q3 = Q1;  q3 = q1;  r3 = r1

    # Equality: reuse the circle from dataC
    P2 = P1;  p2 = p1;  s2 = s1

    H2 = [ 1.0  -1.0   0.0   0.0 ]
    h2 = [ 0.0 ]

    dataD = Dict(
        "n"  => n, "rho"=> ρ,
        "Q0" => zeros(n,n),
        "q0" => [-29.0, 43.0, -30.0, 70.0],
        "Qi" => (Q2, Q3), "qi" => (q2, q3), "ri" => (r2, r3),
        "Pi" => (P2,),    "pi" => (p2,),    "si" => (s2,),
        "A"  => A, "b"=>b,
        "H"  => H2, "h"=>h2,
        "M"  => Mmat
    )

    # small rational example
    Q0 = [
        3//10000    127//1250   79//2500    867//10000;
        127//1250   1//500      1001//10000 1059//10000;
        79//2500    1001//10000 -1//2000    -703//10000;
        867//10000  1059//10000 -703//10000 -1063//10000
    ]
    Q0d = Q0*20000

    q0 = [-1973//10000, -2535//10000, -1967//10000, -973//10000]
    q0d = q0*20000

    A2 = [ 1.0  0.0  0.0  0.0
           0.0  1.0  0.0  0.0
           0.0  0.0  1.0  0.0
           0.0  0.0  0.0  1.0
          -1.0  0.0  0.0  0.0
           0.0 -1.0  0.0  0.0
           0.0  0.0 -1.0  0.0
           0.0  0.0  0.0 -1.0 ]

    b2 = [1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0]

    data_test4 = Dict(
        "n"  => 4,
        "rho"=> 3.0,
        "Q0" => Q0d,
        "q0" => q0d,
        "Qi" => nothing, "qi" => nothing, "ri" => nothing,
        "Pi" => nothing, "pi" => nothing, "si" => nothing,
        "A"  => A2, "b"  => b2,
        "H"  => nothing, "h"  => nothing,
        "M"  => I(4)
    )

    function bomze_stqp_rho3_instance(; bigM_scale::Real = 1.0)
        n   = 6
        rho = 3

        # Q taken from Alper's slide (Bomze et al. example)
        Q = [
            2.6947  -0.2028  -1.1144  -2.4230  -2.1633   0.7710
        -0.2028   5.1998   0.5005  -2.0941   0.7828  -3.3611
        -1.1144   0.5005   5.2918   1.4119  -1.1526  -1.9723
        -2.4230  -2.0941   1.4119   4.1140  -0.2025   0.3132
        -2.1633   0.7828  -1.1526  -0.2025   7.6645  -0.0170
            0.7710  -3.3611  -1.9723   0.3132  -0.0170   3.3040
        ]

        # Our convention is obj = 0.5 x'Q0 x + q0'x, so choose Q0 = 2Q, q0 = 0
        Q0 = 2.0 .* Q
        q0 = zeros(Float64, n)

        # ----------------------------------------------------------
        # Base region: simplex  { x ∈ R^n : x ≥ 0, eᵀx = 1 }.
        # Encode x ≥ 0 as -I x ≤ 0
        # ----------------------------------------------------------
        A = -Matrix{Float64}(I, n, n)
        b = zeros(Float64, n)
        H = ones(Float64, 1, n)
        h = [1.0]

        # Bounds on simplex: each coordinate in [0,1]
        Mminus = zeros(n)
        Mplus  = ones(n)

        # Pack into Dict consistent with generate_instance output
        inst = Dict{String,Any}()

        inst["n"]   = n
        inst["rho"] = rho

        inst["Q0"] = Q0
        inst["q0"] = q0

        # No quadratic (in)equality constraints
        inst["Qi"] = nothing
        inst["qi"] = nothing
        inst["ri"] = nothing

        inst["Pi"] = nothing
        inst["pi"] = nothing
        inst["si"] = nothing

        # Linear constraints of the simplex
        inst["A"] = A
        inst["b"] = b
        inst["H"] = H
        inst["h"] = h

        inst["Mminus"] = Mminus
        inst["Mplus"]  = Mplus

        inst["base_type"] = "simplex"
        inst["want_convex"] = true

        # Eigenvalue pattern: Q0 is PSD, no QI
        inst["neg_eig_counts_input"] = [0]
        inst["neg_eig_counts_used"]  = [0]

        # Seed is arbitrary for this deterministic instance
        inst["seed"] = 0

        # Meta tags
        inst["base_tag"]  = "S"
        inst["convexity"] = "CVX"

        inst["n_LI"] = 0
        inst["n_LE"] = 0
        inst["n_QI"] = 0
        inst["n_QE"] = 0
        inst["bigM_scale"] = bigM_scale

        # Final ID with "fv" tag (final version)
        inst["id"] = "n6_rho3_S_StQP_rho3_CVX_fv"

        return inst
    end

    a_data = bomze_stqp_rho3_instance(; bigM_scale = 1.0)



    for var in ("EXACT","E","EU","I","IU")
        res = build_and_solve(a_data; variant=var)
        println("variant=$var → ", res[1], "  obj=", res[2])
        if res[1] == :OPTIMAL
            if var == "EXACT"
                _, _, x, u = res
                println("x=", x, "  u=", u)
            else
                _, _, x, u, X, R, U = res
                println("x=", x, "  u=", u)
                println("X ="); show(stdout, "text/plain", X); println("\n")
                println("R ="); show(stdout, "text/plain", R); println("\n")
                println("U ="); show(stdout, "text/plain", U); println("\n")
            end
        end
    end
end

end # module RLTBigM


if isinteractive()
    using .RLTBigM
    RLTBigM.demo()
end

