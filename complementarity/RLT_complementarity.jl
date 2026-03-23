module RLTComp

using LinearAlgebra
using JuMP
using MathOptInterface
const MOI = MathOptInterface
using Parameters: @unpack

# ---------- helpers ----------
outer_xy(a, b) = a * transpose(b)

# ============================================================
#  Instance preparation (self-contained)
# ============================================================
"""
    prepare_instance_comp(data::Dict{String,Any})

Converts an instance Dict into a NamedTuple `params` used by the model builder.
Expected keys:
- n, rho, Q0, q0
- optionally Qi, qi, ri (can be `nothing`)
- optionally Pi, pi, si (can be `nothing`)
- optionally A, b (can be `nothing`)
- optionally H, h (can be `nothing`)
- Mminus, Mplus (vectors, length n)
- bigM_scale (optional, default 1.0)
"""
function prepare_instance_comp(data::Dict{String,Any})
    n   = Int(data["n"])
    ρ   = Int(data["rho"])
    Q0  = Matrix{Float64}(data["Q0"])
    q0  = Vector{Float64}(data["q0"])

    Qi  = get(data, "Qi", nothing)
    qi  = get(data, "qi", nothing)
    ri  = get(data, "ri", nothing)

    Pi  = get(data, "Pi", nothing)
    pi  = get(data, "pi", nothing)
    si  = get(data, "si", nothing)

    A   = get(data, "A", nothing)
    b   = get(data, "b", nothing)
    H   = get(data, "H", nothing)
    h   = get(data, "h", nothing)

    # Accept Mminus/Mplus either as length-n vectors or diagonal matrices, and convert them to vectors.
    mminus_raw = data["Mminus"]
    mplus_raw  = data["Mplus"]

    mminus = ndims(mminus_raw) == 1 ? Float64.(mminus_raw) : Float64.(diag(mminus_raw))
    mplus  = ndims(mplus_raw)  == 1 ? Float64.(mplus_raw)  : Float64.(diag(mplus_raw))
    bigM_scale = Float64(get(data, "bigM_scale", 1.0))

    # General bounds: xL <= x <= xU
    xL = -bigM_scale .* mminus
    xU =  bigM_scale .* mplus

    e   = ones(Float64, n)
    eeT = ones(Float64, n, n)

    # Flags for existence of linear systems
    hasA = (A isa AbstractMatrix) && (b isa AbstractVector) && (size(A,1) > 0)
    hasH = (H isa AbstractMatrix) && (h isa AbstractVector) && (size(H,1) > 0)

    # For compatibility with your earlier code, keep ℓ,η as 0/1 flags
    ℓ = hasA ? 1 : 0
    η = hasH ? 1 : 0

    return (
        n=n, ρ=ρ,
        Q0=Q0, q0=q0,
        Qi=Qi, qi=qi, ri=ri,
        Pi=Pi, pi=pi, si=si,
        A=A, b=b, ℓ=ℓ,
        H=H, h=h, η=η,
        mminus=mminus, mplus=mplus, bigM_scale=bigM_scale,
        xL=xL, xU=xU,
        e=e, eeT=eeT
    )
end

# ============================================================
#  Block: FC_comp (common; only ORIGINAL linear rows generate 1st-level RLT)
#   - QCQP lifted via X
#   - Linear rows: Ax<=b, Hx=h
#   - General bounds: xL <= x <= xU
#   - Relaxed lifted complementarity: diag(W)=0
#   - RLT from: (Ax<=b)×(Ax<=b), (Ax<=b)×(x-xL>=0), (Ax<=b)×(xU-x>=0)
#               and (Hx=h)×variables
# ============================================================
function add_FC_comp!(m, x, v, X, V, W, params)
    @unpack n, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, xL, xU = params

    # Original QCQP rows (lifted via X)
    if Qi !== nothing
        for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
            @constraint(m,
                0.5 * sum(Qmat[i,j] * X[i,j] for i=1:n, j=1:n) + qvec' * x + rterm <= 0
            )
        end
    end
    if Pi !== nothing
        for (Pmat, pvec, sterm) in zip(Pi, pi, si)
            @constraint(m,
                0.5 * sum(Pmat[i,j] * X[i,j] for i=1:n, j=1:n) + pvec' * x + sterm == 0
            )
        end
    end

    # Original linear rows
    if ℓ > 0
        @constraint(m, A * x .<= b)
    end
    if η > 0
        @constraint(m, H * x .== h)
    end

    # General x-bounds: xL <= x <= xU
    @constraint(m, xL .<= x)
    @constraint(m, x .<= xU)

    # Relaxed complementarity in the lift: diag(W)=0
    @constraint(m, [i=1:n], W[i,i] == 0)

    # First-level RLT from original linear rows
    if ℓ > 0
        # (Ax<=b)×(Ax<=b): self-RLT
        @constraint(m, A*X*A' .- A*(x*b') .- (b*x')*A' .+ (b*b') .>= 0)

        # (b-Ax)(x-xL)^T >= 0  ==>  -AX + b x^T + A x xL^T - b xL^T >= 0  (SIGNED CORRECTLY)
        @constraint(m, -A*X .+ (b*x') .+ A*(x*xL') .- (b*xL') .>= 0)

        # (b-Ax)(xU-x)^T >= 0  ==>  AX - A x xU^T - b x^T + b xU^T >= 0
        @constraint(m, A*X .- A*(x*xU') .- (b*x') .+ (b*xU') .>= 0)
    end

    # (Hx=h)×variables: equality-products
    if η > 0
        @constraint(m, H * X .== outer_xy(h, x))
        @constraint(m, H * W .== outer_xy(h, v))
    end

    return nothing
end

# ============================================================
#  Block: FE_v (cardinality equality + equality×variable products)
# ============================================================
function add_FE_v!(m, x, v, X, V, W, params)
    @unpack n, ρ, e = params
    @constraint(m, sum(v) == n - ρ)                         # e'v = n-ρ
    @constraint(m, (e' * V) .== (n - ρ) .* (v'))           # e'V = (n-ρ) v'
    @constraint(m, (e' * W') .== (n - ρ) .* (x'))          # e'W' = (n-ρ) x'
    return nothing
end

# ============================================================
#  Block: FB_v (binary lift diag(V)=v)
# ============================================================
function add_FB_v!(m, v, V, params)
    @unpack n = params
    @constraint(m, [i=1:n], V[i,i] == v[i])
    return nothing
end

# ============================================================
#  Block: FU_v (explicit 0<=v<=e and its self-RLT + cross-RLT)
#   Cross-RLT uses GENERAL x-bounds xL<=x<=xU.
# ============================================================
function add_FU_v!(m, x, v, X, V, W, params)
    @unpack n, A, b, ℓ, e, eeT, xL, xU = params

    # Bound rows: 0 <= v <= e
    @constraint(m, 0 .<= v)
    @constraint(m, v .<= e)

    # Self-RLT of 0<=v<=e (V-block)
    @constraint(m, V .>= 0)
    @constraint(m, v*e' .- V .>= 0)
    @constraint(m, V .- v*e' .- e*v' .+ eeT .>= 0)

    # Cross-RLT with general x-bounds:
    # (x - xL)v^T >= 0  ->  W - xL v^T >= 0
    @constraint(m, W .- xL*v' .>= 0)

    # (x - xL)(e - v)^T >= 0  ->  x e^T - W - xL e^T + xL v^T >= 0   
    @constraint(m, x*e' .- W .- xL*e' .+ xL*v' .>= 0)

    # (xU - x)v^T >= 0  ->  xU v^T - W >= 0
    @constraint(m, xU*v' .- W .>= 0)

    # (xU - x)(e - v)^T >= 0  ->  xU e^T - xU v^T - x e^T + W >= 0
    @constraint(m, xU*e' .- xU*v' .- x*e' .+ W .>= 0)

    # Cross-RLT with Ax<=b (unchanged)
    if ℓ > 0
        @constraint(m, b*v' .- A*W .>= 0)
        @constraint(m, b*e' .- (A*x)*e' .- b*v' .+ A*W .>= 0)
    end

    return nothing
end

# ============================================================
#  Build & solve
# ============================================================
"""
    build_and_solve(data; variant="RLT_CompBinBou", build_only=false, optimizer, verbose=false)

Variants (preferred naming):
- "EXACT_CompBin"      : v binary, indicator bounds xL(1-v) <= x <= xU(1-v)
- "EXACT_CompBou"      : v continuous in [0,1], bilinear x[i]*v[i]==0 for all i
- "RLT_CompBin"        : FC_comp + FE_v + FB_v
- "RLT_CompBinBou"     : FC_comp + FE_v + FB_v + FU_v
- "RLT_CompBou"        : FC_comp + FE_v + FU_v

Backward-compatible aliases:
- "EXACT_S2"  -> EXACT_CompBin
- "EXACT_S3"  -> EXACT_CompBou
- "RLT_S2"    -> RLT_CompBin
- "RLT_S2U"   -> RLT_CompBinBou
- "RLT_S3"    -> RLT_CompBou
"""
function build_and_solve(data;
    variant::String="RLT_CompBinBou",
    build_only::Bool=false,
    optimizer,
    verbose::Bool=false
)
    params = prepare_instance_comp(data)

    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η, e, eeT, xL, xU = params

    # Handle aliases
    if variant == "EXACT_S2"; variant = "EXACT_CompBin"; end
    if variant == "EXACT_S3"; variant = "EXACT_CompBou"; end
    if variant == "RLT_S2";   variant = "RLT_CompBin"; end
    if variant == "RLT_S2U";  variant = "RLT_CompBinBou"; end
    if variant == "RLT_S3";   variant = "RLT_CompBou"; end

    m = Model(optimizer)
    verbose ? set_optimizer_attribute(m, "OutputFlag", 1) : set_silent(m)

    # For EXACT_CompBou (nonconvex bilinear), some solvers need this
    try
        set_optimizer_attribute(m, "NonConvex", 2)
    catch
    end

    set_name(m, "SQCQP_Complementarity")

    # =========================================================
    # EXACT_CompBin
    # =========================================================
    if variant == "EXACT_CompBin"
        @variable(m, x[1:n])
        @variable(m, v[1:n], Bin)

        @objective(m, Min, 0.5 * x' * Q0 * x + q0' * x)

        if ℓ > 0; @constraint(m, A * x .<= b); end
        if η > 0; @constraint(m, H * x .== h); end

        # general box
        @constraint(m, xL .<= x)
        @constraint(m, x .<= xU)

        # cardinality on v: sum v = n-ρ
        @constraint(m, sum(v) == n - ρ)

        # indicator bounds: if v[i]=1 -> x[i]=0; if v[i]=0 -> x in [xL[i], xU[i]]
        @constraint(m, [i=1:n], x[i] >= xL[i] * (1 - v[i]))
        @constraint(m, [i=1:n], x[i] <= xU[i] * (1 - v[i]))

        # QCQP rows
        if Qi !== nothing
            for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
                @constraint(m, 0.5 * x' * Qmat * x + qvec' * x + rterm <= 0)
            end
        end
        if Pi !== nothing
            for (Pmat, pvec, sterm) in zip(Pi, pi, si)
                @constraint(m, 0.5 * x' * Pmat * x + pvec' * x + sterm == 0)
            end
        end

        if build_only; return m; end
        optimize!(m)
        term = termination_status(m)
        return term == MOI.OPTIMAL ? (:OPTIMAL, objective_value(m), value.(x), value.(v)) :
                                     (Symbol(term), nothing, nothing, nothing)
    end

    # =========================================================
    # EXACT_CompBou (nonconvex)
    # =========================================================
    if variant == "EXACT_CompBou"
        @variable(m, x[1:n])
        @variable(m, 0 <= v[1:n] <= 1)

        @objective(m, Min, 0.5 * x' * Q0 * x + q0' * x)

        if ℓ > 0; @constraint(m, A * x .<= b); end
        if η > 0; @constraint(m, H * x .== h); end

        @constraint(m, xL .<= x)
        @constraint(m, x .<= xU)

        @constraint(m, sum(v) == n - ρ)
        @constraint(m, [i=1:n], x[i] * v[i] == 0)   # componentwise complementarity

        if Qi !== nothing
            for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
                @constraint(m, 0.5 * x' * Qmat * x + qvec' * x + rterm <= 0)
            end
        end
        if Pi !== nothing
            for (Pmat, pvec, sterm) in zip(Pi, pi, si)
                @constraint(m, 0.5 * x' * Pmat * x + pvec' * x + sterm == 0)
            end
        end

        if build_only; return m; end
        optimize!(m)
        term = termination_status(m)
        return term == MOI.OPTIMAL ? (:OPTIMAL, objective_value(m), value.(x), value.(v)) :
                                     (Symbol(term), nothing, nothing, nothing)
    end

    # =========================================================
    # RLT relaxations (continuous v, lifted X,V,W)
    # =========================================================
    @variable(m, x[1:n])
    @variable(m, v[1:n])                       # continuous in RLT
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, V[1:n,1:n], Symmetric)
    @variable(m, W[1:n,1:n])

    @objective(m, Min, 0.5 * sum(Q0[i,j] * X[i,j] for i=1:n, j=1:n) + q0' * x)

    add_FC_comp!(m, x, v, X, V, W, params)
    add_FE_v!(m, x, v, X, V, W, params)

    if variant == "RLT_CompBin"
        add_FB_v!(m, v, V, params)
    elseif variant == "RLT_CompBinBou"
        add_FB_v!(m, v, V, params)
        add_FU_v!(m, x, v, X, V, W, params)
    elseif variant == "RLT_CompBou"
        add_FU_v!(m, x, v, X, V, W, params)
    else
        error("Unknown variant: $variant. Use EXACT_CompBin, EXACT_CompBou, RLT_CompBin, RLT_CompBinBou, or RLT_CompBou.")
    end

    if build_only
        return m
    end

    optimize!(m)
    term = termination_status(m)
    if term == MOI.OPTIMAL
        return (:OPTIMAL, objective_value(m),
                value.(x), value.(v), value.(X), value.(V), value.(W))
    else
        return (Symbol(term), nothing, nothing, nothing, nothing, nothing, nothing)
    end
end

# ============================================================
# PrettyPrint wrapper
# ============================================================
function solve_and_print(data::Dict{String,Any};
    variant::String="RLT_CompBinBou",
    relaxation = nothing,
    optimizer,
    verbose::Bool=false,
    show_mats::Bool=true,
    pp_kwargs...
)
    if !isdefined(Main, :PrettyPrint)
        error("PrettyPrint is not loaded. Please include(\"instances/prettyprint.jl\") before calling RLTComp.solve_and_print.")
    end

    rel = relaxation === nothing ? (startswith(variant, "EXACT") ? "EXACT" : :RLT) : relaxation

    m = build_and_solve(data;
        variant=variant,
        build_only=true,
        optimizer=optimizer,
        verbose=verbose
    )
    optimize!(m)

    Main.PrettyPrint.print_model_solution(m;
        variant=variant,
        relaxation=rel,
        show_mats=show_mats,
        pp_kwargs...
    )

    return m
end

function demo(data::Dict{String,Any};
    variant::String="RLT_CompBinBou",
    optimizer,
    verbose::Bool=false,
    show_mats::Bool=true
)
    solve_and_print(data; variant=variant, optimizer=optimizer, verbose=verbose, show_mats=show_mats)
    return nothing
end

export build_and_solve, solve_and_print, demo, prepare_instance_comp

end # module RLTComp

using Gurobi
using JuMP

# --- PrettyPrint (your existing helper) ---
include(normpath(joinpath(@__DIR__, "..", "instances", "prettyprint.jl")))
using .PrettyPrint

# --- Instance ---
include(normpath(joinpath(@__DIR__, "..", "instances", "alper_stqp_instance.jl")))
using .AlperStqpInstances
inst = alper_stqp_rho3_instance()

variants = [
    "EXACT_CompBin",
    "EXACT_CompBou",
    "RLT_CompBin",
    "RLT_CompBinBou",
    "RLT_CompBou",
]

println("\n==============================")
println("instance id = ", get(inst, "id", "unknown"))
println("==============================\n")

for var in variants
    println("\n------------------------------")
    println("variant = ", var)
    println("------------------------------")

    m = RLTComp.build_and_solve(
        inst;
        variant=var,
        build_only=true,
        optimizer=Gurobi.Optimizer,
        verbose=false
    )
    optimize!(m)

    PrettyPrint.print_model_solution(
        m;
        variant=var,
        relaxation=(startswith(var, "EXACT") ? "EXACT" : :RLT),
        show_mats=true
    )
end