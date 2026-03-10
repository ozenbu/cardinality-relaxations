
"""
using LinearAlgebra
# --- Data (exact rationals) ---
n  = 4
ρ  = 3//1
e  = ones(Rational{Int}, n)

# Q0 (as rationals)
Q0 = [
    3//10000    127//1250   79//2500    867//10000;
    127//1250   1//500      1001//10000 1059//10000;
    79//2500    1001//10000 -1//2000    -703//10000;
    867//10000  1059//10000 -703//10000 -1063//10000
]
Q0 = Q0*20000

Q0p = max.(Q0,0)
Q0n = max.(-Q0,0)

# q0 (as rationals)
q0 = [-1973//10000, -2535//10000, -1967//10000, -973//10000]
q0 = q0*20000

q0p = max.(q0,0)
q0n = max.(-q0,0)

# Box A,b with M = I
A = [ Matrix{Rational{Int}}(I,n,n) ; -Matrix{Rational{Int}}(I,n,n) ]  # 8×4
b = ones(Rational{Int}, 2n)

# --- Core Γ construction (all elementwise ≥0, ΓUU/ΓLL symmetric) ---
absQ0 = abs.(Q0)
ΓUU = absQ0 .// 2         # 0.5*|Q0|
ΓLL = copy(ΓUU)           # same
ΓUL = zeros(Rational{Int}, n, n)
for i in 1:n, j in 1:n
    if Q0[i,j] < 0
        ΓUL[i,j] = absQ0[i,j]      # 0.5*(|Q0|-Q0) at negatives → |Q0|
    else
        ΓUL[i,j] = 0//1
    end
end
# ΓUL is symmetric here (negatives are symmetric positions)

# --- E-block multipliers & (x,u,U)-related ones ---
ϕ = zeros(Rational{Int}, n)   # phi := 0
γ = -q0                       # γ = -q0  (elementwise ≥ 0)
δ = zeros(Rational{Int}, n)   # δ := 0   (allowed by complementarity)

Λ = zeros(Rational{Int}, n, n)
κ = zeros(Rational{Int}, n)

# Target the dual value you want: D_target = -0.49655  (example)
D_target = -49655//100000   # exact rational
σ = -D_target // ρ          # σ = 49655/300000

Γsum = ΓUU .+ ΓLL .+ ΓUL .+ ΓUL'
t    = γ .+ (Γsum * e) .// 2         # t = γ + 0.5 Γsum e
tbar = sum(t) // n

ψ = -t .+ (2*tbar - σ) .* e

τ = zeros(Rational{Int}, n)                          # bump if you need W ≥ 0
W = (Γsum .// 2) .- Diagonal(τ) .- ((ψ*e' .+ e*ψ') .// 2)
Πp = W .// 2
Πm = W .// 2


# All A-RLT multipliers set to zero by complementarity (slacks > 0 at EU optimum)
μ   = zeros(Rational{Int}, size(A,1))
Θ   = zeros(Rational{Int}, size(A,1), size(A,1))
ΓAU = zeros(Rational{Int}, size(A,1), n)
ΓAL = zeros(Rational{Int}, size(A,1), n)
Ω   = zeros(Rational{Int}, size(A,1), n)

# No H equalities in this instance
λ = Rational{Int}[]                 # empty
Φ = Matrix{Rational{Int}}(undef, 0, n)
Ξ = Matrix{Rational{Int}}(undef, 0, n)
α = Rational{Int}[]                 # no quadratic ineqs
β = Rational{Int}[]                 # no quadratic eqs

# stationarity checks (note the -ρ*ψ term!)
SX = ΓUU .+ ΓLL .- ΓUL .- ΓUL' .- Q0
Sx = q0 .+ γ .- δ .+ (Πp .- Πm) * e
SR = (ΓUU .- ΓLL .+ ΓUL .- ΓUL') .- Πp .+ Πm
Su = .- (γ .+ δ) .+ σ .* e .- τ .- (ρ .* ψ) .- (Πp .+ Πm) * e
SU = .- (Γsum) .// 2 .+ Diagonal(τ) .+ (ψ * e' .+ e * ψ') .// 2 .+ (Πp .+ Πm)

@assert SX == 0 .* SX
@assert Sx == 0 .* Sx
@assert SR == 0 .* SR
@assert Su == 0 .* Su
@assert SU == 0 .* SU


println(−μ'*b−(1/2)*b'*Θ*b− ρ*σ−κ'*e - e'*Ω'*b - (1/2)*e'*Λ*e)





const T = Rational{Int}

# Problem scalars (for reference)
n  = 4
ρ  = 3//1
e  = ones(T, n)

# Empty blocks
α  = T[]                # []
μ  = T[]                # []
λ  = T[]                # []
β  = T[]                # []

Ω  = zeros(T, 0, 4)     # 0×4
Φ  = zeros(T, 0, 4)     # 0×4
ΓAU = zeros(T, 0, 4)    # 0×4
ΓAL = zeros(T, 0, 4)    # 0×4
Ξ  = zeros(T, 0, 4)     # 0×4
Θ  = zeros(T, 0, 0)     # 0×0

# Free / signed vectors (EU)
ψ = T[ 603//2,  603//2, -603//2,  327//2 ]
ϕ = T[ -632//1, -2002//1, 0//1, -3165//2 ]

τ = T[ -1849//2, -4487//2, 613//2, -683//1 ]
σ = 6595//2

# Nonnegative vectors
γ = T[ 349//2, 1204//1, 1300//1, 0//1 ]
δ = T[ 0//1, 0//1, 0//1, 0//1 ]
κ = T[ 0//1, 0//1, 0//1, 0//1 ]

# Γ blocks (≥ 0)
ΓUU = T[
    6//1        8127//4    632//1     1734//1;
    8127//4     40//1      2002//1    7825//4;
    632//1      2002//1    0//1       0//1;
    1734//1     7825//4    0//1       0//1
]

ΓLL = T[
    0//1        1//4       0//1       0//1;
    1//4        0//1       0//1       647//4;
    0//1        0//1       0//1       0//1;
    0//1        647//4     0//1       0//1
]

ΓUL = T[
    0//1  0//1   0//1   0//1;
    0//1  0//1   0//1   0//1;
    0//1  0//1   5//1   703//1;
    0//1  0//1   703//1 1063//1
] # symmetric

# Π blocks (≥ 0)
Πm = T[
    626//1  0//1   0//1        31//4;
    0//1    1962//1 0//1       3297//4;
    0//1    0//1   0//1        0//1;
    0//1    0//1   3165//2     3165//2
]

Πp = T[
    0//1     2799//2  0//1       4439//4;
    59//2    0//1     0//1       2467//4;
    632//1   2002//1  0//1       0//1;
    303//2   212//1   0//1       0//1
]

# Λ (≥ 0, symmetric)
Λ = T[
    0//1  0//1  0//1     0//1;
    0//1  0//1  0//1     0//1;
    0//1  0//1  0//1     77//2;
    0//1  0//1  77//2    0//1
]


# stationarity checks (note the -ρ*ψ term!)
SX = ΓUU .+ ΓLL .- ΓUL .- ΓUL' .- Q0
Sx = q0 .+ γ .- (ρ .* ϕ)  .+ (Πp .- Πm) * e
SR = (ΓUU .- ΓLL .+ ΓUL .- ΓUL') .+ ϕ*e'.- Πp .+ Πm
Su = .- (γ .+ δ) .+ σ .* e .- τ .- (ρ .* ψ) .- (Πp .+ Πm) * e .+ Λ*e
SU = .- (ΓUU .+ ΓLL .+ ΓUL .+ ΓUL') ./ 2 .+ Diagonal(τ) .+ (ψ * e' .+ e * ψ') ./ 2 .- Λ ./ 2 .+ ( Πp .+ Πm .+ Πp' .+ Πm')./ 2 

@assert SX == 0 .* SX
@assert Sx == 0 .* Sx
@assert SR == 0 .* SR
@assert Su == 0 .* Su
@assert SU == 0 .* SU


println(−μ'*b−(1/2)*b'*Θ*b− ρ*σ−κ'*e - e'*Ω'*b - (1/2)*e'*Λ*e)


"""

using LinearAlgebra

const T = Rational{Int}

function main()
    # -------------------------------
    # Instance data (exact rationals)
    # -------------------------------
    n  = 4
    ρ  = 3//1
    e  = ones(T, n)

    # Q0 as rationals, then scale by 20000 (exact)
    Q0 = T[
        3//10000    127//1250   79//2500    867//10000;
        127//1250   1//500      1001//10000 1059//10000;
        79//2500    1001//10000 -1//2000    -703//10000;
        867//10000  1059//10000 -703//10000 -1063//10000
    ]
    Q0 = Q0 * (20000//1)

    # q0 as rationals, then scale by 20000 (exact)
    q0 = T[-1973//10000, -2535//10000, -1967//10000, -973//10000]
    q0 = q0 * (20000//1)

    # Box: A x <= b, with A = [I; -I], b = ones(8)
    In = Matrix{T}(I, n, n)
    A  = [In; -In]               # 8×4
    b  = ones(T, 2n)             # length 8

    # --------------------------------------------
    # Your EU dual certificate values (UNCHANGED)
    # --------------------------------------------

    # Free / signed vectors (EU)
    ψ = T[ 603//2,  603//2, -603//2,  327//2 ]
    ϕ = T[ -632//1, -2002//1, 0//1, -3165//2 ]

    τ = T[ -1849//2, -4487//2, 613//2, -683//1 ]
    σ = 6595//2

    # Nonnegative vectors
    γ = T[ 349//2, 1204//1, 1300//1, 0//1 ]
    δ = T[ 0//1, 0//1, 0//1, 0//1 ]
    κ = T[ 0//1, 0//1, 0//1, 0//1 ]

    # Γ blocks (entrywise ≥ 0)
    ΓUU = T[
        6//1        8127//4    632//1     1734//1;
        8127//4     40//1      2002//1    7825//4;
        632//1      2002//1    0//1       0//1;
        1734//1     7825//4    0//1       0//1
    ]

    ΓLL = T[
        0//1        1//4       0//1       0//1;
        1//4        0//1       0//1       647//4;
        0//1        0//1       0//1       0//1;
        0//1        647//4     0//1       0//1
    ]

    ΓUL = T[
        0//1  0//1   0//1   0//1;
        0//1  0//1   0//1   0//1;
        0//1  0//1   5//1   703//1;
        0//1  0//1   703//1 1063//1
    ] # symmetric

    # Π blocks (entrywise ≥ 0)
    Πm = T[
        626//1  0//1    0//1       31//4;
        0//1    1962//1 0//1       3297//4;
        0//1    0//1    0//1       0//1;
        0//1    0//1    3165//2    3165//2
    ]

    Πp = T[
        0//1     2799//2  0//1       4439//4;
        59//2    0//1     0//1       2467//4;
        632//1   2002//1  0//1       0//1;
        303//2   212//1   0//1       0//1
    ]

    # Λ (entrywise ≥ 0, symmetric)
    Λ = T[
        0//1  0//1  0//1     0//1;
        0//1  0//1  0//1     0//1;
        0//1  0//1  0//1     77//2;
        0//1  0//1  77//2    0//1
    ]

    # -------------------------------------------------------
    # Blocks set to zero (correct dimensions!)
    # -------------------------------------------------------
    μ   = zeros(T, size(A,1))                  # length 8
    Θ   = zeros(T, size(A,1), size(A,1))       # 8×8
    ΓAU = zeros(T, size(A,1), n)               # 8×4
    ΓAL = zeros(T, size(A,1), n)               # 8×4
    Ω   = zeros(T, size(A,1), n)               # 8×4

    # -------------------------------------------------------
    # Stationarity checks (EXACTLY your expressions)
    # -------------------------------------------------------
    SX = ΓUU .+ ΓLL .- ΓUL .- ΓUL' .- Q0

    Sx = q0 .+ γ .- (ρ .* ϕ) .+ (Πp .- Πm) * e

    SR = (ΓUU .- ΓLL .+ ΓUL .- ΓUL') .+ ϕ*e' .- Πp .+ Πm

    Su = .- (γ .+ δ) .+ σ .* e .- τ .- (ρ .* ψ) .- (Πp .+ Πm) * e .+ Λ*e

    SU = .- (ΓUU .+ ΓLL .+ ΓUL .+ ΓUL') ./ 2 .+
         Diagonal(τ) .+
         (ψ * e' .+ e * ψ') ./ 2 .-
         Λ ./ 2 .+
         (Πp .+ Πm .+ Πp' .+ Πm') ./ 2

    @assert SX == 0 .* SX
    @assert Sx == 0 .* Sx
    @assert SR == 0 .* SR
    @assert Su == 0 .* Su
    @assert SU == 0 .* SU

    # -------------------------------------------------------
    # Entrywise nonnegativity checks for required multipliers
    # -------------------------------------------------------
    entrywise_min(X) = minimum(vec(X))

    @assert minimum(μ) >= 0
    @assert minimum(γ) >= 0
    @assert minimum(δ) >= 0
    @assert minimum(κ) >= 0

    @assert entrywise_min(Θ)   >= 0
    @assert entrywise_min(Ω)   >= 0
    @assert entrywise_min(ΓAU) >= 0
    @assert entrywise_min(ΓAL) >= 0

    @assert entrywise_min(ΓUU) >= 0
    @assert entrywise_min(ΓLL) >= 0
    @assert entrywise_min(ΓUL) >= 0
    @assert entrywise_min(Πp)  >= 0
    @assert entrywise_min(Πm)  >= 0
    @assert entrywise_min(Λ)   >= 0

    # -------------------------------------------------------
    # Dual objective (same as your print expression)
    # -------------------------------------------------------
    obj = -(μ' * b) -
          (1//2) * (b' * Θ * b) -
          ρ * σ -
          (κ' * e) -
          (e' * Ω' * b) -
          (1//2) * (e' * Λ * e)

    println("All stationarity asserts passed.")
    println("All nonnegativity asserts passed.")
    println("Dual objective value = ", obj)  # expected: -9931//1

    return obj
end

main()