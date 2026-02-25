module PrettyPrint

using Printf
using LinearAlgebra

export print_instance, print_solution, print_vec, print_mat

# ----------------------------
# Helpers
# ----------------------------
# Makes tiny numbers (|v| < tol) 0.0.
_clean(v::Real; tol::Real=1e-9) = (isfinite(v) && abs(v) < tol) ? 0.0 : Float64(v)

function _fmt(v::Real; digits::Int=6, tol::Real=1e-9, width::Int=12)
    @sprintf("%*.*f", width, digits, _clean(v; tol=tol))
end

# ----------------------------
# Pretty vector / matrix
# ----------------------------
function print_vec(name::AbstractString, v::AbstractVector;
                   digits::Int=6, tol::Real=1e-9, per_line::Bool=true)
    println("\n", name, " =")
    if per_line
        for i in eachindex(v)
            @printf("  %s[%d] = %.*f\n", name, i, digits, _clean(v[i]; tol=tol))
        end
    else
        print("  [")
        for i in eachindex(v)
            s = @sprintf("%.*f", digits, _clean(v[i]; tol=tol))
            print(i == firstindex(v) ? s : ", " * s)
        end
        println("]")
    end
    return nothing
end

function print_mat(name::AbstractString, M::AbstractMatrix;
                   digits::Int=6, tol::Real=1e-9,
                   max_rows::Int=12, max_cols::Int=12, width::Int=12)
    println("\n", name, " =")
    m, n = size(M)

    r_show = min(m, max_rows)
    c_show = min(n, max_cols)

    for i in 1:r_show
        for j in 1:c_show
            print(_fmt(M[i,j]; digits=digits, tol=tol, width=width))
        end
        if c_show < n
            print("  ...")
        end
        print("\n")
    end
    if r_show < m
        println("  ...")
    end
    return nothing
end

# ----------------------------
# Instance printer
# ----------------------------
function print_instance(inst::Dict{String,Any};
                        show_mats::Bool=true, digits::Int=4, tol::Real=1e-12)

    id        = get(inst, "id", "unknown")
    n         = get(inst, "n", missing)
    rho       = get(inst, "rho", missing)
    base_type = get(inst, "base_type", "unknown")
    convexity = get(inst, "convexity", "unknown")
    want_cvx  = get(inst, "want_convex", missing)
    bigM      = get(inst, "bigM_scale", missing)

    println("\n================= INSTANCE =================")
    println("id         = ", id)
    println("n          = ", n)
    println("rho        = ", rho)
    println("base_type  = ", base_type)
    println("convexity  = ", convexity)
    println("want_convex= ", want_cvx)
    println("bigM_scale = ", bigM)

    Q0 = get(inst, "Q0", nothing)
    q0 = get(inst, "q0", nothing)

    if Q0 !== nothing
        Q0sym = Symmetric(Q0)
        evals = eigvals(Matrix(Q0sym))
        println("Q0 eig min = ", minimum(evals))
        println("Q0 eig max = ", maximum(evals))
    end

    if show_mats
        if Q0 !== nothing; print_mat("Q0", Q0; digits=digits, tol=tol) end
        if q0 !== nothing; print_vec("q0", q0; digits=digits, tol=tol, per_line=false) end

        A = get(inst, "A", nothing)
        b = get(inst, "b", nothing)
        H = get(inst, "H", nothing)
        h = get(inst, "h", nothing)
        Mminus = get(inst, "Mminus", nothing)
        Mplus  = get(inst, "Mplus", nothing)

        if A !== nothing; print_mat("A", A; digits=digits, tol=tol) end
        if b !== nothing; print_vec("b", b; digits=digits, tol=tol, per_line=false) end
        if H !== nothing; print_mat("H", H; digits=digits, tol=tol) end
        if h !== nothing; print_vec("h", h; digits=digits, tol=tol, per_line=false) end
        if Mminus !== nothing; print_vec("Mminus", Mminus; digits=digits, tol=tol, per_line=false) end
        if Mplus  !== nothing; print_vec("Mplus",  Mplus;  digits=digits, tol=tol, per_line=false) end
    end

    return nothing
end

# ----------------------------
# Solution printer
# ----------------------------
function print_solution(st, obj, x, u, X, R, U, t;
                        digits::Int=6, tol::Real=1e-9,
                        show_mats::Bool=true, mat_digits::Int=6)

    println("\n================= RESULT =================")
    println("status = ", st)
    println("obj    = ", obj)
    println("time   = ", t, " sec")

    if obj === nothing || x === nothing
        return nothing
    end

    println("sum(x)      = ", sum(x))
    println("sum(u)      = ", sum(u))
    println("nnz(x>1e-6) = ", count(>(1e-6), x))

    print_vec("x", x; digits=digits, tol=tol)
    print_vec("u", u; digits=digits, tol=tol)

    if show_mats
        if X !== nothing; print_mat("X", X; digits=mat_digits, tol=tol) end
        if U !== nothing; print_mat("U", U; digits=mat_digits, tol=tol) end
        if R !== nothing; print_mat("R", R; digits=mat_digits, tol=tol) end
    end

    return nothing
end

end # module