module PrettyPrint

using Printf
using LinearAlgebra
using JuMP
using MathOptInterface
const MOI = MathOptInterface

export print_vec, print_mat, print_model_solution

# ----------------------------
# Small helpers
# ----------------------------
_clean(v::Real; tol::Real=1e-9) = (isfinite(v) && abs(v) < tol) ? 0.0 : Float64(v)

function print_vec(name::AbstractString, v::AbstractVector; digits::Int=6, tol::Real=1e-9)
    println("\n", name, " =")
    for i in eachindex(v)
        @printf("  %s[%d] = %.*f\n", name, i, digits, _clean(v[i]; tol=tol))
    end
    return nothing
end

function print_mat(name::AbstractString, M::AbstractMatrix;
                   digits::Int=6, tol::Real=1e-9,
                   max_rows::Int=20, max_cols::Int=20, width::Int=12)
    println("\n", name, " =")
    m, n = size(M)
    r_show = min(m, max_rows)
    c_show = min(n, max_cols)

    for i in 1:r_show
        for j in 1:c_show
            @printf("%*.*f", width, digits, _clean(M[i,j]; tol=tol))
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

# Detect whether a variable container is binary/continuous
function _var_kind(obj)
    if obj isa JuMP.VariableRef
        return JuMP.is_binary(obj) ? "binary" : (JuMP.is_integer(obj) ? "integer" : "continuous")
    end
    try
        return _var_kind(first(obj))
    catch
        return "unknown"
    end
end

# ----------------------------
# Main: print directly from a JuMP model
# ----------------------------
function print_model_solution(model::JuMP.Model;
                             variant::AbstractString="",
                             relaxation::Union{Symbol,AbstractString}="",
                             digits::Int=6,
                             mat_digits::Int=6,
                             tol::Real=1e-9,
                             show_mats::Bool=true,
                             show_Z::Bool=false)

    st = termination_status(model)

    obj = nothing
    try obj = objective_value(model) catch end

    t = nothing
    try t = JuMP.solve_time(model) catch end

    dict = JuMP.object_dictionary(model)

    has_u = haskey(dict, :u)
    has_v = haskey(dict, :v)

    family = has_u ? "BigM family" : (has_v ? "Complementarity family" : "Unknown family")

    has_lift = any(haskey(dict, s) for s in (:X, :U, :R, :V, :W, :Z))
    level = has_lift ? "Lifted" : "EXACT (no lifting)"

    println("\n================= RESULT =================")
    println("family     = ", family, " | level = ", level)
    if variant != "";    println("variant    = ", variant)    end
    if relaxation != ""; println("relaxation = ", relaxation) end

    if has_u; println("u type     = ", _var_kind(dict[:u])) end
    if has_v; println("v type     = ", _var_kind(dict[:v])) end

    println("status     = ", st)
    println("obj        = ", obj)
    if t !== nothing
        println("time       = ", t, " sec")
    end

    if !(st in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL))
        return nothing
    end

    _get(sym::Symbol) = haskey(dict, sym) ? dict[sym] : nothing
    _val(obj) = obj === nothing ? nothing : value.(obj)

    x = _val(_get(:x))
    u = _val(_get(:u))
    v = _val(_get(:v))

    X = _val(_get(:X))
    U = _val(_get(:U))
    R = _val(_get(:R))

    V = _val(_get(:V))
    W = _val(_get(:W))

    Z = _val(_get(:Z))

    if x !== nothing
        println("sum(x)      = ", sum(x))
        println("nnz(x>1e-6) = ", count(>(1e-6), x))
        print_vec("x", x; digits=digits, tol=tol)
    end
    if u !== nothing
        println("sum(u)      = ", sum(u))
        print_vec("u", u; digits=digits, tol=tol)
    end
    if v !== nothing
        println("sum(v)      = ", sum(v))
        print_vec("v", v; digits=digits, tol=tol)
    end

    if show_mats
        if X !== nothing; print_mat("X", X; digits=mat_digits, tol=tol) end
        if U !== nothing; print_mat("U", U; digits=mat_digits, tol=tol) end
        if R !== nothing; print_mat("R", R; digits=mat_digits, tol=tol) end
        if V !== nothing; print_mat("V", V; digits=mat_digits, tol=tol) end
        if W !== nothing; print_mat("W", W; digits=mat_digits, tol=tol) end
        if show_Z && Z !== nothing
            print_mat("Z", Z; digits=mat_digits, tol=tol, max_rows=12, max_cols=12)
        end
    end

    return nothing
end

end # module