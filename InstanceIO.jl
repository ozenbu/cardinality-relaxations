###############################
# InstanceIO.jl
#
# Shared instance loader. All experiments should `include` this
# file rather than reimplementing JSON parsing.
#
# Usage:
#   include("/home/s2719899/BAM/InstanceIO.jl")
#   instances = load_instances("/home/s2719899/BAM/instances.json")
#
# Prerequisite: RLTBigM must be loaded in the REPL session
# (Main.RLTBigM.prepare_instance is used as the final validity check).
#
# Deliberately not wrapped in a module: include drops the
# functions directly into Main scope, avoiding the `using ..X`
# vs `Main.X` distinction.
###############################

using JSON
using LinearAlgebra: Diagonal, diag

# ------------------------
# Basic coercions
# ------------------------
_as_float(x) = x isa Number ? Float64(x) :
    throw(ArgumentError("Expected number, got $(typeof(x))"))

_as_vec(x) = x === nothing ? nothing : Float64.(collect(x))

function _as_mat(x)
    x === nothing && return nothing
    if x isa AbstractMatrix
        return Float64.(x)
    elseif x isa AbstractVector
        @assert !isempty(x) && x[1] isa AbstractVector
        rows = [permutedims(Float64.(v)) for v in x]
        return reduce(vcat, rows)
    else
        error("Unsupported matrix type: $(typeof(x))")
    end
end

function _as_mat_square(x, n::Int; name::String="Q0")
    M = _as_mat(x)
    (size(M, 1) == n && size(M, 2) == n) ||
        error("$name must be nxn with n=$n, got size=$(size(M))")
    return M
end

function _as_mat_cols_n(x, n::Int; name::String="A")
    M = _as_mat(x)
    if size(M, 2) == n
        return M
    elseif size(M, 1) == n
        return Matrix(M')
    else
        error("$name must have n=$n columns (or be transposed with n rows). Got size=$(size(M)).")
    end
end

_as_tuple_vec(x) = x === nothing ? nothing :
    (x isa Tuple ? x : Tuple(x)) .|> v -> Float64.(collect(v))

function _as_tuple_mat_square(x, n::Int; name::String="Qi")
    x === nothing && return nothing
    xs = x isa Tuple ? x : Tuple(x)
    return Tuple(_as_mat_square(M, n; name=name) for M in xs)
end

_as_tuple_num(x) = x === nothing ? nothing :
    (x isa Tuple ? x : Tuple(x)) .|> _as_float

function _as_diagvec(x, n::Int; name::String="M")
    x === nothing && return nothing
    if x isa AbstractVector && !(x isa AbstractVector{<:AbstractVector})
        v = Float64.(collect(x))
        length(v) == n || error("$name must have length n=$n, got length=$(length(v))")
        return v
    elseif x isa Diagonal
        v = Float64.(diag(x))
        length(v) == n || error("$name size mismatch, need n=$n")
        return v
    elseif x isa AbstractMatrix || x isa AbstractVector
        M = _as_mat(x)
        (size(M, 1) == n && size(M, 2) == n) ||
            error("$name must be nxn (diagonal), got $(size(M))")
        off = M .- Diagonal(diag(M))
        maximum(abs.(off)) <= 1e-12 ||
            error("$name must be diagonal (nonzero off-diagonals found).")
        return Float64.(diag(M))
    else
        error("Unsupported type for $name: $(typeof(x))")
    end
end

function _get_any(d::Dict{String,Any}, keys::Tuple{Vararg{String}})
    for k in keys
        if haskey(d, k) && d[k] !== nothing
            return d[k]
        end
    end
    return nothing
end

# ============================================================
# read_instance!  (in-place coercion + validation)
# ============================================================
function read_instance!(d::Dict{String,Any})
    @assert haskey(d, "n") "Missing key `n`."
    d["n"] = Int(d["n"])
    n = d["n"]

    inst_label = get(d, "id", "(no id)")

    if haskey(d, "rho") && d["rho"] !== nothing
        d["rho"] = Int(round(_as_float(d["rho"])))
    else
        error("Instance $inst_label: missing key `rho`.")
    end

    if haskey(d, "Q0") && d["Q0"] !== nothing
        d["Q0"] = _as_mat_square(d["Q0"], n; name="Q0")
    end
    if haskey(d, "A") && d["A"] !== nothing
        d["A"] = _as_mat_cols_n(d["A"], n; name="A")
        (haskey(d, "b") && d["b"] !== nothing) ||
            error("Instance $inst_label: A provided but b missing.")
    end
    if haskey(d, "H") && d["H"] !== nothing
        d["H"] = _as_mat_cols_n(d["H"], n; name="H")
        (haskey(d, "h") && d["h"] !== nothing) ||
            error("Instance $inst_label: H provided but h missing.")
    end
    for k in ("q0", "b", "h")
        if haskey(d, k) && d[k] !== nothing
            d[k] = _as_vec(d[k])
        end
    end
    if haskey(d, "A") && d["A"] !== nothing && haskey(d, "b") && d["b"] !== nothing
        length(d["b"]) == size(d["A"], 1) ||
            error("Instance $inst_label: length(b)=$(length(d["b"])) != size(A,1)=$(size(d["A"],1)).")
    end
    if haskey(d, "H") && d["H"] !== nothing && haskey(d, "h") && d["h"] !== nothing
        length(d["h"]) == size(d["H"], 1) ||
            error("Instance $inst_label: length(h)=$(length(d["h"])) != size(H,1)=$(size(d["H"],1)).")
    end

    d["Qi"] = _as_tuple_mat_square(get(d, "Qi", nothing), n; name="Qi")
    d["Pi"] = _as_tuple_mat_square(get(d, "Pi", nothing), n; name="Pi")
    d["qi"] = _as_tuple_vec(get(d, "qi", nothing))
    d["pi"] = _as_tuple_vec(get(d, "pi", nothing))
    d["ri"] = _as_tuple_num(get(d, "ri", nothing))
    d["si"] = _as_tuple_num(get(d, "si", nothing))

    mminus_raw = _get_any(d, ("M_minus", "Mminus"))
    mplus_raw  = _get_any(d, ("M_plus",  "Mplus"))
    m_raw      = _get_any(d, ("M",))

    has_pair   = (mminus_raw !== nothing) || (mplus_raw !== nothing)
    has_single = (m_raw !== nothing)

    if has_pair
        (mminus_raw !== nothing && mplus_raw !== nothing) ||
            error("Instance $inst_label: provide BOTH Mminus & Mplus.")
        mminus = _as_diagvec(mminus_raw, n; name="Mminus")
        mplus  = _as_diagvec(mplus_raw,  n; name="Mplus")
        d["Mminus"]  = mminus
        d["Mplus"]   = mplus
        d["M_minus"] = mminus
        d["M_plus"]  = mplus
        d["M"] = max.(mminus, mplus)
    elseif has_single
        m = _as_diagvec(m_raw, n; name="M")
        d["M"]       = m
        d["Mminus"]  = copy(m)
        d["Mplus"]   = copy(m)
        d["M_minus"] = copy(m)
        d["M_plus"]  = copy(m)
    else
        error("Instance $inst_label: missing Big-M data.")
    end

    # Final validity check via single source of truth.
    Main.RLTBigM.prepare_instance(d)

    return d
end

# ============================================================
# load_instances(json_path)
# ============================================================
function load_instances(json_path::AbstractString)
    raw = JSON.parsefile(json_path)
    raw isa AbstractVector ||
        error("$json_path : expected a JSON array of instance objects.")
    instances = Dict{String,Any}[]
    for (idx, rec) in enumerate(raw)
        d = Dict{String,Any}(rec)
        read_instance!(d)
        push!(instances, d)
    end
    return instances
end