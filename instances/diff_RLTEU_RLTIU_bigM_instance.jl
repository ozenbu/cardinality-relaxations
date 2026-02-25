module EUIUdiffinstance

using LinearAlgebra

export euiu_diff_instance

function euiu_diff_instance()
    # small rational example (scaled to integers)
    Q0 = [
        3//10000    127//1250   79//2500    867//10000;
        127//1250   1//500      1001//10000 1059//10000;
        79//2500    1001//10000 -1//2000    -703//10000;
        867//10000  1059//10000 -703//10000 -1063//10000
    ]
    Q0d = Float64.(Q0 .* 20000)

    q0 = [-1973//10000, -2535//10000, -1967//10000, -973//10000]
    q0d = Float64.(q0 .* 20000)

    # Box: -1 <= x_i <= 1  encoded by A x <= b
    A2 = [
         1.0  0.0  0.0  0.0
         0.0  1.0  0.0  0.0
         0.0  0.0  1.0  0.0
         0.0  0.0  0.0  1.0
        -1.0  0.0  0.0  0.0
         0.0 -1.0  0.0  0.0
         0.0  0.0 -1.0  0.0
         0.0  0.0  0.0 -1.0
    ]
    b2 = [1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0]

    # Big-M: easiest is symmetric bounds |x_i| <= 1
    # Your parser accepts either ("M") or ("Mminus","Mplus").
    # Here we use Mminus/Mplus vectors to be explicit & stable.
    Mminus = ones(4)   # x >= -1*u  (with u<=1 it means x >= -u)
    Mplus  = ones(4)   # x <=  1*u

    inst = Dict{String,Any}()

    inst["id"]  = "n4_rho3_EU_IU_diff_rational_scaled_fv"
    inst["n"]   = 4
    inst["rho"] = 3.0

    inst["Q0"] = Q0d
    inst["q0"] = q0d

    inst["Qi"] = nothing; inst["qi"] = nothing; inst["ri"] = nothing
    inst["Pi"] = nothing; inst["pi"] = nothing; inst["si"] = nothing

    inst["A"] = A2
    inst["b"] = b2
    inst["H"] = nothing
    inst["h"] = nothing

    inst["Mminus"] = Mminus
    inst["Mplus"]  = Mplus

    # optional metadata
    inst["base_type"] = "box"
    inst["base_tag"]  = "B"
    inst["convexity"] = "UNK"   # Q0 here is indefinite (likely); you can update if you want
    inst["seed"] = 0

    return inst
end

end # module