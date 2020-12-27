# CheckParameters.jl

function checitemparameters(pars::IRTmodel)
    checkitem(pars)
end

function checkitem(pars::graded)
    p = take(pars)
    for p in take(pars)
        if isnan.(p) || isnan.(p)
            @error "Any parameter is $(typeof(p)). Estimation seems to fail."
            return true
        end
    end
    return false
end

function checkitem(pars::guessing)
    p = take(pars)
    for p in take(pars)
        if isnan.(p) || isnan.(p)
            @error "Any parameter is $p. Estimation seems to fail."
            return true
        end
    end
    # if !(0.0 ≤ p[end] ≤ 1.0)
    #     @warn "Guessing parameter is out of range 0 to 1. Estimation seems to fail"
    #     return true
    # end
    return false
end

function checkitem(pars::LPE)
    p = take(pars)
    for p in take(pars)
        if isnan.(p)
            @error "Any parameter is $(typeof(p)). Estimation seems to fail."
            @show pars
            return true
        end
    end
    return false
end