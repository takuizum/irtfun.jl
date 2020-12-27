# Qmatrix.jl

function explanatoryMIRT(ndims, I)
    q = ones(Bool, I, ndims)
    # Constraints for identification
    if ndims > 1
        for d in 2:ndims
            q[end-d+2, d] = false
        end
    end
    return q
end