include("studentt.jl")

function mrf_nlprior(x::Array{Float64,2}, sigma::Float64, alpha::Float64)
    p = -sum(log(studentt(diff(x,1)[:],sigma,alpha))) -sum(log(studentt(diff(x,2)[:],sigma,alpha)));
    return p::Float64;
end

function grad_mrf_nlprior(x::Array{Float64,2}, sigma::Float64, alpha::Float64)
    p = zeros(Float64, size(x));
    for i in 1:size(x,1)
        for j in 1:size(x,2)
            dp = 0;
            if j < size(x,2)
                dp += (grad_studentt(x[i,j]-x[i,j+1],sigma, alpha) ./ studentt(x[i,j]-x[i,j+1],sigma,alpha));
            end
            if j > 1
                dp += (-grad_studentt(x[i,j-1]-x[i,j],sigma, alpha) ./ studentt(x[i,j-1]-x[i,j],sigma,alpha));
            end
            if i < size(x,1)
                dp += (grad_studentt(x[i,j]-x[i+1,j],sigma, alpha) ./ studentt(x[i,j]-x[i+1,j],sigma,alpha));
            end
            if i > 1
                dp += (-grad_studentt(x[i-1,j]-x[i,j],sigma, alpha) ./ studentt(x[i-1,j]-x[i,j],sigma,alpha));
            end
            p[i,j]=-dp;

        end
    end
    return p::Array{Float64, 2};
end
