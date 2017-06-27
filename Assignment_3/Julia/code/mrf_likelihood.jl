function mrf_denoise_nllh(x::Array{Float64,2}, y::Array{Float64, 2}, sigma)
    llh_factor = -1/(2*(sigma)^2);
    llh_sum = (x-y)^2;
    nllh = (-1)*(sum(llh_sum) .* llh_factor);
    return nllh;
end

function grad_mrf_denoise_nllh(x::Array{Float64,2}, y::Array{Float64,2}, sigma)
    g_nllh = zeros(Float64, size(x))
    factor = 1/( -(sigma^2) )
    for i in 1:size(x, 1)
        for j in 1:size(x, 2)
            g_nllh[i,j]=(x[i,j]-y[i,j])
        end
    end
    g_nllh .*=factor
    return -g_nllh;
end
