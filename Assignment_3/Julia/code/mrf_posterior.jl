function mrf_denoise_nlposterior(x::Array{Float64,2},y::Array{Float64,2}, sigma_noise::Float64, sigma::Float64, alpha::Float64)
    
    return p::Float64;
end

function grad_mrf_denoise_nlposterior(x::Array{Float64,2}, y::Array{Float64,2}, sigma_noise::Float64, sigma::Float64, alpha::Float64)
    
    return g::Float64;
end

function mrf_inpaint_nlposterior(x::Array{Float64,2}, m::BitArray{2}, sigma::Float64, alpha::Float64)
    
    return p::Float64;
end

function grad_mrf_inpaint_nlposterior(x::Array{Float64,2}, m::BitArray{2}, sigma::Float64, alpha::Float64)
    
    return g::Float64;
end