include("mrf_prior.jl")
include("mrf_likelihood.jl")

function mrf_denoise_nlposterior(x::Array{Float64,2},y::Array{Float64,2}, sigma_noise::Float64, sigma::Float64, alpha::Float64)
    p = mrf_nlprior(x, sigma, alpha)+mrf_denoise_nllh(x, y, sigma_noise)
    return p::Float64;
end

function grad_mrf_denoise_nlposterior(x::Array{Float64,2}, y::Array{Float64,2}, sigma_noise::Float64, sigma::Float64, alpha::Float64)
    g = grad_mrf_nlprior(x, sigma, alpha)+grad_mrf_denoise_nllh(x, y, sigma_noise)
    return g::Array{Float64, 2};
end

function mrf_inpaint_nlposterior(x::Array{Float64,2}, m::BitArray{2}, sigma::Float64, alpha::Float64)

    return p::Float64;
end

function grad_mrf_inpaint_nlposterior(x::Array{Float64,2}, m::BitArray{2}, sigma::Float64, alpha::Float64)

    return g::Float64;
end
