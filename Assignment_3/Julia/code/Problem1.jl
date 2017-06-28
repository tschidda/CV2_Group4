using Images
using Optim
using PyPlot

include("add_noise.jl")
include("mrf_posterior.jl")
include("psnr.jl")


function load_images()
  img = 255.*channelview(float64.(Gray.(load("..//data//la.png"))))
  img_noisy = 255.*channelview(float64.(Gray.(load("..//data//la-noisy.png"))))
  return img::Array{Float64, 2}, img_noisy::Array{Float64, 2}
end

function f_denoise(x::Array{Float64, 1}, y::Array{Float64,2}, sigma_noise::Float64, sigma::Float64, alpha::Float64)
    x = reshape(x, size(y))
    return mrf_denoise_nlposterior(x, y, sigma_noise, sigma, alpha)
end

function g_denoise(x::Array{Float64, 1}, y::Array{Float64,2}, sigma_noise::Float64, sigma::Float64, alpha::Float64, storage::Array{Float64, 1})
    x = reshape(x, size(y))
    storage[1:end] = grad_mrf_denoise_nlposterior(x, y, sigma_noise, sigma, alpha)
end

function apply_optimizer(x::Array{Float64, 1}, y::Array{Float64,2}, sigma_noise::Float64, sigma::Float64, alpha::Float64)
  res = optimize(x->f_denoise(x,y,sigma_noise,sigma,alpha),(x,storage)->g_denoise(x,y,sigma_noise,sigma,alpha,storage), x, AcceleratedGradientDescent(), Optim.Options(show_trace=false))
  display(res)
  return Optim.minimizer(res)
end

function denoise(x::Array{Float64, 2}, sigma_noise::Float64, sigma::Float64, alpha::Float64)
  levels = 4
  G_X = gaussian_pyramid(x, levels, 2.0, 1.0)
  img_denoised = zeros(Float64, size(G_X[levels]))
  for idx in levels:-1:1
    img_denoised = apply_optimizer(G_X[idx][:], G_X[idx], sigma_noise ,sigma, alpha)
    img_denoised = reshape(img_denoised, size(G_X[idx]))
    if idx > 1
      img_denoised = imresize(img_denoised, size(G_X[idx-1]))
    end
  end
  return img_denoised
end

function problem1()
  sigma = 12.0
  alpha = 0.7
  sigma_noise = 15.0
  PyPlot.figure()
  img,_  = load_images()
  imshow(img, "gray")
  img_noisy = add_noise(img, sigma_noise)
  PyPlot.figure()
  imshow(img_noisy, "gray")
  img_denoised =@time denoise(img_noisy, sigma_noise, sigma, alpha)
  PyPlot.figure()
  imshow(img_denoised, "gray")
  display(psnr(img, img_denoised))

end

function test_prior()
  sigma = 10.0
  alpha = 1.0
  sigma_noise = 15.0
  img = load_image()
  ideal = fill(10.0, size(img))
  img_noisy = add_noise(img, sigma_noise)
  display(mrf_nlprior(ideal, sigma, alpha))
  display(mrf_nlprior(img, sigma, alpha))
  display(mrf_nlprior(img_noisy, sigma, alpha))
end

function test_likelihood()
  sigma = 10.0
  alpha = 1.0
  sigma_noise = 15.0
  img = load_image()
  ideal = fill(10.0, size(img))
  img_noisy = add_noise(img, sigma_noise)
  display(mrf_denoise_nllh(img,ideal, sigma_noise))
  display(mrf_denoise_nllh(img,img, sigma_noise))
  display(mrf_denoise_nllh(img,img_noisy, sigma_noise))
  display(mrf_denoise_nllh(img_noisy,img_noisy, sigma_noise))
end
