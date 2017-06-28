using Images
using Optim
using PyPlot

include("add_noise.jl")
include("mrf_posterior.jl")
include("psnr.jl")

function load_images()
  img = 255.*channelview(float64.(Gray.(load("..//data//castle.png"))))
  return img::Array{Float64, 2}
end

function mask_image(img::Array{Float64, 2}, value::Float64, ratio::Float64)
  masked_img = copy(img)
  mask = BitArray(size(masked_img)).*false
  indices = range(1,length(img))
  indices = shuffle(indices)[1:convert(Int64,size(indices,1)*ratio)]
  for i in indices
    masked_img[i]=value
    mask[i]=true
  end

  return masked_img::Array{Float64, 2}, mask::BitArray{2}
end

function f_inpaint(x::Array{Float64, 1}, m::BitArray{2}, sigma::Float64, alpha::Float64)
    x = reshape(x, size(m))
    return mrf_inpaint_nlposterior(x, m, sigma, alpha)
end

function g_inpaint(x::Array{Float64, 1}, m::BitArray{2}, sigma::Float64, alpha::Float64, storage::Array{Float64, 1})
    x = reshape(x, size(m))
    storage[1:end] = grad_mrf_inpaint_nlposterior(x, m, sigma, alpha)
end

function apply_optimizer(x::Array{Float64, 1}, m::BitArray{2}, sigma::Float64, alpha::Float64)
  res = optimize(x->f_inpaint(x,m,sigma,alpha),(x,storage)->g_inpaint(x,m,sigma,alpha,storage), x, AcceleratedGradientDescent(), Optim.Options(show_trace=true))
  display(res)
  return Optim.minimizer(res)
end

function inpaint(x::Array{Float64, 2}, m::BitArray{2}, sigma::Float64, alpha::Float64)
  inpainted = apply_optimizer(x[:], m, sigma, alpha)
  return reshape(inpainted, size(x))
end

function problem2()
  sigma = 10.0
  alpha = 1.0
  img = load_images()
  masked_img, mask = mask_image(img, 127.0, 0.5)
  PyPlot.figure()
  imshow(img, "gray")
  PyPlot.figure()
  imshow(masked_img, "gray")
  PyPlot.figure()
  imshow(mask, "gray")
  inpainted = inpaint(masked_img, mask, sigma, alpha)
  PyPlot.figure()
  imshow(inpainted, "gray")
  display(psnr(img, inpainted))

end
