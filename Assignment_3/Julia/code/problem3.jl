using PyPlot
using Optim

include("psnr.jl")
include("mrf_posterior.jl")

function grad_loss(gt::Array{Float64,2}, x::Array{Float64,2})
	return grad_psnr(gt, x)
end

function loss(gt::Array{Float64,2}, x::Array{Float64,2})
	return psnr(gt, x)
end

function prediction(x::Array{Float64,2}, y::Array{Float64,2}, sigma_noise::Float64,
					sigma::Float64, alpha::Float64)
					F = y + grad_mrf_denoise_nlposterior(x, y, sigma_noise, sigma, alpha)
					dsigma = grad2_studentt_sigma(y)
					dalpha = grad2_studentt_alpha(y)
	return F::Array{Float64,2}, dsigma::Array{Float64,2}, dalpha::Array{Float64,2}
end



function grad2_studentt_sigma(input::Array{Float64,2}, sigma::Float64, alpha::Float64)
  height, width = size(input);
  dhor = grad_studentt_sigma(input[:,1:end-1] - input[:,2:end], sigma, alpha);
  phor = hcat(dhor, zeros(height,1)) - hcat(zeros(height,1), dhor);
  dvert = grad_studentt_sigma(input[1:end-1,:] - input[2:end,:], sigma, alpha);
  pvert = vcat(dvert, zeros(1,width)) - vcat(zeros(1,width), dvert);
  return -(phor+pvert);
end


function grad2_studentt_alpha(input::Array{Float64,2}, sigma::Float64, alpha::Float64)
  height, width = size(input);
  dhor = grad_studentt_alpha(input[:,1:end-1] - input[:,2:end], sigma, alpha);
  phor = hcat(dhor, zeros(height,1)) - hcat(zeros(height,1), dhor);
  dvert = grad_studentt_alpha(input[1:end-1,:] - input[2:end,:], sigma, alpha);
  pvert = vcat(dvert, zeros(1,width)) - vcat(zeros(1,width), dvert);
  return -(phor+pvert);
end

function learning_objective(gt::Array{Float64,2}, y::Array{Float64,2},
		sigma_noise::Float64, sigma::Float64, alpha::Float64)
		J = - psnr(gt, prediction(gt, y, sigma_noise, sigma, alpha))
		g = zeros(2)
		g[1] = sum(prediction(gt, y, sigma_noise, sigma, alpha))
		g[2] = sum(prediction(gt, y, sigma_noise, sigma, alpha))
	return J::Float64, g::Array{Float64,1}
end

function find_optimal_params(gt::Array{Float64,2}, y::Array{Float64,2}, sigma_noise::Float64, theta0)
	height, width = size(gt)
  function f(y::Array{Float64,1})
		J, g = learning_objective(gt, y, sigma_noise, y[1], y[2])
    return J
  end
	function g!(y::Array{Float64,1}, storage::Array{Float64,1})
		J, gradient = learning_objective(gt, y, sigma_noise, y[1], y[2])
    storage[:] = gradient[:]
  end;
  minTheta = optimize(f, g!, theta0, GradientDescent())
	return minTheta::Array{Float64,1}
end
