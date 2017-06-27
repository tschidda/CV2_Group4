using PyPlot
using Optim


function grad_loss(gt::Array{Float64,2}, X::Array{Float64,2})

	return G::Array{Float64,2}
end

function prediction(X::Array{Float64,2}, Y::Array{Float64,2}, sigma_noise::Float64,
					sigma::Float64, alpha::Float64)

	return F::Array{Float64,2}, dsigma::Array{Float64,2}, dalpha::Array{Float64,2}
end

function learning_objective(gt::Array{Float64,2}, X::Array{Float64,2}, Y::Array{Float64,2},
							sigma_noise::Float64, sigma::Float64, alpha::Float64)

	return J::Float64, g::Array{Float64,1}
end

function find_optimal_params(gt::Array{Float64,2}, X::Array{Float64,2}, Y::Array{Float64,2}, sigma_noise::Float64, theta0)

	return minTheta::Array{Float64,1}
end


# Problem 3: Image Denoising with Loss-based Training
