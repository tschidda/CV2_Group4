function psnr(X::Array{Float64,2}, Y::Array{Float64,2})
	p = 10*log10(65025/mse(X,Y))
	return p::Float64
end

function mse(X::Array{Float64,2}, Y::Array{Float64,2})
	fac = 1/(size(X,1)*size(X,2))
	result = fac * sum((X-Y).^2)
	return result::Float64
end
