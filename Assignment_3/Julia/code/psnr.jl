function psnr(X::Array{Float64,2}, Y::Array{Float64,2})
	p = 10*log10(65025/(1/(size(X,1)*size(X,2)) * sum((X-Y).^2)));
	return p::Float64;
end

function grad_psnr(x::Array{Float64,2}, y::Array{Float64,2}) 
 		max = max(maximum(x), maximum(y));
 		result = (-20 / log(10)) / ((x-y) * max^2) / (size(x) * size(x);
 		return result;
 end
