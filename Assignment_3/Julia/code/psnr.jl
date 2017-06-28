function psnr(X::Array{Float64,2}, Y::Array{Float64,2})
	p = 10*log10(65025/(1/(size(X,1)*size(X,2)) * sum((X-Y).^2)));
	return p::Float64;
end
