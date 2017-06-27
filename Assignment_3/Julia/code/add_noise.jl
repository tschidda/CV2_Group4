using Distributions

function add_noise(X::Array{Float64,2}, sigma::Float64)
		norm_dist = Normal(0, sigma);
		noise = X + reshape(rand(norm_dist,size(X[:])),size(x));
		noiseAdd = abs(minimum(noise));
		noiseMul = (255/maximum(noise));
    noisy = (noise + noiseAdd)*noiseMul;

    @assert size(noisy) == size(X)
	return noisy::Array{Float64,2}
end
