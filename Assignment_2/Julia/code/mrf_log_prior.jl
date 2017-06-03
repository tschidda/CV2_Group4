## mrf_log_prior ----------------------------------

function mrf_log_prior(x, sigma, alpha)

	# Compute the log of an unnormalized pairwise MRF prior
	# for disparity map x using t-distribution potentials with parameters sigma and alpha.
	# You can freely change the name/type/number of parameters of functions.

	# ----- your code here -----

	vertical = x[1:end-1,:]-x[2:end,:];
	horizontal = x[:,1:end-1]-x[:,2:end];

	lp_h = sum(studentT(horizontal[:],sigma,alpha));
	lp_v = sum(studentT(vertical[:],sigma,alpha));

	lp = lp_h + lp_v;

	return lp;	# return your computed Log-prior here

end


## studentT ---------------------------------------

function studentT(d, sigma, alpha)

	# Calculates the potential based on Student-t distribution
	# You can freely change the name/type/number of parameters of functions.

	# ----- your code here -----

	val = -alpha * log(1+(d.*d)/2*(sigma^2));

	return val;	# return your value here

end
