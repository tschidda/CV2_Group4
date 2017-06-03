## generate a constant disparity map ---------------------------

function constant_disparity(s, a)

  # Create a constant disparity map, sized s, with value a
  # You can freely change the name/type/number of parameters of functions.
  # ----- your code here -----
  disparity_map = ones(s)*a; 

  return disparity_map;	# return your results

end
