function load_images()

    return (img0::Array{Float64,2},img1::Array{Float64,2},disparity::Array{Float64,2});
end


function crop(img0::Array{Float64,2}, img1::Array{Float64,2}, gt::Array{Float64,2})

	return (img0_crop::Array{Float64,2}, img1_crop::Array{Float64,2}, gt_crop::Array{Float64,2})
end


function add_noise(img, percentage)

    return img_noise::Array{Float64,2}
end


function apply_disparity(img::Array{Float64,2}, disparity::Array{Int64,2})

	return img_d::Array{Float64,2}
end


function compute_gaussian_lh(img0::Array{Float64,2}, img1_d::Array{Float64,2}, mu::Float64, sigma::Float64)

    return lh::Float64
end


function compute_gaussian_nllh(img0::Array{Float64,2}, img1_d::Array{Float64,2}, mu::Float64, sigma::Float64)

    return nllh::Float64
end


function compute_laplacian_nllh(img0::Array{Float64,2}, img1_d::Array{Float64,2}, mu::Float64, s::Float64)

    return nllh::Float64
end
