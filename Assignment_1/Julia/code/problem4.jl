using Images 
using PyPlot
include("Common.jl")

function load_images()
	img0 = PyPlot.imread("./downloads/CV2/asgn1/data/i0.png")
	img1 = PyPlot.imread("./downloads/CV2/asgn1/data/i1.png")
	disparity = PyPlot.imread("./downloads/CV2/asgn1/data/gt.png")
	img0 = Common.rgb2gray(img0)
	img1 = Common.rgb2gray(img1)
	img0 = convert(Array{Float64,2}, img0)
	img1 = convert(Array{Float64,2}, img1)
	disparity = convert(Array{Float64,2}, disparity)
    return (img0::Array{Float64,2},img1::Array{Float64,2},disparity::Array{Float64,2});
end


function crop(img0::Array{Float64,2}, img1::Array{Float64,2}, gt::Array{Float64,2})
	img0_crop = img0[19:270, 19:366]
	img1_crop = img1[19:270, 19:366]
	gt_crop = gt[19:270, 19:366]
	return (img0_crop::Array{Float64,2}, img1_crop::Array{Float64,2}, gt_crop::Array{Float64,2})
end


function add_noise(img, percentage)
	img_noise = img
	M, N = size(img)
	num = convert(Int64, round(percentage*M*N/100))
	a = randperm(M*N)
	writedlm("downloads/CV2/asgn1/a.txt", a)
	for index = 1:num
		j = mod(a[index], N)
		if j == 0
			j = N
		end
		i = convert(Int64, (a[index] - j)/N) + 1
		img_noise[i,j] = rand()
	end
    return img_noise::Array{Float64,2}
end


function apply_disparity(img::Array{Float64,2}, disparity::Array{Int64,2})
	M, N = size(img)
	img_d = Array{Float64, 2}(M, N)
	for i = 1:M
		for j = 1:N
			img_d[i, j] = img[i, j - disparity[i,j]]
		end
	end
	return img_d::Array{Float64,2}
end


function compute_gaussian_lh(img0::Array{Float64,2}, img1_d::Array{Float64,2}, mu::Float64, sigma::Float64)
	M, N = size(img0)
	lh::Float64 = 1
	for i = 1:M
		for j = 1:N
			x = img0[i,j] - img1_d[i,j]
			lh = lh * (1/(sqrt(2*pi)*sigma) * exp(-0.5*(((x - mu)/sigma)^2)))
		end
	end
    return lh::Float64
end


function compute_gaussian_nllh(img0::Array{Float64,2}, img1_d::Array{Float64,2}, mu::Float64, sigma::Float64)
	M, N = size(img0)
	nllh::Float64 = 0
	for i = 1:M
		for j = 1:N
			nllh = nllh + (img0[i,j] - img1_d[i,j] - mu)^2
		end
	end
	nllh = nllh * 0.5/(sigma^2) #+ M*N*log(sqrt(2*pi)*sigma)
    return nllh::Float64
end


function compute_laplacian_nllh(img0::Array{Float64,2}, img1_d::Array{Float64,2}, mu::Float64, s::Float64)
	M, N = size(img0)
	nllh::Float64 = 0
	for i = 1:M
		for j = 1:N
			nllh = nllh + abs(img0[i,j] - img1_d[i,j] - mu)
		end
	end
	nllh = nllh/s #+ M*N*log(2*s)
    return nllh::Float64
end

function problem4()
	imgg0, imgg1, disparityg = P4.load_images()
	disparityg = round(disparityg.*255)
	disparityInt = convert(Array{Int64, 2}, disparityg)
	img1_d = P4.apply_disparity(imgg1, disparityInt)
	img0_crop, img_d_crop, gt_crop = P4.crop(imgg0, img1_d, disparityg)
	println("gaussian likelihood without noise:")
	println(P4.compute_gaussian_lh(img0_crop, img_d_crop, 0.0, 1.0))
	println("negative log gaussian without noise:")
	println(P4.compute_gaussian_nllh(img0_crop, img_d_crop, 0.0, 1.0))
	println("negative log laplacian without noise:")
	println(P4.compute_laplacian_nllh(img0_crop, img_d_crop, 0.0, 1.0))

	img1_noise = add_noise(imgg1, 10)
	img1_d_noise = P4.apply_disparity(img1_noise, disparityInt)
	img0_crop, img_d_crop, gt_crop = P4.crop(imgg0, img1_d_noise, disparityg)
	println("gaussian likelihood with noise:")
	println(P4.compute_gaussian_lh(img0_crop, img_d_crop, 0.0, 1.0))
	println("negative log gaussian with noise:")
	println(P4.compute_gaussian_nllh(img0_crop, img_d_crop, 0.0, 1.0))
	println("negative log laplacian with noise:")
	println(P4.compute_laplacian_nllh(img0_crop, img_d_crop, 0.0, 1.0))
end

problem4()

