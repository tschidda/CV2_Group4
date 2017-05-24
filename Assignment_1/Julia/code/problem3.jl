using Images
using PyPlot
using ImageView

include("Common.jl")

function problem3()
	img = PyPlot.imread("./downloads/CV2/asgn1/data/a1p3.png")
	imgg = Common.rgb2gray(img)
	imgg = convert(Array{Float64}, imgg)
	ImageView.imshow(imgg)
	##Bug occurs when ImageView.imshow() and PyPlot.imshow() are used at the same time
	#PyPlot.imshow(imgg, "gray")
	#title("PyPlot")
	println("max value:")
	println(maximum(imgg))
	println("min value:")
	println(minimum(imgg))
	println("mean value:")
	println(mean(imgg))
end

problem3()