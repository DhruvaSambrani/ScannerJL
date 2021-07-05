#@time import ImageFiltering: imfilter
@time using Colors
@time using FileIO
@time using Base.Cartesian

include("./utils.jl")

mean(arr) = sum(arr)/length(arr)
function crop_to_page(image, cutoff, upscale)
	upper = findfirst(1:size(image, 1)) do i; mean(image[i, :])>cutoff end
	lower = size(image, 1) - findfirst(size(image, 1): -1: 1) do i; mean(image[i, :])>cutoff end + 1
	left  = findfirst(1:size(image, 2)) do i; mean(image[:, i])>cutoff end
	right = size(image, 2) - findfirst(size(image, 2): -1: 1) do i; mean(image[:, i])>cutoff end + 1
	upscale_index(x) = upscale*(x-1) + 1
	upper, left, lower, right = upscale_index.([upper, left, lower, right])
	CartesianIndex(upper, left):CartesianIndex(lower, right)
end

sharpen(image) = fastconv(image, [0 -2. 0; -2 9 -2; 0 -2 0])
logistic(x, k) = 1/(1+exp(-k*(x-0.5)))
function contrast(image)
	_min = minimum(image)
	_max = maximum(image)
	map(image) do pixel; logistic((pixel - _min)/(_max-_min), 20); end
end
process = x->(map(gray∘Gray, x) |> contrast |> sharpen .|> t->clamp(t, 0., 1.))


function main()
    path = ARGS[1]
    filepaths = isfile(path) ? [path] : [p for p in readdir(path, join=true) if splitext(p)[2] in [".jpg", ".png"]]
    out_dir = ARGS[2]
    if !isdir(out_dir)
        mkdir(out_dir)
    end
    for filepath in filepaths
        image = load(filepath);
        crop = crop_to_page(
            map(p->HSI(p).i, image[1:20:end, 1:20:end]) |> contrast, 0.3, 20
        );
        sc_image = (process(image[crop]) .|> Gray)
        save(joinpath(out_dir, filepath), sc_image)
    end
end
main()
