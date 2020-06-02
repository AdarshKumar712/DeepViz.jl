im_mean = reshape([0.485, 0.456, 0.406], 1, 1, 3)
global original_size = (224, 224)

im2arr_rgb(img) = permutedims(float.(channelview(imresize(img, (224, 224)))), (3, 2, 1))

"""
    set_training_false()

Function to set the model into `testmode`, so that the output result do not vary while evaluating gradients(or other results) for the models which contain layers like `BatchNormalization`, `Dropout` etc.
"""
function set_training_false() 
    @eval Flux.istraining() = false
    @eval Zygote.@adjoint Flux.istraining() = false, _ -> nothing
end

"""
    set_training_true()

Function to revert back the model into `trainmode`, once the gradients(or other results) are evaluated.
"""
function set_training_true() 
    @eval Flux.istraining() = true
    @eval Zygote.@adjoint Flux.istraining() = true, _ -> nothing
end

# one_hot_encode: Utility to return one_hot array for provided index with size same as that of `preds`
function one_hot_encode(preds, idx)
  one_hot = zeros(eltype(preds), size(preds)[1], 1)
  one_hot[idx.I[1] ,1] = 1
  one_hot |> gpu
end

# get_topk: Utility to return top_k predicted classes with their probabilities
function get_topk(probs; k = 5)
  T = eltype(probs)
  prob = Array{T, 1}()
  idx = []
  while(k!=0)
    push!(idx, argmax(probs))
    push!(prob, probs[idx[end]])
    probs[idx[end]] = 0.0
    k -= 1
  end
  (prob, idx)
end

"""
    save_gradient_images(gradient, file_name; gray=false)

Saves the gradient as an RGB image by default. If `gray` is set to true, gradients are sasved as a gray image with the provided `filename`.
"""
function save_gradient_images(gradient, file_name; gray=false)
  gradient = gradient .- minimum(gradient)
  gradient ./= maximum(gradient)
  gradient = permutedims(dropdims(gradient, dims=4), (3, 2, 1))
  img = colorview(RGB{eltype(gradient)}, gradient)
  if (gray)
      img = Gray.(img)
  end
  display(img)
  @info("Saving Gradient Image......")
  save(file_name, img)
end


"""
    positive_negative_saliency(gradient)

Returns a tuple of positive and negative saliency maps based on the provided gradients.
"""
function positive_negative_saliency(gradient)
  pos_saliency = max.(zero(gradient), gradient) ./ maximum(gradient)
  neg_saliency = max.(zero(gradient), -gradient) ./ maximum(-gradient)
  (pos_saliency, neg_saliency)
end


"""
    image_to_arr(img; preprocess = true)

Convert an RGB type image to an array. If `preprocess` is set as `true`, subtract the mean from the image array.  
"""
function image_to_arr(img; preprocess = true)
  local x = img
  x = Float32.(channelview(img))
  x = permutedims(x, [3,2,1])
  if(preprocess)
    x = x .- im_mean
  end
  x = reshape(x, size(x,1), size(x,2), size(x,3), 1) * 255 |> gpu
end

"""
    load_image(path, resize = false; size_save = true)

Function to load the images from the specified `path`. Returns as an array of image. If `resize` is set as `true`, resize the image to shape (224, 224). 
"""
function load_image(path, resize = false; size_save = true)
  img = load(path)
  if(size_save)
    global original_size = size(img)
  end
  if(resize)
    image_to_arr(imresize(img, (224, 224)))
  else
    image_to_arr(img)
  end
end

"""
     generate_image(x; resize_original=false, original_size=Nothing)

Save the image as an RGB image defined the by the array `x`. If `resize_original` is set to be true, then the image can be resized to the `original_size`. If not provided explicitly, then the image is resized to the `original_size` of the image saved while loading the image. 
"""
function generate_image(x; resize_original = false, original_size=Nothing)
  x = reshape(x, size(x)[1:3]...)/255 |> cpu
  x = x .+ im_mean
  x = permutedims(x, [3,2,1])
  x .-= minimum(x)
  x ./= maximum(x)
  if resize_original
    if original_size!=Nothing
        return imresize(colorview(RGB, x), original_size)
    else
        global original_size
        return imresize(colorview(RGB, x), original_size)
    end
  else
    return colorview(RGB, x)
  end
end

rgb2arr(a) = permutedims(Float32.(channelview(a)), [3,2,1])
arr2rgb(a) = colorview(RGBA{eltype(a)}, permutedims(a, [3,2,1]))

function arr2rgb_normalize(a)
    a .-= minimum(a)
    a ./= maximum(a)
    a = permutedims(a, [3,2,1])
    colorview(RGB{eltype(a)},a)
end

"""
    save_image(path, x)

Save the image defined by `x` to the specified `path`.
"""
save_image(path, x) = save(path, generate_image(x; resize_original = true))

"""
    apple_colormap(org_img, activation_map; map_alpha=0.4, process_img=true)

Creates a `rainbow` based colormap for the image, which is then applied onto `org_img`(original image) with different transparency to generate the combined heatmap with the `org_img` (original image). Function returns a tuple of `combined colormap` and `activation map alone based heatmap`.
"""
function apply_colormap(org_img, activation_map; map_alpha=0.4, process_img=true)
    cm = cgrad(:rainbow)
    no_trans_heat_map = RGB.(cm[activation_map])
    heat_map = copy(rgb2arr(no_trans_heat_map))
    if (process_img==true)
        org_img = rgb2arr(org_img)
    end
    x = org_img*0.7 + heat_map*0.3
    return (arr2rgb_normalize(x), heatmap)
end

