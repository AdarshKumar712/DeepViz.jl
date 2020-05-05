using PlotUtils
using Plots

function one_hot_encode(preds, idx)
  one_hot = zeros(eltype(preds), size(preds)[1], 1)
  one_hot[idx.I[1] ,1] = 1
  one_hot |> gpu
end

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

function positive_negative_saliency(gradient)
  pos_saliency = max.(zero(gradient), gradient) ./ maximum(gradient)
  neg_saliency = max.(zero(gradient), -gradient) ./ maximum(-gradient)
  (pos_saliency, neg_saliency)
end

im2arr_rgb(img) = permutedims(float.(channelview(imresize(img, (224, 224)))), (3, 2, 1))

im_mean = reshape([0.485, 0.456, 0.406], 1, 1, 3)

function image_to_arr(img; preprocess = true)
  local x = img
  x = Float32.(channelview(img))
  x = permutedims(x, [3,2,1])
  if(preprocess)
    x = x .- im_mean
  end
  x = reshape(x, size(x,1), size(x,2), size(x,3), 1) * 255 |> gpu
end

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

function generate_image(x, resize_original = false)
  x = reshape(x, size(x)[1:3]...)/255 |> cpu
  x = x .+ im_mean
  x = permutedims(x, [3,2,1])
  x .-= minimum(x)
  x ./= maximum(x)
  if resize_original
    return imresize(colorview(RGB, x), original_size)
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

save_image(path, x) = save(path, generate_image(x, true))

function apply_colormap(org_img, activation_map; map_alpha=0.4, process_img=true)
    cm = cgrad(:rainbow)
    no_trans_heat_map = RGB.(cm[activation_map])
    heat_map = copy(rgb2arr(no_trans_heat_map))
    if (process_img==true)
        org_img = rgb2arr(org_img)
    end
    x = org_img*0.7 + heat_map*0.3
    return arr2rgb_normalize(x), heatmap
end
