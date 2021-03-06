# Save Gradcam

"""
    save_gradcam(gradient, original_image_path, grad_file_name, heatmap_file_name, combined_file_name)

Function to save the gradient calculated using `gradcam` as a `gray image`, `gradient heatmap` and the `combined heatmap` with different transparency.

# Arguments
 - `gradient`: Gradient calculated using the `viz_gradcam` function
 - `original_image_path`: Path to the original image file, for which the gradient is calculated.
 - `grad_file_name`: Filename with path, to which the gradient image is to be saved as a `gray` image
 - `heatmap_file_name`: Filename to save the gradient heatmap
 - `combined_file_name`: Filename to save the gradient the combined heatmap with original image.

"""
function save_gradcam(gradient, original_image_path, grad_file_name, heatmap_file_name, combined_file_name)
  gradient = max.(gradient, zero(gradient))
  gradient = permutedims((gradient .- minimum(gradient))/(maximum(gradient) - minimum(gradient)), [3, 2, 1])
  img = Gray.(colorview(RGB{eltype(gradient)},gradient))
  img = imresize(img, 224, 224)
  display(img)
  h1 = plot(heatmap(channelview(float.(augment(img, FlipY()))), color = :rainbow))
  try
    display(h1)
  catch
    @info("The heatmap could not be displayed. The file will be saved at $heatmap_file_name")
  end
  mapped = float.(channelview(imresize(load(original_image_path), (224, 224)))) * 0.9 .+ reshape(float.(img), 1, 224, 224) * 2
  mapped .-= minimum(mapped)
  mapped ./= maximum(mapped)
  h2 = plot(heatmap(channelview(float.(Gray.(augment(colorview(RGB{eltype(mapped)}, mapped), FlipY())))), color=:rainbow))
  try
    display(h2)
  catch
    @info("The heatmap could not be displayed. The file will be saved at $combined_file_name")
  end
  save(grad_file_name, img)
  try
    savefig(h1, heatmap_file_name)
    savefig(h2, combined_file_name)
  catch
    @info("Encountered Errors while trying to save heatmaps. So the heatmaps are being returned")
    @info("They need to be saved manually")
    (h1, h2)
  end
end

# GradCAM

normalize_grad(grad) = grad / (sqrt(mean(abs2.(grad))) + eps())

function save_gradient(x, tracker)
    tracker["grad"] = x |>cpu
    x
end

function tracked_model_gc(x, model, layer, tracker)
    x = model[1:layer](x) |> gpu
    tracker["fwd"] = x |> cpu
    model[layer+1:end](Zygote.hook(x->save_gradient(x, tracker), x)) |>gpu
end

"""
    viz_gradcam(img, model, layer; top_k=1)

Layer attribution method that computes the gradients of the target output with respect to the given `layer`, averages for each output channel (dimension 2 of output), and multiplies the average gradient for each channel by the layer activations. The results are summed over all channels.

It returns an array of tuples of (`gradCAM`, `prediction_probability`, `class_index`) for `top_k` predicted classes.
"""
function viz_gradcam(img, model, layer; top_k=1)
    model_ = model[1:end-1]
    tracker = Dict()
    set_training_false()
    preds, back = Zygote.pullback(x -> tracked_model_gc(x, model_, layer, tracker), img)
    set_training_true()
    probs = softmax(preds)
    prob, inds = get_topk(probs, k=top_k)
    grads = []
    for (i, idx) in enumerate(inds)
        back(one_hot_encode(preds, idx))
        weights = reshape(maximum(normalize_grad(tracker["grad"]), dims=[1, 2, 4]), 1, 1, size(tracker["fwd"])[3], 1)
        cam = ones(size(tracker["fwd"])[1:2])
        cam += dropdims(sum(weights .* tracker["fwd"],dims=[3, 4]), dims=(3, 4))
        push!(grads, (cam, probs[i], idx))
    end
    grads
end

# Guided GradCAM

"""
    viz_guidedgradcam(img, model, layer; top_k=1)

It computes the element-wise product of guided backpropagation attributions (evaluated using `viz_guidedbackprop`) with resized (layer) GradCAM attributions (computed using `viz_gradcam`). GradCAM attributions are computed with respect to a given `layer`, and attributions are resized to match the input size.

It returns an array of tuples of (`guidedgradcam`, `prediction_probability`, `class_index`) for `top_k` predicted classes.

Also see: [`viz_guidedbackprop`](@ref), [`viz_guidedgradcam`](@ref).
"""
function viz_guidedgradcam(img, model, layer; top_k=1)
    # Ideally we should not be doing the forward pass twice
    gcam_grads = viz_gradcam(img, model, layer, top_k=1)
    backprop_grads = viz_guidedbackprop(img, model; top_k=1)
    grads = []
    for i in 1:top_k
      normalized_camgrads = gcam_grads[i][1] .- minimum(gcam_grads[i][1])
      normalized_camgrads ./= maximum(normalized_camgrads)
      normalized_camgrads = reshape(channelview(float.(imresize(Gray.(normalized_camgrads), 224, 224))), 224, 224, 1, 1)
      normalized_bpropgrads = backprop_grads[i][1][1] .- minimum(backprop_grads[i][1][1])
      normalized_bpropgrads ./= maximum(normalized_bpropgrads)
      push!(grads, (normalized_camgrads .* normalized_bpropgrads, gcam_grads[i][2], gcam_grads[i][3]))
    end
    grads
end
