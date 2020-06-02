
"""
    viz_backprop(img, model; top_k=1, target_based=false, target_index)

Returns gradient generated with Vanilla Backpropagation for `top_k` classes. It’s the original saliency map algorithm for supervised deep learning from [Simonyan et. al. (2013)](https://arxiv.org/abs/1312.6034). These gradients can further be visualized using utilty functions.

# Arguments
 - `img`: source image for which gradients are to be calculated
 - `model`: model to be used for visualization
 - `top_k=1`: gradients are returned for `top_k` classes as per predictions by the model.
 - `target_based=false`: if set to `true`, function returns the maximizing gradient for the specified `target_index` class along with gradient for `top_k` classes.
 - `target_index`: defines the target class, for which the maximizing gradients are to be calculated. Required when `target_based` is set to `true`. 

The function returns an of gradients for `top_k=1` predicted classes by default. If `target_based` is set to `true` and `target_index` is specified, the function returns a tuple: `(gradients_for_top_k_classes, gradient_for_target_class)`.
"""
function viz_backprop(img, model; top_k=1, target_based=false, target_index=-1)
    model_ = model[1:end-1] |> gpu
    set_training_false()
    preds, back = Zygote.pullback(model_, img)
    set_training_true()
    probs = softmax(preds)
    prob, inds = get_topk(probs, k=top_k)
    grads = []
    for (i, idx) in enumerate(inds)
      push!(grads, (back(one_hot_encode(preds, idx))[1], prob[i], idx))
    end
    if (target_based==true)
        if target_index == -1
            @warn("No Target Class defined!!")
        else
            return (grads, back(one_hot_encode(preds, target_index)))
        end
    end
    grads
end    


# Integrated Backpropagation 

"""
    stepped_images(img, steps)

Utility function for `viz_integrated_gradient`. Generate scaled images, ie. images multiplied by a factor `i` defined in the range of `0` to `1.0` over equal intervals of `1/steps`. 
"""
function stepped_images(img, steps)
    step_size = 1.0/steps
    imgs_in = [Float32.(img.*i) for i in 0+step_size:step_size:1.0]
    return imgs_in
end

"""
    viz_integrated_gradient(img, model, steps; target_based=false, target_index)

Produces gradients generated with integrated gradients from the image scaled over `steps`. This functions uses the `viz_backprop` for the calculation of gradients for each of scaled image and then returns integrated gradients as the average over all the scaled images.

# Arguments 
 - `img`: source image on which the gradient are to be calculated.
 - `model`: model to be used for visualization
 - `steps`: number of steps between 0 to 1.0 for which the image is to be scaled while calculating integrated gradients.
 - `target_based=false`: if set `true`, the integrated gradients are evaluated for the specified `target_index`.
 - `target_index`: index of the target class for which the maximized gradients are to be evaluated. Necessary condition when `target_based` is set as `true`.
 
"""
function viz_integrated_gradient(img, model, steps; target_based=false, target_index=-1)
    imgs_in = stepped_images(img, steps)
    integrated_gradient = zeros(Float32,size(img)...)
    if (target_based==true)
        for img_ in imgs_in
            integrated_gradient .+= viz_backprop(img_, model; top_k=1, target_based=true, target_index=target_index)[2][1]
        end
    else
        for img_ in imgs_in
            grad = viz_backprop(img_, model; top_k=1, target_based=false)[1][1]
            integrated_gradient .+= grad
        end
    end
    Float32.(integrated_gradient/steps)
end

