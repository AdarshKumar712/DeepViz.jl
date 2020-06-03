# SmoothGrad

"""
    smooth_grad(img, model; method=viz_backprop,top_k=1, mean=0, var=1, n_imgs=50, normalized_n_scaled=true)

Method to visually sharpen the gradient based saliency maps. The key idea is to evaluate the average of gradients of images obtained after adding gaussian noise to the original image(`img`). The results suggest the estimated smoothed gradient, leads to visually more coherent sensitivity maps than the unsmoothed gradient, with the resulting visualization aligning better to the human eye with meaningful features.

# Arguments
 - `img`: Original image of interest for visualization
 - `model`: Model to be used for visualization
 - `method=viz_backprop`: Method to be used while evaluating gradients. Can take methods like `viz_guidedbackprop`, `viz_backprop`, `viz_deconvolution` or any other custom function.
 - `top_k=1`: function evaluates the smoothed gradient for `top_k` predictions just as in case of normal `viz_backprop` function.
 - `mean=0`: Mean of the gaussian noise to be added to the image to generate sample space
 - `var=1`: Variance of the gaussian noise to be added to image to generate sample space
 - `n_imgs=50`: Number of images to be generated and used in sample space while evaluation of `smooth_grad`.
 - `normalized_n_scaled=true`: if true, then the noise is first scaled to 0-255 before adding to the image. 

Ref: (https://arxiv.org/pdf/1706.03825.pdf)
"""
function smooth_grad(img, model; method=viz_backprop, top_k=1, mean=0, var=1, n_imgs=50, normalized_n_scaled=true)
    smooth_grad = zeros(size(img)...)
    for i in n_imgs
        noise = (var^0.5).*randn(size(img)...) .+ mean
        if (normalized_n_scaled==false)
            noise *= 255.0
        end
        noise = Float32.(noise)
        noised_img = img + noise
        smooth_grad += method(noised_img, model, top_k = top_k)
    end
    smooth_grad/n_imgs
end 

# Gradient X Image

"""
    grad_times_image(img, model; top_k=1)

An extension to the saliency approach. Returns an array of products of the `img` with the gradient evaluated with respect to the provided `img` for the `model`, for `top_k` prediction classes. 
"""
function grad_times_image(img, model; top_k=1)
    grads = viz_backprop(img, model, top_k=top_k)
    grads_times_image = []
    for i in 1:length(grads)
        push!(grads_times_image, (grads[i][1].*img, grads[i][2],grads[i][3]))
    end
    grads_times_image
end
