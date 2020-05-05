######################################################## Smooth Grad ##############################################

function smooth_grad(img, model; method=viz_backprop,top_k=1, mean=0, var=1, n_imgs=50, normalized_n_scaled=true)
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

######################################################## Grad times Images ############################################

function grad_times_image(img, model; top_k=1)
    grads = viz_backprop(img, model, top_k=top_k)
    grads_times_image = []
    for i in 1:length(grads)
        push!(grads_times_image, (grads[i][1].*img, grads[i][2],grads[i][3]))
    end
    grads_times_image
end
