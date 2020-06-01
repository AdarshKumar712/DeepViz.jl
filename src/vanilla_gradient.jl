############################################################### Viz Backpropagation ####################################

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

############################################################# Integrated Backpropagation #################################### 

function stepped_images(img, steps)
    step_size = 1.0/steps
    imgs_in = [Float32.(img.*i) for i in 0+step_size:step_size:1.0]
    return imgs_in
end

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

