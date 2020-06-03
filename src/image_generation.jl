struct ImageGenerator
  model
  img
  target_class::Int
end

"""
    ImageGenerator(model, target_class)

Function to initialize the ImageGenerator for the `model` and `target_class`. Returns a `struct` with fields `model`, `img` and `target`. Here, the image is randomly generated which is then to be further updated.
"""
function ImageGenerator(model, target_class::Int)
  target_class > 1000 && error("Invalid Target Class")
  img = param(rand(224, 224, 3, 1)) |> gpu
  ImageGenerator(model.layers[1:end-1] |> gpu, img, target_class)
end

"""
    ImageGenerator(save_dir; img_name = "generated", niters::Int = 150, lr = 6.0)

Generates the image for which the prediction probability is maximized for a `target class` after `niter` iterations using backpropagation.
Before using this function, the ImageGenerator must be initialized using `ImageGenerator(model, target_class)`.

# Arguments
 - `save_dir`: Directory to which the generated image is to be saved 
 - `img_name="generated"`: Image name with which file is to be saved withoout file extension
 - `niters::Int=150`: number of iterations for which the generated image is updated with backpropagation
 - `lr=6.0`: Learning rate to be used while updating the generated image with respect to backpropagated gradients to maximize the specific class.

"""
function (IG::ImageGenerator)(save_dir; img_name = "generated", niters::Int = 150, lr = 6.0)
  isdir(save_dir) || error("Storage must be a directory")
  mask = Float32.(zeros(1000, 1)) |> gpu
  mask[IG.target_class, 1] = Float32(1.0)
  local mean_img = Float32.(reshape([0.485, 0.456, 0.406], 1, 1, 3, 1)) |> gpu
  IG.img .= IG.img .- mean_img
  for i in 1:niters
    set_training_false()
    outputs, back = Zygote.pullback(IG.model,IG.img)
    set_training_true()
    loss = outputs .* mask
    grads = back(loss)
    println("Loss after Iteration $i is $(sum(loss)) and Probability is $(softmax(outputs)[IG.target_class, 1])")
    IG.img .= IG.img .+ lr * grads/sqrt(mean(abs2.(grads)))
    save("$(save_dir)/$(img_name)_iteration_$i.png", generate_image(IG.img))
  end
end
