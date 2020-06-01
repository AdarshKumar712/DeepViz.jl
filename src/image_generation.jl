struct ImageGenerator
  model
  img
  target_class::Int
end

function ImageGenerator(model, target_class::Int)
  target_class > 1000 && error("Invalid Target Class")
  img = param(rand(224, 224, 3, 1)) |> gpu
  ImageGenerator(model.layers[1:end-1] |> gpu, img, target_class)
end

function (IG::ImageGenerator)(save_dir, img_name = "generated"; niters::Int = 150, lr = 6.0)
  isdir(save_dir) || error("Storage must be a directory")
  mask = Float32.(zeros(1000, 1)) |> gpu
  mask[IG.target_class, 1] = Float32(1.0)
  local mean_img = Float32.(reshape([0.485, 0.456, 0.406], 1, 1, 3, 1)) |> gpu
  for i in 1:niters
    IG.img .= IG.img .- mean_img
    outputs, back = Zygote.pullback(IG.model,IG.img)
    # This should be removed once the issue with pullback is resolved
    outputs = model(IG.img)
    loss = outputs .* mask
    grads = back(loss)
    println("Loss after Iteration $i is $(sum(loss)) and Probability is $(softmax(outputs)[IG.target_class, 1])")
    IG.img .= IG.img .+ lr * grad/sqrt(mean(abs2.(grad)))
    save("$(save_dir)/$(img_name)_iteration_$i.png", generate_image(IG.img))
  end
end
