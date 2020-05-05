include("vanilla_gradient.jl")

############################################################# Guided Backpropagation ######################################

function change_activation(model::Metalhead.InceptionBlock, activation)
    path_1 = Conv(activation, model.path_1.weight, model.path_1.bias, model_path_1.stride, model.path_1.pad, model.path_1.dilation)

    path_2 = (Conv(activation, model.path_2[1].weight, model.path_2[1].bias, model_path_2[1].stride, model.path_2[1].pad, model.path_2[1].dilation),
            Conv(activation, model.path_2[2].weight, model.path_2[2].bias, model_path_2[2].stride, model.path_2[2].pad, model.path_2[2].dilation))

    path_3 = (Conv(activation, model.path_3[1].weight, model.path_3[1].bias, model_path_3[1].stride, model.path_3[1].pad, model.path_3[1].dilation),
            Conv(activation, model.path_3[2].weight, model.path_3[2].bias, model_path_3[2].stride, model.path_3[2].pad, model.path_3[2].dilation))

    path_4 = (model.path_4[1],
            Conv(activation, model.path_4[2].weight, model.path_4[2].bias, model_path_4[2].stride, model.path_4[2].pad, model.path_4[2].dilation))
    Metalhead.InceptionBlock(path_1, path_2, path_3, path_4)
end

# Resnet model is not supported
function change_activation(model::Chain, activation)
  updated_model = []
  for (i, l) in enumerate(model.layers)
    T = typeof(l)
    if T <: Dense
      push!(updated_model, Dense(l.W, l.b, identity))
      push!(updated_model, activation)
    elseif T <: Conv
      push!(updated_model, Conv(identity, l.weight, l.bias, l.stride, l.pad, l.dilation))
      push!(updated_model, activation)
    elseif T <: BatchNorm
      push!(updated_model, BatchNorm(identity, l.β, l.γ, l.μ, l.σ², l.ϵ, l.momentum, l.active))
      push!(updated_model, activation)
    elseif T <: Chain
      push!(updated_model, change_activation(l, activation))
    elseif T <: ConvTranspose
      push!(updated_model, ConvTranspose(identity, l.weight, l.bias, l.stride, l.pad, l.dilation))
      push!(updated_model, activation)
    elseif T <: DepthwiseConv
      push!(updated_model, DepthwiseConv(identity, l.weight, l.bias, l.stride, l.pad, l.dilation))
      push!(updated_model, activation)
    elseif T <: CrossCor
      push!(updated_model, CrossCor(identity, l.weight, l.bias, l.stride, l.pad, l.dilation))
      push!(updated_model, activation)      
    elseif T <: Metalhead.Bottleneck # To handle the DenseNet Model
      push!(updated_model, Bottleneck(change_activation(l.layer, activation)))
    elseif T <: Metalhead.InceptionBlock # To handle Inception Layer Model
      push!(updated_model, change_activation(l, activation))
    elseif T <: Metalhead.ResidualBlock
      push!(updated_model, change_activation(l, activation))
    else
      push!(updated_model, l)
    end
  end
  Chain(updated_model...)
end

function guided(x)
    max.(zero(x),x)
end

@adjoint guided(x) = guided(x), c̄ -> (Int.(x .> zero(x)) .* max.(zero(c̄), c̄),)

function viz_guidedbackprop(img, model; top_k=1, target_based=false, target_index=-1)
  model1 = change_activation(model, guided)
  print(model1)
  viz_backprop(img, model1; top_k=top_k, target_based=target_based, target_index=target_index)
end

############################################################ DeconvNet ###########################################

function deconv(x)
    max.(zero(x),x)
end

@adjoint deconv(x) = deconv(x), c̄ -> (max.(zero(c̄), c̄),)

function viz_deconvolution(img, model; top_k=1, target_based=false, target_index=-1)
  model = change_activation(model, deconv)
  viz_backprop(img, model; top_k=top_k, target_based=target_based, target_index=target_index)
end
