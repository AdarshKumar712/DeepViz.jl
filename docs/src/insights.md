# Insights

How do neural networks manage to find patterns in data? How do they learn from existing data, and what do the inner layers of these artificial neural networks look like? All of this is studied under fields referred to as model interpretability and visualization, which are currently very active areas of research. This section provides an insight (or a detailed description) into various algorithms available with DeepViz to help understand the model better. These algorithms are further divided into subsections: `Primary Attribution`, `Layer Attribution` and `Others`.

## Primary Attribution

These algorithms deal with the over model as a whole, thus attributing features of the overall model.

#### Vanilla Gradient

It’s the original saliency map algorithm for supervised deep learning from [Simonyan et. al. (2013)](https://arxiv.org/abs/1312.6034). It’s the simplest algorithm among gradient-based³ approaches and very fast to run, which makes it a great place to start to understand how saliency maps work.

The algorithms mainly consist of two steps:
* forward pass the image
* backward pass to input layerto get the gradients for `top kth class` maximization.

Once gradients are calculated for a particular class, these gradients can then be transformed into normalized heatmap, for better visualization

Useful Function(s): [`viz_backprop`](@ref), [`save_gradient_images`](@ref).

#### Guided Backpropagation and Deconvolution

Guided backpropagation and deconvolution compute the gradient of the target output with respect to the input, but backpropagation of ReLU functions is overridden so that only non-negative gradients are backpropagated. In guided backpropagation, the ReLU function is applied to the input gradients, and in deconvolution, the ReLU function is applied to the output gradients and directly backpropagated.

For more details, check the original papers:
* [`Striving for Simplicity: The All Convolutional Net`](https://arxiv.org/abs/1412.6806)
* [`Sanity Checks for Saliency Maps`](https://arxiv.org/abs/1810.03292)

Useful Function(s): [`viz_guidedbackprop`](@ref), [`viz_deconvolution`](@ref)

#### Integrated Gradients

Integrated gradients represents the integral of gradients with respect to inputs along the path from a given baseline to input. Here in this algorithm implementation, the path is chosen to be a straight line, and gradient is evaluated at equal intervals for the scaled images along the straight line. The cornerstones of this approach are two fundamental axioms, namely sensitivity and implementation invariance.

The algorithm consist of following steps:
* Scaling of Input Image with factor between 0.0 to 1.0 along the joing straight line at equal intervals 
* Calculating the gradient for each of the scaled image
* Taking average of the gradients

For more reference, check the original paper: [Original Paper](https://arxiv.org/abs/1703.01365)

Useful Function(s): [`viz_integrated_gradients`](@ref)

#### Gradient x Image

Gradient x Image is an extension of the saliency approach (vanilla gradient), taking the gradients of the output with respect to the input and multiplying by the input feature values. One intuition for this approach considers a linear model. the gradients are simply the coefficients of each input, and the product of the input with a coefficient corresponds to the total contribution of the feature to the linear model's output.

Useful Function(s): [`grad_times_image`](@ref)

## Layer Attribution

These algorithms deal with the specific layer of the model, thus attributing features of that specific layer of the model.

#### GradCAM

GradCAM or Gradient weighted Class Activation Maximization uses the gradients of any target concept, flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the image for predicting the concept. GradCAM is a layer attribution method designed for convolutional neural networks, and is usually applied to the last convolutional layer. GradCAM computes the gradients of the target output with respect to the given layer, averages for each output channel (dimension 2 of output), and multiplies the average gradient for each channel by the layer activations. The results are summed over all channels and a ReLU is applied to the output, returning only non-negative attributions.

For more details, refer to the paper: [`Original Paper`](https://arxiv.org/abs/1610.02391)

Useful Function(s): [`viz_gradcam`](@ref), [`save_gradcam`](@ref)

#### Guided GradCAM

Guided GradCAM computes the element-wise product of guided backpropagation attributions with upsampled (layer) GradCAM attributions. GradCAM attributions are computed with respect to a given layer, and attributions are upsampled to match the input size. This approach is designed for convolutional neural networks.

Guided GradCAM was proposed by the authors of GradCAM as a method to combine the high-resolution nature of Guided Backpropagation with the class-discriminative advantages of GradCAM, which has lower resolution due to upsampling from a convolutional layer.

For more details, refer to the paper: [`Original Paper`](https://arxiv.org/abs/1610.02391)

Useful Function(s): [`viz_guidedgradcam`](@ref), [`save_gradcam`](@ref)

## Others

These are algorithms which can not be properly assigned as primary attributions or layer attributions.

#### SmoothGrad

SmoothGrad is a simple method that can help visually sharpen gradient-based sensitivity maps, and it discusses lessons in the visualization of these maps. The core idea is to take an image of interest, sample similar images by adding Gaussian noise to the image (images in the neighborhood of image of interest), then take the average of the resulting sensitivity maps for each sampled image. The results suggest the estimated smoothed gradient, leads to visually more coherent sensitivity maps than the unsmoothed gradient, with the resulting visualization aligning better to the human eye with meaningful features.

For more details, refer to the original paper: [`Original Image`](https://arxiv.org/pdf/1706.03825.pdf)

Useful Function(s) : [`smooth_grad`](@ref)

#### Image Generator

This algorithms helps to visualise what type of input image maximizes the probability of `target class`. The idea behind Image Generator is simple in hindsight - Generate an input image that maximizes the model output activations for the `target class`.

Useful Function(s): [`ImageGenerator`](@ref)


