# Paper Summary
- [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/pdf/1412.6806.pdf)
## Related Works
- Feature Representation
    - ***Lower layers learn general features with limited amount of invariance, which allows to reconstruct a single pattern that activates them. However, higher layers learn more invariant representations, and there is no single image maximally activating those neurons.*** Hence to get reasonable reconstructions it is necessary to condition on an input image. ***An alternative way of visualizing the part of an image that most activates a given neuron is to use a simple backward pass of the activation of a single neuron after a forward pass through the network; thus computing the gradient of the activation w.r.t. the image.*** The connections between the deconvolution and the backpropagation approach were recently discussed in [2]. In short the both methods differ mainly in the way they handle backpropagation through the rectified linear (ReLU) nonlinearity.
## References
- [1] [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf)
- [2] [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/pdf/1312.6034.pdf)
- [3] https://glassboxmedicine.com/2019/10/06/cnn-heat-maps-gradients-vs-deconvnets-vs-guided-backpropagation/

# Guided Backpropagation
- Figure 1 [4]
    - <img src="https://user-images.githubusercontent.com/67457712/226080571-0e94e14d-49ee-48ad-b32c-d1901ed7889c.png" width="700">
    - $f$ represents a feature map produced by some layer of a CNN, and $R$ represents an intermediate result in the calculation of backpropagation.
    - 'b)' 'Forward pass'
        - Comment: 왼쪽은 $f^{l}_{i}$을, 오른쪽은 $f^{l + 1}_{i}$을 나타냅니다.
    - 'b)' 'Backward pass: backpropagation', 'Backward pass: "deconvnet"' and 'Backward pass: guided backpropagation'
        - Comment: 왼쪽은 $R^{l}_i$을, 오른쪽은 $R^{l + 1}_{i}$을 나타냅니다.
    - Guided Backpropagation basically combines vanilla backpropagation and DeconvNets when handling the ReLU nonlinearity:
        - Like vanilla backpropagation, we restrict ourselves to only positive inputs.
        - Like DeconvNets we only backpropagate positive error signals – i.e. we set the negative gradients to zero. This is the application of the ReLU to the error signal itself during the backward pass.

