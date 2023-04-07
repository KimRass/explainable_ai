# Paper Understanding
- [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf)
## Introduction
- We introduce ***a visualization technique that reveals the input stimuli that excite individual feature maps at any layer in the model.*** It also allows us to observe the evolution of features during training and to diagnose potential problems with the model. The visualization technique we propose uses a multi-layered Deconvolutional Network (deconvnet), as proposed by [1] to project the feature activations back to the input pixel space.
- We also perform a sensitivity analysis of the classifier output by occluding portions of the input image, revealing which parts of the scene are important for classification.
## Methodology
- Figure 1
    - <img src="https://user-images.githubusercontent.com/105417680/229678910-16b9d2db-68fa-4f0e-8ddf-31730de8766a.png" width="500">
    - We present a novel way to map these activities back to the input pixel space, ***showing what input pattern originally caused a given activation in the feature maps.*** We perform this mapping with a Deconvolutional Network (deconvnet) [1]. A deconvnet can be thought of as a convnet model that uses the same components (filtering, pooling) but in reverse, so instead of mapping pixels to features does the opposite. In [1], deconvnets were proposed as a way of performing unsupervised learning.
    - To examine a convnet, a deconvnet is attached to each of its layers, providing a continuous path back to image pixels. ***To start, an input image is presented to the convnet and features computed throughout the layers. To examine a given convnet activation, we set all other activations in the layer to zero and pass the feature maps as input to the attached deconvnet layer.*** Then we successively (i) unpool, (ii) rectify and (iii) filter to reconstruct the activity in the layer beneath that gave rise to the chosen activation. This is then repeated until input pixel space is reached.
    - Unpooling:
        - ***In the convnet, the max pooling operation is non-invertible, however we can obtain an ap- proximate inverse by recording the locations of the maxima within each pooling region in a set of switch variables. In the deconvnet, the unpooling operation uses these switches to place the reconstructions from the layer above into appropriate locations, preserving the structure of the stimulus.***
        - Projecting down from higher layers uses the switch settings generated by the max pooling in the convnet on the way up. As these switch settings are peculiar to a given input image, ***the reconstruction obtained from a single activation thus resembles a small piece of the original input image, with structures weighted according to their contribution toward to the feature activation. Since the model is trained discriminatively, they implicitly show which parts of the input image are discriminative.***
    - Rectification: The convnet uses relu non-linearities, which rectify the feature maps thus ensuring the feature maps are always positive. To obtain valid feature reconstructions at each layer (which also should be positive), we pass the reconstructed signal through a relu non-linearity.
    - Filtering: The convnet uses learned filters to convolve the feature maps from the previous layer. To invert this, the deconvnet uses transposed versions of the same filters.
- Figure 2
    - <img src="https://i.stack.imgur.com/h8YGK.jpg" width="800">
    - Instead of showing the single strongest activation for a given feature map, we show the top 9 activations. Projecting each separately down to pixel space reveals the different structures that excite a given feature map, hence showing its invariance to input deformations. Alongside these visualizations we show the corresponding image patches. These have greater variation than visualizations as the latter solely focus on the discriminant structure within each patch.
    - The projections from each layer show the hierarchical nature of the features in the network. Layer 2 responds to corners and other edge/color conjunctions. Layer 3 has more complex invariances, capturing similar textures (e.g. mesh patterns (Row 1, Col 1); text (R2, C4)). Layer 4 shows significant variation, but is more class-specific: dog faces (R1, C1); bird’s legs (R4, C2). Layer 5 shows entire objects with significant pose variation, e.g. keyboards (R1, C11) and dogs (R4).
## Experiments
- Figure 4: Feature Evolution during Training
    - <img src="https://user-images.githubusercontent.com/105417680/230030551-4e2797d7-449f-43f9-9994-e6f512433bd0.png" width="900">
    - The progression during training of the strongest activation (across all training examples) within a given feature map projected back to pixel space. ***The lower layers of the model can be seen to converge within a few epochs. However, the upper layers only develop develop after a considerable number of epochs (40-50), demonstrating the need to let the models train until fully converged.***
- Figure 5: Feature Invariance
    - <img src="https://shunk031.me/paper-survey/assets/img/cv/Visualizing-and-Understanding-Convolutional-Networks/figure5.png" width="900">
    - (a), (b) and (c): Analysis of vertical translation, scale, and rotation invariance within the model
    - (a1), (b1) and (c1): 5 example images undergoing the transformations
    - (a2), (b2) and (c2):
        - Euclidean distance between feature vectors from the original and transformed images in layers 1
        - Small transformations have a dramatic effect.
    - (a3), (b3) and (c3):
        - Euclidean distance between feature vectors from the original and transformed images in layers 7
        - Small transformations have a lesser effect, being quasi-linear for translation (a3) & scaling (b3).
    - (a4), (b4) and (c4):
        - The probability of the true label for each image, as the image is transformed.
        - The network output is stable to translations (a4) and scalings (b4).
        - In general, the output is not invariant to rotation, except for object with rotational symmetry (e.g. entertainment center).
- Figure 7
    - <img src="https://shunk031.me/paper-survey/assets/img/cv/Visualizing-and-Understanding-Convolutional-Networks/figure7.png" width="800">
    - (a): Three test examples where we systematically cover up different portions of the scene with a gray square
    - (b):
        - ***For each position of the gray scale, we record the total activation in one layer 5 feature map (the one with the strongest response in the unoccluded image).***
        - The first row example shows the strongest feature to be the dog’s face. When this is covered-up the activity in the feature map decreases (blue area).
    - (c):
        - A visualization of this feature map projected down into the input image (black square), along with visualizations of this map from other images.
        - In the 2nd example, text on the car is the strongest feature in layer 5
        - The 3rd example contains multiple objects. The strongest feature in layer 5 picks out the faces
    - (d):
        - A map of correct class probability, as a function of the position of the gray square. E.g. when the dog’s face is obscured, the probability for "pomeranian" drops significantly.
        - In the 2nd example, the classifier is most sensitive to the wheel.
        - In the 3rd example, the classifier is sensitive to the dog (blue region), since it uses multiple feature maps.
    - (e):
        - The most probable label as a function of occluder position.
        - E.g. in the 1st row, for most locations it is "pomeranian", but if the dog’s face is obscured but not the ball, then it predicts "tennis ball".
## References
- [1] [Adaptive Deconvolutional Networks for Mid and High Level Feature Learning]
- [2] [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)