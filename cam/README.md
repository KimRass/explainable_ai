# Paper Reading
## Methodology
- [Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf)
- Figure 2
    - <img src="https://user-images.githubusercontent.com/67457712/230727934-1e4df288-6eca-48d0-9d04-9b6512266d0c.png" width="700">
    - For a given image, let $f_{k}(x, y)$ represent the activation of unit $k$ in the last convolutional layer at spatial location $(x, y)$. Then, for unit $k$, the result of performing global average pooling, $F_{k}$ is $\sum_{x, y}f_{k}(x, y)$. Thus, for a given class $c$, the input to the softmax, $S^{c}$, is $\sum_{k}w^{c}_{k}F_{k}$ where $w^{c}_{k}$ is the weight corresponding to class $c$ for unit $k$. Essentially, $w^{c}_{k}$ indicates the importance of $F_{k}$ for class $c$.
    - ***We explicitly set the input bias of the softmax to 0 as it has little to no impact on the classification performance.***