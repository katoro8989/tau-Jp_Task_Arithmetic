# [$\tau$Jp: a Keystone of Enhancing Task Arithmetic]()

![](images/addition.png)
![](images/negation.png)


## Abstract
Model-editing techniques using task arithmetic have rapidly gained attention and offer the efficient creation of desired models without the need for additional training, simply through arithmetic operations on the weights of pre-trained and fine-tuned models. 
However, task arithmetic faces challenges, such as low reproducibility and the high cost associated with adjusting coefficients in the arithmetic operations on model parameters, which have limited its practical success. 
In this paper, we present three key contributions in the context of task addition and task negation within task arithmetic.
First, we propose a new metric, $\tau$Jp, which can be shown to have a causal relationship with the negative interference that occurs from arithmetic operations. Second, by introducing regularization during fine-tuning to minimize $\tau$Jp, we significantly reduce the interference between task inferences, thus greatly reducing the need for coefficient adjustments. Third, we demonstrate that $\tau$Jp-based regularization is effective not only in a strict way, but also in more practical ways.
We believe that these contributions will lead to significant advancements toward the practical application of model-editing techniques using task arithmetic.

## Setup environment


## Download datasets
Datasets to download:
1. [Cars](https://www.cv-foundation.org/openaccess/content_iccv_workshops_2013/W19/html/Krause_3D_Object_Representations_2013_ICCV_paper.html)
2. [DTD](https://openaccess.thecvf.com/content_cvpr_2014/html/Cimpoi_Describing_Textures_in_2014_CVPR_paper.html)
3. [EuroSAT](https://ieeexplore.ieee.org/abstract/document/8736785)
4. [MNIST](https://yann.lecun.com/exdb/mnist/)
5. [GTSRB](https://ieeexplore.ieee.org/abstract/document/6033395?casa_token=MLJHRCUz8OIAAAAA:9ZMwyQ50HzTzmSkEN1_HBYOFNzPazeKICIKKd3t6g-VgtGc5c7U5tphTVEykAsbcViJUXYFr7UcE)
6. [RESISC45](https://ieeexplore.ieee.org/abstract/document/7891544?casa_token=ZOFbRb8TSDUAAAAA:83ANrYS19nlCWRtLylfeuqD3akKWlSeGE86H0gTFcQkRlENegFj9Brgt-dSBDl_MUcZiUPpdcljp)
7. [SUN397](https://link.springer.com/article/10.1007/s11263-014-0748-y)
8. [SVHN](https://research.google/pubs/reading-digits-in-natural-images-with-unsupervised-feature-learning/)
9. [ImageNet](https://ieeexplore.ieee.org/document/5206848)

We are using the same datasets as [this repository](https://github.com/mlfoundations/task_vectors).

For Cars, the original download link is broken, so please refer to this [issue](https://github.com/pytorch/vision/issues/7545) for a workaround. For DTD, EuroSAT, RESISC45, and SUN397, the datasets need to be manually split after downloading. Please refer to this [issue](https://github.com/mlfoundations/task_vectors/issues/1) for details.

## Avalable IRM variants
1. [IRMv1](https://arxiv.org/abs/1907.02893)
2. [Information Bottleneck based IRM (IB-IRM)](https://arxiv.org/abs/2106.06333)
3. [Pareto IRM (PAIR)](https://arxiv.org/abs/2206.07766)
4. [IRM Game](https://arxiv.org/abs/2002.04692)
5. [Bayesian IRM (BIRM)](https://openaccess.thecvf.com/content/CVPR2022/html/Lin_Bayesian_Invariant_Risk_Minimization_CVPR_2022_paper.html)

## Avalable metrics
1. Accuracy (ACC)
2. [Expected Calibration Error (ECE)](https://ojs.aaai.org/index.php/AAAI/article/view/9602)
3. [Adaptive Calibration Error (ACE)](https://scholar.google.com/scholar_url?url=http://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%2520and%2520Robustness%2520in%2520Deep%2520Visual%2520Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf&hl=en&sa=T&oi=gsr-r-ggp&ct=res&cd=0&d=671990448700625194&ei=gmpxZp_PHoaM6rQP65edyAw&scisig=AFWwaebPo7c5vLkDy-hd7muSkvMn)
4. [Negative Log-Likelihood (NLL)](https://proceedings.neurips.cc/paper/2021/hash/8420d359404024567b5aefda1231af24-Abstract.html)

## Paper Authors
[Kotaro Yoshida](https://github.com/katoro8989)

[Hiroki Naganuma](https://github.com/Hiroki11x)

## Citation

```

```
