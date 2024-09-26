# [$\tau$Jp: a Keystone of Enhancing Task Arithmetic]()

<img width="970" alt="Screenshot 2024-06-18 at 20 11 17" src="https://github.com/katoro8989/IRM_Variants_Calibration/assets/107518964/93e63e05-4352-49e6-bbf9-f92396ce0943">

## Abstract
Model-editing techniques using task arithmetic have rapidly gained attention and offer the efficient creation of desired models without the need for additional training, simply through arithmetic operations on the weights of pre-trained and fine-tuned models. 
However, task arithmetic faces challenges, such as low reproducibility and the high cost associated with adjusting coefficients in the arithmetic operations on model parameters, which have limited its practical success. 
In this paper, we present three key contributions in the context of task addition and task negation within task arithmetic.
First, we propose a new metric, $\tau$Jp, which can be shown to have a causal relationship with the negative interference that occurs from arithmetic operations. Second, by introducing regularization during fine-tuning to minimize $\tau$Jp, we significantly reduce the interference between task inferences, thus greatly reducing the need for coefficient adjustments. Third, we demonstrate that $\tau$Jp-based regularization is effective not only in a strict way, but also in more practical ways.
We believe that these contributions will lead to significant advancements toward the practical application of model-editing techniques using task arithmetic.

## Download datasets
Datasets to download:
1. [Cars]()
2. [DTD]()
3. [EuroSAT]()
4. [MNIST]()
5. [GTSRB]()
6. [RESISC45]()
7. [SUN397]()
8. [SVHN]()

   
```
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data
```

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
TMLR 2024 [OpenReview](https://openreview.net/forum?id=9YqacugDER&noteId=EHiqw76N8t)
```
@misc{yoshida2024understanding,
      title={Towards Understanding Variants of Invariant Risk Minimization through the Lens of Calibration}, 
      author={Kotaro Yoshida and Hiroki Naganuma},
      year={2024},
      eprint={2401.17541},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
