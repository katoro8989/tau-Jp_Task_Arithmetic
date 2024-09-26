# [Towards Understanding Variants of Invariant Risk Minimization through the Lens of Calibration](https://arxiv.org/abs/2401.17541)

<img width="970" alt="Screenshot 2024-06-18 at 20 11 17" src="https://github.com/katoro8989/IRM_Variants_Calibration/assets/107518964/93e63e05-4352-49e6-bbf9-f92396ce0943">

## Abstract
Machine learning models traditionally assume that training and test data are independently and identically distributed. However, in real-world applications, the test distribution often differs from training. This problem, known as out-of-distribution (OOD) generalization, challenges conventional models. Invariant Risk Minimization (IRM) emerges as a solution that aims to identify invariant features across different environments to enhance OOD robustness. However, IRM's complexity, particularly its bi-level optimization, has led to the development of various approximate methods. Our study investigates these approximate IRM techniques, using the consistency and variance of calibration across environments as metrics to measure the invariance aimed for by IRM. Calibration, which measures the reliability of model prediction, serves as an indicator of whether models effectively capture environment-invariant features by showing how uniformly over-confident the model remains across varied environments. Through a comparative analysis of datasets with distributional shifts, we observe that Information Bottleneck-based IRM achieves consistent calibration across different environments. This observation suggests that information compression techniques, such as IB, are potentially effective in achieving model invariance. Furthermore, our empirical evidence indicates that models exhibiting consistent calibration across environments are also well-calibrated. This demonstrates that invariance and cross-environment calibration are empirically equivalent. Additionally, we underscore the necessity for a systematic approach to evaluating OOD generalization. This approach should move beyond traditional metrics, such as accuracy and F1 scores, which fail to account for the model’s degree of over-confidence, and instead focus on the nuanced interplay between accuracy, calibration, and model invariance.

## Download datasets
Datasets to download:
1. MNIST (for ColoredMNIST and RotatedMNIST)
2. PACS
3. VLCS

   
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
