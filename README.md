# Semantic-Segmentation-Loss-Functions (SemSegLoss)
This Repository contains implementation of majority of Semantic Segmentation Loss Functions in Keras. Our paper is available open-source on following sites:

* Survey Paper DOI: [10.1109/CIBCB48159.2020.9277638](10.1109/CIBCB48159.2020.9277638)
* Software Release DOI: https://doi.org/10.1016/j.simpa.2021.100078

In this paper we have summarized 15 such segmentation based loss functions that has been proven to provide state of the art results in different domain datasets.

Recently new los functions have also been added and we are still in process of adding more loss functions, so far we this repo consists of:
1. Binary Cross Entropy
2. Weighted Cross Entropy
3. Balanced Cross Entropy
4. Dice Loss
5. Focal loss
6. Tversky loss
7. Focal Tversky loss
8. log-cosh dice loss (ours)
9. Jaccard/IoU loss
10. SSIM loss
11. [Unet3+](https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf) loss
12. [BASNet](https://arxiv.org/pdf/2101.04704.pdf) loss


This paper is extension of our work on traumatic brain lesion segmentation published at SPIE Medical Imaging'20.

Github Code: https://github.com/shruti-jadon/Traumatic-Brain-Lesions-Segmentation

## Citation
If you find our code useful, please consider citing our work using the bibtex:
```
@inproceedings{jadon2020survey,
  title={A survey of loss functions for semantic segmentation},
  author={Jadon, Shruti},
  booktitle={2020 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB)},
  pages={1--7},
  year={2020},
  organization={IEEE}
}
@article{JADON2021100078,
title = {SemSegLoss: A python package of loss functions for semantic segmentation},
journal = {Software Impacts},
volume = {9},
pages = {100078},
year = {2021},
issn = {2665-9638},
doi = {https://doi.org/10.1016/j.simpa.2021.100078},
url = {https://www.sciencedirect.com/science/article/pii/S2665963821000269},
author = {Shruti Jadon},
keywords = {Deep Learning, Image segmentation, Medical imaging, Loss functions},
abstract = {Image Segmentation has been an active field of research as it has a wide range of applications, ranging from automated disease detection to self-driving cars. In recent years, various research papers proposed different loss functions used in case of biased data, sparse segmentation, and unbalanced dataset. In this paper, we introduce SemSegLoss, a python package consisting of some of the well-known loss functions widely used for image segmentation. It is developed with the intent to help researchers in the development of novel loss functions and perform an extensive set of experiments on model architectures for various applications. The ease-of-use and flexibility of the presented package have allowed reducing the development time and increased evaluation strategies of machine learning models for semantic segmentation. Furthermore, different applications that use image segmentation can use SemSegLoss because of the generality of its functions. This wide range of applications will lead to the development and growth of AI across all industries.}
}
```
## Summarized Loss functions and their use-cases
| #      | Loss Function | Use cases     |
| :---        |    :----:   |    :---: |
| 1      | Binary Cross-Entropy       | Works best in equal data distribution among classes scenarios <br /> Bernoulli distribution based loss function |
| 2      | Loss Function       | Widely used with skewed dataset <br /> Weighs positive examples by Beta coefficient
| 3      | Binary Cross-Entropy       | Similar to weighted-cross entropy, used widely with skewed dataset <br /> weighs both positive as well as negative examples by Beta and 1 - Beta respectively
| 4      | Weighted Cross-Entropy       | Works best with highly-imbalanced dataset down-weight the contribution of <br /> easy examples, enabling model to learn hard examples
| 5      | Balanced Cross-Entropy       | Variant of Cross-Entropy <br /> Used for hard-to-segment boundaries
| 6      | Focal Loss       | Inspired from Dice Coefficient, a metric to evaluate segmentation results. <br /> As Dice Coefficient is non-convex in nature, it has been modified to make it more tractable.
| 7      | Distance map derived loss penalty term       | Inspired from Sensitivity and Specificity metrics <br /> Used for cases where there is more focus on True Positives.
| 8      | Dice Loss       | Variant of Dice Coefficient <br /> Add weight to False positives and False negatives.
| 9      | Sensitivity-Specificity Loss       | Variant of Tversky loss with focus on hard examples
| 10      | Tversky Loss       | Variant of Dice Loss and inspired regression log-cosh approach for smoothing <br /> Variations can be used for skewed dataset
| 11      | Focal Tversky Loss       | Inspired by Hausdorff Distance metric used for evaluation of segmentation <br /> Loss tackle the non-convex nature of Distance metric by adding some variations
| 12      | Log-Cosh Dice Loss(ours)       | Variant of Dice Loss and inspired regression log-cosh approach for smoothing <br /> Variations can be used for skewed dataset
| 13      | Hausdorff Distance loss       | Inspired by Hausdorff Distance metric used for evaluation of segmentation <br /> Loss tackle the non-convex nature of Distance metric by adding some variations
| 14      | Shape aware loss       | Variation of cross-entropy loss by adding a shape based coefficient <br /> used in cases of hard-to-segment boundaries.
| 15      | Combo Loss       | Combination of Dice Loss and Binary Cross-Entropy <br /> used for lightly class imbalanced by leveraging benefits of BCE and Dice Loss
| 16      | Exponential Logarithmic Loss       | Combined function of Dice Loss and Binary Cross-Entropy <br /> Focuses on less accurately predicted cases
| 18      | Correlation Maximized Structural Similarity Loss       | Focuses on Segmentation Structure. <br /> Used in cases of structural importance such as medical images.
| 19      | Jaccard/IoU loss       | Works well on balanced data <br />  Emphasizes more on large foreground regions 
| 20      | SSIM loss       | Captures the structural information in an image. <br /> Focuses on only boundaries of an object