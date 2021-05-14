# Semantic-Segmentation-Loss-Functions (SemSegLoss)
This Repository is implementation of majority of Semantic Segmentation Loss Functions in Keras. Our Survey paper is available on arxiv: https://arxiv.org/abs/2006.14822

In this paper we have summarized 15 such segmentation based loss functions that has been proven to provide state of results in different domain datasets.

We are still in process of adding more loss functions, so far we this repo consists of:
1. Binary Cross Entropy
2. Weighted Cross Entropy
3. Balanced Cross Entropy
4. Dice Loss
5. Focal loss
6. Tversky loss
7. Focal Tversky loss
8. log-cosh dice loss (ours)

This paper is extension of our work on traumatic brain lesion segmentation published at SPIE Medical Imaging'20.

Survey Paper DOI: https://doi.org/10.1117/12.2566332
Software Release DOI: https://doi.org/10.1016/j.simpa.2021.100078

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
![alt text](https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/summary.png)
