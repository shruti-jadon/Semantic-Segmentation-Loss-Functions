# Semantic-Segmentation-Loss-Functions
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

DOI: https://doi.org/10.1117/12.2566332

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
@inproceedings{jadon2020comparative,
  title={A comparative study of 2D image segmentation algorithms for traumatic brain lesions using CT data from the ProTECTIII multicenter clinical trial},
  author={Jadon, Shruti and Leary, Owen P and Pan, Ian and Harder, Tyler J and Wright, David W and Merck, Lisa H and Merck, Derek L},
  booktitle={Medical Imaging 2020: Imaging Informatics for Healthcare, Research, and Applications},
  volume={11318},
  pages={113180Q},
  year={2020},
  organization={International Society for Optics and Photonics}
}
```
## Summarized Loss functions and their use-cases
![alt text](https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/summary.png)


## Other Similar Awesome Paper/Code. 
1. Segmentation Loss Odyssey, https://github.com/JunMa11/SegLoss
```
@article{SegLossOdyssey,
  title={Segmentation Loss Odyssey},
  author={Ma Jun},
  journal={arXiv preprint arXiv:2005.13449},
  year={2020}
}
```
