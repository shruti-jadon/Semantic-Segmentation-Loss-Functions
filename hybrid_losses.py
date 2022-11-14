"""
    @Author: Hamid Ali
    @Date: 11/14/2022
    @GitHub: https://github.com/hamidriasat
    @Gmail: hamidriasat@gmail.com
"""

import tensorflow as tf
from loss_functions import Semantic_loss_functions


class HybridLossFunctions(object):
    """
    A hybrid loss function is simply a loss function which consisted
    of two or more different loss functions. Recent research has
    shown a combination of loss functions can yield better results.
    For further details about each hybrid loss please check the corresponding
    research paper.
    """

    def unet3p_hybrid_loss(self, y_true, y_pred):
        """
        Hybrid loss proposed in UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
        Hybrid loss for segmentation in three-level hierarchy – pixel, patch and map-level,
        which is able to capture both large-scale and fine structures with clear boundaries.
        :param y_true:
        :param y_pred:
        :return:
        """
        semantic_lf = Semantic_loss_functions()
        focal_loss = semantic_lf.focal_loss(y_true, y_pred)
        ms_ssim_loss = semantic_lf.ssim_loss(y_true, y_pred)
        jacard_loss = semantic_lf.jacard_loss(y_true, y_pred)

        return focal_loss + ms_ssim_loss + jacard_loss

    def basnet_hybrid_loss(self, y_true, y_pred):
        """
        Hybrid loss proposed in BASNET (https://arxiv.org/pdf/2101.04704.pdf)
        The hybrid loss is a combination of the binary cross entropy, structural similarity
        and intersection-over-union losses, which guide the network to learn
        three-level (i.e., pixel-, patch- and map- level) hierarchy representations.
        :param y_true:
        :param y_pred:
        :return:
        """
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        bce_loss = bce_loss(y_true, y_pred)

        semantic_lf = Semantic_loss_functions()
        ms_ssim_loss = semantic_lf.ssim_loss(y_true, y_pred)
        jacard_loss = semantic_lf.jacard_loss(y_true, y_pred)

        return bce_loss + ms_ssim_loss + jacard_loss
