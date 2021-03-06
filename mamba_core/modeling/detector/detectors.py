# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .generalized_rcnn_dff import GeneralizedRCNNDFF
from .generalized_rcnn_fgfa import GeneralizedRCNNFGFA
from .generalized_rcnn_mamba import GeneralizedRCNNMAMBA
from .generalized_rcnn_mega import GeneralizedRCNNMEGA
from .generalized_rcnn_rdn import GeneralizedRCNNRDN
from .generalized_rcnn_selsa import GeneralizedRCNNSELSA

_DETECTION_META_ARCHITECTURES = {
    "GeneralizedRCNN": GeneralizedRCNN,
    "GeneralizedRCNNRDN": GeneralizedRCNNRDN,
    "GeneralizedRCNNMEGA": GeneralizedRCNNMEGA,
    "GeneralizedRCNNFGFA": GeneralizedRCNNFGFA,
    "GeneralizedRCNNDFF": GeneralizedRCNNDFF,
    "GeneralizedRCNNMAMBA": GeneralizedRCNNMAMBA,
    "GeneralizedRCNNSELSA": GeneralizedRCNNSELSA,
}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
