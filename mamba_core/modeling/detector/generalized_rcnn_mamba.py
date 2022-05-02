# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Implements the Generalized R-CNN framework."""

import torch
from mamba_core.structures.image_list import to_image_list
from PIL import Image
from torch import nn

from ..backbone import build_backbone
from ..mem_bank.memory_bank_pix import MemoryBankPix
from ..roi_heads.roi_heads import build_roi_heads
from ..rpn.rpn import build_rpn

# from ..mem_bank.memory_bank_ins import MemoryBankIns


class GeneralizedRCNNMAMBA(nn.Module):
    """Main class for Generalized R-CNN. Currently supports boxes and masks. It
    consists of three main parts:

    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.MODEL.DEVICE
        self.use_pix_mem = cfg.MODEL.VID.MAMBA.PIX_MEM.ENABLE
        self.use_ins_mem = cfg.MODEL.VID.MAMBA.INS_MEM.ENABLE
        self.backbone = build_backbone(cfg)

        # build pixel-level memory bank
        if self.use_pix_mem:
            self.pix_mem = MemoryBankPix(cfg)

        self.rpn = build_rpn(cfg, self.backbone.out_channels)

        # build instance-level memory bank inside roi_heads
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.initial_frame_num = cfg.MODEL.VID.MAMBA.INITIAL_FRAME_NUM
        self.num_proposals_ref = cfg.MODEL.VID.RPN.REF_POST_NMS_TOP_N

        # freeze parameters if using feature alignment
        if self.use_pix_mem and cfg.MODEL.VID.MAMBA.PIX_MEM.GEO.OUTPUT_TYPE in (
            "alignment",
        ):
            self.backbone.requires_grad_(False)
            self.rpn.requires_grad_(False)
            self.roi_heads.requires_grad_(False)

    def forward(self, images, targets=None):
        """
        Arguments:
            #images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            all_img = to_image_list(
                [images["cur"], *[img.tensors[0] for img in images["ref_g"]]]
            )
            images["cur"] = to_image_list(all_img.tensors[0])
            images["ref_g"] = [to_image_list(image) for image in all_img.tensors[1:]]
            # modify target size
            targets[0].size = tuple(images["cur"].image_sizes[0][::-1])

            return self._forward_train(images["cur"], images["ref_g"], targets)
        else:
            images["cur"] = to_image_list(images["cur"])
            infos = images.copy()
            infos.pop("cur")

            return self._forward_test(images["cur"], infos)

    def _forward_train(self, img_cur, imgs_g, targets):
        concat_imgs = torch.cat(
            [img_cur.tensors, *[img.tensors for img in imgs_g]], dim=0
        )
        concat_feats = self.backbone(concat_imgs)[0]

        num_imgs = 1 + len(imgs_g)
        # [key, ref_1, ref_2, ..., ref_k]
        feats_list = list(torch.chunk(concat_feats, num_imgs, dim=0))

        if self.use_pix_mem:
            # last 2 ref images as pixel memory
            pix_mem = torch.cat(feats_list[-2:], dim=0)
            for i in range(len(feats_list) - 2):
                feats_list[i] = self.pix_mem(feats_list[i], pix_mem)

        proposals, proposal_losses = self.rpn(img_cur, (feats_list[0],), targets)
        proposals_list = [proposals]

        if self.use_ins_mem:
            # first 2 ref images as instance memory
            for i in range(2):
                proposals_ref = self.rpn(imgs_g[i], (feats_list[i + 1],), version="ref")
                proposals_list.append(proposals_ref[0])
        else:
            feats_list = (feats_list[0],)
            proposals_list = proposals_list[0]

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(
                feats_list, proposals_list, targets
            )
        else:
            detector_losses = {}

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def _init_memory(self, infos, imgs):
        if self.use_pix_mem:
            self.pix_mem.reset()
        if self.use_ins_mem:
            self.roi_heads.box.feature_extractor.ins_mem.reset()

        # init features in memory bank
        features_list = []
        proposals_list = []
        if self.use_ins_mem or self.use_pix_mem:
            for _ in range(self.initial_frame_num):
                self.end_id = min(self.end_id + 1, self.seg_len - 1)
                if "shuffled_index" in infos.keys():
                    # shuffle test
                    end_filename = infos["pattern"] % int(
                        infos["shuffled_index"][self.end_id]
                    )
                else:
                    end_filename = infos["pattern"] % self.end_id
                end_image = Image.open(infos["img_dir"] % end_filename).convert(
                    "RGB"
                )

                end_image = infos["transforms"](end_image)
                if isinstance(end_image, tuple):
                    end_image = end_image[0]
                img = end_image.view(1, *end_image.shape).to(self.device)

                features = self.backbone(img)[0]
                features_list.append(features)
                proposals, _ = self.rpn(imgs, (features,), version="key")
                proposals_list.append(proposals)

        if self.use_ins_mem:
            for _feat, _proposal in zip(features_list, proposals_list):
                _proposal = [_proposal[0][: self.num_proposals_ref]]
                _proposal_feat = self.roi_heads.box.feature_extractor(
                    _feat, _proposal, pre_calculate=True
                )
                self.roi_heads.box.feature_extractor.ins_mem.update(_proposal_feat)

        if self.use_pix_mem:
            for _feat, _proposal in zip(features_list, proposals_list):
                if not self.use_ins_mem:
                    _, detections, _ = self.roi_heads((_feat,), _proposal, None)
                else:
                    proposals_feat = self.roi_heads.box.feature_extractor(
                        _feat, _proposal, pre_calculate=True
                    )
                    proposals_feat_ref = (
                        self.roi_heads.box.feature_extractor.ins_mem.sample()
                    )
                    # no memory
                    if len(proposals_feat_ref) == 0:
                        proposals_feat_ref = proposals_feat
                    proposals_list = [_proposal, proposals_feat, proposals_feat_ref]
                    _, detections, _ = self.roi_heads(None, proposals_list, None)
                self.pix_mem.write_operation(_feat, detections)

    def _forward_test(self, imgs, infos, targets=None):
        if targets is not None:
            raise ValueError("In testing mode, targets should be None")

        # a new video, init pix and ins memory
        if infos["frame_category"] == 0:
            self.seg_len = infos["seg_len"]
            self.end_id = 0
            # reset mem
            video = infos["filename"].split("/")[-2]
            video_id = int(video.split("_")[-1])
            if video_id % 1000 == 0:
                # share memory across videos
                # print(video)
                self._init_memory(infos, imgs)

        # if no instance memory, init pix and ins memory
        if self.use_ins_mem and len(self.roi_heads.box.feature_extractor.ins_mem) == 0:
            self._init_memory(infos, imgs)

        feats = self.backbone(imgs.tensors)[0]
        # enhance with pixel-level memory bank
        if self.use_pix_mem:
            feats = self.pix_mem(feats)
        proposals, _ = self.rpn(imgs, (feats,), None)

        if self.use_ins_mem:
            proposals_feat = self.roi_heads.box.feature_extractor(
                feats, proposals, pre_calculate=True
            )
            proposals_feat_ref = self.roi_heads.box.feature_extractor.ins_mem.sample()
            # no memory
            if len(proposals_feat_ref) == 0:
                proposals_feat_ref = proposals_feat
            proposals_list = [proposals, proposals_feat, proposals_feat_ref]
            feats = (feats,)
        else:
            proposals_list = proposals
            feats = (feats,)

        x, result, detector_losses = self.roi_heads(feats, proposals_list, None)

        # update memory bank
        if self.use_pix_mem:
            self.pix_mem.write_operation(feats[0], result)
        if self.use_ins_mem:
            proposals_feat_ref = proposals_feat[: self.num_proposals_ref]
            self.roi_heads.box.feature_extractor.ins_mem.update(proposals_feat_ref)

        return result
