import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .generalized_enhancement import GEO
from .memory_bank import MemoryBank

# max num of pixels of an object
PIXEL_NUM = 100


def get_l1_norm(k, q):
    """
    q [b, q, c]; k [b, k, c]
    return: [b, k, q]
    """
    assert len(q.size()) == len(k.size()) == 3
    k = k.unsqueeze(2)
    q = q.unsqueeze(1)
    return torch.norm(k - q, p=1, dim=-1)


class MemoryBankPix(MemoryBank):
    def __init__(self, cfg):
        super().__init__(
            max_num=cfg.MODEL.VID.MAMBA.MEMORY_BANK.MAX_NUM,
            key_num=cfg.MODEL.VID.MAMBA.MEMORY_BANK.KEY_NUM,
            sampling_policy=cfg.MODEL.VID.MAMBA.MEMORY_BANK.SAMPLING_POLICY,
            updating_policy=cfg.MODEL.VID.MAMBA.MEMORY_BANK.UPDATING_POLICY,
        )

        num_enhancement = cfg.MODEL.VID.MAMBA.PIX_MEM.NUM_ENHANCEMENT
        geo_list = []
        for _ in range(num_enhancement):
            geo = GEO(cfg.MODEL.VID.MAMBA.PIX_MEM.GEO)
            geo_list.append(geo)

        self.add_module("geos", nn.Sequential(*geo_list))

        self.score_thresh = cfg.MODEL.VID.MAMBA.PIX_MEM.SCORE_THRESH
        self.stride = cfg.MODEL.VID.MAMBA.PIX_MEM.STRIDE
        self.use_relu = cfg.MODEL.VID.MAMBA.PIX_MEM.USE_RELU
        self.sample_train = cfg.MODEL.VID.MAMBA.PIX_MEM.SAMPLE_TRAIN

        # initialize pixel memory
        self.obj_irr_mem = None

    def reset(self):
        self.feat = None
        self.obj_irr_mem = None

    def get_mem(self):
        """
        :return: [memory: [1, c, l, 1]]
        """
        """
        :return: [memory: [1, c, l, 1]]
        """
        mem_temp = []

        # have obj
        if self.__len__() > 0:
            mem_temp.append(self.sample())

        if self.obj_irr_mem is not None:
            mem_temp.append(self.obj_irr_mem)

        if len(mem_temp) > 0:
            # [1, c, l, 1]
            mem = torch.cat(mem_temp, dim=0)
            mem_temp = mem.permute(1, 0).unsqueeze(0).unsqueeze(-1)
        return mem_temp

    def forward(self, x, mem=None):
        """
        :param x: [1, C, H, W]
        :param mem: [b, C, H, W]
        :return: out: [1, C, H, W]
        """
        if not self.training:
            assert mem is None
            # mem [1, C, L, 1]
            mem = self.get_mem()
            if len(mem) == 0:
                # no memory
                # self enhancement
                mem = x

        b, c, h, w = x.size()
        assert b == 1
        # [1, c, h, w] -> [1*h*w, c]
        query = x.permute(1, 0, 2, 3).reshape(c, -1).transpose(1, 0)
        # [b, c, h, w] -> [b*h*w, c]
        key = mem.permute(1, 0, 2, 3).reshape(c, -1).transpose(1, 0)

        if self.training and self.sample_train:
            _sampled_ind = torch.randperm(key.size(0))[: self.key_num]
            key = key[_sampled_ind]

        # enhancement
        for _geo in self.geos:
            query = _geo(query, key)
            if self.use_relu:
                query = F.relu(query)

        # [h*w, c] -> [1, c, h, w]
        out = query.reshape(h, w, c).permute(2, 0, 1).unsqueeze(0)
        return out

    def write_operation(self, x, proposals=None):
        """save pixels within detected boxes into memory.

        :param x: [[N, C, H, W]]
        :param proposals: [BoxList]
        :return:
        """
        N, C, H, W = x.size()
        if self.training:
            raise NotImplementedError

        proposals = proposals[0]
        # [N, C, H, W] -> [N, H*W, C] -> [H*W, C]
        x = x.view(N, C, -1).permute(0, 2, 1).squeeze(0)

        self.write_test(x, proposals, W)

    def write_test(self, x, proposals, width):
        """
        :param x: [H*W, self.c]
        :param width: width of original feature map
        :param proposals: BoxList
        :return:
        """
        # no obj detected
        if len(proposals) == 0:
            # do nothing
            return

        pred_labels = proposals.get_field("labels").detach().cpu().numpy()
        pred_scores = proposals.get_field("scores").detach().cpu().numpy()
        boxes = proposals.bbox.detach()
        # stride = 16
        boxes = (boxes / self.stride).int().cpu().numpy()
        temp_obj_pixels = []

        if pred_scores.max() < self.score_thresh:
            # no high quality obj -> do nothing
            return

        for box, pred_score, pred_label in zip(boxes, pred_scores, pred_labels):
            if pred_score >= self.score_thresh:
                # 1. map pixels in box to new index on x_box [H*W, C]
                # box [x1, y1, x2, y2] -> [ind_1, ind_2, ind_3, ... ]
                inds = sorted(self.box_to_inds_list(box, width))
                # 2. get mem_dict

                # save part obj
                if len(inds) > PIXEL_NUM:
                    inds = np.asarray(inds)
                    inds = np.random.choice(inds, PIXEL_NUM, replace=False)

                pixels = x[inds]

                self.update(pixels)

            elif pred_score >= 0.5:
                # quality [0.5, 0.9)
                inds = sorted(self.box_to_inds_list(box, width))

                # save part obj
                if len(inds) > PIXEL_NUM:
                    inds = np.asarray(inds)
                    inds = np.random.choice(inds, PIXEL_NUM, replace=False)

                pixels = x[inds]
                temp_obj_pixels.append(pixels)

        # obj irr pixels
        obj_irr_pixels = self.get_obj_irr_pixels(x)
        # save part of irr pixels
        if len(obj_irr_pixels) > PIXEL_NUM:
            inds = np.arange(len(obj_irr_pixels))
            inds = np.random.choice(inds, PIXEL_NUM, replace=False)
            obj_irr_pixels = obj_irr_pixels[inds]

        if len(temp_obj_pixels) > 0:
            # low quality obj
            obj_temp_pixels = torch.cat(temp_obj_pixels, dim=0)
            obj_irr_pixels = torch.cat([obj_temp_pixels, obj_irr_pixels])
        self.obj_irr_mem = obj_irr_pixels
        return

    @staticmethod
    def box_to_inds_list(box, w):
        inds = []
        for x_i in range(box[0], box[2] + 1):
            for y_j in range(box[1], box[3] + 1):
                inds.append(int(x_i + y_j * w))
        return inds

    @staticmethod
    def get_obj_irr_pixels(x, scale=1.0):
        """get object irrelevant features.

        :param x: [n, c]
        :param scale: factor to control threshold
        :return: [m, c]
        """
        n, c = x.size()
        l2_norm = x.pow(2).sum(dim=1).sqrt() / np.sqrt(c)
        keep_irrelevant = F.softmax(l2_norm, dim=0) > scale / n
        pixels = x[keep_irrelevant]
        return pixels.detach().clone()
