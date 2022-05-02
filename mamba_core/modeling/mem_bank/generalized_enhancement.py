import math
import torch
from mamba_core.modeling.make_layers import Conv2d, make_fc
from torch import nn
from torch.nn import functional as F


class GEO(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        in_channels = cfg.IN_CHANNELS
        emb_channels = cfg.EMB_CHANNELS
        num_heads = cfg.NUM_HEADS
        output_type = cfg.OUTPUT_TYPE
        query_weight = cfg.QUERY_WEIGHT

        self.output_type = output_type
        if output_type == "alignment":
            self.query_weight = query_weight

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.emb_channels = emb_channels
        self.num_heads = num_heads

        self.wq = make_fc(self.in_channels, self.emb_channels)
        self.wk = make_fc(self.in_channels, self.emb_channels)

        self.wv = Conv2d(
            self.emb_channels * self.num_heads,
            self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.num_heads,
        )

        # init_weight
        if self.output_type == "alignment" and cfg.USE_EYE_INIT:
            torch.nn.init.eye_(self.wv.weight[:, :, 0, 0])
            torch.nn.init.constant_(self.wv.bias, 0)
        else:
            torch.nn.init.normal_(self.wv.weight, std=0.01)
            torch.nn.init.constant_(self.wv.bias, 0)

    def forward(self, query, key=None):
        """
        :param query: [Q, C]
        :param key: [K, C]
        :return: out: [Q, C]
        """
        # multi-head attentions
        if not self.training:
            out = self.enhance_with_mem_test(query, key)
        else:
            out = self.enhance_with_mem_train(query, key)

        return out

    def attention_module_multi_head(self, query, key):
        feat_dim = self.in_channels
        dim = (self.in_channels, self.in_channels, self.in_channels)
        group = self.num_heads
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
        # multi head
        assert dim[0] == dim[1]

        q_data = self.wq(query)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, num_rois, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)

        k_data = self.wk(key)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, num_nongt_rois, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [num_nongt_rois, feat_dim]
        v_data = key

        # aff, [group, num_rois, num_nongt_rois]
        aff = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        # [group, q, k]
        # aff_scale, [num_rois, group, num_nongt_rois]
        aff_scale = aff_scale.permute(1, 0, 2)

        # weighted_aff, [num_rois, group, num_nongt_rois]
        weighted_aff = aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)

        aff_softmax_reshape = aff_softmax.reshape(
            aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2]
        )

        # output_t, [num_rois * group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, group * feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = self.wv(output_t)

        output = linear_out.squeeze(3).squeeze(2)

        return output

    def enhance_with_mem_train(self, query, key):
        att = self.attention_module_multi_head(query, key)

        if self.output_type == "attention":
            out = query + att
        elif self.output_type == "alignment":
            out = att
        else:
            raise NotImplementedError

        return out

    def enhance_with_mem_test(self, query, key):
        att = self.attention_module_multi_head(query, key)

        if self.output_type == "attention":
            out = query + att
        elif self.output_type == "alignment":
            out = self.query_weight * query + (1 - self.query_weight) * att
        else:
            raise NotImplementedError

        return out
