from torch import nn

from .generalized_enhancement import GEO
from .memory_bank import MemoryBank


class MemoryBankIns(MemoryBank):
    def __init__(self, cfg):
        super().__init__(
            max_num=cfg.MODEL.VID.MAMBA.MEMORY_BANK.MAX_NUM,
            key_num=cfg.MODEL.VID.MAMBA.MEMORY_BANK.KEY_NUM,
            sampling_policy=cfg.MODEL.VID.MAMBA.MEMORY_BANK.SAMPLING_POLICY,
            updating_policy=cfg.MODEL.VID.MAMBA.MEMORY_BANK.UPDATING_POLICY,
        )

        num_enhancement = cfg.MODEL.VID.MAMBA.INS_MEM.NUM_ENHANCEMENT
        geo_list = []
        for _ in range(num_enhancement):
            geo = GEO(cfg.MODEL.VID.MAMBA.INS_MEM.GEO)
            geo_list.append(geo)

        self.add_module("geos", nn.Sequential(*geo_list))

    def forward(self, x, mem=None):
        raise NotImplementedError
