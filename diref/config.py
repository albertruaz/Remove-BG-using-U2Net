"""Minimal configuration values required for BiRefNet inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Config:
    # Core architecture choices (matching BiRefNet DIS checkpoint)
    auxiliary_classification: bool = False
    batch_size: int = 4
    bb: str = "swin_v1_l"
    dec_blk: str = "BasicDecBlk"
    dec_att: str = "ASPPDeformable"
    dec_ipt: bool = True
    dec_ipt_split: bool = True
    dec_channels_inter: str = "fixed"
    freeze_bb: bool = False
    ms_supervision: bool = True
    mul_scl_ipt: str = "cat"
    out_ref: bool = True
    refine: str = ""
    squeeze_block: str = "BasicDecBlk_x1"
    ender: str = ""
    size: Tuple[int, int] = (1024, 1024)
    SDPA_enabled: bool = False

    # Auxiliary values derived from architecture choices
    lateral_channels_in_collection: List[int] = field(init=False)
    cxt_num: int = 3
    cxt: List[int] = field(init=False)
    lat_blk: str = "BasicLatBlk"
    weights: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        channels_map = {
            "vgg16": [512, 512, 256, 128],
            "vgg16bn": [512, 512, 256, 128],
            "resnet50": [2048, 1024, 512, 256],
            "pvt_v2_b0": [256, 160, 64, 32],
            "pvt_v2_b1": [512, 320, 128, 64],
            "pvt_v2_b2": [512, 320, 128, 64],
            "pvt_v2_b5": [512, 320, 128, 64],
            "swin_v1_t": [1536, 768, 384, 192],
            "swin_v1_s": [1536, 768, 384, 192],
            "swin_v1_b": [2048, 1024, 512, 256],
            "swin_v1_l": [3072, 1536, 768, 384],
        }
        if self.bb not in channels_map:
            raise ValueError(f"Unsupported backbone '{self.bb}' for lightweight config")
        self.lateral_channels_in_collection = channels_map[self.bb]

        if self.cxt_num:
            self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:]
        else:
            self.cxt = []
