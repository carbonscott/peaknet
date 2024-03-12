from dataclasses import dataclass

@dataclass
class BNConfig:
    eps     : float = 1e-5
    momentum: float = 1e-1


@dataclass
class FusionConfig:
    eps: float = 1e-5


@dataclass
class BiFPNBlockConfig:
    relu_inplace     : bool         = False
    down_scale_factor: float        = 0.5
    up_scale_factor  : int          = 2
    num_features     : int          = 256
    num_levels       : int          = 4
    base_level       : int          = 2    # ...ResNet50 uses 2, EfficientNet uses 3
    bn               : BNConfig     = BNConfig()
    fusion           : FusionConfig = FusionConfig()


@dataclass
class BiFPNConfig:
    num_blocks: int = 1
    block     : BiFPNBlockConfig = BiFPNBlockConfig()
