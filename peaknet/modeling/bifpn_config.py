from dataclasses import dataclass

@dataclass
class BNConfig:
    EPS     : float = 1e-5
    MOMENTUM: float = 1e-1


@dataclass
class FusionConfig:
    EPS: float = 1e-5


@dataclass
class BiFPNBlockConfig:
    RELU_INPLACE     : bool         = False
    DOWN_SCALE_FACTOR: float        = 0.5
    UP_SCALE_FACTOR  : int          = 2
    NUM_BLOCKS       : int          = 1
    NUM_FEATURES     : int          = 256
    NUM_LEVELS       : int          = 4
    BASE_LEVEL       : int          = 2    # ...ResNet50 uses 2, EfficientNet uses 3
    BN               : BNConfig     = BNConfig()
    FUSION           : FusionConfig = FusionConfig()


@dataclass
class BiFPNConfig(BiFPNBlockConfig):
    pass
