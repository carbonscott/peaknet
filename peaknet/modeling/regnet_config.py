from dataclasses import dataclass

@dataclass
class BNConfig:
    EPS     : float = 1e-5
    MOMENTUM: float = 1e-1


@dataclass
class ResStemConfig:
    BN          : BNConfig = BNConfig()
    RELU_INPLACE: bool = False


@dataclass
class ResBlockConfig:
    BN           : BNConfig = BNConfig()
    RELU_INPLACE : bool = False
    USES_RES_V1p5: bool = True


@dataclass
class ResStageConfig:
    RESBLOCK: ResBlockConfig = ResBlockConfig()


@dataclass
class ResNet50Config:
    RESSTEM : ResStemConfig  = ResStemConfig()
    RESSTAGE: ResStageConfig = ResStageConfig()
