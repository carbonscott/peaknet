from .utils import Configurator

CONFIG = Configurator()
with CONFIG.enable_auto_create():
    CONFIG.RESSTEM.BN.EPS       = 1e-5
    CONFIG.RESSTEM.BN.MOMENTUM  = 1e-1
    CONFIG.RESSTEM.RELU_INPLACE = True

with CONFIG.enable_auto_create():
    CONFIG.RESBLOCK.BN.EPS       = 1e-5
    CONFIG.RESBLOCK.BN.MOMENTUM  = 1e-1
    CONFIG.RESBLOCK.RELU_INPLACE = True

CONFIG.USES_RES_V1p5 = True

with CONFIG.enable_auto_create():
    CONFIG.BIFPN.BN.EPS            = 1e-5
    CONFIG.BIFPN.BN.MOMENTUM       = 1e-1
    CONFIG.BIFPN.RELU_INPLACE      = True
    CONFIG.BIFPN.DOWN_SCALE_FACTOR = 0.5
    CONFIG.BIFPN.UP_SCALE_FACTOR   = 2
    CONFIG.BIFPN.FUSION.EPS        = 1e-5

