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
