from .utils import Configurator

CONFIG = Configurator()
with CONFIG.enable_auto_create():
    CONFIG.RESSTEM.BN.EPS = 1e-5
    CONFIG.RESSTEM.BN.MOMENTUM = 1e-1
