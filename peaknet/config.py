from .configurator import Configurator

CONFIG = Configurator()
with CONFIG.enable_auto_create():
    CONFIG.RESSTEM.BN.EPS       = 1e-5
    CONFIG.RESSTEM.BN.MOMENTUM  = 1e-1
    CONFIG.RESSTEM.RELU_INPLACE = False

with CONFIG.enable_auto_create():
    CONFIG.RESBLOCK.BN.EPS       = 1e-5
    CONFIG.RESBLOCK.BN.MOMENTUM  = 1e-1
    CONFIG.RESBLOCK.RELU_INPLACE = False

CONFIG.USES_RES_V1p5 = True

with CONFIG.enable_auto_create():
    CONFIG.BIFPN.BN.EPS            = 1e-5
    CONFIG.BIFPN.BN.MOMENTUM       = 1e-1
    CONFIG.BIFPN.RELU_INPLACE      = False
    CONFIG.BIFPN.DOWN_SCALE_FACTOR = 0.5
    CONFIG.BIFPN.UP_SCALE_FACTOR   = 2
    CONFIG.BIFPN.FUSION.EPS        = 1e-5

with CONFIG.enable_auto_create():
    CONFIG.RES_ATT_UNET.CHANNELS = {
        "fusion layers" : (
            (2048, 1024),
            (1024, 512 ),
            (512,  256 ),
            (256,  64  ),
        ),
        "head_segmask_layer" : (64, 3),
    }

with CONFIG.enable_auto_create():
    CONFIG.RESNET_ENCODER.SAVES_LAYER = {
        "stem"   : True,
        "layer1" : True,
        "layer2" : True,
        "layer3" : True,
        "layer4" : True,
    }

    CONFIG.RESNET_ENCODER.OUTPUT_CHANNELS = {
        "stem"   : 64,
        "layer1" : 256,
        "layer2" : 512,
        "layer3" : 1024,
        "layer4" : 2048,
    }

    CONFIG.SEG_HEAD.UP_SCALE_FACTOR = [
        2,  # stem
        4,  # layer1
        8,  # layer2
        16, # layer3
        32, # layer4
    ]

    CONFIG.SEG_HEAD.Q3_UP_SCALE_FACTOR = 2

    CONFIG.SEG_HEAD.Q3_IN_CHANNELS   = 64
    CONFIG.SEG_HEAD.FUSE_IN_CHANNELS = 64 * 5
    CONFIG.SEG_HEAD.OUT_CHANNELS     = 3

    CONFIG.SEG_HEAD.USES_Q3 = True
