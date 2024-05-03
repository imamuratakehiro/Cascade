"""modelファイルの実体化"""

from .csn import ConditionalSimNet2d, ConditionalSimNet1d

from .tripletnet import CS_Tripletnet

from .unet.model_unet import UNet
from .unet.unet import PL_UNet

from .nnet.nnet import NNet

from .to1d.model_avgp import AVGPooling
from .to1d.model_linear import To1D640
from .to1d.model_embedding import EmbeddingNet

from .jnet.model_jnet_128_embnet import JNet128Embnet
from .jnet.model_jnet_cross_mlp import JNetCrossMLP
from .jnet.model_jnet_selfattpool import JNetSAPMLP
from .jnet.model_jnet_transformer_attpool import TransformerSAPMLP
from .jnet.model_conv_transformer_selfattpool import ConvTransformerSAPMLP
from .jnet.model_transformer_bpm import TransformerWithBPM
from .jnet.model_transformer_bpm_cls import TransformerWithBPMCLS

__all__ = ["ConditionalSimNet2d",
            "ConditionalSimNet1d"
            "UNetcsnde5",
            "UNetNormal",
            "UNetForTriplet640De5",
            "UNetForTriplet640De1",
            "UNetForTriplet128De1",
            "AVGPooling",
            "To1D",
            "UNetnotcsnde5",
            "UNetForTripletInst",
            "model_ae",
            "AE"]
