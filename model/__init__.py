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

__all__ = [
    "ConditionalSimNet2d",
    "ConditionalSimNet1d",
    "CS_Tripletnet",
    "UNet",
    "PL_UNet",
    "NNet",
    "AVGPooling",
    "To1D640",
    "EmbeddingNet",
    "JNet128Embnet",
    "JNetCrossMLP",
    "JNetSAPMLP",
    "TransformerSAPMLP",
    "ConvTransformerSAPMLP",
    "TransformerWithBPM",
    "TransformerWithBPMCLS"
]
