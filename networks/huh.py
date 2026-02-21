import torch
from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
m = SwinTransformerSys(img_size=512, patch_size=4, in_chans=3)
x = torch.randn(2,3,512,512)
y = m(x)
print(y.shape)