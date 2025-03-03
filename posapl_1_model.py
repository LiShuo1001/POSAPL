import os
import re
import json
import math
import torch
import logging
import numpy as np
from torch import nn
from pathlib import Path
from copy import deepcopy
import torch.nn.functional as F
from dataclasses import dataclass
from alisuretool.Tools import Tools
from collections import OrderedDict
from typing import Optional, Tuple, Union, Callable
from posapl_0_utils import to_2tuple
from open_clip.tokenizer import tokenize


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    pass


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    pass


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out

    pass


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

    pass


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()
        pass

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)
            pass

        pass

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x

    pass


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

    pass


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

    pass


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x

    pass


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0, ls_init_value: float = None,
                 act_layer: Callable = nn.GELU, norm_layer: Callable = nn.LayerNorm,
                 layer_id: int = 0, adapter=None, adapter2=None):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value else nn.Identity()

        # --------------------------------------------------------------------
        self.adapter = adapter[layer_id] if adapter is not None else None
        self.adapter2 = adapter2[layer_id] if adapter2 is not None else None
        # --------------------------------------------------------------------
        pass

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, pos=None, attn_mask: Optional[torch.Tensor] = None):
        x_a = self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.ls_1(x_a)

        # --------------------------------------------------------------------
        if self.adapter2 is not None and pos is not None:
            x = x + self.adapter2(pos)
            pass
        # --------------------------------------------------------------------

        x_mlp = self.mlp(self.ln_2(x))

        # --------------------------------------------------------------------
        if self.adapter is not None:
            x_mlp = x_mlp + self.adapter(x_mlp)
            pass
        # --------------------------------------------------------------------

        x = x + self.ls_2(x_mlp)
        return x

    pass


class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: float = 4.0, ls_init_value: float = None,
                 act_layer: Callable = nn.GELU, norm_layer: Callable = nn.LayerNorm, adapter=None, adapter2=None):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList([ResidualAttentionBlock(
            width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer,
            layer_id=layer_id, adapter=adapter, adapter2=adapter2) for layer_id in range(layers)])
        pass

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, pos=None, attn_mask: Optional[torch.Tensor] = None):
        feature_list = []
        step, total = 2, len(self.resblocks) - 1
        for i, r in enumerate(self.resblocks):
            x = r(x, pos=pos, attn_mask=attn_mask)

            if i in [total - step * 3, total - step * 2, total - step, total]:
                feature_list.append(x.permute(1, 0, 2))
            pass
        return feature_list, x

    pass


class PatchMean(nn.Module):

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        pass

    def forward(self, x):
        if len(x.shape) == 2:
            return x
        return x.mean(dim=self.dim)

    pass


class VisionTransformer(nn.Module):

    def __init__(self, image_size: int, patch_size: int, width: int, layers: int,
                 heads: int, mlp_ratio: float, ls_init_value: float = None, output_dim: int = 512,
                 act_layer: Callable = nn.GELU, norm_layer: Callable = nn.LayerNorm, adapter=None, config=None):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)

        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(width, layers, heads, mlp_ratio, ls_init_value=ls_init_value,
                                       act_layer=act_layer, norm_layer=norm_layer, adapter=adapter)

        self.ln_post = norm_layer(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # --------------------------------------------------------------------
        self.config = config
        if self.config.abl_model_has_adapter_fpn_image:
            self.adapter = nn.ModuleDict({
                "adapter_1": nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True), PatchMean(dim=1)),  # 1
                "adapter_2": nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True), PatchMean(dim=1)),  # 2
                "adapter_3": nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True), PatchMean(dim=1)),  # 3
                "adapter_4": nn.Sequential(PatchMean(dim=1), nn.Linear(width, width)),  # 4
            })
            self.adapter["adapter_4"][-1].weight.data.zero_()
            self.adapter["adapter_4"][-1].bias.data.zero_()
            pass
        # --------------------------------------------------------------------
        pass

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        feature_list, x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 0, :]

        x = self.ln_post(x)

        # --------------------------------------------------------------------
        if self.config.abl_model_has_adapter_fpn_image:
            x1, x2, x3, x4 = feature_list
            x1a = self.adapter["adapter_1"](x1)
            if self.config.abl_model_is_fpn_end_add:
                x2a = self.adapter["adapter_2"](x2)
                x3a = self.adapter["adapter_3"](x3)
                x4a = self.adapter["adapter_4"](x1a + x2a + x3a)
            else:
                x2a = self.adapter["adapter_2"](x1a.unsqueeze(1) + x2)
                x3a = self.adapter["adapter_3"](x2a.unsqueeze(1) + x3)
                x4a = self.adapter["adapter_4"](x3a.unsqueeze(1) + x4)
                pass
            x = x + x4a
            pass
        # --------------------------------------------------------------------

        # x = self.ln_post(x)
        x = x @ self.proj if self.proj is not None else x
        return x

    pass


class TextTransformer(nn.Module):

    def __init__(self, context_length: int = 77, vocab_size: int = 49408, width: int = 512, heads: int = 8,
                 layers: int = 12, ls_init_value: float = None, output_dim: int = 512, act_layer: Callable = nn.GELU,
                 norm_layer: Callable = nn.LayerNorm, adapter=None, adapter2=None, config=None):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.transformer = Transformer(width=width, layers=layers, heads=heads, ls_init_value=ls_init_value,
                                       act_layer=act_layer, norm_layer=norm_layer, adapter=adapter, adapter2=adapter2)
        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

        # --------------------------------------------------------------------
        self.config = config
        if self.config.abl_model_has_adapter_fpn_text:
            self.adapter = nn.ModuleDict({
                "adapter_1": nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True)),  # adapter 1
                "adapter_2": nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True)),  # adapter 2
                "adapter_3": nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True)),  # adapter 3
                "adapter_4": nn.Sequential(nn.Linear(width, width))   # adapter 4
            })
            self.adapter["adapter_4"][0].weight.data.zero_()
            self.adapter["adapter_4"][0].bias.data.zero_()
            pass

        if self.config.abl_model_has_adapter_pos:
            self.adapter_mlp = AdapterMLPZeroInit(512, 512)
            pass
        # --------------------------------------------------------------------
        pass

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            pass
        pass

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, pos=None):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)

        # --------------------------------------------------------------------
        if self.config.abl_model_has_adapter_pos:
            if pos is not None:
                pos = self.token_embedding(pos).to(cast_dtype)
                pos = pos + self.positional_embedding.to(cast_dtype)
                pos = self.adapter_mlp(pos)
                x = x + pos
                pos = pos.permute(1, 0, 2)
                pass
        else:
            pos = None
        # --------------------------------------------------------------------

        x = x.permute(1, 0, 2)  # NLD -> LND
        feature_list, x = self.transformer(x, pos=pos, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x)

        # --------------------------------------------------------------------
        if self.config.abl_model_has_adapter_fpn_text:
            x1, x2, x3, x4 = feature_list
            x1a = self.adapter["adapter_1"](x1)
            if self.config.abl_model_is_fpn_end_add:
                x2a = self.adapter["adapter_2"](x2)
                x3a = self.adapter["adapter_3"](x3)
                x4a = self.adapter["adapter_4"](x1a + x2a + x3a)
            else:
                x2a = self.adapter["adapter_2"](x1a + x2)
                x3a = self.adapter["adapter_3"](x2a + x3)
                x4a = self.adapter["adapter_4"](x3a + x4)
                pass
            x = x + x4a
            pass
        # --------------------------------------------------------------------

        # x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    pass


class AdapterMLPZeroInit(nn.Module):

    def __init__(self, hidden_dim, down_dim=128):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, down_dim)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(down_dim, hidden_dim)
        self.init_weights()
        pass

    def init_weights(self):
        self.l2.weight.data.zero_()
        self.l2.bias.data.zero_()
        pass

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))

    pass


class AdapterSAZeroInit(nn.Module):

    def __init__(self, hidden_dim, num_heads, down_dim=128):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, down_dim)
        self.l2 = nn.Linear(down_dim, hidden_dim)
        self.msa = nn.MultiheadAttention(down_dim, num_heads)
        self.init_weights()
        pass

    def init_weights(self):
        self.l2.weight.data.zero_()
        self.l2.bias.data.zero_()
        pass

    def forward(self, x):
        x = self.l1(x)
        attn, _ = self.msa(x, x, x)
        attn = attn + x
        x = self.l2(attn)
        return x

    pass


class CLIP(nn.Module):

    def __init__(self, embed_dim: int, vision_cfg: CLIPVisionCfg, text_cfg: CLIPTextCfg, quick_gelu=False, config=None):
        super().__init__()
        self.config = config

        # --------------------------------------------------------------------
        self.adapter_img = None
        if self.config.abl_model_has_adapter_image:
            self.adapter_img = nn.ModuleList([
                AdapterSAZeroInit(768, 8,  self.config.abl_param_down_dim) for l_id in range(12)])
        self.adapter_text = None
        if self.config.abl_model_has_adapter_text:
            self.adapter_text = nn.ModuleList([
                AdapterSAZeroInit(512, 8, self.config.abl_param_down_dim) for l_id in range(12)])
        # ################################################################################################
        self.adapter_pos = None
        if self.config.abl_model_has_adapter_pos:
            if self.config.abl_model_has_pos_side:
                self.adapter_pos = nn.ModuleList([
                    AdapterMLPZeroInit(512, self.config.abl_param_down_dim_pos) for l_id in range(12)])
        # ################################################################################################
        # --------------------------------------------------------------------

        self.visual = self._build_vision_tower(embed_dim, vision_cfg, quick_gelu,
                                               adapter_img=self.adapter_img, config=self.config)
        self.text = self._build_text_tower(embed_dim, text_cfg, quick_gelu, adapter_text=self.adapter_text,
                                           adapter_pos=self.adapter_pos, config=self.config)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        pass

    @staticmethod
    def _build_vision_tower(embed_dim: int, vision_cfg, quick_gelu=False, adapter_img=None, config=None):
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        if isinstance(vision_cfg.layers, (tuple, list)):
            visual = ModifiedResNet(layers=vision_cfg.layers, output_dim=embed_dim,
                                    heads=vision_cfg.width * 32 // vision_cfg.head_width,
                                    image_size=vision_cfg.image_size, width=vision_cfg.width)
        else:
            visual = VisionTransformer(
                image_size=vision_cfg.image_size, patch_size=vision_cfg.patch_size, width=vision_cfg.width,
                layers=vision_cfg.layers, heads=vision_cfg.width // vision_cfg.head_width,
                mlp_ratio=vision_cfg.mlp_ratio, ls_init_value=vision_cfg.ls_init_value, output_dim=embed_dim,
                act_layer=QuickGELU if quick_gelu else nn.GELU, norm_layer=nn.LayerNorm,
                adapter=adapter_img, config=config)
        return visual

    @staticmethod
    def _build_text_tower(embed_dim, text_cfg, quick_gelu=False, adapter_text=None, adapter_pos=None, config=None):
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)
        text = TextTransformer(context_length=text_cfg.context_length, vocab_size=text_cfg.vocab_size,
                               width=text_cfg.width, heads=text_cfg.heads, layers=text_cfg.layers,
                               ls_init_value=text_cfg.ls_init_value, output_dim=embed_dim,
                               act_layer=QuickGELU if quick_gelu else nn.GELU, norm_layer=nn.LayerNorm,
                               adapter=adapter_text, adapter2=adapter_pos, config=config)
        return text

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, pos=None, normalize: bool = False):
        features = self.text(text, pos=pos)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, self.logit_scale.exp()

    pass


class CreateModel(object):

    def __init__(self, config):
        self.config = config

        self._MODEL_CONFIG_PATHS = [Path(__file__).parent / f"open_clip/model_configs/"]
        self._MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs
        self._rescan_model_configs()  # initial populate of model config registry

        self.OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        self.OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
        self.tokenize = tokenize
        pass

    def create_model(self, model_name, pretrained, pretrain_path):
        model_name = model_name.replace('/', '-')

        # 创建模型
        model = CLIP(**self.get_model_config(model_name), config=self.config)
        model.visual.image_mean = self.OPENAI_DATASET_MEAN
        model.visual.image_std = self.OPENAI_DATASET_STD

        # 加载要用的权重
        state_dict = self.load_state_dict(pretrain_path)
        if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
            state_dict = self.convert_to_custom_text_state_dict(state_dict)
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        return model

    def _rescan_model_configs(self):
        def _natural_key(string_):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

        config_ext = ('.json',)
        config_files = []
        for config_path in self._MODEL_CONFIG_PATHS:
            if config_path.is_file() and config_path.suffix in config_ext:
                config_files.append(config_path)
            elif config_path.is_dir():
                for ext in config_ext:
                    config_files.extend(config_path.glob(f'*{ext}'))

        for cf in config_files:
            with open(cf, 'r') as f:
                model_cfg = json.load(f)
                if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                    self._MODEL_CONFIGS[cf.stem] = model_cfg

        self._MODEL_CONFIGS = {k: v for k, v in sorted(self._MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}
        pass

    def get_model_config(self, model_name):
        if model_name in self._MODEL_CONFIGS:
            return deepcopy(self._MODEL_CONFIGS[model_name])
        else:
            return None
        pass

    @staticmethod
    def load_state_dict(checkpoint_path: str, map_location='cpu'):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        if next(iter(state_dict.items()))[0].startswith('module'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        return state_dict

    @staticmethod
    def convert_to_custom_text_state_dict(state_dict: dict):
        if 'text_projection' in state_dict:
            new_state_dict = {}
            for k, v in state_dict.items():
                if any(k.startswith(p) for p in ('text_projection', 'positional_embedding',
                                                 'token_embedding', 'transformer', 'ln_final')):
                    k = 'text.' + k
                new_state_dict[k] = v
            return new_state_dict
        return state_dict

    pass
