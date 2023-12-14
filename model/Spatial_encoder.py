## Our model was revised from https://github.com/zczcwh/PoseFormer/blob/main/common/model_poseformer.py

import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath

from common.opt import opts

opt = opts().parse()
device = torch.device("cuda")


#######################################################################################################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


#######################################################################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.edge_embedding = nn.Linear(17*17, 17*17)

    def forward(self, x, edge_embedding):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        edge_embedding = self.edge_embedding(edge_embedding)
        edge_embedding = edge_embedding.reshape(1, 17, 17).unsqueeze(0).repeat(B, self.num_heads, 1, 1)
        # print(edge_embedding.shape)

        attn = attn + edge_embedding


        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#######################################################################################################################
class CVA_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.Qnorm = nn.LayerNorm(dim)
        self.Knorm = nn.LayerNorm(dim)
        self.Vnorm = nn.LayerNorm(dim)
        self.QLinear = nn.Linear(dim, dim)
        self.KLinear = nn.Linear(dim, dim)
        self.VLinear = nn.Linear(dim, dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.edge_embedding = nn.Linear(17*17, 17*17)




    def forward(self, x, CVA_input, edge_embedding):
        B, N, C = x.shape
        # CVA_input = self.max_pool(CVA_input)
        # print(CVA_input.shape)
        q = self.QLinear(self.Qnorm(CVA_input)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.KLinear(self.Knorm(CVA_input)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.VLinear(self.Vnorm(x)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        edge_embedding = self.edge_embedding(edge_embedding)
        edge_embedding = edge_embedding.reshape(1, 17, 17).unsqueeze(0).repeat(B, self.num_heads, 1, 1)



        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#######################################################################################################################
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), edge_embedding))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#######################################################################################################################
class Multi_Out_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


        self.norm_hop1 = norm_layer(dim)
        self.norm_hop2 = norm_layer(dim)
        self.mlp_hop = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, hops, edge_embedding):
        MSA = self.drop_path(self.attn(self.norm1(x), edge_embedding))
        MSA = self.norm_hop1(hops) * MSA

        x = x + MSA
        x = x + self.drop_path(self.mlp(self.norm2(x)))


        hops = hops + MSA
        hops = hops + self.drop_path(self.mlp_hop(self.norm_hop2(hops)))
        

        return x, hops, MSA


#######################################################################################################################
class Multi_In_Out_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cva_attn = CVA_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.max_pool = nn.MaxPool1d(3, stride=1, padding=1, dilation=1, return_indices=False, ceil_mode=False)


        self.norm_hop1 = norm_layer(dim)
        self.norm_hop2 = norm_layer(dim)
        self.mlp_hop = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, hops, CVA_input, edge_embedding):
        MSA = self.drop_path(self.cva_attn(x, CVA_input, edge_embedding))
        MSA = self.norm_hop1(hops) * MSA

        x = x + MSA
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        hops = hops + MSA
        hops = hops + self.drop_path(self.mlp_hop(self.norm_hop2(hops)))
        return x, hops, MSA


#######################################################################################################################
class First_view_Spatial_features(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))


        self.hop_to_embedding = nn.Linear(68, embed_dim_ratio)
        self.hop_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.block1 = Multi_Out_Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
                                      norm_layer=norm_layer)
        self.block2 = Multi_Out_Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
                                      norm_layer=norm_layer)
        self.block3 = Multi_Out_Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[2],
                                      norm_layer=norm_layer)
        self.block4 = Multi_Out_Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[3],
                                      norm_layer=norm_layer)

        self.Spatial_norm = norm_layer(embed_dim_ratio)

        self.hop_norm = norm_layer(embed_dim_ratio)

    def forward(self, x, hops, edge_embedding):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        
        hops = rearrange(hops, 'b c f p  -> (b f) p  c', )
        hops = self.hop_to_embedding(hops)
        hops += self.hop_pos_embed
        hops = self.pos_drop(hops)


        x, hops, MSA1 = self.block1(x, hops, edge_embedding)
        x, hops, MSA2 = self.block2(x, hops, edge_embedding)
        x, hops, MSA3 = self.block3(x, hops, edge_embedding)
        x, hops, MSA4 = self.block4(x, hops, edge_embedding)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)

        hops = self.hop_norm(hops)
        hops = rearrange(hops, '(b f) w c -> b f (w c)', f=f)

        return x, hops, MSA1, MSA2, MSA3, MSA4


#######################################################################################################################
class Spatial_features(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.hop_to_embedding = nn.Linear(68, embed_dim_ratio)
        self.hop_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.block1 = Multi_In_Out_Block(
            dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        self.block2 = Multi_In_Out_Block(
            dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer)
        self.block3 = Multi_In_Out_Block(
            dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[2], norm_layer=norm_layer)
        self.block4 = Multi_In_Out_Block(
            dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[3], norm_layer=norm_layer)

        self.Spatial_norm = norm_layer(embed_dim_ratio)

        self.hop_norm = norm_layer(embed_dim_ratio)

    def forward(self, x, hops, MSA1, MSA2, MSA3, MSA4, edge_embedding):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)


        hops = rearrange(hops, 'b c f p  -> (b f) p  c', )
        hops = self.hop_to_embedding(hops)
        hops += self.hop_pos_embed
        hops = self.pos_drop(hops)


        x, hops, MSA1 = self.block1(x, hops, MSA1, edge_embedding)
        x, hops, MSA2 = self.block2(x, hops, MSA2, edge_embedding)
        x, hops, MSA3 = self.block3(x, hops, MSA3, edge_embedding)
        x, hops, MSA4 = self.block4(x, hops, MSA4, edge_embedding)


        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)

        hops = self.hop_norm(hops)
        hops = rearrange(hops, '(b f) w c -> b f (w c)', f=f)

        return x, hops, MSA1, MSA2, MSA3, MSA4