## Our model was revised from https://github.com/zczcwh/PoseFormer/blob/main/common/model_poseformer.py

import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath

from common.opt import opts

from model.Spatial_encoder import First_view_Spatial_features, Spatial_features
from model.Temporal_encoder import Temporal__features

opt = opts().parse()
device = torch.device("cuda")



#######################################################################################################################
class sgraformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        embed_dim = embed_dim_ratio * num_joints
        out_dim = num_joints * 3  #### output dimension is num_joints * 3
        ##Spatial_features
        self.SF1 = First_view_Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                               num_heads, mlp_ratio, qkv_bias, qk_scale,
                                               drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF2 = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                    num_heads, mlp_ratio, qkv_bias, qk_scale,
                                    drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF3 = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                    num_heads, mlp_ratio, qkv_bias, qk_scale,
                                    drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF4 = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                    num_heads, mlp_ratio, qkv_bias, qk_scale,
                                    drop_rate, attn_drop_rate, drop_path_rate, norm_layer)

        ## MVF
        self.view_pos_embed = nn.Parameter(torch.zeros(1, 4, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.1),
            nn.Conv2d(4, 1, kernel_size=opt.mvf_kernel, stride=1, padding=int(opt.mvf_kernel // 2), bias=False),
            nn.ReLU(inplace=True),
        )


        self.conv_hop = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.1),
            nn.Conv2d(4, 1, kernel_size=opt.mvf_kernel, stride=1, padding=int(opt.mvf_kernel // 2), bias=False),
            nn.ReLU(inplace=True),
        )

        self.conv_norm = nn.LayerNorm(embed_dim)

        self.conv_hop_norm = nn.LayerNorm(embed_dim)


        # Time Serial
        self.TF = TemTemporal__features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                        num_heads, mlp_ratio, qkv_bias, qk_scale,
                                        drop_rate, attn_drop_rate, drop_path_rate, norm_layer)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self.hop_w0 = nn.Parameter(torch.ones(17, 17))
        self.hop_w1 = nn.Parameter(torch.ones(17, 17))
        self.hop_w2 = nn.Parameter(torch.ones(17, 17))
        self.hop_w3 = nn.Parameter(torch.ones(17, 17))
        self.hop_w4 = nn.Parameter(torch.ones(17, 17))

        self.hop_global = nn.Parameter(torch.ones(17, 17))

        self.linear_hop = nn.Linear(8, 2)
        # self.max_pool = nn.MaxPool1d(2)

        self.edge_embedding = nn.Linear(17*17*4, 17*17)

    def forward(self, x, hops):
        b, f, v, j, c = x.shape

        edge_embedding = self.edge_embedding(hops[0].reshape(1, -1))

        ###############golbal feature#################
        x_hop_global = x.unsqueeze(3).repeat(1, 1, 1, 17, 1, 1)
        x_hop_global = x_hop_global - x_hop_global.permute(0, 1, 2, 4, 3, 5)
        x_hop_global = torch.sum(x_hop_global ** 2, dim=-1)
        hop_global = x_hop_global / torch.sum(x_hop_global, dim=-1).unsqueeze(-1)
        hops = hops.unsqueeze(1).unsqueeze(2).repeat(1, f, v, 1, 1, 1)
        hops1 = hop_global * hops[:, :, :, 0]
        hops2 = hop_global * hops[:, :, :, 1]
        hops3 = hop_global * hops[:, :, :, 2]
        hops4 = hop_global * hops[:, :, :, 3]
        # hops = torch.cat((hops1,hops2,hops3,hops4), dim=-1)
        hops = torch.cat((hops1,hops2,hops3,hops4), dim=-1)

        x1 = x[:, :, 0]
        x2 = x[:, :, 1]
        x3 = x[:, :, 2]
        x4 = x[:, :, 3]

        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)
        x4 = x4.permute(0, 3, 1, 2)

        hop1 = hops[:, :, 0]
        hop2 = hops[:, :, 1]
        hop3 = hops[:, :, 2]
        hop4 = hops[:, :, 3]

        hop1 = hop1.permute(0, 3, 1, 2)
        hop2 = hop2.permute(0, 3, 1, 2)
        hop3 = hop3.permute(0, 3, 1, 2)
        hop4 = hop4.permute(0, 3, 1, 2)

        ### Semantic graph transformer encoder
        x1, hop1, MSA1, MSA2, MSA3, MSA4 = self.SF1(x1, hop1, edge_embedding)
        x2, hop2, MSA1, MSA2, MSA3, MSA4 = self.SF2(x2, hop2, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        x3, hop3, MSA1, MSA2, MSA3, MSA4 = self.SF3(x3, hop3, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        x4, hop4, MSA1, MSA2, MSA3, MSA4 = self.SF4(x4, hop4, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        
        ### Multi-view cross-channel fusion
        x = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)), dim=1) + self.view_pos_embed
        x = self.pos_drop(x)
        x = self.conv(x).squeeze(1) + x1 + x2 + x3 + x4
        x = self.conv_norm(x)

        hop = torch.cat((hop1.unsqueeze(1), hop2.unsqueeze(1), hop3.unsqueeze(1), hop4.unsqueeze(1)), dim=1) + self.view_pos_embed
        hop = self.pos_drop(hop)
        # hop = self.conv_hop(hop).squeeze(1) + hop1 + hop2 + hop3 + hop4
        # hop = self.conv_hop_norm(hop)
        hop = self.conv(hop).squeeze(1) + hop1 + hop2 + hop3 + hop4
        hop = self.conv_norm(hop)

        x = x * hop


        ### Temporal transformer encoder
        x = self.TF(x)
        
        x = self.head(x)
        x = x.view(b, opt.frames, j, -1)
        return x


# x = torch.rand((8, 27, 4, 17 , 2))
# hops = torch.rand((8,4,17,17))
# mvft = hmvformer(num_frame=opt.frames, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
#                             num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
# print(mvft(x, hops).shape)