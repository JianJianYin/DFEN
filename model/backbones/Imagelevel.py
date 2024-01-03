'''
Function:
    Implementation of ImageLevelContext
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .selfattention import SelfAttentionBlock


'''ImageLevelContext'''
class ImageLevelContext(nn.Module):
    def __init__(self, feats_channels, transform_channels, concat_input=False, align_corners=False):
        super(ImageLevelContext, self).__init__()
        self.align_corners = align_corners
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.correlate_net = SelfAttentionBlock(
            key_in_channels=feats_channels * 2,
            query_in_channels=feats_channels,
            transform_channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
        )
        if concat_input:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(feats_channels),
                nn.ReLU(inplace=True),
                #BuildNormalization(placeholder=feats_channels, norm_cfg=norm_cfg),
                #BuildActivation(act_cfg),
            )
    '''forward'''
    def forward(self, x):
        # x_global = self.global_avgpool(x)
        b,c,h,w = x.size()
        x = x.view(b,c,(h * w))
        x = x.flatten(2)
        template = x
        k = torch.softmax(x,dim=2)
        x_global = torch.mul(k, template)
        x_global = torch.mean(x_global,dim=2)
        x_global = x_global.unsqueeze(dim=3)
        x_global = F.interpolate(x_global, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners)
        feats_il = self.correlate_net(x, torch.cat([x_global, x], dim=1))
        if hasattr(self, 'bottleneck'):
            feats_il = self.bottleneck(torch.cat([x, feats_il], dim=1))
        return feats_il
    