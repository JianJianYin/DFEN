import torch
from torch import nn

from model.encoder import build_encoder
from model.decoder import build_decoder
from model.aspp import build_aspp
from .backbones.Imagelevel import ImageLevelContext
from .backbones.Semanticlevel import SemanticLevelContext

class SwinDeepLab(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = build_encoder(encoder_config) # 建立编码器 输出的低层次的特征图  输出的特征是384维度的
        ilc_cfg = {
            'feats_channels': self.encoder.high_level_dim, # 512
            'transform_channels': self.encoder.high_level_dim//2, # 256
            'concat_input': True, # True
            'align_corners': False, # False
        }
        self.ilc_net = ImageLevelContext(**ilc_cfg)

        slc_cfg = {
            'feats_channels': self.encoder.high_level_dim,#512
            'transform_channels': self.encoder.high_level_dim//2,#256
            'concat_input': True, #True
            'output_dim': self.encoder.low_level_dim,
        }
        self.slc_net = SemanticLevelContext(**slc_cfg)

        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(self.encoder.high_level_dim, self.encoder.high_level_dim, kernel_size=1, stride=1, padding=0, bias=False),
            #BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            #BuildActivation(act_cfg),
            nn.BatchNorm2d(self.encoder.high_level_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.encoder.high_level_dim, decoder_config.num_classes, kernel_size=1, stride=1, padding=0)
        )

        # self.aspp = build_aspp(input_size=self.encoder.high_level_size,
        #                        input_dim=self.encoder.high_level_dim,
        #                        out_dim=self.encoder.low_level_dim, config=aspp_config)
        self.decoder = build_decoder(input_size=self.encoder.high_level_size,
                                     input_high_dim = self.encoder.high_level_dim,  # 输入的维度也应该是384维度
                                     input_dim=self.encoder.low_level_dim,
                                     config=decoder_config)

    def run_encoder(self, x):
        low_level,middle_level, high_level = self.encoder(x)
        return low_level,middle_level, high_level
    
    def run_aspp(self, x):
        return self.aspp(x)

    def run_decoder(self, low_level,middle_level,high_level, aspp):
        return self.decoder(low_level,middle_level,high_level, aspp)

    def run_upsample(self, x):
        return self.upsample(x)

    def forward(self, x):
        low_level, middle_level, high_level = self.run_encoder(x) # 14 × 14 × 384
        feat_il = self.ilc_net(high_level.permute(0,3,1,2))
        pred = self.decoder_stage1(high_level.permute(0,3,1,2))
        feat_sl = self.slc_net(high_level.permute(0,3,1,2),pred,feat_il)

        x = self.run_decoder(low_level,middle_level,high_level, aspp = feat_sl.permute(0,2,3,1)) # x:  
        
        return x
