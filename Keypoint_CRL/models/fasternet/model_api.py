import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytorch_lightning as pl
from timm.models import create_model


class LitModel(pl.LightningModule):
    def __init__(self, num_classes, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        if 'fasternet' in hparams.model_name:
            self.model = create_model(
                hparams.model_name,
                mlp_ratio=hparams.mlp_ratio,
                embed_dim=hparams.embed_dim,
                depths=hparams.depths,
                pretrained=hparams.pretrained,
                n_div=hparams.n_div,
                feature_dim=hparams.feature_dim,
                patch_size=hparams.patch_size,
                patch_stride=hparams.patch_stride,
                patch_size2=hparams.patch_size2,
                patch_stride2=hparams.patch_stride2,
                num_classes=num_classes,
                layer_scale_init_value=hparams.layer_scale_init_value,
                drop_path_rate=hparams.drop_path_rate,
                norm_layer=hparams.norm_layer,
                act_layer=hparams.act_layer,
                pconv_fw_type=hparams.pconv_fw_type
            )
        else:
            self.model = create_model(
                hparams.model_name,
                pretrained=hparams.pretrained,
                num_classes=num_classes
            )

    def forward(self, x):
        return self.model(x)
