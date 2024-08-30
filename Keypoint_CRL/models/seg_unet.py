if __name__ == "__main__" and (__package__ is None or __package__ == ''):
    import os
    import sys
    project_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, project_dir)
    # import utility.sub_region_detector  # noqa: F401
    __package__ = "Keypoint_CRL.models"
from .modules import *


class SegUNet(nn.Module):
    def __init__(self, num_class):
        super(SegUNet, self).__init__()
        self.num_class = num_class
        self.feature_extraction = FeaExtra()
        self.decoder = Decoder(inchannels=[768, 384, 192, 96], num_class=num_class)
        self.pred_head = nn.Sequential(
            nn.Conv2d(96, num_class, 1, 1, 0, groups=num_class)
        )
        self.x_layer = nn.Sequential(
            nn.Conv2d(96, num_class, 1, 1, 0, groups=num_class)
        )
        self.y_layer = nn.Sequential(
            nn.Conv2d(96, num_class, 1, 1, 0, groups=num_class)
        )
        for module in [self.decoder, self.pred_head, self.x_layer, self.y_layer]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                    nn.init.normal_(m.weight, mean=0.0, std=1.0)
                    nn.init.constant_(m.bias, val=0.0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        feas = self.feature_extraction(x)
        decoded = self.decoder(feas)
        pred = self.pred_head(decoded)
        offset_x = self.x_layer(decoded)
        offset_y = self.y_layer(decoded)
        return pred, offset_x, offset_y


if __name__ == '__main__':
    # model = SegUNet(num_class=3).cuda()
    # # print(model)
    # import time
    # start = time.time()
    # for i in range(1000):
    #     x = torch.rand([1, 3, 480, 640]).cuda()
    #     y = model(x)
    # end = time.time()
    # print('inference time: ', end - start)

    model = SegUNet(num_class=3)
    from thop import profile

    input1 = torch.randn(1, 3, 480, 640)
    flops, params = profile(model, inputs=(input1,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    total = sum([param.nelement() for param in model.feature_extraction.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
