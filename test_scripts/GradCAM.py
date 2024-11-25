import cv2
import sys
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

sys.path.append("..")
from Keypoint_CRL.models.seg_unet import SegUNet


def imRead(ori_img, image_size):
    ori_img = cv2.imdecode(np.fromfile(ori_img, dtype=np.uint8), cv2.IMREAD_COLOR)
    if ori_img.shape[-1] == 4:
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGRA2BGR)
    elif len(ori_img.shape) == 2:
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
    ori_h, ori_w, _ = ori_img.shape
    r = min(image_size[0] / ori_h, image_size[1] / ori_w)
    resize_w, resize_h = int(round(ori_w * r)), int(round(ori_h * r))
    img = cv2.resize(ori_img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    pad_width = image_size[1] - resize_w
    pad_height = image_size[0] - resize_h
    dw = pad_width / 2
    dh = pad_height / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_np = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img_np = img_np.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    return img_np, img_tensor, [top, bottom, left, right]


if __name__ == '__main__':
    model = SegUNet(num_class=3)
    state_dict = torch.load(r'')
    model.load_state_dict(state_dict, strict=True)
    model = model.eval()

    import os
    im_root = r''
    for root, dirs, files in os.walk(im_root):
        for file in files:
            if file.split('.')[-1].lower() in ['jpg', 'bmp', 'png', 'tif', 'jpeg']:
                imNp, imTensor, imPad = imRead(os.path.join(root, file), (480, 640))
                target_layers1 = [model.decoder.InfoSimplify31.group1]
                targets1 = [RawScoresOutputTarget()]
                with GradCAM(model=model, target_layers=target_layers1) as cam:
                    grayscale_cam = cam(input_tensor=imTensor, targets=targets1, eigen_smooth=True)
                    grayscale_cam = grayscale_cam[0, :]
                    visualization = show_cam_on_image(imNp, grayscale_cam, use_rgb=False)
                cv2.imwrite(os.path.join(r'E:\NTCRL\vis', file.split('.')[0] + '_11.jpg'), visualization)

                target_layers2 = [model.decoder.InfoSimplify31.group2]
                targets2 = [RawScoresOutputTarget()]
                with GradCAM(model=model, target_layers=target_layers2) as cam:
                    grayscale_cam = cam(input_tensor=imTensor, targets=targets2, eigen_smooth=True)
                    grayscale_cam = grayscale_cam[0, :]
                    visualization = show_cam_on_image(imNp, grayscale_cam, use_rgb=False)
                cv2.imwrite(os.path.join(r'E:\NTCRL\vis', file.split('.')[0] + '_12.jpg'), visualization)
