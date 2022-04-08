from genericpath import exists
import nntplib
import torch
import torch.nn as nn

from mmcv.runner import load_checkpoint
from depth_mit import mit_b4


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class GLPDepth(nn.Module):
    def __init__(self, max_depth=10.0, is_train=False, ckpt_path=None, n_layers=3):
        super().__init__()
        self.max_depth = max_depth

        self.encoder = mit_b4()
        
        channels_in = [512, 320, 128]
        channels_out = 64

        self.decoder = Decoder(channels_in, channels_out)
    
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):                
        conv1, conv2, conv3, conv4 = self.encoder(x)
        out = self.decoder(conv1, conv2, conv3, conv4)
        out_depth = self.last_layer_depth(out)
        out_depth = torch.sigmoid(out_depth)
        # out_depth = out_depth * self.max_depth

        return {'pred_d': out_depth, 'enc_feat': [conv1, conv2, conv3, conv4]}


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bot_conv = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        self.skip_conv1 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)
        self.fusion3 = SelectiveFeatureFusion(out_channels)

    def forward(self, x_1, x_2, x_3, x_4):
        x_4_ = self.bot_conv(x_4)
        out = self.up(x_4_)

        x_3_ = self.skip_conv1(x_3)
        out = self.fusion1(x_3_, out)
        out = self.up(out)

        x_2_ = self.skip_conv2(x_2)
        out = self.fusion2(x_2_, out)
        out = self.up(out)

        out = self.fusion3(x_1, out)
        out = self.up(out)
        out = self.up(out)

        return out


class SelectiveFeatureFusion(nn.Module):
    def __init__(self, in_channel=64):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel*2), out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=int(in_channel / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel / 2)),
            nn.ReLU())
        self.conv3 = nn.Conv2d(in_channels=int(in_channel / 2), out_channels=2, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_local, x_global):
        x = torch.cat((x_local, x_global), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        attn = self.sigmoid(x)

        out = x_local * attn[:, 0, :, :].unsqueeze(1) + \
              x_global * attn[:, 1, :, :].unsqueeze(1)

        return out


def check_and_make_dirs(paths):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

# kb cropping
def cropping(img, crop_size=(900, 1200)):
    h_im, w_im = img.shape[:2]

    margin_top = int(h_im - crop_size[0])
    margin_left = int((w_im - crop_size[1]) / 2)

    img = img[margin_top: margin_top + crop_size[0],
                margin_left: margin_left + crop_size[1]]
    return img


if __name__ == '__main__':
    import os
    import argparse
    import cv2 as cv
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from collections import OrderedDict
    import torchvision.transforms as transforms
    to_tensor = transforms.ToTensor()
    image_size=(896, 1184)

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='./data/CASE_NAME')
    parser.add_argument('--img_folder', type=str, default='preprocessed')
    parser.add_argument('--depth_ckpt', type=str, default='./ckpts/best_model_kitti.ckpt')
    parser.add_argument('--max_depth', type=float, default=10.0)
    
    args = parser.parse_args()
    device = torch.device('cuda')
    torch.cuda.set_device(args.gpu)

    case = args.case
    data_dir = args.data_dir.replace('CASE_NAME', case)
    img_folder = os.path.join(data_dir, args.img_folder)

    result_path_feat = os.path.join(img_folder, 'depth_feats')
    check_and_make_dirs(result_path_feat)
    print("Saving result feats in to %s" % result_path_feat)
    result_path_img = os.path.join(img_folder, 'depth_imgs')
    check_and_make_dirs(result_path_img)
    print("Saving result images in to %s" % result_path_img)

    images_lis = sorted(glob(os.path.join(img_folder, 'image/*.png')))
    n_images = len(images_lis)
    print('n_images:', n_images)
    
    # model = GLPDepth(max_depth=args.max_depth, is_train=False, n_layers=3).to(device)
    # model_weight = torch.load(args.depth_ckpt)
    # if 'module' in next(iter(model_weight.items()))[0]:
    #     model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    # model.load_state_dict(model_weight)
    # model.eval()

    # for im_name in tqdm(images_lis):
    #     image = cv.imread(im_name)
    #     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    #     image = cropping(image, crop_size=image_size)
    #     image = to_tensor(image).to(device).unsqueeze(0)

    #     with torch.no_grad():
    #         pred = model(image)
    #     # print(pred_d_numpy.shape)
    #     enc_feats = pred['enc_feat']
        
    #     im_name = os.path.basename(im_name)
    #     for idx, feat in enumerate(enc_feats):
    #         feat_numpy = torch.sigmoid(feat).squeeze().cpu().numpy()
    #         if not os.path.exists(os.path.join(result_path_feat, str(idx))):
    #             os.makedirs(os.path.join(result_path_feat, str(idx)), exist_ok=True)
    #         save_path = os.path.join(result_path_feat, str(idx), im_name[:-4]+'.npy')
    #         np.save(save_path, feat_numpy)

    #     pred_d = pred['pred_d']
    #     pred_d_numpy = pred_d.squeeze().cpu().numpy()
    #     depth_save_path = os.path.join(result_path_feat, im_name[:-4]+'.npy')
    #     np.save(depth_save_path, pred_d_numpy)

    #     pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
    #     pred_d_numpy = pred_d_numpy.astype(np.uint8)
    #     pred_d_color = cv.applyColorMap(pred_d_numpy, cv.COLORMAP_RAINBOW)
    #     im_save_path = os.path.join(result_path_img, im_name)
    #     cv.imwrite(im_save_path, pred_d_color)

    model = mit_b4().to(device)
    model_weight = torch.load(args.depth_ckpt)
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[15:], v) for k, v in model_weight.items() if 'encoder' in k)
    model.load_state_dict(model_weight)
    model.eval()
    resolution_level = 4

    for im_name in tqdm(images_lis):
        image = cv.imread(im_name)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cropping(image, crop_size=image_size)
        image = cv.resize(image, (image_size[1]//resolution_level, image_size[0]//resolution_level))
        image = to_tensor(image).to(device).unsqueeze(0)
        # print(image.shape)

        with torch.no_grad():
            pred = model(image)
        # print(pred_d_numpy.shape)
        enc_feats = pred
        
        im_name = os.path.basename(im_name)
        for idx, feat in enumerate(enc_feats):
            # feat_numpy = torch.sigmoid(feat).squeeze().cpu().numpy()
            feat_numpy = feat.squeeze().cpu().numpy()
            if feat_numpy.shape[-2:] != image.shape[-2:]:
                break
            # print(feat_numpy.shape)
            if not os.path.exists(os.path.join(result_path_feat, str(idx))):
                os.makedirs(os.path.join(result_path_feat, str(idx)), exist_ok=True)
            save_path = os.path.join(result_path_feat, str(idx), im_name[:-4]+'.npy')
            np.save(save_path, feat_numpy)