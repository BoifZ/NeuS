import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from models.poses import LearnPose, LearnIntrin, RaysGenerator


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', img_dir='image', npz_postfix='d', is_continue=False, latest_ckpt=None):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        conf_text = conf_text.replace('IMG_DIR', img_dir)
        conf_text = conf_text.replace('TYPE', npz_postfix)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.conf['dataset.img_dir'] = self.conf['dataset.img_dir'].replace('IMG_DIR', img_dir)
        self.conf['dataset.render_cameras_name'] = self.conf['dataset.render_cameras_name'].replace('TYPE', npz_postfix)
        self.conf['dataset.object_cameras_name'] = self.conf['dataset.object_cameras_name'].replace('TYPE', npz_postfix)
        self.conf['general.base_exp_dir'] = self.conf['general.base_exp_dir'].replace('TYPE', npz_postfix)
        # if img_dir != 'image':
        self.base_exp_dir = self.conf['general.base_exp_dir'] + img_dir.split('image')[-1]
        print(self.base_exp_dir)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0
        self.poses_iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_int('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_int('train.anneal_end', default=0.0)

        params_to_train = []
        self.learnable = self.conf.get_bool('train.focal_learnable')
        if self.learnable:
            if not os.path.exists(os.path.join(self.base_exp_dir, 'cam_poses')):
                os.makedirs(os.path.join(self.base_exp_dir, 'cam_poses'), exist_ok=True)
            self.savepose_freq = self.conf.get_int('train.savepose_freq')
            self.init_focal = self.conf.get_bool('train.init_focal')
            self.init_poses = self.conf.get_bool('train.init_poses')
            self.focal_lr = self.conf.get_float('train.focal_lr')
            self.pose_lr = self.conf.get_float('train.focal_lr')
            self.focal_lr_gamma = self.conf.get_float('train.focal_lr_gamma')
            self.pose_lr_gamma = self.conf.get_float('train.focal_lr_gamma')
            self.step_size = self.conf.get_int('train.step_size')

            self.start_refine_pose_iter = self.conf.get_int('train.start_refine_pose_iter')
            self.start_refine_focal_iter = self.conf.get_int('train.start_refine_focal_iter')

            init_focal = self.dataset.focal if self.init_focal else None
            init_poses = self.dataset.pose_all if self.init_poses else None
            # learn focal parameter
            self.intrin_net = LearnIntrin(self.dataset.H, self.dataset.W, **self.conf['model.focal'], init_focal=init_focal).to(self.device)
            # learn pose for each image
            self.pose_param_net = LearnPose(self.dataset.n_images, **self.conf['model.pose'], init_c2w=init_poses).to(self.device)
            # params_to_train += list(self.intrin_net.parameters())
            # params_to_train += list(self.pose_param_net.parameters())

            self.optimizer_focal = torch.optim.Adam(self.intrin_net.parameters(), lr=self.focal_lr)
            self.optimizer_pose = torch.optim.Adam(self.pose_param_net.parameters(), lr=self.pose_lr)

            self.scheduler_focal = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_focal, milestones=(self.warm_up_end, self.end_iter, self.step_size),
                                                                gamma=self.focal_lr_gamma)
            self.scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_pose, milestones=range(self.warm_up_end, self.end_iter, self.step_size),
                                                                gamma=self.pose_lr_gamma)
        else:
            self.intrin_net = self.dataset.intrinsics_all
            self.pose_param_net = self.dataset.pose_all
  
        self.rays_generator = RaysGenerator(self.dataset.images_lis, self.dataset.masks_lis, self.pose_param_net, self.intrin_net, learnable=self.learnable)
  
        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if latest_ckpt is not None and (latest_ckpt[:-4]=='.pth' and  \
                latest_ckpt in os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))):
            is_continue = False
            latest_model_name = latest_ckpt

        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        
        if self.learnable:
            if self.poses_iter_step >= self.start_refine_pose_iter:
                self.pose_param_net.train()
            else:
                self.pose_param_net.eval()

            if self.poses_iter_step >= self.start_refine_focal_iter:
                self.intrin_net.train()
            else:
                self.intrin_net.eval()
        
        for iter_i in tqdm(range(res_step)):
            if self.learnable:
                if self.poses_iter_step == self.start_refine_pose_iter:
                    self.pose_param_net.train()
                if self.poses_iter_step == self.start_refine_focal_iter:
                    self.intrin_net.train()

            img_idx = image_perm[self.iter_step % len(image_perm)]
            # data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)
            data = self.rays_generator.gen_random_rays_at(img_idx, self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            if self.learnable:
                self.optimizer_focal.zero_grad()
                self.optimizer_pose.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.learnable:
                self.optimizer_focal.step()
                self.optimizer_pose.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            # pose_history_milestone = list(range(0, 100, 5)) + list(range(100, 1000, 100)) + list(range(1000, 10000, 1000))
            # if self.poses_iter_step in pose_history_milestone:
            #     self.save_pnf_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                res = 512 if self.iter_step % 50000==0 else 64
                self.validate_mesh(resolution=res)
            
            if self.learnable and self.iter_step % self.savepose_freq == 0:
                self.store_current_pose()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

        if self.learnable:
            self.scheduler_focal.step()
            self.scheduler_pose.step()

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        if self.learnable:
            self.load_pnf_checkpoint(checkpoint_name.replace('ckpt', 'pnf'))
        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
        if self.learnable:
            self.save_pnf_checkpoint()

    def load_pnf_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'pnf_checkpoints', checkpoint_name), map_location=self.device)
        self.intrin_net.load_state_dict(checkpoint['intrin_net'])
        self.pose_param_net.load_state_dict(checkpoint['pose_param_net'])
        self.optimizer_focal.load_state_dict(checkpodnt['optimizer_pose'])
        self.poses_iter_step = checkpoint['poses_iter_step']

    def save_pnf_checkpoint(self):
        pnf_checkpoint = {
            'intrin_net': self.intrin_net.state_dict(),
            'pose_param_net': self.pose_param_net.state_dict(),
            'optimizer_focal': self.optimizer_focal.state_dict(),
            'optimizer_pose': self.optimizer_pose.state_dict(),
            'poses_iter_step': self.poses_iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'pnf_checkpoints'), exist_ok=True)
        torch.save(pnf_checkpoint, os.path.join(self.base_exp_dir, 'pnf_checkpoints', 'pnf_{:0>6d}.pth'.format(self.iter_step)))

    def store_current_pose(self):
        self.pose_param_net.eval()
        num_cams = self.pose_param_net.module.num_cams if isinstance(self.pose_param_net, torch.nn.DataParallel) else self.pose_param_net.num_cams

        c2w_list = []
        for i in range(num_cams):
            c2w = self.pose_param_net(i)  # (4, 4)
            c2w_list.append(c2w)
            
        c2w_list = torch.stack(c2w_list)  # (N, 4, 4)
        c2w_list = c2w_list.detach().cpu().numpy()
        np.save(os.path.join(self.base_exp_dir, 'cam_poses', 'pose_{:0>6d}.npy'.format(self.iter_step)), c2w_list)
        return

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        # rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        rays_o, rays_d = self.rays_generator.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            if self.learnable:
                rot = np.linalg.inv(self.pose_param_net(idx)[:3, :3].detach().cpu().numpy())
            else:
                rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        # rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        rays_o, rays_d = self.rays_generator.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--val_res', type=int, default=256)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--ckpt', type=str, nargs='+', default=None)
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('-c', '--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='', required=True)
    parser.add_argument('-d', '--img_dir', type=str, default='image')
    parser.add_argument('-psfx', '--npz_postfix', type=str, default='d')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, img_dir=args.img_dir, npz_postfix=args.npz_postfix, is_continue=args.is_continue, latest_ckpt=args.ckpt)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'gen_meshes':
        for k in args.ckpt:
            # model_name = os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_'+k+'.pth')
            model_name = 'ckpt_'+k+'.pth'
            runner.load_checkpoint(model_name)
            runner.validate_mesh(world_space=True, resolution=args.val_res, threshold=args.mcube_threshold)
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=args.val_res, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
