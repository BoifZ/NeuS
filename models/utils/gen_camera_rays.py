# import numpy as np
# import torch
# import torch.nn.functional as F



# def gen_rays(self, pose, intrinsic, resolution_level=1):
#     """
#     Generate rays at world space from one camera.
#     """
#     l = resolution_level
#     tx = torch.linspace(0, self.W - 1, self.W // l)
#     ty = torch.linspace(0, self.H - 1, self.H // l)
#     pixels_x, pixels_y = torch.meshgrid(tx, ty)
#     p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
#     p = torch.matmul(intrinsic[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
#     rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
#     rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
#     rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
#     return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


# def gen_random_rays(self, img_idx, pose, intrinsic, batch_size):
#     """
#     Generate random rays at world space from one camera.
#     """
#     pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
#     pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
#     color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
#     mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
#     p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
#     p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
#     rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
#     rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
#     rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
#     return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10


# def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
#         """
#         Interpolate pose between two cameras.
#         """
#         l = resolution_level
#         tx = torch.linspace(0, self.W - 1, self.W // l)
#         ty = torch.linspace(0, self.H - 1, self.H // l)
#         pixels_x, pixels_y = torch.meshgrid(tx, ty)
#         p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
#         p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
#         rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
#         trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
#         pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
#         pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
#         pose_0 = np.linalg.inv(pose_0)
#         pose_1 = np.linalg.inv(pose_1)
#         rot_0 = pose_0[:3, :3]
#         rot_1 = pose_1[:3, :3]
#         rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
#         key_times = [0, 1]
#         slerp = Slerp(key_times, rots)
#         rot = slerp(ratio)
#         pose = np.diag([1.0, 1.0, 1.0, 1.0])
#         pose = pose.astype(np.float32)
#         pose[:3, :3] = rot.as_matrix()
#         pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
#         pose = np.linalg.inv(pose)
#         rot = torch.from_numpy(pose[:3, :3]).cuda()
#         trans = torch.from_numpy(pose[:3, 3]).cuda()
#         rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
#         rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
#         return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


# def near_far_from_sphere(self, rays_o, rays_d):
#     a = torch.sum(rays_d**2, dim=-1, keepdim=True)
#     b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
#     mid = 0.5 * (-b) / a
#     near = mid - 1.0
#     far = mid + 1.0
#     return near, far


