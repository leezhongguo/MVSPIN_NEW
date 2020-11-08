import torch
import torch.nn as nn
import numpy as np 
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2
from utils import BaseTrainer

from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation

from datasets import MixedDataset
from datasets import BaseDataset
from models import hmr, SMPL
from smplify import SMPLify 
from utils.renderer import Renderer
from .fits_dict import FitsDict

import config
import constants
import os
class Trainer_li(BaseTrainer):
    def init_fn(self):
        #self.dataset = 'h36m'
        #self.train_ds = BaseDataset(self.options, self.dataset)   # training dataset
        self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)
        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)      # feature extraction model
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                            lr=5e-5,
                                            weight_decay=0)
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size = 16,
                         create_transl=False).to(self.device)
        # per vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # keypoints loss including 2D and 3D
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # SMPL parameters loss if we have
        self.criterion_regr = nn.MSELoss().to(self.device)

        self.models_dict = {'model':self.model}
        self.optimizers_dict = {'optimizer':self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH
        # initialize SMPLify
        self.smplify = SMPLify(step_size=1e-2, batch_size=16, num_iters=100,focal_length=self.focal_length)
        print(self.options.pretrained_checkpoint)
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file = self.options.pretrained_checkpoint)
        #load dictionary of fits
        self.fits_dict = FitsDict(self.options, self.train_ds)
        # create renderer
        self.renderer = Renderer(focal_length=self.focal_length, img_res = 224, faces=self.smpl.faces)

    def finalize(self):
        self.fits_dict.save()
    
    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        """
        conf = gt_keypoints_2d[:,:,-1].unsqueeze(-1).clone()
        conf[:,:25] *= openpose_weight
        conf[:,25:] *= gt_weight
        loss = (conf*self.criterion_keypoints(pred_keypoints_2d,gt_keypoints_2d[:,:,:-1])).mean()
        return loss
    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        pred_keypoints_3d = pred_keypoints_3d[:,25:,:]
        conf = gt_keypoints_3d[:,:,-1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:,:,:-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d==1]
        conf = conf[has_pose_3d==1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d==1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)
    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
        #print(pred_rotmat_valid.size(),gt_rotmat_valid.size())
        #input()
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas
    
    def train_step(self, input_batch):
        self.model.train()
        # get data from batch
        has_smpl = input_batch['has_smpl'].bool()
        has_pose_3d = input_batch['has_pose_3d'].bool()
        gt_pose = input_batch['pose'] # SMPL pose parameters
        gt_betas = input_batch['betas'] # SMPL beta parameters
        dataset_name = input_batch['dataset_name']
        indices = input_batch['sample_index'] # index of example inside its dataset
        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        #print(rot_angle)
        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices
        # Get current best fits from the dictionary
        opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)
        opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:,3:], global_orient=opt_pose[:,:3])
        opt_vertices = opt_output.vertices
        opt_joints = opt_output.joints
        
        pred_keypoints_2d_mv, pred_joints_mv = [], [] # 2d and 3d joint points
        pred_pose_mv, pred_betas_mv, pred_camera_mv,pred_cam_t_mv,pred_vertices_mv,pred_rotmat_mv = [], [], [], [],[], [] # SMPL parameters
        gt_keypoints_2d_orig_mv = []
        opt_cam_t_mv,opt_cam_rot_mv = [],[]
        opt_pose_mv, opt_vertices_mv = [], []
        for i in range(4):
            # images
            images = input_batch['img_%d' % i] # for the i-th view
            #np.save('img_%d' % i, images.detach().cpu())
            batch_size = images.shape[0]
            #import matplotlib.pyplot as plt
            #img = images[0][0].cpu()
            #plt.imshow(img)
            #plt.show()
            # Feed images in the network to predict camera and SMPL parameters
            pred_rotmat, pred_betas, pred_camera = self.model(images)
            pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            #print(pred_rotmat.size())
            #input()
            pred_vertices = pred_output.vertices  # predicted SMPL model vertices
            pred_joints = pred_output.joints  # predicted 3D joint points
            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t = torch.stack([pred_camera[:,1],
                                      pred_camera[:,2],
                                      2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)
            
            camera_center = torch.zeros(batch_size, 2, device=self.device)
            #camera_center = 0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device)
            #predicted 2d joint points
            pred_keypoints_2d = perspective_projection(pred_joints,
                                                        rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                        translation=pred_cam_t,
                                                        focal_length=self.focal_length,
                                                        camera_center=camera_center)
            
            #np.save('pred_keypoints_%d' % i, pred_keypoints_2d.detach().cpu())
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)
            
            # 2d joint points 
            gt_keypoints_2d = input_batch['keypoints_%d' % i]
            gt_keypoints_2d_orig = gt_keypoints_2d.clone()
            gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)    
            #np.save('gt_keypoints_%d' % i, gt_keypoints_2d_orig.detach().cpu())
            gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)
            opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)
            opt_cam_rot = torch.zeros(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
            # add to list
            pred_keypoints_2d_mv.append(pred_keypoints_2d)
            pred_joints_mv.append(pred_joints)
            pred_rotmat_mv.append(pred_rotmat)
            #pred_pose_mv.append(pred_pose)
            pred_betas_mv.append(pred_betas)
            pred_cam_t_mv.append(pred_cam_t)
            pred_camera_mv.append(pred_camera)
            pred_vertices_mv.append(pred_vertices)
            a  = opt_vertices.clone()
            opt_vertices_mv.append(a)
            b = opt_pose.clone()
            opt_pose_mv.append(b)
            gt_keypoints_2d_orig_mv.append(gt_keypoints_2d_orig)
            opt_cam_t_mv.append(opt_cam_t)
            #opt_cam_rot_mv.append(opt_cam_rot)
        #input()
        opt_joint_loss = self.smplify.get_fitting_loss(opt_pose, opt_betas, opt_cam_t_mv,
                                                       0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device),
                                                       gt_keypoints_2d_orig_mv).mean(dim=-1)
        # After get the pose, theta, camera of four images, we run the smplify on the four images
        
        if self.options.run_smplify:
            # convert predicted rotation matrix to axis-angle
            pred_pose_mv,pred_betas_detach_mv,pred_cam_t_detach_mv = [],[],[]
            for i in range(4):
                pred_rotmat_hom = torch.cat([pred_rotmat_mv[i].detach().view(-1, 3, 3).detach(), torch.tensor([0,0,1], dtype=torch.float32,
                                            device=self.device).view(1, 3, 1).expand(batch_size * 24, -1, -1)], dim=-1)
                pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch_size, -1)
                pred_pose[torch.isnan(pred_pose)] = 0.0
                #print(pred_pose.size())
                #input()
                pred_pose_mv.append(pred_pose.detach())
                pred_betas_detach_mv.append(pred_betas_mv[i].detach())
                pred_cam_t_detach_mv.append(pred_cam_t_mv[i].detach())
                #np.save('pred_pose_%d' % i, pred_pose.detach().cpu())
                #np.save('pred_beta_%d' % i, pred_betas_mv[i].detach().cpu())
                #np.save('pred_cam_t_%d' % i, pred_cam_t_mv[i].detach().cpu())
            # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
            #pred_pose[torch.isnan(pred_pose)] = 0.0
            # Run SMPLify optimization starting from the network prediction
            new_opt_vertices, new_opt_joints,\
            new_opt_pose, new_opt_betas,\
            new_opt_cam_t, new_opt_joint_loss = self.smplify(
                                        pred_pose_mv, pred_betas_detach_mv,
                                        pred_cam_t_detach_mv,
                                        0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device),
                                        gt_keypoints_2d_orig_mv)
            #print(new_opt_cam_rot)
            #input()
            new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)
            # Will update the dictionary for the examples where the new loss is less than the current one
            update = (new_opt_joint_loss < opt_joint_loss)
            opt_joint_loss[update] = new_opt_joint_loss[update]
            opt_joints[update, :] = new_opt_joints[update, :]
            #print(opt_pose.size(),new_opt_pose.size())
            opt_betas[update, :] = new_opt_betas[update, :]
            for i in range(4):
                opt_pose_mv[i][update, :] = new_opt_pose[i][update, :]
                #print(i, opt_pose_mv[i])
                opt_vertices_mv[i][update, :] = new_opt_vertices[i][update, :]
                opt_cam_t_mv[i][update, :] = new_opt_cam_t[i][update, :]
                #opt_cam_rot_mv[i][update, :] = new_opt_cam_rot[i][update, :]
            #input()
            self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu(), update.cpu())] = (opt_pose_mv[0].cpu(), opt_betas.cpu())
        # now we comput the loss on the four images
        # Replace the optimized parameters with the ground truth parameters, if available
        #for i in range(4):
            #print('Here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
        #    opt_vertices_mv[i][has_smpl, :, :] = gt_vertices[has_smpl, :, :]
        #    opt_pose_mv[i][has_smpl, :] = gt_pose[has_smpl, :]
        #opt_cam_t[has_smpl, :] = gt_cam_t[has_smpl, :]
        #opt_joints[has_smpl, :, :] = gt_model_joints[has_smpl, :, :]
        #opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

        # Assert whether a fit is valid by comparing the joint loss with the threshold
        valid_fit = (opt_joint_loss < self.options.smplify_threshold).to(self.device)
        # Add the examples with GT parameters to the list of valid fits
        valid_fit = valid_fit | has_smpl
        
        # Compute 3D keypoint loss
        
        loss_keypoints_mv, loss_regr_pose_mv, loss_keypoints_3d_mv, loss_regr_betas_mv = [],[], [], []
        loss_shape_mv = []
        for i in range(4):
            # Compute 2D reprojection loss for the keypoints
            gt_keypoints_2d = input_batch['keypoints_%d' % i]
            #print(pred_keypoints_2d_mv[i].size(),gt_keypoints_2d.size())
            loss_keypoints = self.keypoint_loss(pred_keypoints_2d_mv[i], gt_keypoints_2d,0,1)
            loss_keypoints_mv.append(loss_keypoints)
            gt_joints = input_batch['pose_3d_%d' % i] # 3D pose
            loss_keypoints_3d = self.keypoint_3d_loss(pred_joints_mv[i], gt_joints, has_pose_3d)
            loss_keypoints_3d_mv.append(loss_keypoints_3d)
            # Compute loss on SMPL parameters
            loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat_mv[i], pred_betas_mv[i], opt_pose_mv[i], opt_betas, valid_fit)
            loss_regr_pose_mv.append(loss_regr_pose)
            loss_regr_betas_mv.append(loss_regr_betas)
            # Per-vertex loss for the shape
            loss_shape = self.shape_loss(pred_vertices_mv[i], opt_vertices_mv[i], valid_fit)
            loss_shape_mv.append(loss_shape)
        loss_keypoints_sum = (loss_keypoints_mv[0]+loss_keypoints_mv[1]+loss_keypoints_mv[2]+loss_keypoints_mv[3])/4.
        loss_keypoints_3d_sum =  (loss_keypoints_3d_mv[0]+loss_keypoints_3d_mv[1]+loss_keypoints_3d_mv[2]+loss_keypoints_3d_mv[3])/4.
        loss_regr_pose_sum = (loss_regr_pose_mv[0]+loss_regr_pose_mv[1]+loss_regr_pose_mv[2]+loss_regr_pose_mv[3])/4.
        loss_regr_betas_sum = (loss_regr_betas_mv[0]+loss_regr_betas_mv[1]+loss_regr_betas_mv[2]+loss_regr_betas_mv[3])/4.
        loss_shape_sum = (loss_shape_mv[0]+loss_shape_mv[1]+loss_shape_mv[2]+loss_shape_mv[3])/4.
        #print(loss_shape_sum,loss_keypoints_sum,loss_keypoints_3d_sum,loss_regr_pose_sum,loss_regr_betas_sum)
        #input()
        loss_all = 0 * loss_shape_sum +\
                   50. * loss_keypoints_sum +\
                   50. * loss_keypoints_3d_sum +\
                   loss_regr_pose_sum + 0.001* loss_regr_betas_sum +\
                   ((torch.exp(-pred_camera_mv[0][:,0]*10)) ** 2 ).mean()
        loss_all *= 100
        print(loss_all)
        
        # Do backprop
        self.optimizer.zero_grad()
        loss_all.backward()
        self.optimizer.step()
        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': pred_vertices_mv,
                  'opt_vertices': opt_vertices_mv,
                  'pred_cam_t': pred_cam_t_mv,
                  'opt_cam_t': opt_cam_t_mv
                  }
        #losses = {'loss': loss_all.detach().item(),
        #          'loss_keypoints': loss_keypoints_sum.detach().item(),
        #          'loss_regr_pose': loss_regr_pose_sum.detach().item(),
        #          'loss_regr_betas': loss_regr_betas_sum.detach().item(),
        #          'loss_shape': loss_shape_sum.detach().item()}
        losses = {'loss': loss_all.detach().item(),
                  'loss_keypoints': loss_keypoints_sum.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d_sum.detach().item(),
                  'loss_regr_pose': loss_regr_pose_sum.detach().item(),
                  'loss_regr_betas': loss_regr_betas_sum.detach().item(),
                  'loss_shape': loss_shape_sum.detach().item()}
        #print(opt_cam_rot_mv,opt_cam_t_mv)
        #np.save('opt_vertices.npy',opt_vertices.detach().cpu())
        #np.save('pred_vertices.npy',pred_vertices.detach().cpu())
        #np.save('opt_beta.npy', opt_betas.detach().cpu())
        #for i in range(4):
        #    np.save('opt_vertices_%d.npy' % i,opt_vertices_mv[i].detach().cpu())
        #    np.save('opt_pose_%d.npy' % i, opt_pose_mv[i].detach().cpu())
            #np.save('opt_rot_%d.npy' % i,opt_cam_rot_mv[i].detach().cpu())
        #    np.save('opt_cam_t_%d.npy' % i,opt_cam_t_mv[i].detach().cpu())
        return output, losses  

    def train_summaries(self, input_batch, output, losses):
        pred_vertices = output['pred_vertices']
        opt_vertices = output['opt_vertices']
        #opt_vertices = [opt_vertices_single,opt_vertices_single,opt_vertices_single,opt_vertices_single]
        images = input_batch['img_%d' % 0]
        #images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        #images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        pred_cam_t = output['pred_cam_t']
        #rot = torch.eye(3, device=self.device).unsqueeze(0).expand(images.shape[0], -1, -1)
        #pred_cam_rot = [rot,rot,rot,rot]
        opt_cam_t = output['opt_cam_t']
        #opt_cam_rot = output['opt_cam_rot']
        #pred_cam_t[:,0] *= -1 
        #print(opt_cam_t,opt_cam_rot)
        #input()
        #images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t,pred_cam_rot, images)
        #images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, opt_cam_rot, images)
        images_pred = self.renderer.visualize_tb(pred_vertices,pred_cam_t, input_batch)
        images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, input_batch)
        
        self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
        self.summary_writer.add_image('opt_shape', images_opt, self.step_count)
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
