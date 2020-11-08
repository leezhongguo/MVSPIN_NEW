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
from mvsmplify import MVSMPLify 
from utils.renderer import Renderer
from .fits_dict import FitsDict

import config
import constants
import os
class Trainer(BaseTrainer):
    def init_fn(self):
        self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)
        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)      # feature extraction model
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                            lr = self.options.lr,
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
        # initialize MVSMPLify
        self.mvsmplify = MVSMPLify(step_size=1e-2, batch_size=16, num_iters=100,focal_length=self.focal_length)
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
        # get data from the batch
        has_smpl = input_batch['has_smpl'].bool()
        has_pose_3d = input_batch['has_pose_3d'].bool()
        gt_pose1 = input_batch['pose'] # SMPL pose parameters
        gt_betas1 = input_batch['betas'] # SMPL beta parameters
        dataset_name = input_batch['dataset_name']
        indices = input_batch['sample_index'] # index of example inside its dataset
        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_betas = torch.cat((gt_betas1,gt_betas1,gt_betas1,gt_betas1),0)
        gt_pose = torch.cat((gt_pose1,gt_pose1,gt_pose1,gt_pose1),0)
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # Get current best fits from the dictionary, since we do not have the best fits dictionary, we set it randomly in the file
        opt_pose1, opt_betas1 = self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = torch.cat((opt_pose1.to(self.device),opt_pose1.to(self.device),opt_pose1.to(self.device),opt_pose1.to(self.device)),0)
        opt_betas = torch.cat((opt_betas1.to(self.device),opt_betas1.to(self.device),opt_betas1.to(self.device),opt_betas1.to(self.device)),0)
        opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:,3:], global_orient=opt_pose[:,:3])
        opt_vertices = opt_output.vertices
        opt_joints = opt_output.joints

        # images, form the four-view images as a small batch
        images = torch.cat((input_batch['img_0'],input_batch['img_1'],input_batch['img_2'],input_batch['img_3']),0)
        batch_size = input_batch['img_0'].shape[0] # here is the batch_size we set in the code

        # Output of CNN
        pred_rotmat, pred_betas, pred_camera = self.model(images)
        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints # 3D joints of SMPL model
        pred_cam_t = torch.stack([pred_camera[:,1],
                                pred_camera[:,2],
                                2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)
        camera_center = torch.zeros(batch_size*4, 2, device=self.device)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                        rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size*4, -1, -1),
                                                        translation=pred_cam_t,
                                                        focal_length=self.focal_length,
                                                        camera_center=camera_center)
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)
        # 2d joint points
        gt_keypoints_2d = torch.cat((input_batch['keypoints_0'],input_batch['keypoints_1'],input_batch['keypoints_2'],input_batch['keypoints_3']),0)
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)
        gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)
        opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)
        opt_joint_loss = self.mvsmplify.get_fitting_loss(opt_pose, opt_betas, opt_cam_t,
                                                       0.5 * self.options.img_res * torch.ones(batch_size*4, 2, device=self.device),
                                                       gt_keypoints_2d_orig).mean(dim=-1)
        # do mvsmplify
        if self.options.run_mvsmplify:
            pred_rotmat_hom = torch.cat([pred_rotmat.detach().view(-1, 3, 3).detach(), torch.tensor([0,0,1], dtype=torch.float32,
                                            device=self.device).view(1, 3, 1).expand(batch_size*4 * 24, -1, -1)], dim=-1)
            pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch_size*4, -1) # convert rotation matrix to rotation vector
            pred_pose[torch.isnan(pred_pose)] = 0.0
            new_opt_vertices, new_opt_joints,\
            new_opt_pose, new_opt_betas,\
            new_opt_cam_t, new_opt_joint_loss = self.mvsmplify(
                                        pred_pose.detach(), pred_betas.detach(),
                                        pred_cam_t.detach(),
                                        0.5 * self.options.img_res * torch.ones(batch_size*4, 2, device=self.device),
                                        gt_keypoints_2d_orig)
            new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)
            # Will update the dictionary for the examples where the new loss is less than the current one
            update = (new_opt_joint_loss < opt_joint_loss)
            update1 = torch.cat((update,update,update,update),0)
            opt_joint_loss[update] = new_opt_joint_loss[update]
            opt_joints[update1, :] = new_opt_joints[update1, :]
            opt_betas[update1, :] = new_opt_betas[update1, :]
            opt_pose[update1, :] = new_opt_pose[update1, :]
            opt_vertices[update1, :] = new_opt_vertices[update1, :]
            opt_cam_t[update1, :] = new_opt_cam_t[update1, :]
        # now we comput the loss on the four images
        # Replace the optimized parameters with the ground truth parameters, if available
        has_smpl1 = torch.cat((has_smpl,has_smpl,has_smpl,has_smpl),0)
        opt_vertices[has_smpl1, :, :] = gt_vertices[has_smpl1, :, :]
        opt_pose[has_smpl1, :] = gt_pose[has_smpl1, :]
        opt_cam_t[has_smpl1, :] = gt_cam_t[has_smpl1, :]
        opt_joints[has_smpl1, :, :] = gt_model_joints[has_smpl1, :, :]
        opt_betas[has_smpl1, :] = gt_betas[has_smpl1, :]
        # Assert whether a fit is valid by comparing the joint loss with the threshold
        valid_fit1 = (opt_joint_loss < self.options.mvsmplify_threshold).to(self.device)
        # Add the examples with GT parameters to the list of valid fits
        valid_fit = torch.cat((valid_fit1,valid_fit1,valid_fit1,valid_fit1),0) | has_smpl1

        # 2d joint points loss
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,0,1)
        # 3d joint points loss
        gt_joints = torch.cat((input_batch['pose_3d_0'],input_batch['pose_3d_1'],input_batch['pose_3d_2'],input_batch['pose_3d_3']),0)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints, torch.cat((has_pose_3d,has_pose_3d,has_pose_3d,has_pose_3d),0))
        # theta and beta loss
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)
        # mesh loss
        loss_shape = self.shape_loss(pred_vertices, opt_vertices, valid_fit)

        loss_all = self.options.shape_loss_weight * loss_shape +\  
                   self.options.keypoint_loss_weight * loss_keypoints +self.options.keypoint_loss_weight * loss_keypoints_3d\
                   self.options.pose_loss_weight * loss_regr_pose + self.options.beta_loss_weight * loss_regr_betas +\
                   ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()
        #loss_all = 5. * loss_keypoints + 5. * loss_keypoints_3d +((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()
                   #0. * loss_keypoints_3d +\
                   
        loss_all *= 60
        print(loss_all)
        
        # Do backprop
        self.optimizer.zero_grad()
        loss_all.backward()
        self.optimizer.step()
        output = {'pred_vertices': pred_vertices,
                  'opt_vertices': opt_vertices,
                  'pred_cam_t': pred_cam_t,
                  'opt_cam_t': opt_cam_t
                  }
        #losses = {'loss': loss_all.detach().item(),
        #          'loss_keypoints': loss_keypoints.detach().item(),
        #          'loss_keypoints_3d': loss_keypoints_3d.detach().item()}
        losses = {'loss': loss_all.detach().item(),
                  'loss_keypoints': loss_keypoints.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                  'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas.detach().item(),
                  'loss_shape': loss_shape.detach().item()}
        #'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
        return output, losses
       
    def train_summaries(self, input_batch, output, losses):
        pred_vertices = output['pred_vertices']
        opt_vertices = output['opt_vertices']
        pred_cam_t = output['pred_cam_t']
        opt_cam_t = output['opt_cam_t']
        images_pred = self.renderer.visualize_tb(pred_vertices,pred_cam_t, input_batch)
        images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, input_batch)
        
        self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
        self.summary_writer.add_image('opt_shape', images_opt, self.step_count)
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
