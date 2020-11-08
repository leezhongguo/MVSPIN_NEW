import torch
import os

from models.smpl import SMPL
#from .losses import camera_fitting_loss, body_fitting_loss
from .losses_mv import camera_fitting_loss, body_fitting_loss
import config
import constants
from utils.geometry import perspective_projection, batch_rodrigues
import numpy as np
# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from .prior import MaxMixturePrior

class MVSMPLify():
    """Implementation of single-stage MVSMPLify.""" 
    def __init__(self, 
                 step_size=1e-2,
                 batch_size=66,
                 num_iters=100,
                 focal_length=5000,
                 device=torch.device('cuda')):

        # Store options
        self.device = device
        self.focal_length = focal_length
        self.step_size = step_size

        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [constants.JOINT_IDS[i] for i in ign_joints]
        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder='data',
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # Load SMPL model
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=batch_size,
                         create_transl=False).to(self.device)

    def __call__(self, init_pose, init_betas, init_cam_t, camera_center, keypoints_2d):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """
        batch_size = int(init_pose.shape[0])
        camera_translation = init_cam_t.clone()
        # get joint confidence
        joints_2d = keypoints_2d[:,:,:2]
        joints_conf = keypoints_2d[:,:,-1]
        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:,3:].detach().clone()
        global_orient = init_pose[:,:3].detach().clone()    # This is the body orientation
        betas = init_betas.detach().clone()
        # Step 1:optimize camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True     
        camera_translation.requires_grad = True
        # for each human, we use the average of four-view parameters as initilization, and we only update the average pose and shape for each body
        body_pose_1 = (body_pose[:int(batch_size/4)].clone()+body_pose[int(batch_size/4):int(batch_size/2)].clone()+\
                      body_pose[int(batch_size/2):int(batch_size/4*3)].clone()+body_pose[int(batch_size/4*3):].clone())/4.
        body_pose_1.requires_grad=False
        betas_1 = (betas[:int(batch_size/4)].clone()+betas[int(batch_size/4):int(batch_size/2)].clone()+\
                   betas[int(batch_size/2):int(batch_size/4*3)].clone()+betas[int(batch_size/4*3):].clone())/4.
        betas_1.requires_grad=False  
        camera_opt_params = [global_orient,camera_translation]  # update body orientation and camera translation
        camera_optimizer = torch.optim.Adam(camera_opt_params,lr=self.step_size, betas=(0.9,0.999))
        # optimize camera translation
        for j in range(self.num_iters):
            smpl_output1 = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
            model_joints_1 = smpl_output1.joints
            loss = camera_fitting_loss(model_joints_1, camera_translation,init_cam_t,camera_center,joints_2d,joints_conf,focal_length = self.focal_length)
            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        # step 2: Optimize body joints
        body_pose_1.requires_grad=True
        betas_1.requires_grad=True
        # updata pose, shape, body orientaiton, camera translation
        # The global orientation is four-view orientaiton
        body_opt_params = [body_pose_1,betas_1,global_orient,camera_translation] 
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        for i in range(self.num_iters):
            body_pose1 = torch.cat((body_pose_1,body_pose_1,body_pose_1,body_pose_1),0)  # the four-view body pose and shape should keep the same
            betas1 = torch.cat((betas_1,betas_1,betas_1,betas_1),0)
            smpl_output1 = self.smpl(global_orient=global_orient,
                                     body_pose=body_pose1,
                                     betas=betas1)
            model_joints1 = smpl_output1.joints
            loss = body_fitting_loss(body_pose_1, betas_1, model_joints1, camera_translation, camera_center,
                                     joints_2d, joints_conf, self.pose_prior,
                                     focal_length=self.focal_length)
            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()
        camera_translation.requires_grad = False
        # After fitting, compute the loss
        with torch.no_grad():
            global_orient.requires_grad = False
            body_pose_1.requires_grad=False
            betas_1.requires_grad=False
            body_pose1 = torch.cat((body_pose_1,body_pose_1,body_pose_1,body_pose_1),0)
            betas1 = torch.cat((betas_1,betas_1,betas_1,betas_1),0)
            smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
           
            vertices = smpl_output.vertices.detach()
            reprojection_loss = body_fitting_loss(body_pose_1, betas_1, model_joints, camera_translation, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')
        joints = model_joints.detach()
        pose = torch.cat([global_orient, body_pose1], dim=-1).detach()

        return vertices, joints, pose, betas1.detach(), camera_translation, reprojection_loss
        

    def get_fitting_loss(self, pose, betas, cam_t, camera_center, keypoints_2d):
        """Given body and camera parameters, compute reprojection loss value.
        Input:
            pose: SMPL pose parameters
            betas: SMPL beta parameters
            cam_t: Camera translation
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            reprojection_loss: Final joint reprojection loss
        """

        batch_size = pose.shape[0]

        # Get joint confidence
        joints_2d = keypoints_2d[:,:,:2]
        joints_conf = keypoints_2d[:,:,-1]
        # Split SMPL pose to body pose and global orientation
        body_pose = pose[:int(batch_size/4), 3:]
        global_orient = pose[:, :3]
        betas_1 = betas[:int(batch_size/4)]

        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient[:int(batch_size/4)],
                                    body_pose=body_pose,
                                    betas=betas_1, return_full_pose=True)
            model_joints_1 = smpl_output.joints
            smpl_output = self.smpl(global_orient=global_orient[int(batch_size/4):int(batch_size/2)],
                                        body_pose=body_pose,
                                        betas=betas_1)
            model_joints_2 = smpl_output.joints
            smpl_output = self.smpl(global_orient=global_orient[int(batch_size/2):int(batch_size/4*3)],
                                        body_pose=body_pose,
                                        betas=betas_1)
            model_joints_3 = smpl_output.joints
            smpl_output = self.smpl(global_orient=global_orient[int(batch_size/4*3):],
                                        body_pose=body_pose,
                                        betas=betas_1)
            model_joints_4 = smpl_output.joints
            model_joints = torch.cat((model_joints_1,model_joints_2,model_joints_3,model_joints_4),0)
            reprojection_loss = body_fitting_loss(body_pose, betas_1, model_joints, cam_t, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')

        return reprojection_loss
