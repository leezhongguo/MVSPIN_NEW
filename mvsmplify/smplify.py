import torch
import os

from models.smpl import SMPL
from .losses import camera_fitting_loss, body_fitting_loss
import config
import constants
from utils.geometry import perspective_projection, batch_rodrigues
import numpy as np
# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from .prior import MaxMixturePrior

class SMPLify():
    """Implementation of single-stage SMPLify.""" 
    def __init__(self, 
                 step_size=1e-2,
                 batch_size=66,
                 num_iters=50,
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
        global_orient = init_pose[:,:3].detach().clone()
        betas = init_betas.detach().clone()
        # Step 1:optimize camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True
        #print(body_pose.device,betas.device,global_orient.device)
        camera_opt_params = [global_orient,camera_translation]
        camera_optimizer = torch.optim.Adam(camera_opt_params,lr=self.step_size, betas=(0.9,0.999))
        for j in range(self.num_iters):
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose = body_pose,
                                    betas = betas)
            model_joints = smpl_output.joints
            loss = camera_fitting_loss(model_joints, camera_translation,init_cam_t,camera_center,joints_2d,joints_conf,focal_length = self.focal_length)
            camera_optimizer.zero_grad()
            loss.backward()
            #print(j,loss)
            camera_optimizer.step()
        camera_translation.requires_grad = False
        global_orient.requires_grad = False
        # step 2: Optimize body joints
        body_pose_1 = body_pose[:int(batch_size/4)].clone()
        #print(body_pose_1.is_leaf)
        body_pose_1.requires_grad=True
        #print(body_pose_1.is_leaf)
        betas_1 = betas[:int(batch_size/4)].clone()
        betas_1.requires_grad=True
        global_orient1 = global_orient[:int(batch_size/4)].clone()
        global_orient1.requires_grad=True
        global_orient2 = global_orient[int(batch_size/4):int(batch_size/2)].clone()
        global_orient2.requires_grad=True
        global_orient3 = global_orient[int(batch_size/2):int(batch_size/4*3)].clone()
        global_orient3.requires_grad=True
        global_orient4 = global_orient[int(batch_size/4*3):].clone()
        global_orient4.requires_grad=True
        
        body_opt_params = [body_pose_1,betas_1,global_orient1,global_orient2,global_orient3,global_orient4]
        #print(body_pose_1.is_leaf,betas_1.is_leaf, global_orient.is_leaf)
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        #print('Here!!!!!!!!')
        #input()
        for i in range(self.num_iters):
            smpl_output1 = self.smpl(global_orient=global_orient1,
                                        body_pose=body_pose_1,
                                        betas=betas_1)
            #model_joints_1 = smpl_output1.joints
            #print(model_joints_1.device)
            smpl_output2 = self.smpl(global_orient=global_orient2,
                                        body_pose=body_pose_1,
                                        betas=betas_1)
            #model_joints_2 = smpl_output2.joints
            smpl_output3 = self.smpl(global_orient=global_orient3,
                                        body_pose=body_pose_1,
                                        betas=betas_1)
            #model_joints_3 = smpl_output3.joints
            smpl_output4 = self.smpl(global_orient=global_orient4,
                                        body_pose=body_pose_1,
                                        betas=betas_1)
            #model_joints_4 = smpl_output4.joints
            model_joints = torch.cat((smpl_output1.joints,smpl_output2.joints,smpl_output3.joints,smpl_output4.joints),0)
            #model_joints = torch.cat((model_joints_1,model_joints_2,model_joints_3,model_joints_4),0)
            loss = body_fitting_loss(body_pose_1, betas_1, model_joints, camera_translation, camera_center,
                                     joints_2d, joints_conf, self.pose_prior,
                                     focal_length=self.focal_length)
            body_optimizer.zero_grad()
            loss.backward()
            #print(i,loss)
            body_optimizer.step()
        with torch.no_grad():
            global_orient1.requires_grad = False
            global_orient2.requires_grad = False
            global_orient3.requires_grad = False
            global_orient4.requires_grad = False
            smpl_output1 = self.smpl(global_orient=global_orient1,
                                        body_pose=body_pose_1,
                                        betas=betas_1, return_full_pose=True)
            smpl_output2 = self.smpl(global_orient=global_orient2,
                                        body_pose=body_pose_1,
                                        betas=betas_1, return_full_pose=True)
            smpl_output3 = self.smpl(global_orient=global_orient3,
                                        body_pose=body_pose_1,
                                        betas=betas_1, return_full_pose=True)
            smpl_output4 = self.smpl(global_orient=global_orient4,
                                        body_pose=body_pose_1,
                                        betas=betas_1, return_full_pose=True)
            model_joints = torch.cat((smpl_output1.joints,smpl_output2.joints,smpl_output3.joints,smpl_output4.joints),0)
            vertices = torch.cat((smpl_output1.vertices,smpl_output2.vertices,smpl_output3.vertices,smpl_output4.vertices),0).detach()
            pose1 = torch.cat([global_orient[:int(batch_size/4)], body_pose_1], dim=-1).detach()
            pose2 = torch.cat([global_orient[int(batch_size/4):int(batch_size/2)], body_pose_1], dim=-1).detach()
            pose3 = torch.cat([global_orient[int(batch_size/2):int(batch_size/4*3)], body_pose_1], dim=-1).detach()
            pose4 = torch.cat([global_orient[int(batch_size/4*3):], body_pose_1], dim=-1).detach()
            pose = torch.cat((pose1,pose2,pose3,pose4),0)
            betas = torch.cat((betas_1,betas_1,betas_1,betas_1),0).detach()
            reprojection_loss = body_fitting_loss(body_pose_1, betas_1, model_joints, camera_translation, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')
        #vertices = smpl_output.vertices.detach()
        joints = model_joints.detach()
        #pose = torch.cat([global_orient_mv[0], body_pose_mv[0]], dim=-1).detach()
        #betas = betas_1.detach()
        #np.save('pose.npy',pose.cpu())
        #np.save('betas.npy',betas.cpu())
        return vertices, joints, pose, betas, camera_translation, reprojection_loss
        """
        body_pose_mv, global_orient_mv, betas_mv, camera_trans_mv = [], [], [], []
        joints_2d_mv, joints_conf_mv = [],  []
        for i in range(4):
            batch_size = init_pose[i].shape[0]
            # make camera translation a learnable parameter
            camera_translation = init_cam_t[i].clone()
            # get joint confidence
            joints_2d = keypoints_2d[i][:,:,:2]
            joints_conf = keypoints_2d[i][:,:,-1]
            # Split SMPL pose to body pose and global orientation
            body_pose = init_pose[i][:,3:].detach().clone()
            global_orient = init_pose[i][:,:3].detach().clone()
            betas = init_betas[i].detach().clone()
            body_pose_mv.append(body_pose)
            global_orient_mv.append(global_orient)
            betas_mv.append(betas)
            joints_2d_mv.append(joints_2d)
            joints_conf_mv.append(joints_conf)
            # Step 1:optimize camera translation and body orientation
            body_pose.requires_grad = False
            betas.requires_grad = False
            global_orient.requires_grad = True
            camera_translation.requires_grad = True

            camera_opt_params = [global_orient,camera_translation]
            camera_optimizer = torch.optim.Adam(camera_opt_params,lr=self.step_size, betas=(0.9,0.999))
            for j in range(self.num_iters):
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose = body_pose,
                                        betas = betas)
                model_joints = smpl_output.joints
                loss = camera_fitting_loss(model_joints, camera_translation,init_cam_t[i],camera_center,joints_2d,joints_conf,focal_length = self.focal_length)
                camera_optimizer.zero_grad()
                loss.backward()
                #print(j,loss)
                camera_optimizer.step()
            # Fix camera translation after optimizing camera
            #input()
            camera_translation.requires_grad = False
            camera_trans_mv.append(camera_translation)
            
        # step 2: Optimize body joints
        model_joint_mv,camera_rot_mv = [],[]
        for i in range(4):
            body_pose_mv[i].requires_grad=True
            betas_mv[i].requires_grad=True
            #global_orient_mv[i].requires_grad = False
            #camera_rot = -global_orient_mv[i].detach()
            #print(camera_rot.size)
            #camera_rot.requires_grad = True
            #camera_rot_mv.append(camera_rot)
            global_orient_mv[i].requires_grad=True
            #camera_trans_mv[i].requires_grad=True
            #print()
            #camera_trans_mv[i].requires_grad = False
            
            #camera_rot.requires_grad = True
            #camera_rot_mv.append(camera_rot)
            #print(camera_rot.size())
            #input()
            #camera_rot = global_orient_mv[i].clone()
            camera_rot = torch.zeros(batch_size, 3, device=body_pose.device)
            camera_rot.requires_grad = False
            camera_rot_mv.append(camera_rot)
            #print(camera_rot)
            #camera_rot_mat = batch_rodrigues(camera_rot)
            #print(camera_rot_mat)
            #input()
        #camera_rot =  torch.zeros(batch_size, 3, device=body_pose.device)
        #camera_rot.requires_grad = True
        #camera_rot_mv[0] = camera_rot
        body_opt_params = [body_pose_mv[0],betas_mv[0],global_orient_mv[0],global_orient_mv[1],global_orient_mv[2],global_orient_mv[3]]
        
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        #print(self.num_iters)
        for i in range(self.num_iters):
            smpl_output = self.smpl(global_orient=global_orient_mv[0],
                                        body_pose=body_pose_mv[0],
                                        betas=betas_mv[0])
            model_joints_1 = smpl_output.joints
            smpl_output = self.smpl(global_orient=global_orient_mv[1],
                                        body_pose=body_pose_mv[0],
                                        betas=betas_mv[0])
            model_joints_2 = smpl_output.joints
            smpl_output = self.smpl(global_orient=global_orient_mv[2],
                                        body_pose=body_pose_mv[0],
                                        betas=betas_mv[0])
            model_joints_3 = smpl_output.joints
            smpl_output = self.smpl(global_orient=global_orient_mv[3],
                                        body_pose=body_pose_mv[0],
                                        betas=betas_mv[0])
            model_joints_4 = smpl_output.joints
            loss = body_fitting_loss(body_pose_mv[0], betas_mv[0], [model_joints_1,model_joints_2,model_joints_3,model_joints_4], camera_trans_mv, camera_center,
                                     joints_2d_mv, joints_conf_mv, self.pose_prior,
                                     focal_length=self.focal_length)
            body_optimizer.zero_grad()
            loss.backward()
            #print(i,loss)
            body_optimizer.step()
        #input()
        #for i in range(4):
        #    global_orient_mv[i].requires_grad = False
            #camera_trans_mv[i].requires_grad = False
            #camera_rot_mv[i].requires_grad = False
            #camera_rot_mat = batch_rodrigues(camera_rot_mv[i])
            #camera_rot_mat_mv.append(camera_rot_mat)
            #rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
            
            #smpl_output = self.smpl(global_orient=global_orient_mv[i],
            #                            body_pose=body_pose_mv[0],
            #                            betas=betas_mv[0])
            #model_joints = smpl_output.joints
            #pred_keypoints_2d = perspective_projection(smpl_output.vertices.detach(),
            #                                            rotation=camera_rot_mat,
            #                                            translation=camera_trans_mv[i],
            #                                            focal_length=self.focal_length,
            #                                            camera_center=camera_center)
            #pred_keypoints_2d = perspective_projection(model_joints,
            #                                           rotation=rotation,
            #                                            translation=camera_trans_mv[i],
            #                                            focal_length=self.focal_length,
            #                                            camera_center=camera_center)
            #np.save('opt_keypoints_%d' % i, pred_keypoints_2d.detach().cpu())
        #for i in range(4):
        #    camera_rot_mat = batch_rodrigues(camera_rot_mv[i])
        #    pred_keypoints_2d = perspective_projection(model_joints,
        #                                                rotation=camera_rot_mat,
        #                                                translation=camera_trans_mv[i],
        #                                                focal_length=self.focal_length,
        #                                                camera_center=camera_center)
        #    np.save('pred_keypoints_%d' % i, pred_keypoints_2d.detach().cpu())
        #input()
        # Get final loss value
        vertices_mv,pose_mv = [],[]
        with torch.no_grad():
            for i in range(4):
                global_orient_mv[i].requires_grad = False
                smpl_output = self.smpl(global_orient=global_orient_mv[i],
                                        body_pose=body_pose_mv[0],
                                        betas=betas_mv[0], return_full_pose=True)
                model_joint_mv.append(smpl_output.joints)
                vertices_mv.append(smpl_output.vertices.detach())
                pose_mv.append(torch.cat([global_orient_mv[i], body_pose_mv[0]], dim=-1).detach())
            reprojection_loss = body_fitting_loss(body_pose_mv[0], betas_mv[0], model_joint_mv, camera_trans_mv, camera_center,
                                                  joints_2d_mv, joints_conf_mv, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')
        """
        

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
        """
        joints_2d_mv, joints_conf_mv = [], []
        #camera_rot_mv = []
        for i in range(4):
            joints_2d = keypoints_2d[i][:, :, :2]
            joints_conf = keypoints_2d[i][:, :, -1]
            # For joints ignored during fitting, set the confidence to 0
            joints_conf[:, self.ign_joints] = 0.
            #camera_rot = torch.zeros(batch_size,3, device=pose[0].device)
            joints_2d_mv.append(joints_2d)
            joints_conf_mv.append(joints_conf)
            #camera_rot_mv.append(camera_rot)
        """
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
