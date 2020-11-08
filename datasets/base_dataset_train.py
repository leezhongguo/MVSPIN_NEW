from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from os.path import join

import config
import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = config.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(config.DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        
        # Get gt SMPL parameters, if available
        #try:
        #    self.pose = self.data['pose'].astype(np.float)
        #    self.betas = self.data['shape'].astype(np.float)
        #    if 'has_smpl' in self.data:
        #        self.has_smpl = self.data['has_smpl']
        #    else:
        #        self.has_smpl = np.ones(len(self.imgname))
        #except KeyError:
        #    self.has_smpl = np.zeros(len(self.imgname))
        self.has_smpl = np.zeros(len(self.imgname))
        
        # Get gt 3D pose, if available
        self.pose_3d = self.data['S']
        self.has_pose_3d = 1
        #
        #ry:
        #    self.pose_3d = self.data['S']
        #    self.has_pose_3d = 1
        #except KeyError:
        #    self.has_pose_3d = 0
        # you can choose if trainning the network with 3d pose
        if ignore_3d:
            self.has_pose_3d = 0
        
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 4, 25, 3)) # for training we have 4 views
            #keypoints_openpose = np.zeros((len(self.imgname), 25, 3)) # for testing we don't use 4 views
        #print(keypoints_openpose.shape,keypoints_gt.shape)
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=2 ) # for training we have 4 views

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)
        
        self.length = self.scale.shape[0]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                    max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()
        # Load image
        view_number = 4 # here we use four-view images
        for i in range(view_number): # four views
            if type(self.imgname[index][i]).__name__!="str_":
                imgname = join(self.img_dir,  str(self.imgname[index][i],encoding='utf-8'))
            else:
                imgname = join(self.img_dir,self.imgname[index][i])
            try:
                img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
            except KeyError:
                print(imgname)
            item['orig_shape_%d' % i] = np.array(img.shape)[:2] # HxW
            img = self.rgb_processing(img, center[i], sc*scale[i], rot, flip, pn)
            img = torch.from_numpy(img).float()
            item['img_%d' % i] = self.normalize_img(img)
            item['imgname_%d' % i] = imgname
        # Get 3D pose of different view
        S = self.pose_3d[index].copy()
        for i in range(4): # four views
            item['pose_3d_%d' % i] = torch.from_numpy(self.j3d_processing(S[i],rot,flip)).float()
        # Get 2D pose
        keypoints = self.keypoints[index].copy()
        for i in range(view_number):
            item['keypoints_%d' % i] = torch.from_numpy(self.j2d_processing(keypoints[i],center[i], sc*scale[i], rot, flip)).float()      
            item['scale_%d' % i] = float(scale[i])
            item['center_%d' % i] = center[i].astype(np.float32)
        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)
        item['has_pose_3d'] = self.has_pose_3d
        item['has_smpl'] = self.has_smpl[index]
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
        return item 

    def __len__(self):
        return len(self.imgname)
