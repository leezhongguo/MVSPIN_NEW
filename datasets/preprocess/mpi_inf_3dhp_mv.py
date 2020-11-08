import os
import sys
import cv2
import glob
import h5py
import json
import numpy as np
import scipy.io as sio
import scipy.misc
#from .read_openpose import read_openpose

def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i*7+5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i*7+6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3,:3]
        T = RT[:3,3]/1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts
    
def train_data(dataset_path, openpose_path, out_path, joints_idx, scaleFactor, extract_img=False, fits_3d=None):

    joints17_idx = [4, 18, 19, 20, 23, 24, 25, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]

    h, w = 2048, 2048
    

    # training data
    user_list = range(1,2)
    seq_list = range(1,3)
    #vid_list = list(range(1)) + list(range(4,9))
    vid_list = [0,2,7,8]
    counter = 0

    for user_i in user_list:
        print("Subject: %d " % user_i)
        imgnames_, scales_, centers_ = [], [], []
        parts_, Ss_, openposes_ = [], [], []
        for seq_i in seq_list:
            print("Sequence: %d " % seq_i)
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))
            # mat file with annotations
            annot_file = os.path.join(seq_path, 'annot.mat')
            print(annot_file)
            annot2 = sio.loadmat(annot_file)['annot2']
            annot3 = sio.loadmat(annot_file)['annot3']
            # calibration file and camera parameters
            calib_file = os.path.join(seq_path, 'camera.calibration')
            Ks, Rs, Ts = read_calibration(calib_file, vid_list)

            for j, vid_i in enumerate(vid_list):
                print("Video: %d" % vid_i)
                # image folder
                imgs_path = os.path.join(seq_path,    
                                         'imageFrames',
                                         'video_' + str(vid_i))

                # extract frames from video file
                if extract_img:
                    # if doesn't exist
                    if not os.path.isdir(imgs_path):
                        os.makedirs(imgs_path)

                    # video file
                    vid_file = os.path.join(seq_path,
                                            'imageSequence',
                                            'video_' + str(vid_i) + '.avi')
                    vidcap = cv2.VideoCapture(vid_file)

                    # process video
                    frame = 0
                    while 1:
                        # extract all frames
                        success, image = vidcap.read()
                        if not success:
                            break
                        frame += 1
                        # image name
                        imgname = os.path.join(imgs_path,
                            'frame_%06d.jpg' % frame)
                        ok_pts = []
                        for k, v_index in enumerate(vid_list):
                            joints = np.reshape(annot2[v_index][0][frame-1], (28, 2))[joints17_idx]
                            x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
                            y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
                            ok_pts.append(np.logical_and(x_in, y_in))
                        if (np.sum(ok_pts[0]) < len(joints_idx)) | (np.sum(ok_pts[1])<len(joints_idx)) | (np.sum(ok_pts[2])<len(joints_idx)) | (np.sum(ok_pts[3])<len(joints_idx)):
                            continue
                        if frame % 10 != 1:
                            continue
                        # save image
                        cv2.imwrite(imgname, image)
                        joints = np.reshape(annot2[vid_i][0][frame-1], (28, 2))[joints17_idx]
                        S17 = np.reshape(annot3[vid_i][0][frame-1], (28, 3))/1000
                        S17 = S17[joints17_idx] - S17[4] # 4 is the root
                        bbox = [min(joints[:,0]), min(joints[:,1]),
                                max(joints[:,0]), max(joints[:,1])]
                        center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                        scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
                        part = np.zeros([24,3])
                        part[joints_idx] = np.hstack([joints, np.ones([17,1])])
                        S = np.zeros([24,4])
                        S[joints_idx] = np.hstack([S17, np.ones([17,1])])
                        img_name = imgname.split('/')[-1]
                        img_view = os.path.join('S' + str(user_i),
                                            'Seq' + str(seq_i),
                                            'imageFrames',
                                            'video_' + str(vid_i),
                                            img_name)
                        imgnames_.append(img_view)
                        centers_.append(center)
                        scales_.append(scale)
                        parts_.append(part)
                        Ss_.append(S)
                       
        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        file_name = 'mpi_inf_3dhp_train_%d.npz' % user_i
        out_file = os.path.join(out_path, file_name)
        if fits_3d is not None:
            fits_3d = np.load(fits_3d)
            np.savez(out_file, imgname=imgnames_,
                            center=centers_,
                            scale=scales_,
                            part=parts_,
                            pose=fits_3d['pose'],
                            shape=fits_3d['shape'],
                            has_smpl=fits_3d['has_smpl'],
                            S=Ss_)
        else:
            np.savez(out_file, imgname=imgnames_,
                            center=centers_,
                            scale=scales_,
                            part=parts_,
                            S=Ss_)        
        
        
def test_data(dataset_path, out_path, joints_idx, scaleFactor):

    joints17_idx = [14, 11, 12, 13, 8, 9, 10, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4]

    imgnames_, scales_, centers_, parts_,  Ss_ = [], [], [], [], []

    # training data
    user_list = range(1,7)

    for user_i in user_list:
        seq_path = os.path.join(dataset_path,
                                'mpi_inf_3dhp_test_set',
                                'TS' + str(user_i))
        # mat file with annotations
        annot_file = os.path.join(seq_path, 'annot_data.mat')
        mat_as_h5 = h5py.File(annot_file, 'r')
        annot2 = np.array(mat_as_h5['annot2'])
        annot3 = np.array(mat_as_h5['univ_annot3'])
        valid = np.array(mat_as_h5['valid_frame'])
        for frame_i, valid_i in enumerate(valid):
            if valid_i == 0:
                continue
            img_name = os.path.join('mpi_inf_3dhp_test_set',
                                   'TS' + str(user_i),
                                   'imageSequence',
                                   'img_' + str(frame_i+1).zfill(6) + '.jpg')

            joints = annot2[frame_i,0,joints17_idx,:]
            S17 = annot3[frame_i,0,joints17_idx,:]/1000
            S17 = S17 - S17[0]

            bbox = [min(joints[:,0]), min(joints[:,1]),
                    max(joints[:,0]), max(joints[:,1])]
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
            scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200

            # check that all joints are visible
            img_file = os.path.join(dataset_path, img_name)
            I = scipy.misc.imread(img_file)
            h, w, _ = I.shape
            x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
            y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
            ok_pts = np.logical_and(x_in, y_in)
            if np.sum(ok_pts) < len(joints_idx):
                continue

            part = np.zeros([24,3])
            part[joints_idx] = np.hstack([joints, np.ones([17,1])])

            S = np.zeros([24,4])
            S[joints_idx] = np.hstack([S17, np.ones([17,1])])

            # store the data
            imgnames_.append(img_name)
            centers_.append(center)
            scales_.append(scale)
            parts_.append(part)
            Ss_.append(S)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'mpi_inf_3dhp_test.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       S=Ss_)    

def mpi_inf_3dhp_extract(dataset_path, openpose_path, out_path, mode, extract_img=False, static_fits=None):

    scaleFactor = 1.2
    joints_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]
    
    if static_fits is not None:
        fits_3d = os.path.join(static_fits, 
                               'mpi-inf-3dhp_mview_fits.npz')
    else:
        fits_3d = None
    
    if mode == 'train':
        train_data(dataset_path, openpose_path, out_path, 
                   joints_idx, scaleFactor, extract_img=extract_img, fits_3d=fits_3d)
    elif mode == 'test':
        test_data(dataset_path, out_path, joints_idx, scaleFactor)


dataset_path = "/usr/vision/data/mpi_inf_3dhp/"
openpose_path = "test/"
out_path = "/usr/vision/data/mpi_inf_3dhp/"
mode = "train"
extract_img=True
static_fits = "/home/li/human_body/SPIN/data/static_fits/"
mpi_inf_3dhp_extract(dataset_path, openpose_path, out_path, mode, extract_img=extract_img)