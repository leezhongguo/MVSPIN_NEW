"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

#H36M_ROOT = '/usr/vision/data/Human36m/images_test'
H36M_ROOT = '/usr/vision/data/Human36m/images'
MPI_INF_3DHP_ROOT = '/usr/vision/data/mpi_inf_3dhp/'
PW3D_ROOT = '/usr/vision/data/3DPW/'

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# We do not use openpose to regenerate the annotations. So we comment this.
# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
# OPENPOSE_PATH = 'datasets/openpose'

# Path to test/train npz files
DATASET_FILES = [ {'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                  },

                  {'h36m': join(DATASET_NPZ_PATH, 'h36m_train_full_3d.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train_mv.npz')
                  }
                ]

DATASET_FOLDERS = {'h36m': H36M_ROOT,
                   'h36m-p1': H36M_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   '3dpw': PW3D_ROOT,
                }

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
