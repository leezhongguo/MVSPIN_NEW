# basic modules
import numpy as np
import cv2
import argparse
import pickle as pkl
import json

# model modules
import torch
from torchvision.transforms import Normalize
import torchgeometry as tgm
from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer import Renderer
import config
import constants

parser = argparse.ArgumentParser()
parser.add_argument('--trained_model', required=True, help='Path to pretrained model')
parser.add_argument('--test_image', type=str, required=True, help='Path to the test image')
parser.add_argument('--bbox',type=str, default=None,help='Path to .json containing bounding box of the person')
parser.add_argument('--output', type=str, default=None, help='Path to output')


def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def process_image(img_file, bbox_file, input_res=224):
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() 
    if bbox_file is None:
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        center, scale = bbox_from_json(bbox_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

if __name__ =='__main__':
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load trained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.trained_model)
    model.load_state_dict(checkpoint['model'], strict=False)
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1, create_transl=False).to(device)
    model.eval()
    # Generate rendered image
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES,faces=smpl.faces)
    # Processs the image and predict the parameters
    img, norm_img = process_image(args.test_image, args.bbox, input_res=constants.IMG_RES)
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()
    #Convert rotation matrix of joint points to rotate vector 
    rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
    rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(1 * 24, -1, -1)), dim=-1)
    pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
    pred_theta  = pred_pose.cpu().numpy()
    pred_beta = pred_betas.cpu().numpy()
    pred_param = {'pose':pred_theta,'shape':pred_beta}
    print(pred_theta.shape,pred_beta.shape)
    img = img.permute(1,2,0).cpu().numpy()
    # Rendered result
    img_shape = renderer(pred_vertices, camera_translation, img)
    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    center = pred_vertices.mean(axis=0)
    rot_vertices = np.dot((pred_vertices - center), aroundy) + center
    # The other side result
    img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img))
    # Output filename
    outfile = args.test_image.split('.')[0] if args.output is None else args.output
    # Save results
    cv2.imwrite(outfile + '_shape.png', 255 * img_shape[:,:,::-1])            # image
    cv2.imwrite(outfile + '_shape_side.png', 255 * img_shape_side[:,:,::-1])  # other side image
    outmesh = outfile+'_smpl.obj'                                             # output mesh
    outparmeters = outfile+'_param.pkl'                                       # output pose and shape parameters of SMPL model
    # Save mesh
    with open(config.SMPL_MODEL_DIR,'rb') as fp:
        u = pkl._Unpickler(fp)
        u.encoding = 'latin1'
        model = u.load()
    pred_faces = model['f']
    with open( outmesh, 'w') as fp:
        for v in pred_vertices:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
        for f in pred_faces+1: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
    # Save parameters
    with open(outparmeters,'wb') as fp:
        pkl.dump(pred_param,fp)
