import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
#os.environ["PYOPENGL_PLATFORM"] = "egl"
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces

    #def visualize_tb(self, vertices, camera_translation, camera_rotation, images):
    def visualize_tb(self, vertices_mv, camera_translation_mv,  input_batch):
        batch_size = input_batch['img_%d' % 0].shape[0]
        rend_imgs = []
        for i in range(4):
            images = input_batch['img_%d' % i]
            images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
            images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
            vertices = vertices_mv[i].detach().cpu().numpy()
            camera_translation = camera_translation_mv[i].detach().cpu().numpy()
            #camera_rotation = camera_rotation_mv[i].detach().cpu().numpy()
            #print(camera_rotation.shape)
            #input()
            images = images.cpu()
            images_np = np.transpose(images.numpy(), (0,2,3,1))
            for j in range(vertices.shape[0]):
                rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[j], camera_translation[j], images_np[j]), (2,0,1))).float()
                #np.save('rend_img.npy',rend_img)
                #np.save('image.npy',images[i])
                #input()
                rend_imgs.append(images[j])
                rend_imgs.append(rend_img)
        rend_img_news = []
        for i in range(batch_size):
            for j in range(4):
                rend_img_news.append(rend_imgs[i*2+batch_size*2*j])
                rend_img_news.append(rend_imgs[i*2+batch_size*2*j+1])
        rend_img_news= make_grid(rend_img_news, nrow=8)
        return rend_img_news

    def __call__(self, vertices, camera_translation, image):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        #rot1 = np.eye(4)
        #rot1[:3,:3] = camera_rotation.T
        #mesh.apply_transform(rot1)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        #camera_pose[:3,:3] = camera_rotation
        #print(camera_pose)
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        #camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        scene.add(camera, pose=camera_pose)

        #light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
        #                    innerConeAngle=np.pi/16.0,
        #                    outerConeAngle=np.pi/6.0)
        #scene.add(light, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        #scene.add(light, pose=camera_pose)
        light_pose = np.eye(4)
        #light_pose[:3,:3] = camera_rotation

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img
