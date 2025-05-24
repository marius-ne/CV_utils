from scipy.spatial.transform import Rotation
import numpy as np
import cv2
import pyrender
import trimesh

def invert_pose(pose_matrix: np.ndarray):
    return np.linalg.inv(pose_matrix)

def render_depth_map(pose: np.ndarray,
                     mesh_path: str,
                     K: np.ndarray,
                     width: int,
                     height: int) -> np.ndarray:
        """
        Loads an OBJ mesh, place a camera defined by intrinsic & extrinsic,
        and renders a depth map.

        Args:
            mesh_path:        Path to the .obj file.
            intrinsic:       3x3 camera intrinsic matrix (numpy array).
                            [[fx,  0, cx],
                            [ 0, fy, cy],
                            [ 0,  0,  1]]
            pose:           4x4 passive camera-to-world pose matrix (in cam coordinates).
                            Transforms points from camera coords into world coords.
            width:           Width of the rendered image in pixels.
            height:          Height of the rendered image in pixels.

        Returns:
            A (heightxwidth) numpy float32 array representing per-pixel depth
            in the camera's view (in the same units as your mesh).
        """
        # 1. Load mesh
        mesh = trimesh.load(mesh_path, force='mesh')
        render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

        # 2. Build scene and add mesh
        scene = pyrender.Scene()
        scene.add(render_mesh)

        # 3. Create camera
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

        # Note: pyrender expects camera pose as a 4×4 world‐to‐camera transform.
        
        # Going from OpenCV to OpenGL camera coordinates
        # Source: https://github.com/mmatl/pyrender/issues/228
        opengl_pose = pose.copy()
        opengl_pose[[1, 2]] *= -1

        # Going from passive world2cam to passive cam2world
        cam_pose = np.linalg.inv(opengl_pose)

        scene.add(camera, pose=cam_pose)

        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                            innerConeAngle=np.pi/16.0,
                            outerConeAngle=np.pi/6.0)

        scene.add(light, pose=cam_pose)

        # 4. Offscreen renderer
        r = pyrender.OffscreenRenderer(viewport_width=width,
                                    viewport_height=height,
                                    point_size=1.0)

        # 5. Render depth only
        depth = r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        r.delete()  

        # depth is a float32 array (height×width) with z in camera coords
        return depth