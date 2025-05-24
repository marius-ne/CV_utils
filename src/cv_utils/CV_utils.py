import numpy as np
import cv2
import pyrender
import trimesh

from typing import Tuple, Sequence
from scipy.spatial.transform import Rotation

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

    # Note: pyrender expects camera pose as a 4x4 world‐to‐camera transform.
    
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

    # depth is a float32 array (heightxwidth) with z in camera coords
    return depth

def depth_estimate(
    self,
    pose: np.ndarray,
    K: np.ndarray,
    center: Tuple[float, float],
    diag: float,
    physical_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate a 3D point and its uncertainty along the viewing ray,
    using the full passive world→camera pose (R, t).

    Args:
        pose:   4x4 cam2world passive transformation matrix.
        K:      camera intrinsics matrix.
        center: (x, y) pixel coordinates of the feature.
        diag:   Observed feature diagonal length in pixels.
        physical_size: Physical size of the object in meters.

    Returns:
        mu_world:    (3,) Estimated point in world coordinates.
        sigma_world: (3,) World-space 1σ uncertainty along the viewing ray.
    """
    # 1) Intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    f_mean = 0.5 * (fx + fy)

    # 2) Depth and its std (10% relative error)
    Z = f_mean * physical_size / diag
    Z_std = 0.1 * Z

    # 3) Back‐project pixel into camera‐frame point
    x, y = center
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    cam_pt = np.array([X, Y, Z])

    # 4) Recover world‐space point:
    #    world_pt satisfies R @ world_pt + t = cam_pt
    #   The pose is passive, therefore the points get transformed
    #   from coordinate frames but stay in place.
    R = pose[:3,:3]
    t = pose[:3,3]
    mu_world = R.T @ (cam_pt - t)

    # 5) Build unit viewing‐ray in camera frame
    ray_cam = np.array([(x - cx) / fx, (y - cy) / fy, 1.0])
    ray_cam /= np.linalg.norm(ray_cam)

    # 6) Rotate ray into world frame (translation drops out)
    ray_world = R.T @ ray_cam
    ray_world /= np.linalg.norm(ray_world)

    # 7) Scale by depth‐std to get world‐space σ
    sigma_world = ray_world * Z_std

    return mu_world, sigma_world

def triangulate(
    pt1: Sequence[float],
    pt2: Sequence[float],
    K1: np.ndarray,
    pose1: np.ndarray,
    K2: np.ndarray,
    pose2: np.ndarray
) -> np.ndarray:
    """
    Triangulate a 3D point from two image observations given camera intrinsics and extrinsics.

    Args:
        pt1: (x, y) pixel in image 1.
        pt2: (x, y) pixel in image 2.
        K1: (3x3) camera intrinsic matrix for camera 1.
        pose1: (4x4) passive world2cam
        K2: (3x3) camera intrinsic matrix for camera 2.
        pose2: (4x4) passive world2cam

    Returns:
        (3,) Euclidean 3D point in world coordinates.
    """
    # Get Rs and ts
    R1 = pose1[:3,:3]
    t1 = pose1[:3,3]
    R2 = pose2[:3,:3]
    t2 = pose2[:3,3]
    
    # Build the 3x4 projection matrices
    P1 = K1 @ np.hstack((R1, t1.reshape(3, 1)))
    P2 = K2 @ np.hstack((R2, t2.reshape(3, 1)))

    # Convert points to 2xN shape (N=1 here)
    pts1 = np.asarray(pt1, float).reshape(2, 1)
    pts2 = np.asarray(pt2, float).reshape(2, 1)

    # Perform triangulation (results in 4xN homogeneous points)
    hom_pts = cv2.triangulatePoints(P1, P2, pts1, pts2)

    # Convert homogeneous to Euclidean coordinates
    euclid = (hom_pts[:3] / hom_pts[3]).reshape(3)
    return euclid


