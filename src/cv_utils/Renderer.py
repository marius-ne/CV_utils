import os
import json

import numpy as np
import cv2
import pyrender
import trimesh

from glob import glob

import plotly.graph_objs as go
from plotly.offline import plot as plotlyPlot

from typing import Tuple, Sequence
from scipy.spatial.transform import Rotation
from scipy.interpolate import splprep, splev

# Our own imports
from cv_utils.core import invert_pose


class Renderer:

    def __init__(self,
                 start_pose: str = None,
                 end_pose: str = None,
                 bg_video_filepath: str = None,
                 mesh_filepath: str = None,
                 K: np.ndarray = None):
        self.video_filepath = bg_video_filepath
        self.K = K
        width = K[0,2] * 2
        height = K[1,2] * 2

        self.start_pose = start_pose
        self.end_pose = end_pose

        self.media_dirpath = r"D:\DATASETS\RENDERED\CV_utils"
        self.images_dirpath = os.path.join(self.media_dirpath, "images")
        self.videos_dirpath = os.path.join(self.media_dirpath, "videos")

        os.makedirs(self.images_dirpath, exist_ok=True)
        os.makedirs(self.videos_dirpath, exist_ok=True)

        self.bg_video_path = bg_video_filepath
        
        # Prepare renderer ================================
        # 1.) Load mesh
        self.mesh = trimesh.load(mesh_filepath, force='mesh')
        self.render_mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=False)
        print("Loaded mesh.")
        # 2) Build scene and add mesh
        self.scene = pyrender.Scene()
        self.scene.add(self.render_mesh)

        # 3) Create (and add) camera node once
        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]
        self.camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        # placeholder pose; we’ll override it in render_image()
        self.camera_node = self.scene.add(self.camera, pose=np.eye(4))

        # 4) Create (and add) light node once
        light_color = np.ones(3)
        light_intensity = 10
        self.light = pyrender.SpotLight(color=light_color,
                                        intensity=light_intensity,
                                        innerConeAngle=np.pi/16.0,
                                        outerConeAngle=np.pi/6.0)
        self.light_node = self.scene.add(self.light, pose=np.eye(4))

        # 5) Offscreen renderer once
        self.renderer = pyrender.OffscreenRenderer(viewport_width=width,
                                                   viewport_height=height,
                                                   point_size=1.0)
        # =================================================

        #start_pose_world = invert_pose(start_pose)
        #end_pose_world = invert_pose(end_pose)
        start_point = start_pose[:3,3]
        end_point = end_pose[:3,3] 

        self.traj = self.random_spline_trajectory(start_point, end_point, plot=False)

        self.render_scene(bg_video=True)

    def render_scene(self, fps: float = 30.0, video_name: str = "out.mp4", bg_video = False):
        """
        Renders video by first writing frame PNGs, then packing them into an MP4.

        Args:
            fps:        Frames per second for the output video.
            video_name: Filename (in image_dirpath) for the final video.
        """
        pose = self.start_pose.copy()

        if bg_video:
            cap = cv2.VideoCapture(self.bg_video_path)

        # 1) Write out each frame
        for ix, point in enumerate(self.traj):
            pose[:3, 3] = point
            img, mask = self._render_image(pose)
            if bg_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, ix)
                ret, bg_frame = cap.read()
                if ret:
                    bg_frame = cv2.resize(bg_frame, (img.shape[1], img.shape[0]))
                else:
                    bg_frame = np.zeros_like(img) 
                # Expand mask to 3 channels
                mask_3c = np.repeat(mask[:, :, None], img.shape[2], axis=2)     
                composite = img.copy()

                composite[~mask_3c] = bg_frame[~mask_3c]
                img = composite


            image_filepath = os.path.join(self.images_dirpath, f"img{ix:05d}.png")
            cv2.imwrite(image_filepath, img)
            print(f"Wrote image #{ix} at {image_filepath}.")

        cap.release()

        # 2) Gather the written frame files in order
        pattern = os.path.join(self.images_dirpath, "img*.png")
        frame_files = sorted(glob(pattern))

        if not frame_files:
            raise RuntimeError(f"No frames found with pattern {pattern}")

        # 3) Read the first frame to get size
        first_frame = cv2.imread(frame_files[0])
        height, width = first_frame.shape[:2]

        # 4) Set up VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or "XVID", "H264", etc.
        video_path = os.path.join(self.videos_dirpath, video_name)
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # 5) Write each frame into the video
        for fpath in frame_files:
            frame = cv2.imread(fpath)
            writer.write(frame)

        # 6) Finalize
        writer.release()
        print(f"Video saved to {video_path}")

    def _render_image(self, pose: np.ndarray) -> np.ndarray:
        """
        Given a new 4x4 world→camera passive pose, update camera & light,
        render, and return the color image.
        """
        # Convert OpenCV→OpenGL and passive world→cam→cam→world as before
        opengl_pose = pose.copy()
        opengl_pose[[1,2]] *= -1
        cam_pose = np.linalg.inv(opengl_pose)

        # 1) Update camera & light poses in the existing scene
        self.scene.set_pose(self.camera_node, pose=cam_pose)
        self.scene.set_pose(self.light_node,  pose=cam_pose)

        # 2) Render
        color, depth = self.renderer.render(self.scene)

        # 3: Create a mask: valid where depth > 0 (foreground)
        mask = (depth > 0).astype(bool)

        return color, mask
    
    def __del__(self):
        # Clean up OpenGL context
        self.renderer.delete()
    
    def random_spline_trajectory(
            self,
            A: np.ndarray,
            B: np.ndarray,
            fineness: int = None,
            num_ctrl: int = 3,
            seed: int = None,
            plot: bool = False
        ) -> np.ndarray:
        """
        Generate a smooth random trajectory from A to B, contained in the sphere
        for which A and B are antipodes, by fitting a cubic B-spline to random
        interior control points.

        Args:
            A, B        : array-like, shape (3,). Start & end points (antipodes).
            fineness    : int. Number of samples along the spline (including endpoints).
            num_ctrl    : int. Number of random interior control points.
            seed        : Optional RNG seed for reproducibility.

        Returns:
            trajectory  : ndarray, shape (fineness, 3).  
                        A smooth path from A → B within the sphere.
        """
        A = np.array(A, float)
        B = np.array(B, float)
        C = 0.5 * (A + B)               # sphere center
        R = np.linalg.norm(A - C)       # sphere radius

        if fineness is None:
            # Compute based on distance between points, goal: 1 image per 10 centimeters
            fineness = np.ceil(np.linalg.norm(A - B)*10).astype(np.int32)

        if seed is not None:
            np.random.seed(seed)

        # 1) Build control points: include endpoints + random interior pts
        ctrl_params = np.linspace(0, 1, num_ctrl + 2)
        control_pts = [None] * (num_ctrl + 2)
        control_pts[0] = A
        control_pts[-1] = B

        for i, t in enumerate(ctrl_params[1:-1], start=1):
            # straight‐line interp
            L = (1 - t) * A + t * B
            # how far L is from the center
            d_lin = np.linalg.norm(L - C)
            # max possible radial offset
            d_max = np.sqrt(max(R * R - d_lin * d_lin, 0.0))
            # random direction
            v = np.random.normal(size=3)
            v /= np.linalg.norm(v)
            # random magnitude in [0, d_max]
            offset = v * np.random.uniform(0, d_max)
            control_pts[i] = L + offset

        ctrl_arr = np.vstack(control_pts).T  # shape (3, K)

        # 2) Fit a parametric B-spline of degree k=3
        #    s=0 forces interpolation through all ctrl points
        tck, u = splprep(ctrl_arr, s=0, k=min(3, ctrl_arr.shape[1] - 1))

        # 3) Sample along the spline
        u_fine = np.linspace(0, 1, fineness)
        spline_pts = splev(u_fine, tck)      # list of 3 arrays, each length=fineness
        traj =  np.vstack(spline_pts).T

        # 4) Optional Plotly visualization
        if plot:
            # Create a meshgrid of spherical angles
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            uu, vv = np.meshgrid(u, v)
            xs = C[0] + R * np.cos(uu) * np.sin(vv)
            ys = C[1] + R * np.sin(uu) * np.sin(vv)
            zs = C[2] + R * np.cos(vv)

            # Build the Plotly figure
            fig = go.Figure()

            # 1) Sphere
            fig.add_trace(go.Mesh3d(
                x=xs.flatten(),
                y=ys.flatten(),
                z=zs.flatten(),
                alphahull=0,
                opacity=0.2,
                color='lightgrey',
                showscale=False,
                name='Enclosing Sphere'
            ))

            # 2) Trajectory
            traj_x, traj_y, traj_z = traj[:,0], traj[:,1], traj[:,2]
            fig.add_trace(go.Scatter3d(
                x=traj_x, y=traj_y, z=traj_z,
                mode='lines',
                line=dict(width=4, color='blue'),
                name='Trajectory'
            ))

            # 3) Control points
            ctrl_arr = np.vstack(control_pts)
            fig.add_trace(go.Scatter3d(
                x=ctrl_arr[:,0], y=ctrl_arr[:,1], z=ctrl_arr[:,2],
                mode='markers',
                marker=dict(size=6, color='red'),
                name='Control Points'
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                title="Random Spline Trajectory in Enclosing Sphere"
            )
            plotlyPlot(fig, auto_open=True)

        return traj 
    
if __name__ == "__main__":
    #base_path_airbus = r"E:\ESA\VBN_DataSets\Data\AIRBUS_GSTP\MAN-DATA-L1"
    #mesh_path = r"C:\Users\meneu\Documents\Projekte\PoseSampler\assets\envisat.glb"
    mesh_path = r"C:\Users\meneu\Documents\Projekte\PoseSampler\assets\fpv_drone.glb"
    #camera_filepath = os.path.join(base_path_airbus, 'metadata', 'camera.json')
    camera_filepath = r"C:\Users\meneu\Documents\Projekte\CV_utils\camera.json"
    with open(camera_filepath, 'r') as f:
        camera_dict = json.load(f)
    
    fx = camera_dict["focalLengthX"]  # focal length[m]
    fy = camera_dict["focalLengthY"]  # focal length[m]
    cx = camera_dict["principalPointX"]
    cy = camera_dict["principalPointY"]
    k = [[fx,  0, cx],
            [0,  fy, cy],
            [0,  0,  1]]
    K = np.array(k)

    R = np.eye(3)
    start_t = np.array([0, 0, 1])
    end_t = np.array([0, 0, 20])
    pose1 = np.zeros(shape=(4,4))
    pose2 = np.zeros(shape=(4,4))
    pose1[:3,:3] = R
    pose2[:3,:3] = R
    pose1[:3,3] = start_t
    pose2[:3,3] = end_t
    pose1[3,3] = 1
    pose2[3,3] = 1

    bg_video_path = r"D:\DATASETS\RENDERED\CV_utils\bg_videos\forest_cinematic.mp4"

    Renderer(pose1, pose2, bg_video_path, mesh_path, K)