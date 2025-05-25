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
from cv_utils.core import invert_pose, backproject
import datetime


class Renderer:

    def __init__(self,
                 start_pose: str = None,
                 end_pose: str = None,
                 bg_video_filepath: str = None,
                 mesh_filepath: str = None,
                 K: np.ndarray = None):
        self.video_filepath = bg_video_filepath
        self.K = K
        self.width = int(K[0,2] * 2)
        self.height = int(K[1,2] * 2)

        self.object, extension = os.path.splitext(os.path.basename(mesh_filepath))
        self.mesh_filepath = mesh_filepath

        self.start_pose = start_pose
        self.end_pose = end_pose

        self.media_dirpath = r"D:\DATASETS\RENDERED\CV_utils"
        self.images_dirpath = os.path.join(self.media_dirpath, "images")
        self.labels_dirpath = os.path.join(self.media_dirpath, "labels")
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
        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.width,
                                                   viewport_height=self.height,
                                                   point_size=1.0)
        # =================================================

        #start_pose_world = invert_pose(start_pose)
        #end_pose_world = invert_pose(end_pose)
        start_point = start_pose[:3,3]
        end_point = end_pose[:3,3] 

        self.traj = self.random_spline_trajectory(start_point, end_point, plot=True)
        #self.traj = self.random_spline_trajectory_in_viewport(start_pose,plot=True)

        self.render_scene(bg_video=True)

    def render_scene(self, 
                     fps: float = 30.0, 
                     video_name: str = "out.mp4", 
                     bg_video: bool = False,
                     draw_bbox: bool = True):
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

        num_points = self.traj.shape[0]
        self.image_filepaths = []
        self.poses = np.zeros((num_points,4,4))
        self.bboxs = np.zeros((num_points,4))

        for ix, point in enumerate(self.traj):
            pose[:3, 3] = point

            img, mask, bbox = self._render_image(pose)
            self.poses[ix,...] = pose
            self.bboxs[ix,...] = bbox

            # Optionally add in BG using mask
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

            # Optionally draw the bounding box on the image
            if draw_bbox and bbox is not None:
                x, y, w, h = bbox
                bbox_color = (0, 0, 255)  # Red box (BGR)
                thickness = 10
                bbox_color = tuple(int(c) for c in bbox_color)
                bbox_img = img.copy()
                cv2.rectangle(bbox_img, (x, y), (x + w, y + h), bbox_color, thickness)
                img = bbox_img

            image_filepath = os.path.join(self.images_dirpath, f"img{ix:05d}.png")
            success = cv2.imwrite(image_filepath, img)
            if not success:
                raise RuntimeError(f"cv2.imwrite failed for {image_filepath!r}: "
                                f"dtype={img.dtype}, shape={img.shape}")
            else:
                print(f"Wrote image #{ix} at {image_filepath}.")
            self.image_filepaths.append(image_filepath)

        cap.release()

        # 2) Gather the written frame files in order
        pattern = os.path.join(self.images_dirpath, "img*.png")
        frame_files = sorted(glob(pattern))

        if not frame_files:
            raise RuntimeError(f"No frames found with pattern {pattern}")

        # 3) Read the first frame to get size
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            raise IOError(f"Failed to reread just-written {image_filepath!r}")
        height, width = first_frame.shape[:2]

        # 4) Set up VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or "XVID", "H264", etc.
        video_path = os.path.join(self.videos_dirpath, video_name)
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # 5) Write each frame into the video
        for fpath in frame_files:
            frame = cv2.imread(fpath)
            if frame is None:
                raise IOError(f"Failed to re-read written {fpath!r}")
            writer.write(frame)

        # 6) Finalize
        writer.release()
        print(f"Video saved to {video_path}")

        self.has_sequence = True

        # 7) Write GT to COCO label file
        self.save_to_coco()

    def _render_image(self, pose: np.ndarray, draw_bbox: bool = True) -> np.ndarray:
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
        color_img, depth = self.renderer.render(self.scene)

        # 3: Create a mask: valid where depth > 0 (foreground)
        mask = (depth > 0).astype(bool)

        # 4: Get bounding box
        # Find contours and get bounding box of largest object
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            bbox = (x, y, w, h)
        else:
            bbox = None

        return color_img, mask, bbox
    
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
            plotlyPlot(fig, auto_open=True, filename=r"media\plots\temp-plot.html")

        return traj 
    
    def random_spline_trajectory_in_viewport(
        self,
        pose: np.ndarray,
        max_len: float = 100,
        near: float = 0.001,
        far: float = 100,
        fineness: int = 200,
        num_ctrl: int = 3,
        seed: int = None,
        plot: bool = False
    ) -> np.ndarray:
        """
        Generate a smooth random trajectory inside the camera's view frustum.

        Args:
            pose      : 4x4 cam→world transform.
            K         : 3x3 camera intrinsics.
            width     : Image width in px.
            height    : Image height in px.
            near, far : Distances for near/far planes.
            max_len   : Max allowed distance between endpoints.
            fineness  : Number of samples along the spline.
            num_ctrl  : Number of random interior control points.
            seed      : RNG seed.
            plot      : If True, show Plotly 3D scene.

        Returns:
            traj : (fineness,3) array of world-space points.
        """
        if seed is not None:
            np.random.seed(seed)

        # Decompose pose and intrinsics
        R = pose[:3, :3]
        t = pose[:3,  3]
        K = self.K
        width = self.width
        height = self.height

        # 1) Compute the 8 corners of the view frustum in WORLD space
        corners_px = np.array([[0,0],[width,0],[width,height],[0,height]])
        
        corners_cam = np.vstack([
            [*backproject(u,v,near,K)] for (u,v) in corners_px
        ] + [
            [*backproject(u,v,far,K)]  for (u,v) in corners_px
        ])  # shape (8,3)
        corners_world = (R @ corners_cam.T).T + t  # (8,3)

        # helper: sample 1 random point inside frustum
        def sample_point():
            Z = np.random.uniform(near, far)
            u = np.random.uniform(0, width)
            v = np.random.uniform(0, height)
            cam_pt = backproject(u, v, Z, K)
            return R @ cam_pt + t

        # 2) Pick endpoints A,B with ||A-B|| <= max_len
        # TODO LOL make this more efficient
        for _ in range(1000):
            A = sample_point()
            B = sample_point()
            if np.linalg.norm(A-B) <= max_len:
                break
        else:
            raise RuntimeError("Couldn't sample endpoints within max_len")

        # 3) Build control points (including A,B)
        control_pts = [A] + [sample_point() for _ in range(num_ctrl)] + [B]
        ctrl_arr = np.vstack(control_pts).T  # shape (3, K)

        # 4) Fit & sample a cubic B‐spline
        tck, _ = splprep(ctrl_arr, s=0, k=min(3, ctrl_arr.shape[1]-1))
        u_fine = np.linspace(0, 1, fineness)
        pts = splev(u_fine, tck)
        traj = np.vstack(pts).T   # (fineness,3)

        # 5) Optional Plotly visualization
        if plot:
            fig = go.Figure()

            # ——— Frustum edges ———
            # near‐plane loop
            for i in range(4):
                fig.add_trace(go.Scatter3d(
                    x=[corners_world[i,0], corners_world[(i+1)%4,0]],
                    y=[corners_world[i,1], corners_world[(i+1)%4,1]],
                    z=[corners_world[i,2], corners_world[(i+1)%4,2]],
                    mode='lines', line=dict(color='gray'), showlegend=False
                ))
            # far‐plane loop
            for i in range(4, 8):
                j = 4 + (i+1-4)%4
                fig.add_trace(go.Scatter3d(
                    x=[corners_world[i,0], corners_world[j,0]],
                    y=[corners_world[i,1], corners_world[j,1]],
                    z=[corners_world[i,2], corners_world[j,2]],
                    mode='lines', line=dict(color='gray'), showlegend=False
                ))
            # connect near→far
            for i in range(4):
                fig.add_trace(go.Scatter3d(
                    x=[corners_world[i,0], corners_world[i+4,0]],
                    y=[corners_world[i,1], corners_world[i+4,1]],
                    z=[corners_world[i,2], corners_world[i+4,2]],
                    mode='lines', line=dict(color='gray'), showlegend=False
                ))

            # ——— Trajectory ———
            fig.add_trace(go.Scatter3d(
                x=traj[:,0], y=traj[:,1], z=traj[:,2],
                mode='lines', line=dict(width=4, color='blue'),
                name='Trajectory'
            ))

            # ——— Control points ———
            cp = np.vstack(control_pts)
            fig.add_trace(go.Scatter3d(
                x=cp[:,0], y=cp[:,1], z=cp[:,2],
                mode='markers', marker=dict(size=6, color='red'),
                name='Control Points'
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                    aspectmode='data'
                ),
                title="Random Spline Trajectory in Camera Frustum"
            )
            fig.show()

        return traj
    
    def save_to_coco(self):
        """
        Saves pose labels for images in coco format.
        """
        current_year = datetime.datetime.now().year
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        coco_json = {
            "info": None,
            "categories": None,
            "images": None,
            "annotations": None,
            "licenses": None
        }
        coco_json["info"] = {
            "year": current_year,
            "version": "1.0.0",
            "description": f"COCO labels for rendered dataset of object {self.object}.",
            "contributor": "Marius Neuhalfen",
            "url": "https://marius-ne.github.io/",
            "date_created": current_date
        }
        # TODO: add support for multiple objects in frame
        coco_json["categories"] = [
            {
                "id": 0,
                "name": self.object
            }
        ]
        coco_json["licenses"] = [
            {
                "id": "3.0",
                "name": "GPL",
                "url": "https://www.gnu.org/licenses/gpl-3.0.en.html"
            }
        ]
        if not hasattr(self,"has_sequence") or not self.has_sequence:
            raise ValueError("Cannot write labels to COCO file if " \
                "no self.images or self.labels are present.")
        
        # Create annotations and image list
        annotations = []
        image_dicts = []
        for (image_filepath, pose, bbox) in zip(self.image_filepaths, self.poses, self.bboxs):
            # Image
            image_id_text, extension = os.path.splitext(os.path.basename(image_filepath))
            image_id = int(''.join(char for char in image_id_text if char.isdigit()))
            image_dict = {
                "id": image_id,
                "file_name": image_filepath,
                "width": self.width,
                "height": self.height,
                "date_captured": current_date
            }
            image_dicts.append(image_dict)
        
            # Annotation
            q = Rotation.from_matrix(pose[:3,:3]).as_quat()
            t = pose[:3,3]
            area = bbox[2] * bbox[3]
            annotation = {
                "id": image_id,
                "image_id": image_id,
                "image_filepath": image_filepath,
                "q_world2cam_passive": q.tolist(),
                "t_world2cam_passive": t.tolist(),
                "bbox": bbox.tolist(),
                "area": float(area),
                "categories": 0
            }
            annotations.append(annotation)

        coco_json["images"] = image_dicts
        coco_json["annotations"] = annotations

        # Write COCO JSON to file
        coco_filepath = os.path.join(self.labels_dirpath, f"{self.object}_coco.json")
        os.makedirs(self.labels_dirpath,exist_ok=True)
        with open(coco_filepath, "w") as f:
            json.dump(coco_json, f, indent=4)
        print(f"COCO annotations saved to {coco_filepath}")
    
if __name__ == "__main__":
    #base_path_airbus = r"E:\ESA\VBN_DataSets\Data\AIRBUS_GSTP\MAN-DATA-L1"
    #mesh_path = r"C:\Users\meneu\Documents\Projekte\PoseSampler\assets\envisat.glb"
    mesh_path = r"C:\Users\meneu\Documents\Projekte\PoseSampler\assets\fpv_drone.glb"
    #camera_filepath = os.path.join(base_path_airbus, 'metadata', 'camera.json')
    camera_filepath = r"C:\Users\meneu\Documents\Projekte\CV_utils\assets\camera.json"
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
    end_t = np.array([0, 0, 10])
    pose1 = np.zeros(shape=(4,4))
    pose2 = np.zeros(shape=(4,4))
    pose1[:3,:3] = R
    pose2[:3,:3] = R
    pose1[:3,3] = start_t
    pose2[:3,3] = end_t
    pose1[3,3] = 1
    pose2[3,3] = 1

    bg_video_path = r"D:\DATASETS\RENDERED\CV_utils\bg_videos\forest_cinematic.mp4"

    Renderer(pose1, pose2, None, mesh_path, K)