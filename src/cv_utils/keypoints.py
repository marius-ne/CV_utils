import trimesh
import numpy as np
import json
import os
import cv2



def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Standard farthest point sampling among a set of points.

    Parameters:
        points (np.ndarray): (M, 3) array of candidate point coordinates.
        num_samples (int): Number of points to pick.

    Returns:
        np.ndarray: (num_samples, 3) array of selected coordinates.
    """
    M = points.shape[0]
    if num_samples >= M:
        # If requested more points than exist, just return all (or shuffle)
        return points.copy()

    # Start with a random index
    selected_indices = [np.random.randint(M)]
    
    # Precompute pairwise distances for efficiency (M×M)
    # If M is very large, you could do this blockwise or compute distances on the fly.
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    
    for _ in range(1, num_samples):
        # For each candidate, find distance to closest already-selected point
        min_distances = np.min(dists[:, selected_indices], axis=1)
        # Pick the candidate whose min-distance is largest
        next_idx = np.argmax(min_distances)
        selected_indices.append(int(next_idx))
    
    return points[selected_indices]


def sample_interior_points(mesh: trimesh.Trimesh, num_candidates: int) -> np.ndarray:
    """
    Generate a set of points uniformly at random in the bounding box of `mesh`,
    keep only those that lie strictly inside (via mesh.contains), until we have
    num_candidates of them.

    Parameters:
        mesh (trimesh.Trimesh): 3D mesh (closed manifold).
        num_candidates (int): how many interior points to collect.

    Returns:
        np.ndarray: (num_candidates, 3) array of interior points.
    """
    # 1. Compute axis-aligned bounding box
    bounds_min, bounds_max = mesh.bounds  # both shape (3,)
    interior_pts = []

    # We'll do rejection sampling in batches of, say, 5× the remainder each loop.
    # This is just to avoid one‐point‐at‐a‐time calls to mesh.contains.
    batch_size = max(num_candidates * 5, 1000)

    while len(interior_pts) < num_candidates:
        # Sample `batch_size` random points in the AABB
        pts = np.random.rand(batch_size, 3)  # uniform in [0,1]
        # Scale to actual bounding box
        pts = pts * (bounds_max - bounds_min)[None, :] + bounds_min[None, :]
        
        # Test which are inside the mesh (mesh.contains returns a boolean mask)
        mask = mesh.contains(pts)  # shape (batch_size,)
        
        inside = pts[mask]
        if inside.shape[0] > 0:
            interior_pts.append(inside)
        
        # If we still don't have enough, loop again
    interior_pts = np.vstack(interior_pts)
    # Truncate to exactly num_candidates
    return interior_pts[:num_candidates]


def select_farthest_keypoints_from_obj(obj_path: str,
                                       num_keypoints: int = 10,
                                       candidate_multiplier: int = 10) -> np.ndarray:
    """
    Selects `num_keypoints` points _inside_ a 3D mesh (.obj), maximizing mutual distance.

    The procedure:
      1. Load mesh.
      2. Generate a pool of interior‐only candidate points via rejection sampling.
      3. Run farthest‐point sampling on that pool to pick the final keypoints.

    Parameters:
        obj_path (str): path to the .obj file (must be a watertight/closed mesh).
        num_keypoints (int): how many interior keypoints to output.
        candidate_multiplier (int):
            We will generate (num_keypoints * candidate_multiplier) interior candidates,
            then pick `num_keypoints` via FPS. If your mesh is very skinny or complicated,
            you might need to raise this multiplier so rejection sampling actually
            yields enough points.

    Returns:
        np.ndarray: (num_keypoints, 3) array of keypoint coordinates (all inside).
    """
    # 1. Load the mesh
    mesh = trimesh.load(obj_path)
    if not mesh.is_watertight:
        print("Warning: mesh is not watertight. 'contains' may be unreliable.")
    print(f"Loaded mesh '{obj_path}' with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")

    # 2. Sample a larger pool of interior points
    num_candidates = num_keypoints * candidate_multiplier
    interior_points = sample_interior_points(mesh, num_candidates)
    print(f"  → Collected {interior_points.shape[0]} interior candidate points "
          f"(requested {num_candidates}).")

    # 3. Farthest‐point sample among just those interior points
    keypoints = farthest_point_sampling(interior_points, num_keypoints)

    print(f"Selected {num_keypoints} farthest-distributed keypoints inside the mesh.")
    return keypoints


def add_visibility_to_keypoints(keypoints_2d, image_size):
    """
    Adds a third column to 2D keypoints indicating if each keypoint is inside the image frame.

    Parameters:
        keypoints_2d (np.ndarray): (N, 2) array of 2D keypoint coordinates (x, y).
        image_size (tuple): (width, height) of the image.

    Returns:
        np.ndarray: (N, 3) array where the third column is 1 if inside frame, 0 otherwise.
    """
    width, height = image_size
    x_in = (keypoints_2d[:, 0] >= 0) & (keypoints_2d[:, 0] < width)
    y_in = (keypoints_2d[:, 1] >= 0) & (keypoints_2d[:, 1] < height)
    in_frame = (x_in & y_in).astype(int)
    return np.hstack([keypoints_2d, in_frame[:, None]])

def add_coco_visibility_to_keypoints(
    keypoints_2d: np.ndarray,
    keypoints_3d: np.ndarray,
    depth_map: np.ndarray,
    image_size: tuple,
    occlusion_tol: float = 0.05
) -> np.ndarray:
    """
    Adds a third column to 2D keypoints indicating visibility according to COCO convention:
      - 2 = visible
      - 1 = occluded (behind depth map beyond tolerance)
      - 0 = out of frame

    A 3D keypoint is considered visible if:
      depth_3d <= depth_map[pixel] + tolerance,
    where tolerance = occlusion_tol * (depth_map.max() - depth_map.min()).

    Parameters:
        keypoints_2d (np.ndarray): (N, 2) array of 2D keypoint coordinates (x, y).
        keypoints_3d (np.ndarray): (N, 3) array of 3D keypoint coordinates (X, Y, Z),
                                   where Z is the depth in the same units as depth_map.
        depth_map (np.ndarray): (H, W) array of depth values per pixel.
        image_size (tuple): (width, height) of the image (W, H).
        occlusion_tol (float): Fraction of (max_depth - min_depth) used as tolerance.

    Returns:
        np.ndarray: (N, 3) array where each row is [x, y, v], and
                    v ∈ {0, 1, 2} per COCO:
                      0 = out of frame,
                      1 = occluded,
                      2 = visible.
    """
    width, height = image_size
    N = keypoints_2d.shape[0]
    assert keypoints_3d.shape[0] == N, "keypoints_2d and keypoints_3d must have the same length"
    H, W = depth_map.shape

    # Precompute depth range and tolerance
    depth_min = np.nanmin(depth_map)
    depth_max = np.nanmax(depth_map)
    depth_tol = occlusion_tol * (depth_max - depth_min)

    visibilities = np.zeros((N,), dtype=int)

    for i in range(N):
        x_float, y_float = keypoints_2d[i]
        z3d = keypoints_3d[i, 2]

        # Check if (x, y) is inside the image frame
        if x_float < 0 or x_float >= width or y_float < 0 or y_float >= height:
            visibilities[i] = 0
            continue

        # Round to nearest integer pixel indices
        x_pixel = int(round(x_float))
        y_pixel = int(round(y_float))

        # Clamp to valid pixel range in case rounding pushes to edge
        x_pixel = min(max(x_pixel, 0), W - 1)
        y_pixel = min(max(y_pixel, 0), H - 1)

        depth_at_pixel = depth_map[y_pixel, x_pixel]

        # If depth_at_pixel is NaN or zero, we cannot compare meaningfully → mark as occluded
        if np.isnan(depth_at_pixel) or depth_at_pixel <= 0:
            visibilities[i] = 1
            continue

        # Compare 3D keypoint depth to depth map value + tolerance
        if z3d <= depth_at_pixel + depth_tol:
            visibilities[i] = 2  # visible
        else:
            visibilities[i] = 1  # occluded

    # Stack to (N, 3): [x, y, v]
    return np.concatenate([keypoints_2d, visibilities[:, None]], axis=1)    

def draw_kps_on_img(img,points2D):
    """
    Draws keypoints onto image including numbered labels.

    Drawing is done INPLACE! So directly supply output image.

    Args:
        points2D (np.ndarray): Either keypoints with or without visibility 
            with shape [N,2] or [N,3].
    """
    # If the last dimension is 2 then we add visibility
    if points2D.shape[-1] == 2:
        h,w = img.shape[:2]
        points2D = keypoints.add_visibility_to_keypoints(points2D,(w,h))
        
    for idx, kp in enumerate(points2D):
        x, y, visible = kp[:3]
        if visible:
            # Draw a larger, slightly darker magenta circle
            cv2.circle(img, (int(x), int(y)), 12, (180, 0, 180), -1)
            # Put the label centered within the circle
            text = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = int(x) - text_size[0] // 2
            text_y = int(y) + text_size[1] // 2
            cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    obj_file = r"E:\ESA\VBN_DataSets\Data\AIRBUS_GSTP\MAN-DATA-L1\metadata\model.obj"
    N = 12  # Number of keypoints to select

    keypoints = select_farthest_keypoints_from_obj(obj_file, num_keypoints=N)
    print("Farthest-distributed keypoints:\n", keypoints)

    # Save keypoints to a JSON file 
    keypoints_list = keypoints.tolist()
    model_name = os.path.basename(obj_file)
    model_name, extension = os.path.splitext(model_name)
    output_path = f"assets/keypoints/{model_name}_keypoints.json"
    with open(output_path, "w") as f:
        json.dump(keypoints_list, f, indent=2)
    print(f"Keypoints saved to {output_path}")


