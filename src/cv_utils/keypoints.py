import trimesh
import numpy as np
import json
import os
import cv2

def farthest_point_sampling(vertices, num_samples):
    """
    Selects num_samples vertices from the input vertices array
    so that they are as far apart from each other as possible.
    
    Parameters:
        vertices (np.ndarray): (N, 3) array of vertex coordinates.
        num_samples (int): Number of points to select.
    
    Returns:
        np.ndarray: (num_samples, 3) array of selected vertex coordinates.
    """
    N = len(vertices)
    
    # Start with a random point
    selected_indices = [np.random.randint(N)]
    
    # Precompute distance matrix for efficiency
    dists = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=2)
    
    for _ in range(1, num_samples):
        # Compute the distance from each vertex to the closest selected point
        min_distances = np.min(dists[:, selected_indices], axis=1)
        # Select the point that is farthest from the already selected set
        next_index = np.argmax(min_distances)
        selected_indices.append(next_index)
    
    return vertices[selected_indices]

def select_farthest_keypoints_from_obj(obj_path, num_keypoints=10):
    """
    Selects N keypoints from a 3D mesh loaded from an OBJ file, so that
    they are as far from each other as possible.
    
    Parameters:
        obj_path (str): Path to the .obj file.
        num_keypoints (int): Number of keypoints to select.

    Returns:
        np.ndarray: (num_keypoints, 3) array of keypoint coordinates.
    """
    mesh = trimesh.load(obj_path)
    vertices = mesh.vertices
    print(f"Loaded mesh with {len(vertices)} vertices.")

    # Farthest Point Sampling
    keypoints = farthest_point_sampling(vertices, num_keypoints)
    
    print(f"Selected {num_keypoints} farthest-distributed keypoints.")
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
    obj_file = r"E:\ESA\VBN_DataSets\Data\SHIRT\models\TangoV12.obj" 
    N = 10  # Number of keypoints to select

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


