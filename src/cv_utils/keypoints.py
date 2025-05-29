import trimesh
import numpy as np
import json
import os

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
