import numpy as np
import glob
import os
import open3d as o3d
from sklearn.decomposition import PCA

def candidate_bud_top_n(txt_path: str, max_angle=45, min_points=30, top_n=3):
    data = np.loadtxt(txt_path)
    xyz = data[:, :3]
    labels = data[:, -1].astype(int)

    unique_labels = np.unique(labels)
    axis_distances = {}

    for label in unique_labels:
        pts = xyz[labels == label]
        if len(pts) < min_points:
            continue
        center = np.mean(pts, axis=0)
        r = np.sqrt(center[0]**2 + center[1]**2)
        axis_distances[label] = r

    if not axis_distances:
        return None, None, None, xyz, labels

    sorted_labels = sorted(axis_distances.items(), key=lambda x: x[1])[:top_n]

    candidate_buds = []

    for label, dist in sorted_labels:
        pts = xyz[labels == label]
        pca = PCA(n_components=3)
        pca.fit(pts)
        v1 = pca.components_[0]
        z_axis = np.array([0, 0, 1])
        cos_theta = np.dot(v1, z_axis) / (np.linalg.norm(v1) * np.linalg.norm(z_axis))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(abs(cos_theta)))

        if angle <= max_angle:
            # Record candidate bud: label, horizontal distance, angle, first principal component length
            length_pc1 = np.ptp(pts @ v1)
            candidate_buds.append((label, dist, angle, length_pc1))

    if not candidate_buds:
        return None, None, None, xyz, labels

    # If multiple candidate buds, select the one with the shortest first principal component length
    candidate_buds.sort(key=lambda x: x[3])
    chosen_bud = candidate_buds[0]
    label, dist, angle, length_pc1 = chosen_bud
    return label, dist, angle, xyz, labels

def visualize_candidate(xyz, labels, bud_label, window_name="Candidate Bud", width=1024, height=768):
    unique_labels = np.unique(labels)
    colors = np.zeros((len(labels), 3))
    rng = np.random.default_rng(42)
    label_colors = {label: rng.random(3) for label in unique_labels}

    for i, label in enumerate(labels):
        colors[i] = label_colors[label]

    if bud_label is not None:
        colors[labels == bud_label] = [1, 0, 0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def process_folder_top_n(folder_path: str, max_angle=45, top_n=3):
    files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
    if not files:
        print("No txt files found")
        return

    print(f"Found {len(files)} txt files in folder {folder_path}\n")

    for file in files:
        bud_label, dist, angle, xyz, labels = candidate_bud_top_n(file, max_angle=max_angle, top_n=top_n)
        if bud_label is not None:
            print(f"[{os.path.basename(file)}] Final candidate bud instance = {bud_label}, "
                  f"Z-axis horizontal distance = {dist:.4f}, Angle with Z-axis = {angle:.2f}°")
        else:
            print(f"[{os.path.basename(file)}] No bud satisfying conditions found")
        visualize_candidate(xyz, labels, bud_label, window_name=os.path.basename(file), width=1024, height=768)

if __name__ == "__main__":
    folder_path = r"./processed"
    process_folder_top_n(folder_path, max_angle=35, top_n=3)