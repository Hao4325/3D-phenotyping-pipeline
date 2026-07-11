import numpy as np
import glob
import os
import open3d as o3d
from sklearn.decomposition import PCA
from math import sqrt

def estimate_curvature(points, kdtree, idx, k=30):
    # Use the smallest eigenvalue ratio from local PCA as curvature estimate
    [_, idxs, _] = kdtree.search_knn_vector_3d(points[idx], k)
    nb = points[idxs]
    cov = np.cov(nb.T)
    w, _ = np.linalg.eigh(cov)
    # w sorted ascending (because eigh), smallest is w[0]
    # curvature measure: smallest / sum
    curv = w[0] / (w.sum() + 1e-12)
    return curv

def filter_robust_neighbors(center_pt, center_normal, neighbors_pts, keep_ratio=0.9):
    num_pts = len(neighbors_pts)
    if num_pts < 10:
        return neighbors_pts
    
    # 1. Calculate the vertical distance from each neighbor point to the tangent plane of the center point
    n = center_normal / (np.linalg.norm(center_normal) + 1e-12)
    rel = neighbors_pts - center_pt
    dist_to_plane = np.abs(rel.dot(n))
    
    # 2. Keep only the top 90% of points closest to the local tangent plane
    num_keep = int(np.ceil(num_pts * keep_ratio))
    num_keep = max(6, num_keep)  # Safety limit: ensure at least 6 points for quadratic fitting degrees of freedom
    
    keep_indices = np.argsort(dist_to_plane)[:num_keep]
    return neighbors_pts[keep_indices]

def local_quadratic_fit_and_project(center_pt, center_normal, neighbors_pts):
    # Build local coordinate system: n as z-axis, pick arbitrary vector as x-axis
    n = center_normal / (np.linalg.norm(center_normal) + 1e-12)
    # Construct arbitrary tangent basis e1,e2
    arbitrary = np.array([1.0,0.0,0.0])
    if abs(np.dot(n, arbitrary)) > 0.9:
        arbitrary = np.array([0.0,1.0,0.0])
    e1 = np.cross(n, arbitrary)
    e1 /= (np.linalg.norm(e1)+1e-12)
    e2 = np.cross(n, e1)
    e2 /= (np.linalg.norm(e2)+1e-12)

    # Coordinate transformation: neighbors relative to center
    rel = neighbors_pts - center_pt
    u = rel.dot(e1)
    v = rel.dot(e2)
    z = rel.dot(n)   # height along normal

    # Design matrix for quad: [u^2, u*v, v^2, u, v, 1]
    A = np.stack([u*u, u*v, v*v, u, v, np.ones_like(u)], axis=1)
    # least squares solve
    try:
        coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
    except np.linalg.LinAlgError:
        return center_pt  # fallback
    # evaluate at u=0, v=0 -> z = coeff[5] (constant term)
    z0 = coeff[5]
    # projected point in 3D:
    proj = center_pt + z0 * n  # because u=v=0 -> no e1/e2 component
    return proj

def thin_leaf_by_adaptive_local_quad(pcd,
                                     base_k=30,
                                     C=0.05,
                                     r_min=0.5,
                                     r_max=5.0,
                                     iter_rounds=2):
    """
    Input: open3d pointcloud, Output: thin point cloud (numpy Nx3) + original color (Nx3)
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    normals = np.asarray(pcd.normals)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    N = points.shape[0]
    curvatures = np.zeros(N)
    for i in range(N):
        curvatures[i] = estimate_curvature(points, kdtree, i, k=base_k)

    pts_curr = points.copy()
    colors_curr = colors.copy() if colors is not None else None

    for it in range(iter_rounds):
        print(f"[layering]Iteration {it+1}/{iter_rounds} ...")
        pts_next = np.zeros_like(pts_curr)

        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(pts_curr)
        if colors_curr is not None:
            pcd_tmp.colors = o3d.utility.Vector3dVector(colors_curr)
        pcd_tmp.normals = o3d.utility.Vector3dVector(normals)
        tree_tmp = o3d.geometry.KDTreeFlann(pcd_tmp)

        for i in range(N):
            curv = curvatures[i]
            r = C / (curv + 1e-9)
            r = max(min(r, r_max), r_min)
            [_, idxs, _] = tree_tmp.search_radius_vector_3d(pcd_tmp.points[i], r)
            if len(idxs) < 6:
                [_, idxs, _] = tree_tmp.search_knn_vector_3d(pcd_tmp.points[i], min(30, N))
            nb_pts = np.asarray(pcd_tmp.points)[idxs]
            
            n = normals[i]
            nb_pts = filter_robust_neighbors(pts_curr[i], n, nb_pts, keep_ratio=0.9)
            
            proj = local_quadratic_fit_and_project(pts_curr[i], n, nb_pts)
            pts_next[i] = proj

        # Update point coordinates, keep colors unchanged
        pts_curr = pts_next
        pcd_tmp.points = o3d.utility.Vector3dVector(pts_curr)
        pcd_tmp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        normals = np.asarray(pcd_tmp.normals)

    return pts_curr, colors_curr


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

    # Sort by horizontal distance, take top_n
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
    parser = argparse.ArgumentParser(description='Leaf thinning with growing point recognition')
    parser.add_argument('--folder_path', type=str, default="./processed", help='Folder path for processing')
    parser.add_argument('--max_angle', type=float, default=35, help='Maximum angle with Z-axis for bud candidate')
    parser.add_argument('--top_n', type=int, default=3, help='Number of top candidates to consider')
    parser.add_argument('--txt_path', type=str, default="./leaf/54.txt", help='Path to input txt file')
    parser.add_argument('--save_path_name', type=str, default="leaf_thin_adaptive_quad", help='Output file name prefix')
    parser.add_argument('--show', action='store_true', default=True, help='Show visualization')
    parser.add_argument('--no_show', dest='show', action='store_false', help='Disable visualization')
    parser.add_argument('--voxel_down_sample_size', type=float, default=0.075, help='Voxel downsampling size')
    parser.add_argument('--base_k', type=int, default=60, help='Base k for curvature estimation')
    parser.add_argument('--C', type=float, default=0.003, help='Curvature scaling factor for radius calculation')
    parser.add_argument('--r_min', type=float, default=0.1, help='Minimum search radius')
    parser.add_argument('--r_max', type=float, default=1.0, help='Maximum search radius')
    parser.add_argument('--iter_rounds', type=int, default=7, help='Number of thinning iterations')
    
    args = parser.parse_args()
    
    folder_path = args.folder_path
    max_angle = args.max_angle
    top_n = args.top_n
    
    # Thinning parameters
    txt_path = args.txt_path
    save_path_name = args.save_path_name
    show = args.show
    voxel_down_sample_size = args.voxel_down_sample_size
    base_k = args.base_k
    C = args.C
    r_min = args.r_min
    r_max = args.r_max
    iter_rounds = args.iter_rounds
    
    print("=== Growing Point Recognition ===")
    bud_label, _, _, _, _ = candidate_bud_top_n(txt_path, max_angle=max_angle, top_n=top_n)
    if bud_label is not None:
        print(f"Growing point identified: instance {bud_label}")

        data = np.loadtxt(txt_path)
        xyz = data[:, 0:3].astype(np.float64)
        labels = data[:, 6].astype(np.int32)

        bud_z_max = np.max(xyz[labels == bud_label][:, 2])
        print(f"Growing point instance {bud_label} max Z value = {bud_z_max:.3f}")

    else:
        print("No growing point identified")

    
    print("\n=== Leaf Thinning Processing ===")
    data = np.loadtxt(txt_path)
    xyz = data[:, 0:3].astype(np.float64)
    rgb = data[:, 3:6].astype(np.int32)
    labels = data[:, 6].astype(np.int32)

    unique_labels = np.unique(labels)
    print("[layering]Number of instances found:", len(unique_labels))
    
    filtered_labels = []
    for lbl in unique_labels:
        mask = (labels == lbl)
        point_count = np.sum(mask)
        if point_count >= 70:
            filtered_labels.append(lbl)
        else:
            print(f"[layering]Ignoring instance {lbl}, point count: {point_count} (<100)")

    unique_labels = np.array(filtered_labels)
    print(f"[layering]Number of instances after filtering: {len(unique_labels)}")


    all_pts = []
    all_colors = []
    all_labels = []

    for lbl in unique_labels:
        print(f"[layering]Processing instance {lbl} ...")
        mask = (labels == lbl)
        xyz_inst = xyz[mask]
        rgb_inst = rgb[mask]

        print(f"[layering]Original point count: {xyz_inst.shape[0]}")

        # If it is a growing point, skip downsampling and thinning, use original data directly
        if lbl == bud_label:
            print(f"[layering]Instance {lbl} is the growing point, skipping downsampling and thinning")
            pts_thin = xyz_inst
            colors_thin = rgb_inst / 255.0
        else:
            # Non-growing points: perform downsampling and thinning
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_inst)
            pcd.colors = o3d.utility.Vector3dVector(rgb_inst / 255.0)

            pcd_down = pcd.voxel_down_sample(voxel_down_sample_size)
            print(f"[layering]Point count after downsampling: {len(pcd_down.points)}")
            
            pts_thin, colors_thin = thin_leaf_by_adaptive_local_quad(
                pcd_down, base_k, C, r_min, r_max, iter_rounds
            )
            print(f"[layering]Point count after thinning: {pts_thin.shape[0]}")

        all_pts.append(pts_thin)
        all_colors.append(colors_thin)
        # Add label for all points in current instance
        all_labels.append(np.full(pts_thin.shape[0], lbl))

    all_pts = np.vstack(all_pts)
    all_colors = np.vstack(all_colors)
    all_labels = np.concatenate(all_labels)

    # Ensure color data format is correct
    if all_colors.max() <= 1.0:
        all_colors = (all_colors * 255).astype(np.uint8)
    else:
        all_colors = all_colors.astype(np.uint8)

    output_data = np.hstack((all_pts, all_colors, all_labels.reshape(-1, 1)))
    np.savetxt(save_path_name + '.txt', output_data, fmt='%.8f %.8f %.8f %d %d %d %d')
    
    print(f"[layering]Results saved to {save_path_name}.txt, total {output_data.shape[0]} points")

    if show:
        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(all_pts)

        # ---------------------------
        # Color growing points in red
        all_colors_float = all_colors / 255.0
        if bud_label is not None:
            mask_bud = (all_labels == bud_label)
            all_colors_float[mask_bud] = np.array([1.0, 0.0, 0.0])  # Red
        # ---------------------------

        pcd_out.colors = o3d.utility.Vector3dVector(all_colors_float)

        pcd_out.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        o3d.io.write_point_cloud(save_path_name + ".ply", pcd_out)

        # Add coordinate axis
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

        o3d.visualization.draw_geometries([pcd_out, axis])

        print("[layering]Total points after merging all instances:", len(pcd_out.points))
        print("[layering]Color information:", "Yes" if pcd_out.has_colors() else "No")

    # Save instance files
    input_txt = save_path_name + '.txt'
    output_dir = "instances"
    os.makedirs(output_dir, exist_ok=True)

    data = np.loadtxt(input_txt)

    instance_ids = np.unique(data[:, 6])

    for inst_id in instance_ids:
        instance_points = data[data[:, 6] == inst_id]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(instance_points[:, :3])
        
        colors = instance_points[:, 3:6] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        output_file = os.path.join(output_dir, f"instance_{int(inst_id)}.ply")
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"[layering]Saved {len(instance_points)} points to {output_file}")