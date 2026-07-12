import numpy as np
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, Delaunay
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from shapely.geometry import MultiLineString
from shapely.ops import cascaded_union, polygonize
import glob
import os

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def compute_leaf_inclination_angle(points: np.ndarray) -> float:
    pca = PCA(n_components=3)
    pca.fit(points)
    v = pca.components_[0]
    vz = abs(v[2])
    angle_rad = np.arccos(vz / np.linalg.norm(v))
    inclination = 90.0 - np.degrees(angle_rad)
    return inclination

def process_point_cloud(txt_path: str) -> dict:
    data = np.loadtxt(txt_path)
    xyz = data[:, :3]
    labels = data[:, -1].astype(int)
    unique_labels = np.unique(labels)
    results = {}

    for label in unique_labels:
        pts = xyz[labels == label]
        if len(pts) < 10:
            continue
        results[label] = compute_leaf_inclination_angle(pts)
    return results

def build_kdtree_edges(points: np.ndarray, k: int = 16) -> nx.Graph:
    tree = cKDTree(points)
    edges = []
    for idx, p in enumerate(points):
        dists, idxs = tree.query(p, k+1)
        for neighbor_idx, dist in zip(idxs[1:], dists[1:]):
            edges.append((idx, neighbor_idx, dist))
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G

def find_geodesic_farthest_pair(points: np.ndarray, k: int = 16):
    G = build_kdtree_edges(points, k)
    n = len(points)
    max_dist = 0
    idx1, idx2 = 0, 0
    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(G, i, weight='weight')
        j = max(lengths, key=lengths.get)
        if lengths[j] > max_dist:
            max_dist = lengths[j]
            idx1, idx2 = i, j
    return points[idx1], points[idx2]

def alpha_shape_2D(points: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    if len(points) < 4:
        return None
    tri = Delaunay(points)
    edges = set()
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[ia], points[ib], points[ic]
        a, b, c = np.linalg.norm(pa - pb), np.linalg.norm(pb - pc), np.linalg.norm(pc - pa)
        s = (a + b + c) / 2
        area = max(s*(s-a)*(s-b)*(s-c), 0)**0.5
        circum_r = a*b*c / (4.0*area + 1e-12)
        if circum_r < alpha:
            edges.update([(ia, ib), (ib, ic), (ic, ia)])
    edge_lines = [[points[i], points[j]] for i,j in edges]
    
    # If no edges, return None
    if not edge_lines:
        return None
        
    polygons = list(polygonize(MultiLineString(edge_lines)))
    if not polygons:
        return None
    
    # Fix: Handle MultiPolygon case
    union_result = cascaded_union(polygons)
    
    if union_result.geom_type == 'MultiPolygon':
        # Select the largest polygon
        largest_polygon = max(union_result.geoms, key=lambda p: p.area)
        return np.array(largest_polygon.exterior.coords)
    elif union_result.geom_type == 'Polygon':
        return np.array(union_result.exterior.coords)
    else:
        return None
        
def compute_leaf_width_along_vein(points_3d: np.ndarray, show=False) -> float:
    """
    Calculate the maximum leaf width by smoothing along the main vein direction
    points_3d: Flattened leaf 3D point cloud (Nx3)
    Returns LW: Maximum leaf width
    """
    from scipy.interpolate import splprep, splev

    # Main vein approximation: sort along X direction, take middle Y points as vein
    x_sorted_idx = np.argsort(points_3d[:, 0])
    points_sorted = points_3d[x_sorted_idx]
    # Median Y of vein points
    vein_y = np.median(points_sorted[:, 1])
    vein_points = np.column_stack((points_sorted[:, 0], np.full(len(points_sorted), vein_y)))

    # Smooth main vein (B-spline)
    tck, u = splprep([vein_points[:,0], vein_points[:,1]], s=1.0)
    u_fine = np.linspace(0, 1, len(vein_points)*10)
    vein_smooth = np.array(splev(u_fine, tck)).T  # Nx2

    # Calculate local normals (normal vector: perpendicular to X axis)
    normals = np.zeros_like(vein_smooth)
    tangents = np.gradient(vein_smooth, axis=0)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True)+1e-12)

    # Find maximum distance on both sides along normal direction
    max_width = 0
    for vp, n in zip(vein_smooth, normals):
        # Project points onto normal direction
        proj = (points_3d[:, :2] - vp) @ n
        d_left  = -proj[proj<0].min() if np.any(proj<0) else 0
        d_right = proj[proj>0].max()  if np.any(proj>0) else 0
        local_width = d_left + d_right
        if local_width > max_width:
            max_width = local_width

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.scatter(points_3d[:,0], points_3d[:,1], s=2, c='green')
        plt.plot(vein_smooth[:,0], vein_smooth[:,1], 'r-', lw=2, label='Smoothed Vein')
        plt.title(f"Leaf Width along Vein = {max_width:.3f}")
        plt.axis('equal')
        plt.show()

    return max_width
    
def compute_leaf_curvature(path_points: np.ndarray) -> float:
    """
    Calculate normalized leaf curvature (LC)
    Normalized by path length to eliminate sampling density effects
    """
    n = len(path_points)
    if n < 3:
        return 0.0
    
    vectors = []
    segment_lengths = []
    
    # Calculate vectors and segment lengths
    for i in range(n - 1):
        v = path_points[i + 1] - path_points[i]
        vectors.append(v)
        segment_lengths.append(np.linalg.norm(v))
    
    total_curvature = 0.0
    total_path_length = sum(segment_lengths)
    
    # If path length is too short, return 0
    if total_path_length < 1e-12:
        return 0.0
    
    for i in range(len(vectors) - 1):
        v_i = vectors[i]
        v_i1 = vectors[i + 1]
        
        dot_product = np.dot(v_i, v_i1)
        norm_v_i = np.linalg.norm(v_i)
        norm_v_i1 = np.linalg.norm(v_i1)
        
        if norm_v_i < 1e-12 or norm_v_i1 < 1e-12:
            continue
        
        cos_theta = np.clip(dot_product / (norm_v_i * norm_v_i1), -1.0, 1.0)
        theta_i = np.arccos(cos_theta)
        
        # Weighted by segment length
        avg_segment_length = (segment_lengths[i] + segment_lengths[i+1]) / 2
        weighted_curvature = theta_i * (avg_segment_length / total_path_length)
        
        total_curvature += weighted_curvature
    
    return total_curvature


###########################
# - Leaf Length
# - Leaf Width (not applicable for unfolded buds)
# - Leaf Inclination Angle
# - Leaf Area (not applicable for unfolded buds)
# - - Delaunay Area (flattened leaf 2D projection + triangulation + area accumulation)
# - - Shoelace Area (Gauss area formula)
# - Leaf Perimeter (not applicable for unfolded buds)
# - Leaf Curvature
###########################

if __name__ == "__main__":

    txt_path = r".\leaf_thin_adaptive_quad.txt" ## Total
    
    path = "./instances"
    
    show = True
    
    files = sorted(glob.glob(os.path.join(path, "*.ply")))
    results = process_point_cloud(txt_path)
    for label, angle in results.items():
        print(f"[calculate_leaf_param]Instance {label}: Leaf Inclination Angle = {angle:.2f}°")
    for file in files:
        
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
        if show:
            o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

        # PCA transform for leaf: points_pca = (points - mean) · eigvecs
        mean = np.mean(points, axis=0)
        points_centered = points - mean
        cov = np.cov(points_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
        points_pca = points_centered @ eigvecs
        
        # Create three colored lines for PCA principal axes
        scale = 0.5 * np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        axes = []
        colors_axes = [[1,0,0],[0,1,0],[0,0,1]]
        for k in range(3):
            start = mean - eigvecs[:,k]*scale
            end = mean + eigvecs[:,k]*scale
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([start,end]),
                lines=o3d.utility.Vector2iVector([[0,1]])
            )
            line.colors = o3d.utility.Vector3dVector([colors_axes[k]])
            axes.append(line)
        if show:
            o3d.visualization.draw_geometries([pcd,*axes], window_name="PCA Axes")
        
        # In PCA coordinate system, apply "band clipping" along leaf width direction (PC2)
        proj_2 = points_pca[:, 1]
        mask = (proj_2 >= -0.55) & (proj_2 <= 0.55)

        filtered_points_pca = points_pca[mask]
        filtered_idx = np.where(mask)[0]

        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points_pca)
        pcd_filtered.paint_uniform_color([0,1,0])
        if show:
            o3d.visualization.draw_geometries([pcd_filtered], window_name="Filtered by 2nd PCA")


        point1, point2 = find_geodesic_farthest_pair(filtered_points_pca)

        # Map back to original coordinate system to determine leaf base/tip
        point1_orig = point1 @ eigvecs.T + mean
        point2_orig = point2 @ eigvecs.T + mean

        if np.linalg.norm(point1_orig) < np.linalg.norm(point2_orig):
            base_orig, tip_orig = point1_orig, point2_orig
        else:
            base_orig, tip_orig = point2_orig, point1_orig
            
        # Base point refinement
        base_pca_init = (base_orig - mean) @ eigvecs
        x_range_mask = (filtered_points_pca[:, 0] >= base_pca_init[0] - 0.01) & \
                       (filtered_points_pca[:, 0] <= base_pca_init[0] + 0.01)
        candidate_points = filtered_points_pca[x_range_mask]

        if len(candidate_points) > 0:
            y_mid = (candidate_points[:, 1].max() + candidate_points[:, 1].min()) / 2
            base_pca_idx = np.argmin(np.abs(candidate_points[:, 1] - y_mid))
            new_base_pca = candidate_points[base_pca_idx]
            base_orig = new_base_pca @ eigvecs.T + mean

        # Map back to PCA coordinate system for subsequent path calculation
        base_pca = (base_orig - mean) @ eigvecs
        tip_pca  = (tip_orig - mean) @ eigvecs

        sphere_start = o3d.geometry.TriangleMesh.create_sphere(radius=0.01*scale)
        sphere_start.translate(base_pca @ eigvecs.T + mean)
        sphere_start.paint_uniform_color([0,1,0])

        sphere_end = o3d.geometry.TriangleMesh.create_sphere(radius=0.01*scale)
        sphere_end.translate(tip_pca @ eigvecs.T + mean)
        sphere_end.paint_uniform_color([1,0,0])

        G = build_kdtree_edges(filtered_points_pca, k=20)

        # Use base_pca and tip_pca to find closest indices
        idx1 = np.argmin(np.linalg.norm(filtered_points_pca - base_pca, axis=1))
        idx2 = np.argmin(np.linalg.norm(filtered_points_pca - tip_pca, axis=1))

        path_idx = nx.shortest_path(G, source=idx1, target=idx2, weight="weight")
        path_points_pca = filtered_points_pca[path_idx]
        path_global_idx = filtered_idx[path_idx]
        
        # Calculate leaf curvature
        leaf_curvature = compute_leaf_curvature(path_points_pca)
        leaf_curvature_deg = np.degrees(leaf_curvature)

        print(f"[calculate_leaf_param]Leaf Normalized Curvature (LC): {leaf_curvature:.4f} rad ({leaf_curvature_deg:.2f}°)")

        colors[path_global_idx] = [1,0,0]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if show:
            o3d.visualization.draw_geometries([pcd, sphere_start, sphere_end], window_name="Geodesic Path")

        points_pca_full = (np.asarray(pcd.points) - mean) @ eigvecs
        pcd_pca = o3d.geometry.PointCloud()
        pcd_pca.points = o3d.utility.Vector3dVector(points_pca_full)
        pcd_pca.colors = pcd.colors
        o3d.io.write_point_cloud("leaf_with_path.ply", pcd_pca)
        print("[calculate_leaf_param]Saved PCA coordinate point cloud: leaf_with_path.ply")


        # X-Z projection length (PCA)
        # === Calculate 3D main vein curve length ===
        path_points_3d = path_points_pca  # Main vein in PCA coordinate system

        # Calculate total curve length in 3D space
        path_length_3d = np.sum(np.linalg.norm(path_points_3d[1:] - path_points_3d[:-1], axis=1))
        print("[calculate_leaf_param] 3D main vein curve length (PCA coordinate):", path_length_3d)

        # === Visualize 3D main vein curve ===
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(
            path_points_3d[:, 0], path_points_3d[:, 1], path_points_3d[:, 2],
            '-o', color='purple', markersize=4, linewidth=2, label='Principal Vein (Smoothed)'
        )
        ax.scatter(path_points_3d[0, 0], path_points_3d[0, 1], path_points_3d[0, 2],
                   color='green', s=60, label="Start")
        ax.scatter(path_points_3d[-1, 0], path_points_3d[-1, 1], path_points_3d[-1, 2],
                   color='red', s=60, label="End")

        ax.set_xlabel("PC1 axis")
        ax.set_ylabel("PC2 axis")
        ax.set_zlabel("PC3 axis")
        ax.set_title(f"3D Principal Vein Path (Length = {path_length_3d:.2f})")
        ax.legend()
        plt.tight_layout()
        plt.savefig("path_3d_pca_curve.png", dpi=800)
        if show:
            plt.show()
            
        path_points_original = path_points_pca @ eigvecs.T + mean

        save_path = file[:-4] + "kp.txt"

        np.savetxt(save_path, path_points_original, fmt="%.8f", delimiter="\t")
        print(f"[calculate_leaf_param]Saved original coordinate system path points: {save_path}")

        # --------------------------
        # Flatten Leaf
        # --------------------------
        pcd = o3d.io.read_point_cloud("leaf_with_path.ply")
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        print("[calculate_leaf_param]Total points:", len(points))

        isomap = Isomap(n_neighbors=70, n_components=2)
        points_2d = isomap.fit_transform(points)
        points_2d_aligned = PCA(n_components=2).fit_transform(points_2d)

        leaf_length = points_2d_aligned[:,0].max() - points_2d_aligned[:,0].min()
        leaf_width_2d  = points_2d_aligned[:,1].max() - points_2d_aligned[:,1].min()
        print(f"[calculate_leaf_param]Leaf Width (2D): {leaf_width_2d:.3f}")
        leaf_width = compute_leaf_width_along_vein(points_2d_aligned, show=True)
        print(f"[calculate_leaf_param]Leaf Length: {leaf_length:.3f}, Leaf Width: {leaf_width:.3f}")

        plt.figure(figsize=(8,6))
        plt.scatter(points_2d_aligned[:,0], points_2d_aligned[:,1], s=2, c=colors)
        plt.title("Flattened Lettuce Leaf Point Cloud with Isomap")
        plt.xlabel("X axis (flattened)")
        plt.ylabel("Y axis (flattened)")
        plt.axis("equal")
        plt.grid(True)
        plt.savefig("leaf_flat_isomap.png", dpi=800)
        if show:
            plt.show()

        points_flat = np.column_stack((points_2d_aligned, np.zeros(points_2d_aligned.shape[0])))
        pcd_flat = o3d.geometry.PointCloud()
        pcd_flat.points = o3d.utility.Vector3dVector(points_flat)
        pcd_flat.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud("leaf_flat_isomap.ply", pcd_flat)
        print("[calculate_leaf_param]Saved flattened leaf point cloud: leaf_flat_isomap.ply")

        # --------------------------
        # Delaunay Area Calculation
        # --------------------------
        points_2d_for_area = points_2d_aligned
        tri = Delaunay(points_2d_for_area)
        simplices = tri.simplices
        vertices = points_2d_for_area[simplices]

        max_edge_threshold = 0.15
        valid_triangles, valid_areas = [], []

        for simplex in vertices:
            edges = [np.linalg.norm(simplex[0]-simplex[1]),
                     np.linalg.norm(simplex[1]-simplex[2]),
                     np.linalg.norm(simplex[2]-simplex[0])]
            if max(edges) <= max_edge_threshold:
                valid_triangles.append(simplex)
                area = 0.5 * abs(
                    simplex[0,0]*(simplex[1,1]-simplex[2,1]) +
                    simplex[1,0]*(simplex[2,1]-simplex[0,1]) +
                    simplex[2,0]*(simplex[0,1]-simplex[1,1])
                )
                valid_areas.append(area)

        leaf_area_delaunay = np.sum(valid_areas)
        print(f"[calculate_leaf_param]Leaf area after filtering long edges = {leaf_area_delaunay:.3f}, valid triangles: {len(valid_triangles)}/{len(simplices)}")

        plt.figure(figsize=(8,6))
        plt.triplot(points_2d_for_area[:,0], points_2d_for_area[:,1], simplices, color="lightgray", alpha=0.5)
        for tri_valid in valid_triangles:
            plt.fill(tri_valid[:,0], tri_valid[:,1], color="green", alpha=0.4)
        plt.scatter(points_2d_for_area[:,0], points_2d_for_area[:,1], c="black", s=3)
        plt.title(f"Filtered Delaunay Triangulation\nLeaf Area={leaf_area_delaunay:.3f}")
        plt.axis("equal")
        plt.grid(False)
        plt.savefig("leaf_delaunay_area.png", dpi=800)
        if show:
            plt.show()

        # --------------------------
        # Alpha Shape Boundary & Perimeter
        # --------------------------
        boundary_coords = alpha_shape_2D(points_2d_for_area, alpha=0.08)
        if boundary_coords is None:
            print("[calculate_leaf_param]Alpha Shape generation failed")
        else:
            perimeter = np.sum(np.linalg.norm(boundary_coords[1:] - boundary_coords[:-1], axis=1))
            print(f"[calculate_leaf_param]Leaf 2D perimeter: {perimeter:.3f}")

            x, y = boundary_coords[:,0], boundary_coords[:,1]
            shoelace_area = 0.5 * np.abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))
            print(f"[calculate_leaf_param]Leaf 2D Shoelace area: {shoelace_area:.3f}")

            plt.figure(figsize=(8,6))
            plt.scatter(points_2d_for_area[:,0], points_2d_for_area[:,1], s=2, c='green', label='Leaf points')
            plt.plot(boundary_coords[:,0], boundary_coords[:,1], 'r-', lw=2, label='Leaf boundary')
            plt.title(f"Leaf Alpha Shape Boundary (Perimeter={perimeter:.3f}, Area={shoelace_area:.3f})")
            plt.axis("equal")
            plt.legend()
            if show:
                plt.show()

            tree_2d = cKDTree(points_2d_for_area)
            _, idxs = tree_2d.query(boundary_coords)
            boundary_3d = points[idxs]
            perimeter_3d = np.sum(np.linalg.norm(boundary_3d[1:] - boundary_3d[:-1], axis=1))
            print(f"[calculate_leaf_param]Leaf 3D perimeter: {perimeter_3d:.3f}")
            if show:
                lines = [[i, i+1] for i in range(len(boundary_3d)-1)]
                line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(boundary_3d),
                                                lines=o3d.utility.Vector2iVector(lines))
                line_set.colors = o3d.utility.Vector3dVector([[1,0,0]] * len(lines))
                o3d.visualization.draw_geometries([pcd, line_set], window_name="3D Leaf Perimeter & Shoelace Area")
        print('*' * 100)
        # print(f"{leaf_area_delaunay:.3f}, {shoelace_area:.3f}, {path_length_3d:.3f}, {leaf_length:.3f}, {leaf_width:.3f}, {perimeter:.3f}, {perimeter_3d:.3f}")
        print(f"{shoelace_area:.3f}, {path_length_3d:.3f}, {leaf_width:.3f}, {perimeter:.3f}")
        # print(f"Leaf Curvature (LC): {leaf_curvature:.4f} rad ({leaf_curvature_deg:.2f}°)")
        print('*' * 100)