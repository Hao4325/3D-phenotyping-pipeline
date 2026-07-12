import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import distance_matrix
import miniball
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import trimesh
from collections import defaultdict
import random
import matplotlib
matplotlib.use("TkAgg")


def triangle_area(tri_pts):
        a = np.linalg.norm(tri_pts[0] - tri_pts[1])
        b = np.linalg.norm(tri_pts[1] - tri_pts[2])
        c = np.linalg.norm(tri_pts[2] - tri_pts[0])
        s = (a + b + c) / 2
        return np.sqrt(s * (s - a) * (s - b) * (s - c))

def get_projected_area(points, voxel_size=0.02, max_edge=0.12,
                       save_path="plant_projected_delaunay_area.png", show=False):
                           

    # max_edge=0.2
    
    xy_points = points[:, :2]

    # -----------------------------
    #   Voxel downsampling
    # -----------------------------
    grid_idx = np.floor(xy_points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(grid_idx, axis=0, return_index=True)
    points_xy_voxel = xy_points[unique_idx]

    # Point count protection (must be >=3)
    if len(points_xy_voxel) < 3:
        print("Insufficient points for Delaunay triangulation")
        return 0.0

    # -----------------------------
    #   Build Delaunay structure
    # -----------------------------
    tri = Delaunay(points_xy_voxel)
    triangles = points_xy_voxel[tri.simplices]

    # -----------------------------
    #   Triangle area calculation
    # -----------------------------
    def triangle_area(t):
        return 0.5 * abs(np.cross(t[1] - t[0], t[2] - t[0]))

    areas = np.array([triangle_area(t) for t in triangles])

    # -----------------------------
    #   Edge length filtering
    # -----------------------------
    edge_lengths = np.linalg.norm(triangles[:,0,:] - triangles[:,1,:], axis=1)
    edge_lengths = np.maximum(edge_lengths, np.linalg.norm(triangles[:,1,:] - triangles[:,2,:], axis=1))
    edge_lengths = np.maximum(edge_lengths, np.linalg.norm(triangles[:,2,:] - triangles[:,0,:], axis=1))

    mask_valid = edge_lengths < max_edge
    areas_filtered = areas[mask_valid]

    projected_area = np.sum(areas_filtered)

    # -----------------------------
    #   Plotting (using the first style)
    # -----------------------------
    plt.figure(figsize=(8, 8))

    # Draw Delaunay structure (light gray lines)
    plt.triplot(points_xy_voxel[:, 0], points_xy_voxel[:, 1], tri.simplices,
                color="lightgray", alpha=0.5)

    # Draw valid triangle areas (green fill)
    for t, valid in zip(triangles, mask_valid):
        if valid:
            plt.fill(t[:, 0], t[:, 1], color="green", alpha=0.4)

    # Point cloud scatter (black)
    plt.scatter(points_xy_voxel[:, 0], points_xy_voxel[:, 1], c="black", s=3)

    # Title
    plt.title(f"Projected Leaf Area\nArea = {projected_area:.3f}")

    # Equal aspect ratio
    plt.axis("equal")

    # Remove grid
    plt.grid(False)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Save high-resolution image
    plt.savefig(save_path, dpi=800, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()

    return projected_area


def open_point_cloud_txt(txt_path, voxel_size=0.01, xy_radius=0.2, z_height=0.05, z_step=0.01):
    """Process single point cloud file, translate origin using cylinder search"""
    data = np.loadtxt(txt_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0
    labels = data[:, 6].astype(int)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size)
    points = np.asarray(pcd.points)

    tree_orig = cKDTree(data[:, :3])
    _, idxs = tree_orig.query(points, k=1)
    colors = data[idxs, 3:6]
    labels = data[idxs, 6]

    origin = np.array([0.0, 0.0, 0.0])
    max_z = points[:, 2].max()
    new_origin = origin.copy()

    while new_origin[2] <= max_z:
        # XY plane distance
        dx = points[:, 0] - new_origin[0]
        dy = points[:, 1] - new_origin[1]
        dz = points[:, 2] - new_origin[2]
        mask = (dx**2 + dy**2 <= xy_radius**2) & (np.abs(dz) <= z_height/2)
        idxs = np.where(mask)[0]
        if len(idxs) > 0:
            break
        new_origin[2] += z_step

    points_translated = points - new_origin

    processed_data = np.hstack([points_translated, colors, labels[:, None]])

    return processed_data

def get_crown_width(points, save_path = "plant_crown_width.png", show=False):
    xy_points = points[:, :2]
    center, squared_radius = miniball.get_bounding_ball(xy_points)
    radius = np.sqrt(squared_radius)


    plt.figure(figsize=(8,8))
    plt.scatter(xy_points[:,0], xy_points[:,1], s=2, c='green')

    if show:
        # Draw minimum enclosing circle
        circle = plt.Circle(center, radius, color='red', fill=False, linewidth=2)
        plt.gca().add_patch(circle)

        plt.axis('equal')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title(f'Crown width: {2*radius:.3f}')
        plt.grid(False)

        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(top=False, right=False)

        plt.savefig(save_path, dpi=500)
        plt.show()
        plt.close()

    crown_width = 2 * radius

    
    return crown_width

def get_plant_concave_volume(txt_path, alpha=0.3, alpha_step=0.05, max_alpha=1.0, show=False):
    # Read point cloud
    data = np.loadtxt(txt_path)
    points = data[:, :3]
    instance_ids = data[:, 6].astype(int)
    
    # Group by instance
    instances = defaultdict(list)
    for pt, iid in zip(points, instance_ids):
        instances[iid].append(pt)
    
    mesh_list = []
    volumes = {}
    
    # Generate alpha shape mesh for each instance
    for iid, pts in instances.items():
        instance_points = np.array(pts)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(instance_points)
        
        cur_alpha = alpha
        volume = 0.0
        mesh = None
        
        while cur_alpha <= max_alpha:
            try:
                # Attempt to generate alpha shape
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, cur_alpha)
                mesh.compute_vertex_normals()
                
                # Calculate volume
                volume = mesh.get_volume()   # Will throw exception if mesh is not watertight
                break
            except Exception:
                ##
                cur_alpha += alpha_step
        
        if mesh is not None and volume > 0:
            # Random color to distinguish instances
            color = [random.random(), random.random(), random.random()]
            mesh.paint_uniform_color(color)
            mesh_list.append(mesh)
        else:
            print(f"[calculate_plant_param][Failed] Instance {iid} could not generate watertight mesh, volume set to 0")
        
        volumes[iid] = volume
        print(f"[calculate_plant_param]Instance {iid} concave hull volume: {volume:.4f} (final alpha={cur_alpha:.2f})")
    
    # Total volume
    total_volume = sum(volumes.values())
    
    if show and mesh_list:
        o3d.visualization.draw(mesh_list, show_ui=True, title="Concave Hulls of Instances")
    
    return volumes, total_volume, mesh_list

#########################
##########################

#######################
#### 
#### Plant Height
#### Crown Width
#### Projected Area
#### 
#######################
####### Additional #######
#### 
#### Compactness
#### 
#######################

txt_path  = r".\leaf_thin_adaptive_quad.txt"
show = False

points = open_point_cloud_txt(txt_path, voxel_size = 0.1)

plant_height = points[:,2].max()
print(f"[calculate_plant_param]Plant Height: {plant_height:.3f}")
    
crown_width = get_crown_width(points, "plant_crown_width.png", show)
print(f"[calculate_plant_param]Crown Width (minimum enclosing circle diameter, using miniball): {crown_width:.3f}")

compactness = crown_width / plant_height
print(f"[calculate_plant_param]Compactness (crown_width / plant_height): {compactness:.3f}")

projected_area = get_projected_area(points, 0.02, 0.12, "plant_projected_delaunay_area.png", show)
print(f"[calculate_plant_param]Leaf Projected Area (Delaunay, with voxel pre-processing + edge filtering): {projected_area:.3f}")
    
_, total_volume, _ = get_plant_concave_volume(txt_path, 0.25, 0.05, 1.0, show)
print('*' * 100)
print(f"[calculate_plant_param]Plant Volume (concave hull): {total_volume:.6f}")
print('*' * 100)
    
print()