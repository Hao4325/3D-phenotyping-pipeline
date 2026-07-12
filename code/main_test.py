######################################################################
# 3D Reconstruction
import os

if os.path.exists("test_dense.ply"):
    print("File exists")
else:
    print("File does not exist")
    print("Reconstruct? [y/n]", end='')
    if str(input()) == "y":
        os.chdir("ws")
        os.system("python colmap_pipeline.py -s data")
        os.chdir("..")
    else:
        exit(-1)


######################################################################
# Coordinate Correction and Scale Recovery

from xy_correction import xy_correction
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

pcd = o3d.io.read_point_cloud("test_dense.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)


black_thresh = 0.25 # ！！！
black_mask = np.all(colors < black_thresh, axis=1)
black_idx = np.where(black_mask)[0]

# Construct sub-point cloud containing only "black points"
pcd_black = pcd.select_by_index(black_idx)
print(f"Number of black points: {len(black_idx)} / Total points: {len(points)}")

plane_model, inliers_black = pcd_black.segment_plane(
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=1000
)
a, b, c, d = plane_model
print(f"[Black point cloud] Maximum plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

# Map black point cloud inliers back to original point cloud indices
inliers = black_idx[inliers_black]

plane_normal = np.array([a, b, c])
plane_normal /= np.linalg.norm(plane_normal)
target_normal = np.array([0, 0, 1])

# Rotation axis & angle
rot_axis = np.cross(plane_normal, target_normal)
if np.linalg.norm(rot_axis) < 1e-6:
    rot_axis = np.array([0,0,1])  # Already parallel
else:
    rot_axis /= np.linalg.norm(rot_axis)
angle = np.arccos(np.clip(np.dot(plane_normal, target_normal), -1,1))
print(f"Rotation angle: {np.degrees(angle):.2f}°")

rotation_mat = R.from_rotvec(rot_axis * angle).as_matrix()

rotated_pts = (rotation_mat @ points.T).T
rotated_plane_pts = rotated_pts[inliers]
mean_z = np.mean(rotated_plane_pts[:,2])
print(f"Mean z of rotated plane points: {mean_z:.4f}")

# Translate to z=0
corrected_pts = rotated_pts.copy()
corrected_pts[:,2] -= mean_z

# Reconstruct corrected point cloud
pcd_corrected = o3d.geometry.PointCloud()
pcd_corrected.points = o3d.utility.Vector3dVector(corrected_pts)
pcd_corrected.colors = o3d.utility.Vector3dVector(colors)

# Plane points (red)
plane_pts = corrected_pts[inliers]
plane_colors = np.tile(np.array([1.0,0.0,0.0]), (len(inliers),1))
pcd_plane = o3d.geometry.PointCloud()
pcd_plane.points = o3d.utility.Vector3dVector(plane_pts)
pcd_plane.colors = o3d.utility.Vector3dVector(plane_colors)

pcd_without = pcd_corrected.select_by_index(inliers, invert=True)

print("Displaying corrected point cloud (ground plane removed) and red plane points...")
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
o3d.visualization.draw_geometries([pcd, pcd_without, pcd_plane, frame], window_name="Corrected Point Cloud (Ground Plane Removed) and Red Plane Points", width=1200, height=900)

# Save
o3d.io.write_point_cloud("dense.ply", pcd_without)
o3d.io.write_point_cloud("plane.ply", pcd_plane)

scale_factor, mid = xy_correction("plane.ply", "dense.ply", scale = 15.556)
print("Scale factor:", scale_factor)

# Re-read plane.ply to get z range
pcd_plane = o3d.io.read_point_cloud("plane.ply")
z_vals = np.asarray(pcd_plane.points)[:,2]
z_max, z_min = z_vals.max(), z_vals.min()
print(f"plane.ply z_max={z_max:.6f}, z_min={z_min:.6f}")

# Apply depth filtering to dense.ply (e.g., offset 0.05)
pcd_dense = o3d.io.read_point_cloud("dense.ply")
pts = np.asarray(pcd_dense.points)
cls = np.asarray(pcd_dense.colors)
mask = (pts[:,2] < z_min - 0.05) | (pts[:,2] > z_max + 0.05)

filtered_pts = pts[mask]
filtered_colors = cls[mask] if cls.shape[0]==pts.shape[0] else None


filtered_pts = filtered_pts - mid

pcd_filtered = o3d.geometry.PointCloud()
pcd_filtered.points = o3d.utility.Vector3dVector(filtered_pts)
if filtered_colors is not None:
    pcd_filtered.colors = o3d.utility.Vector3dVector(filtered_colors)

o3d.io.write_point_cloud("dense.ply", pcd_filtered)
print(f"Original points: {len(pts)}, Remaining after filtering: {len(filtered_pts)}")

# ######################################################################
# Point Cloud Processing

from pc_post_processing import pc_post_processing, filter_black_but_keep_greenish
from remove_middle_files import remove_ws_files, remove_mid_files
from scipy.spatial import ConvexHull, distance_matrix
import cv2

def compute_crown_width_hull(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    xy = np.asarray(pcd.points)[:, :2]
    hull = ConvexHull(xy)
    hull_pts = xy[hull.vertices]
    dists = distance_matrix(hull_pts, hull_pts)
    crown_width = np.max(dists)
    print(f"Convex hull crown width: {crown_width:.4f}")

def compute_crown_width(ply_file, method='aabb'):
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)

    # Use only xy coordinates for projection
    xy = points[:, :2]

    if method == 'aabb':
        # Calculate AABB bounding box
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        crown_width = max(x_max - x_min, y_max - y_min)
        print(f"AABB projected crown width: {crown_width:.4f}")
    elif method == 'obb':
        # Open3D OBB is 3D, but here we use minimum rotated rectangle for more accurate 2D crown width
        from shapely.geometry import MultiPoint
        from shapely.affinity import rotate
        from scipy.spatial import ConvexHull
        hull = ConvexHull(xy)
        hull_pts = xy[hull.vertices]
        from shapely.geometry import Polygon
        poly = Polygon(hull_pts)
        min_rect = poly.minimum_rotated_rectangle
        # Get the four vertices of the rectangle
        xys = np.array(min_rect.exterior.coords)
        edge_lengths = np.linalg.norm(xys[1:] - xys[:-1], axis=1)
        crown_width = max(edge_lengths)
        print(f"OBB projected crown width: {crown_width:.4f}")
    else:
        raise ValueError("method should be 'aabb' or 'obb'")

def show_crown_circle_open3d(ply_file, circle_color=[1, 0, 0]):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    xy = np.asarray(pcd.points)[:, :2].astype(np.float32)

    # Calculate minimum enclosing circle
    (x_center, y_center), radius = cv2.minEnclosingCircle(xy)
    diameter = 2 * radius
    print(f"Minimum enclosing circle diameter: {diameter:.4f}")

    # Generate circle polygon in xy plane (e.g., 100 points)
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_pts = np.vstack([
        x_center + radius * np.cos(theta),
        y_center + radius * np.sin(theta),
        np.zeros_like(theta)  # z=0
    ]).T

    # Use Open3D LineSet to draw the circle
    lines = [[i, (i + 1) % 100] for i in range(100)]  # Closed polygon
    circle_line_set = o3d.geometry.LineSet()
    circle_line_set.points = o3d.utility.Vector3dVector(circle_pts)
    circle_line_set.lines = o3d.utility.Vector2iVector(lines)
    circle_line_set.colors = o3d.utility.Vector3dVector([circle_color for _ in lines])

    # Optional: Display circle center
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=diameter * 0.01)
    sphere.paint_uniform_color([1, 0.5, 0])
    sphere.translate([x_center, y_center, 0])

    # Display
    o3d.visualization.draw_geometries([pcd, circle_line_set, sphere])

input_file = r"dense.ply"
outputfile = r"ok.ply"
first_voxel_size = 0.006
second_voxel_size = 0.026
radius_filter_nb_points, radius_filter_radius = 5, 0.045 # 10 -> 5
# radius_filter_nb_points, radius_filter_radius = 3, 0.05
# radius_filter_nb_points, radius_filter_radius = 10, 0.1

pc_post_processing(input_file, outputfile, first_voxel_size,second_voxel_size, radius_filter_nb_points, radius_filter_radius)

pcd = o3d.io.read_point_cloud("ok.ply")
points = np.asarray(pcd.points)
print("Points after point cloud post-processing:", points.shape)
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)
o3d.visualization.draw_geometries([pcd, frame], window_name="Point Cloud Post-Processing Result", width=1200, height=900)

# filter_black_but_keep_greenish(outputfile, outputfile, black_thresh=0.15, green_thresh_high=0.3, green_thresh_low=0.01, thresh=0.825)
# 0.22  0.3  0.1 0.8
filter_black_but_keep_greenish(outputfile, outputfile, black_thresh=0.35, green_thresh_high=0.35, green_thresh_low=0.2, thresh=0.)

pcd = o3d.io.read_point_cloud("ok.ply")
pcd, ind = pcd.remove_radius_outlier(5, 0.135) # 13 -> 5
points = np.asarray(pcd.points)
print("Points after color filtering:", points.shape)
o3d.visualization.draw_geometries([pcd, frame], window_name="Point Cloud Post-Processing Result", width=1200, height=900)
o3d.io.write_point_cloud("ok.ply", pcd)

max_z = np.max(points[:, 2]) + 0.05
min_z = np.min(points[:, 2])
print("max_z:", max_z, "\tmin_z:", min_z)
print(f"Height (with fixed cotton): {max_z:.4f}")
compute_crown_width_hull(outputfile)
compute_crown_width(outputfile, method='aabb')
compute_crown_width(outputfile, method='obb')
# show_crown_circle_open3d(outputfile)

print()
print("Clear cache? (y/n):", end='')
if str(input()) == 'y':
    remove_mid_files()
    remove_ws_files()