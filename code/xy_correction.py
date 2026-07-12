import open3d as o3d
import numpy as np

from scipy.spatial import ConvexHull


def max_distance_xy_convex_hull(points):
    # Project to XY plane
    points_xy = points[:, :2]

    # Calculate convex hull
    hull = ConvexHull(points_xy)
    hull_vertices = hull.vertices  # Convex hull vertex indices

    # Extract points on the convex hull
    hull_points = points[hull_vertices]

    # Calculate distances between all pairs of points
    max_dist = 0
    point1 = None
    point2 = None

    for i in range(len(hull_points)):
        for j in range(i + 1, len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[j]
            dist = np.linalg.norm(p1[:2] - p2[:2])  # Only calculate XY plane distance
            if dist > max_dist:
                max_dist = dist
                point1 = p1
                point2 = p2

    # Calculate midpoint
    mid_point = (point1 + point2) / 2

    return max_dist, point1, point2, mid_point


def scale_pcd_file(input_path, output_path, scale_factor):
    """
    Scale point cloud file and save (supports PLY/PCD/XYZ formats)
    :param scale_factor: Scale factor (scalar or per-axis factor, e.g., [1.5, 1.5, 1.0])
    """
    pcd = o3d.io.read_point_cloud(input_path)
    pcd.scale(scale_factor, center=(0, 0, 0))  # Scale with origin as center
    o3d.io.write_point_cloud(output_path, pcd)


def plot_points_and_line_open3d(points, p1, p2, mid, length):
    """
    Visualize point cloud, the two points with maximum distance,
    the line between them, and the midpoint using Open3D.
    """
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 0])
    # pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(points), 3)))

    # Create line segment between the two points
    line_points = np.array([p1, p2])
    lines = [[0, 1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 1])  # Purple line

    # Create sphere markers for the two maximum distance points
    sphere_p1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
    sphere_p1.translate(p1 - sphere_p1.get_center())
    sphere_p1.paint_uniform_color([1, 0, 0])  # Red

    sphere_p2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
    sphere_p2.translate(p2 - sphere_p2.get_center())
    sphere_p2.paint_uniform_color([1, 0, 0])  

    # Create sphere marker for the midpoint
    sphere_mid = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
    sphere_mid.translate(mid - sphere_mid.get_center())
    sphere_mid.paint_uniform_color([0, 1, 0])  # Green

    # Visualize point cloud, line, and markers
    o3d.visualization.draw_geometries([pcd, line_set, sphere_p1, sphere_p2, sphere_mid],
                                      window_name="Max Distance and Mid Point Visualization [" + str(length) + "]",
                                      width=1000, height=900)


def xy_correction(plane_file="./plane.ply", dest_file="ok.ply", scale=15.556):
    pcd = o3d.io.read_point_cloud(plane_file)
    pcd = pcd.voxel_down_sample(0.009)
    pcd, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.07)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.015)
    
    
    pcd = pcd.select_by_index(ind)
    
    pcd = pcd.voxel_down_sample(0.0025)
    pcd, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.065)
    
    

    points = np.asarray(pcd.points)
    length, p1, p2, mid = max_distance_xy_convex_hull(points)

    print("Maximum distance (convex hull method):", length)
    print("Corresponding point 1:", p1)
    print("Corresponding point 2:", p2)
    print("Midpoint coordinates:", mid)

    plot_points_and_line_open3d(points, p1, p2, mid, length)

    # obb = pcd.get_oriented_bounding_box()
    # obb.color = (0, 1, 0)  # Set to green
    #
    #
    # center = obb.center
    # extent = obb.extent  # Length, width, height (dimensions in three directions)
    # R = obb.R             # 3x3 rotation matrix
    #
    # # Print information
    # print("OBB center coordinates:", center)
    # print("OBB dimensions (LxWxH):", extent)
    #
    #
    # # Create coordinate frame helper
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    #
    # # Display point cloud and its OBB bounding box
    # o3d.visualization.draw_geometries([pcd, obb, coordinate_frame])

    scale_factor = scale / length

    scale_pcd_file(dest_file, dest_file, scale_factor)
    scale_pcd_file(plane_file, plane_file, scale_factor)

    return scale_factor, mid * scale_factor
