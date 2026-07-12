import numpy as np
# from numpy.ma.bench import maskx
from sklearn.neighbors import NearestNeighbors
import open3d as o3d


def filter_black_but_keep_greenish(input_file, output_file,
                                   black_thresh=0.7,
                                   green_thresh_high=0.25,
                                   green_thresh_low=0.15,
                                   thresh=0.5):
    pcd = o3d.io.read_point_cloud(input_file)
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)

    z = points[:, 2]
    # Use different green_thresh for points in different z regions
    green_thresh = np.where(z > thresh, green_thresh_high, green_thresh_low)

    # Define "black": R, G, B all below black_thresh
    is_black = np.all(colors < black_thresh, axis=1)
    # Determine "greenish-black": G component is above region threshold
    keep_greenish = colors[:, 1] > green_thresh

    # Final retention: not black points, or black but high G
    mask = ~is_black | keep_greenish

    # Apply mask filtering
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    o3d.io.write_point_cloud(output_file, filtered_pcd)


def pc_post_processing(input_file, outputfile,
                       first_voxel_size, second_voxel_size, radius_filter_nb_points=20, radius_filter_radius=0.12):
    pcd = o3d.io.read_point_cloud(input_file)

    pcd = pcd.voxel_down_sample(first_voxel_size)

    # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.8)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    count_positive_z = np.sum(points[:, 2] > 0)
    count_negative_z = np.sum(points[:, 2] < 0)

    print(f"[Z > 0 : {count_positive_z}]", f"[Z < 0 : {count_negative_z}]")

    if count_positive_z > count_negative_z:
        print("Conclusion: More points > 0, keep as is, remove points < 0.")
        # Keep points with z >= 0
        mask = points[:, 2] >= 0
        points = points[mask]
        colors = colors[mask]
    else:
        # Flip Z axis and remove negative points
        points[:, 2] = -points[:, 2]
        print("Conclusion: More points <= 0, flip Z axis and remove points < 0.")
        mask = points[:, 2] >= 0
        points = points[mask]
        colors = colors[mask]

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # points = np.asarray(pcd.points)
    # print("Count before filtering:", points.shape)
    pcd, ind = pcd.remove_radius_outlier(radius_filter_nb_points, radius_filter_radius, print_progress=True)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    z_values = points[:, 2]
    z_min = np.min(z_values)
    # print("Original minimum z value:", z_min)
    points[:, 2] = points[:, 2] - z_min

    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))

    pcd = pcd.voxel_down_sample(second_voxel_size)

    o3d.io.write_point_cloud(outputfile, pcd)