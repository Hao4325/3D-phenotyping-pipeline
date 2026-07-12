import os
import glob
import open3d as o3d

def visualize_ply_files(folder_path):
    # Find all .ply files
    ply_files = glob.glob(os.path.join(folder_path, "*.ply"))
    
    if not ply_files:
        print(f"No PLY files found in {folder_path}")
        return
    
    # Sort by filename
    ply_files.sort()
    
    print(f"Found {len(ply_files)} PLY files")
    print("Press Q or ESC to close current window and continue to next")
    print("-" * 50)
    
    for i, ply_file in enumerate(ply_files, 1):
        print(f"[{i}/{len(ply_files)}] Displaying: {os.path.basename(ply_file)}")
        
        try:
            # Read point cloud
            pcd = o3d.io.read_point_cloud(ply_file)
            
            if len(pcd.points) == 0:
                print(f"  Warning: File is empty, skipping")
                continue
            
            print(f"  Points: {len(pcd.points)}")
            
            # Create window and display
            window_name = f"{os.path.basename(ply_file)} ({i}/{len(ply_files)})"
            
            # Create visualization window
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name, width=1024, height=768)
            vis.add_geometry(pcd)
            
            # Add coordinate axis (optional)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.55)
            vis.add_geometry(coord_frame)
            
            # Run visualization (manual close)
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            print(f"  Error: Unable to read file - {e}")
            continue
    
    print("-" * 50)
    print("All PLY files have been displayed")

def visualize_ply_with_controls(folder_path):
    """
    Interactive browsing with keyboard controls
    Press N: Next, P: Previous, Q: Quit
    """
    import numpy as np
    
    ply_files = glob.glob(os.path.join(folder_path, "*.ply"))
    
    if not ply_files:
        print(f"No PLY files found in {folder_path}")
        return
    
    ply_files.sort()
    current_idx = 0
    total = len(ply_files)
    
    print(f"Found {total} PLY files")
    print("Keyboard controls: N=Next, P=Previous, Q=Quit")
    print("-" * 50)
    
    def update_geometry(vis, idx):
        """Update displayed point cloud"""
        vis.clear_geometries()
        
        ply_file = ply_files[idx]
        print(f"[{idx+1}/{total}] Displaying: {os.path.basename(ply_file)}")
        
        pcd = o3d.io.read_point_cloud(ply_file)
        if len(pcd.points) > 0:
            print(f"  Points: {len(pcd.points)}")
            vis.add_geometry(pcd)
            
            # Add coordinate axis
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis.add_geometry(coord_frame)
        else:
            print(f"  Warning: File is empty")
        
        return False
    
    def key_callback(vis, action, mods):
        nonlocal current_idx
        
        if action == 1:  # Key pressed
            key = chr(mods) if 32 <= mods <= 126 else None
            
            if key == 'n' or key == 'N':
                # Next
                if current_idx < total - 1:
                    current_idx += 1
                    update_geometry(vis, current_idx)
                else:
                    print("Already at last file")
                    
            elif key == 'p' or key == 'P':
                # Previous
                if current_idx > 0:
                    current_idx -= 1
                    update_geometry(vis, current_idx)
                else:
                    print("Already at first file")
                    
            elif key == 'q' or key == 'Q':
                # Quit
                vis.close()
                return True
        
        return False
    
    # Create visualization window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="PLY Viewer (N=Next, P=Prev, Q=Quit)", width=1024, height=768)
    
    # Register keyboard callbacks
    vis.register_key_callback(ord('N'), key_callback)
    vis.register_key_callback(ord('P'), key_callback)
    vis.register_key_callback(ord('Q'), key_callback)
    
    # Display first point cloud
    update_geometry(vis, current_idx)
    
    # Run
    vis.run()
    vis.destroy_window()
    
    print("Exited")

if __name__ == "__main__":
    # Use current folder
    folder_path = "."

    visualize_ply_files(folder_path)
    
    # visualize_ply_with_controls(folder_path)