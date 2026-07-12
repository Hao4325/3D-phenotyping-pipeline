import os
import shutil

def remove_ws_files():
    target_dir = r"ws"

    # Files and folders to keep
    keep_files = {'colmap_pipeline.py', 'pro.py'}
    keep_dirs = {'data'}

    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)

        if item in keep_files or item in keep_dirs:
            continue

        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Deleted file: {item}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Deleted folder: {item}")

    print("All operations completed!")

def remove_mid_files():
    if os.path.exists("test_dense.ply"):
        os.remove("test_dense.ply")
        print("Deleted: test_dense.ply")
    if os.path.exists("plane.ply"):
        os.remove("plane.ply")
        print("Deleted: plane.ply")
    if os.path.exists("dense.ply"):
        os.remove("dense.ply")
        print("Deleted: dense.ply")
    if os.path.exists("ok.ply"):
        os.remove("ok.ply")
        print("Deleted: ok.ply")