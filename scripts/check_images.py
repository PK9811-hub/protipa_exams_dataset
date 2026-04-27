import os
import json
from pathlib import Path

def check_image_paths():
    # Updated to point to the data directory in the current linux environment
    base_data_dir = Path(__file__).parent.parent / "data"
    missing_images = []
    total_images_checked = 0
    json_files_checked = 0

    print(f"Scanning {base_data_dir} for JSON files...")
    
    for json_path in base_data_dir.rglob("*.json"):
        json_files_checked += 1
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    continue
                
                for item in data:
                    images = item.get("images", [])
                    if not images:
                        continue
                    
                    for img in images:
                        rel_path = img.get("path")
                        if not rel_path:
                            continue
                        
                        total_images_checked += 1
                        # The path is relative to the JSON file's directory
                        # Normalize backslashes (Windows) to forward slashes (Linux)
                        normalized_rel_path = rel_path.replace('\\', '/')
                        abs_image_path = json_path.parent / normalized_rel_path
                        
                        if not abs_image_path.exists():
                            # Path doesn't exist. Check for common issues like Latin vs Greek "E"
                            # We can check if the folder exists with a different 'E'
                            # For simplicity in this script, we'll just report it as missing for now
                            # but the backslash normalization is now handled.
                            missing_images.append({
                                "json": str(json_path.relative_to(base_data_dir)),
                                "missing_path": rel_path,
                                "id": item.get("id")
                            })
            except Exception as e:
                print(f"Error reading {json_path}: {e}")

    print("\n--- Validation Summary ---")
    print(f"JSON files checked: {json_files_checked}")
    print(f"Total image references checked: {total_images_checked}")
    
    if missing_images:
        print(f"Found {len(missing_images)} missing images:")
        for missing in missing_images:
            print(f"  - In {missing['json']} (ID: {missing['id']}): {missing['missing_path']}")
    else:
        print("✓ All image paths are correct!")

if __name__ == "__main__":
    check_image_paths()
