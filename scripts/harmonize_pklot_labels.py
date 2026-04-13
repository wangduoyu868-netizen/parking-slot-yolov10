import os
import glob

def harmonize_labels(base_dir):
    print(f"Starting label harmonization for {base_dir}")
    total_files = 0
    total_modified = 0
    # Include test split as well if it exists
    for split in ["train", "valid", "test"]:
        label_dir = os.path.join(base_dir, split, "labels")
        if not os.path.exists(label_dir):
            continue
        txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
        count = 0
        mod_count = 0
        for path in txt_files:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            new_lines = []
            modified = False
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] != "0": 
                    parts[0] = "0"
                    modified = True
                new_lines.append(" ".join(parts) + "\n")
            
            if modified:
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                mod_count += 1
            count += 1
            
        print(f"Split [{split}]: Processed {count} files. Modified {mod_count} files to downgrade objects to class 0.")
        total_files += count
        total_modified += mod_count
        
    print(f"Harmonization Complete. Evaluated {total_files} files, updated {total_modified} files.")

if __name__ == "__main__":
    harmonize_labels(r"E:\谷歌下载\PKLot.v2-640.yolov8")
