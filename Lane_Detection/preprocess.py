
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import config

def process_tusimple_data(data_dir, output_dir, json_files):
    """
    Parses TuSimple JSON files and generates segmentation masks.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for json_file in json_files:
        json_path = os.path.join(data_dir, json_file)
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found. Skipping.")
            continue

        with open(json_path, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Processing {json_file}"):
            info = json.loads(line)
            raw_file = info['raw_file']
            lanes = info['lanes']
            h_samples = info['h_samples']

            # Image path
            img_path = os.path.join(data_dir, raw_file)
            if not os.path.exists(img_path):
                # Try to find it relative to data_dir if raw_file starts with clips/
                # TuSimple structure can vary.
                pass 
            
            # Create mask
            # Assuming image size 1280x720. 
            # We will read the image to get dimensions just to be safe, or assume standard TuSimple.
            # Reading every image might be slow, let's assume 1280x720 for mask creation
            mask = np.zeros((720, 1280), dtype=np.uint8)

            # Filter out empty lanes (all -2)
            valid_lanes = []
            for lane in lanes:
                if any(x != -2 for x in lane):
                    valid_lanes.append(lane)
            
            # Sort lanes from left to right based on their x-coordinate at the bottom of the image
            # We can pick a y-sample near the bottom to sort.
            # Or just use the order provided if consistent. TuSimple usually provides left-to-right?
            # Let's verify by checking the x-coordinate at the last valid point.
            
            # For simplicity, let's just assign IDs 1..N based on the order in the list for now.
            # Ideally we should sort them.
            
            lane_centroids = []
            for lane in valid_lanes:
                # find average x of valid points
                valid_x = [x for x in lane if x != -2]
                if valid_x:
                    lane_centroids.append(np.mean(valid_x))
                else:
                    lane_centroids.append(0) # Should not happen due to filter above
            
            # Get indices that would sort the centroids
            sorted_indices = np.argsort(lane_centroids)

            for i, idx in enumerate(sorted_indices):
                lane = valid_lanes[idx]
                lane_id = i + 1 # 1-based ID
                if lane_id >= config.NUM_CLASSES:
                    break # Limit to supported classes

                # Draw lane on mask
                # We have points (x, y). We can draw lines between them.
                points = []
                for x, y in zip(lane, h_samples):
                    if x != -2:
                        points.append((x, y))
                
                if len(points) > 1:
                    cv2.polylines(mask, [np.array(points, dtype=np.int32)], isClosed=False, color=lane_id, thickness=10)

            # Save mask
            # Construct output path. We want to mirror the folder structure or just use a flat structure with unique names.
            # raw_file is like "clips/0313-1/60/20.jpg"
            # We can replace .jpg with .png and save in processed dir.
            
            mask_rel_path = raw_file.replace('.jpg', '.png')
            mask_save_path = os.path.join(output_dir, mask_rel_path)
            
            os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
            cv2.imwrite(mask_save_path, mask)

if __name__ == "__main__":
    # Example usage
    # Assuming train_set is extracted in data/tusimple/train_set
    train_json_files = ['label_data_0313.json', 'label_data_0531.json', 'label_data_0601.json']
    process_tusimple_data(config.TRAIN_SET_DIR, config.PROCESSED_DATA_DIR, train_json_files)
    
    # For test set, TuSimple usually provides test_tasks_0627.json but without labels (it's for submission).
    # If we have ground truth for test, we can process it too.
