import os
import json
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm

def subset_dataset(input_dir, output_dir, percentage, label_file_path=None):
    """
    Creates a subset of the TuSimple dataset.

    Args:
        input_dir (str): Path to the original dataset.
        output_dir (str): Path to save the reduced dataset.
        percentage (float): Percentage of data to keep (0.0 to 1.0).
        label_file_path (str): Optional path to a specific label file.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    if output_path.exists():
        print(f"Warning: Output directory '{output_dir}' already exists.")
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    # List of label files to process
    if label_file_path:
        label_files = [Path(label_file_path).name]
        # We need to handle the case where the label file is not in input_dir
        # We will read from label_file_path directly
    else:
        label_files = [
            "label_data_0313.json",
            "label_data_0531.json",
            "label_data_0601.json",
            "test_label.json"
        ]

    print(f"Scanning for label files...")
    
    all_clips = []
    
    # First pass: Collect all clips from all label files
    if label_file_path:
        # Process single external label file
        file_path = Path(label_file_path)
        if file_path.exists():
            print(f"Processing external label file: {file_path}")
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        data = json.loads(line)
                        raw_file = data['raw_file']
                        clip_path = str(Path(raw_file).parent)
                        all_clips.append({
                            'clip_path': clip_path,
                            'data': data,
                            'label_file': file_path.name # Use basename for output
                        })
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode line in {file_path}")
                        continue
        else:
             print(f"Error: Label file '{label_file_path}' does not exist.")
             return
    else:
        # Process default files in input_dir
        for label_file in label_files:
            file_path = input_path / label_file
            if file_path.exists():
                print(f"Processing {file_path}")
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        try:
                            data = json.loads(line)
                            raw_file = data['raw_file']
                            clip_path = str(Path(raw_file).parent)
                            all_clips.append({
                                'clip_path': clip_path,
                                'data': data,
                                'label_file': label_file
                            })
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode line in {label_file}")
                            continue

    total_clips = len(all_clips)
    print(f"Found {total_clips} total labeled frames.")

    if total_clips == 0:
        print("No labeled data found. Exiting.")
        return

    unique_clip_folders = list(set(item['clip_path'] for item in all_clips))
    num_unique_clips = len(unique_clip_folders)
    print(f"Found {num_unique_clips} unique clip sequences.")
    
    num_to_keep = int(num_unique_clips * percentage)
    print(f"Selecting {num_to_keep} clips ({percentage*100}%)...")

    selected_clip_folders = set(random.sample(unique_clip_folders, num_to_keep))

    # Prepare new label files content
    # If using external label file, we map its name to the list
    if label_file_path:
         new_labels = {Path(label_file_path).name: []}
    else:
         new_labels = {lf: [] for lf in label_files}

    print("Filtering labels and copying files...")
    
    copied_folders = set()

    for item in tqdm(all_clips, desc="Processing"):
        if item['clip_path'] in selected_clip_folders:
            # Add to new labels
            new_labels[item['label_file']].append(item['data'])
            
            src_clip_dir = input_path / item['clip_path']
            dst_clip_dir = output_path / item['clip_path']
            
            if item['clip_path'] not in copied_folders:
                if src_clip_dir.exists():
                    shutil.copytree(src_clip_dir, dst_clip_dir, dirs_exist_ok=True)
                    copied_folders.add(item['clip_path'])
                else:
                    # Try checking if input_dir is already inside 'clips' or parent
                    # Sometimes raw_file is relative to dataset root
                    pass 
                    # For now assume input_dir is the root containing 'clips' folder
                    if not src_clip_dir.exists():
                         print(f"Warning: Source clip directory not found: {src_clip_dir}")

    # Write new label files
    print("Writing new label files...")
    for label_file, data_list in new_labels.items():
        if data_list:
            output_label_path = output_path / label_file
            with open(output_label_path, 'w') as f:
                for entry in data_list:
                    json.dump(entry, f)
                    f.write('\n')
            print(f"Created {label_file} with {len(data_list)} entries.")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a subset of the TuSimple dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the original dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the reduced dataset.")
    parser.add_argument("--percentage", type=float, default=0.5, help="Percentage of data to keep (0.0 to 1.0).")
    parser.add_argument("--label_file", type=str, help="Optional path to a specific label file to process.")
    
    args = parser.parse_args()
    
    subset_dataset(args.input_dir, args.output_dir, args.percentage, args.label_file)
