#!/usr/bin/env python3
import os
import json
import re
import argparse
from pathlib import Path

def create_dataset_json(input_dir, output_file):
    """
    Scans a directory for images and creates a dataset.json file.
    It extracts the label from the filename and saves it as a list of lists.
    
    Args:
        input_dir (str): The directory containing image files.
        output_file (str): The name of the output JSON file.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: Directory not found at {input_dir}")
        return

    # Regular expression to match filenames like 'label123_000001.png'
    pattern = re.compile(r"label(\d+)_\d+\..+", re.IGNORECASE)

    labels = []
    processed_count = 0

    print(f"Scanning directory: {input_dir}")
    
    # Iterate through all files in the input directory
    for image_file in sorted(input_path.iterdir()):
        filename = image_file.name
        
        # Check if the filename matches our pattern
        match = pattern.match(filename)
        if match:
            # Extract the captured label group and convert it to an integer
            label = int(match.group(1))
            
            # Create a list for this image and label
            labels.append([filename, label])
            processed_count += 1
            
    if not labels:
        print(f"Warning: No files matching the pattern were found in {input_dir}.")
        return

    # Write the list of lists to a JSON file under the "labels" key
    with open(output_file, 'w') as f:
        json.dump({"labels": labels}, f, indent=4)
        
    print(f"\nSuccessfully created {output_file} with {processed_count} entries.")

def main():
    parser = argparse.ArgumentParser(
        description="Create a dataset.json from image files with 'label<id>_*.ext' names."
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=True,
        help="Path to the directory containing your images (e.g., './imagenet_syn/ns150k/train')."
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="dataset.json",
        help="Name of the output JSON file (default: dataset.json)."
    )
    
    args = parser.parse_args()
    create_dataset_json(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()