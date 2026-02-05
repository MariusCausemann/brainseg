import nibabel as nib
import fastremap
import numpy as np
import argparse
import sys
import os

def load_label_map(file_path):
    """Parses the txt file to create a dictionary of {name: id}."""
    label_to_id = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                # ID is the first element, Name is the second
                label_id = int(parts[0])
                label_name = parts[1]
                label_to_id[label_name] = label_id
        return label_to_id
    except FileNotFoundError:
        print(f"Error: Label file not found at {file_path}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Exchange NIfTI labels from an old schema to a new schema using fastremap."
    )
    
    # Required arguments
    parser.add_argument("-i", "--input", required=True, help="Path to the input .nii.gz file")
    parser.add_argument("-o", "--output", required=True, help="Path to save the remapped .nii.gz file")
    parser.add_argument("--old-txt", required=True, help="Text file containing old label mapping")
    parser.add_argument("--new-txt", required=True, help="Text file containing new label mapping")
    
    # Optional flag
    parser.add_argument("--inplace", action="store_true", help="Perform remap in-place to save memory")

    args = parser.parse_args()

    print(f"--- Processing: {os.path.basename(args.input)} ---")

    # 1. Load NIfTI
    img = nib.load(args.input)
    data = img.get_fdata().astype(np.int32)
    
    # 2. Parse label files
    old_labels = load_label_map(args.old_txt)
    new_labels = load_label_map(args.new_txt)
    
    # 3. Create mapping {old_id: new_id}
    mapping = {}
    for name, old_id in old_labels.items():
        if name in new_labels:
            mapping[old_id] = new_labels[name]
        else:
            # Default missing labels to 0 (Background)
            mapping[old_id] = 0

    # 4. Perform the remap
    remapped_data = fastremap.remap(data, mapping, in_place=args.inplace)
    
    # 5. Save
    new_img = nib.Nifti1Image(remapped_data, img.affine, img.header)
    nib.save(new_img, args.output)
    
    print(f"Success! Saved to: {args.output}")

if __name__ == "__main__":
    main()