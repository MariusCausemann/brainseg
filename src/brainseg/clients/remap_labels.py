import argparse
import os
from brainseg.remap import remap_file

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

    remap_file(args.input, args.old_txt, args.new_txt, args.output)
    print(f"Success! Saved to: {args.output}")

if __name__ == "__main__":
    main()