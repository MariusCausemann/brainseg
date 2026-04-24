import argparse
import nibabel as nib
import nibabel.processing

def resample_image(input_path, output_path, voxel_size):
    print(f"Loading: {input_path}")
    img = nib.load(input_path)
    
    print(f"Conforming to {voxel_size}mm isotropic (RAS orientation)...")
    # This re-orients the image to standard RAS and resamples it to the target voxel size
    conformed_img = nib.processing.resample_to_output(
        img, 
        voxel_sizes=(voxel_size, voxel_size, voxel_size)
    )
    
    print(f"Saving to: {output_path}")
    nib.save(conformed_img, output_path)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Conform NIfTI to a specific isotropic resolution.")
    parser.add_argument("-i", "--input", required=True, help="Input NIfTI file")
    parser.add_argument("-o", "--output", required=True, help="Output NIfTI file")
    parser.add_argument("-v", "--voxel-size", type=float, default=0.5, help="Isotropic voxel size (e.g., 0.5)")
    
    args = parser.parse_args()
    resample_image(args.input, args.output, args.voxel_size)

if __name__ == "__main__":
    main()