import argparse
import nibabel as nib
import numpy as np
from skimage.filters import threshold_li
from skimage.measure import label

def extract_csf_mask(t2_stripped_path, output_path):
    print(f"Loading skull-stripped T2: {t2_stripped_path}")
    img = nib.load(t2_stripped_path)
    data = img.get_fdata()

    # 1. Li Thresholding
    # We only want to calculate the threshold based on the actual brain/fluid voxels,
    # so we exclude the dark background (zeros) from the calculation.
    brain_voxels = data[data > 0]
    
    if len(brain_voxels) == 0:
        raise ValueError("The input image appears to be empty (all zeros).")

    li_thresh = threshold_li(brain_voxels)
    print(f"Calculated Li Threshold for CSF: {li_thresh:.2f}")

    # Create the initial binary mask of everything above the threshold
    binary_fluid_mask = data > li_thresh

    # 2. Keep Only the Largest Connected Component
    print("Running 3D connected component analysis...")
    labeled_mask = label(binary_fluid_mask, connectivity=2)
    
    # Count the size of each component
    counts = np.bincount(labeled_mask.ravel())
    # We set the count of the background (label 0) to 0  so it isn't accidentally chosen as the largest component.
    counts[0] = 0  
    
    # Find the label with the maximum volume
    largest_cc_label = counts.argmax()
    
    # Extract only that component
    final_csf_mask = (labeled_mask == largest_cc_label).astype(np.uint8)
    
    csf_volume_voxels = np.sum(final_csf_mask)
    print(f"Largest CSF component isolated. Size: {csf_volume_voxels} voxels.")

    # 3. Save the result
    print(f"Saving CSF mask to: {output_path}")
    new_img = nib.Nifti1Image(final_csf_mask, img.affine)
    nib.save(new_img, output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a CSF mask from a skull-stripped T2 using Li thresholding."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the skull-stripped T2 NIfTI file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the binary CSF mask.")
    
    args = parser.parse_args()
    extract_csf_mask(args.input, args.output)