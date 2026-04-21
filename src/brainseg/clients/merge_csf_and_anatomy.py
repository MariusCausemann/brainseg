import nibabel as nib
import numpy as np
import argparse
from nbmorph import dilate_labels_spherical, close_labels_spherical, open_labels_spherical
import nibabel.processing

def merge_csf_and_anatomy(seg_path, csf_mask_path, out_path, csf_label=24, 
                          fill_by_dilation=False):
    print(f"Loading GOUHFI parcellation: {seg_path}")
    seg = nib.load(seg_path)
    seg_data = seg.get_fdata().astype(np.int32)
    
    print(f"Loading T2 CSF Mask: {csf_mask_path}")
    csf_img = nib.load(csf_mask_path)
    if seg.shape != csf_img.shape or not np.allclose(seg.affine, csf_img.affine):
        print("Mismatched dimensions/affine detected. Resampling CSF mask to segmentation space...")
        # order=0 ensures nearest-neighbor interpolation
        csf_img = nib.processing.resample_from_to(csf_img, seg, order=0)

    csf_data = csf_img.get_fdata() > 0
    # Create a copy for our final output
    combined_data = np.copy(seg_data)

    # Erase the old CSF label (Label 24)
    combined_data[combined_data == csf_label] = 0

    # Define the "Brain Anatomy" mask 
    brain_anatomy_mask = combined_data > 0

    #  Add the CSF only where there is NO solid brain tissue
    combined_data[(csf_data == True) & (brain_anatomy_mask == False)] = csf_label

    total_volume_mask = combined_data > 0
    filled_volume_mask = open_labels_spherical(total_volume_mask, radius=1)
    filled_volume_mask = close_labels_spherical(total_volume_mask, radius=5)
    combined_data[~filled_volume_mask] = 0
    # Find exactly where the holes were
    internal_holes = (filled_volume_mask == True) & (total_volume_mask == False)

    num_holes_filled = np.sum(internal_holes)
    
    while num_holes_filled > 0:
        print(f"Found {num_holes_filled} unsegmented background voxels within the cranium.")
        if fill_by_dilation:
            dilated_data = dilate_labels_spherical(combined_data, radius=5)
            combined_data[internal_holes] = dilated_data[internal_holes]
        else: combined_data[internal_holes] = 24
        total_volume_mask = combined_data > 0
        internal_holes = (filled_volume_mask == True) & (total_volume_mask == False)
        num_holes_filled = np.sum(internal_holes)

    print(f"Saving merged output to: {out_path}")
    new_img = nib.Nifti1Image(combined_data, seg.affine, seg.header)
    nib.save(new_img, out_path)


def main():
    parser = argparse.ArgumentParser(description="Merge CSF mask with tissue segmentation.")
    parser.add_argument("--seg", required=True, help="Path to segmentation file")
    parser.add_argument("--csf", required=True, help="Path to binary CSF mask")
    parser.add_argument("--out", required=True, help="Path to save merged NIfTI")
    args = parser.parse_args()
    
    merge_csf_and_anatomy(args.seg, args.csf, args.out)

if __name__ == "__main__":
    main()