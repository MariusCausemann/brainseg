import nibabel as nib
import numpy as np
import argparse
import nibabel.processing


def merge_csf_and_anatomy(
    seg_path, csf_mask_path, out_path, csf_label=24
):

    print(f"Loading GOUHFI parcellation: {seg_path}")
    seg = nib.load(seg_path)
    seg_data = seg.get_fdata().astype(np.int32)

    print(f"Loading T2 CSF Mask: {csf_mask_path}")
    csf_img = nib.load(csf_mask_path)
    if seg.shape != csf_img.shape or not np.allclose(seg.affine, csf_img.affine):
        print(
            "Mismatched dimensions/affine detected. Resampling CSF mask to segmentation space..."
        )
        # order=0 ensures nearest-neighbor interpolation
        csf_img = nib.processing.resample_from_to(csf_img, seg, order=0)

    csf_data = csf_img.get_fdata() > 0
    # Create a copy for our final output
    combined_data = np.copy(seg_data)

    # Erase the old CSF label (Label 24)
    combined_data[combined_data == csf_label] = 0

    ventricles = [
        4,  # Left lateral ventricle
        5,  # Left inferior lateral ventricle
        14,  # Third ventricle
        15,  # Fourth ventricle
        43,  # Right lateral ventricle
        44,  # Right inferior lateral ventricle
        63,
        31, # Add also choroid plexus labels to avoid overwriting them with CSF completely
    ]

    combined_data[(csf_data == True) & (combined_data==0)] = csf_label

    print(f"Saving merged output to: {out_path}")
    new_img = nib.Nifti1Image(combined_data, seg.affine, seg.header)
    nib.save(new_img, out_path)



         
def main():
    parser = argparse.ArgumentParser(
        description="Merge CSF mask with tissue segmentation."
    )
    parser.add_argument("--seg", required=True, help="Path to segmentation file")
    parser.add_argument("--csf", required=True, help="Path to binary CSF mask")
    parser.add_argument("--out", required=True, help="Path to save merged NIfTI")
    args = parser.parse_args()

    merge_csf_and_anatomy(args.seg, args.csf, args.out)


if __name__ == "__main__":
    main()
