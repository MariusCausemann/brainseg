import argparse
import math
import matplotlib.pyplot as plt
from nilearn import plotting
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import nibabel as nib
plt.style.use('dark_background')


lut_path = "resources/freesurfer-label-list-lut.txt"

def create_exact_colormap(lut_path, alpha=0.5):
    """
    Creates a sparse ListedColormap where the index matches the label ID exactly.
    Indices not found in the LUT are transparent (alpha=0).
    """

    # Read LUT (handles whitespace and comments)
    df = pd.read_csv(
        lut_path, 
        sep=r"\s+", 
        comment="#", 
        header=None, 
        names=["index", "name", "r", "g", "b", "a"],
        dtype={"index": int, "r": int, "g": int, "b": int}
    )
    
    # 1. Determine the size of the colormap needed
    # We need an array large enough to hold the highest label ID found.
    # e.g., if max ID is 14175, we need 14176 entries (0 to 14175).
    max_id = df["index"].max()

    df.loc[df["name"]=="CSF", ['r', 'g', 'b']] = [0, 255, 255]
    
    # 2. Initialize RGBA array with Transparent (0,0,0,0)
    # nilearn will overlay this on the T1, so 0 alpha means "show T1"
    lut_colors = np.zeros((max_id + 1, 4))
    
    # 3. Fill in the specific indices defined in the LUT
    # Normalize 0-255 RGB to 0-1 for Matplotlib
    indices = df["index"].values
    rgbs = df[["r", "g", "b"]].values / 255.0
    
    # Set RGB colors
    lut_colors[indices, 0:3] = rgbs
    # Set Alpha to 1.0 (Opaque) for defined labels
    lut_colors[indices, 3] = alpha
    
    # Create the discrete colormap
    # 'N' is implicitly len(lut_colors), ensuring 1:1 mapping
    custom_cmap = ListedColormap(lut_colors, name="FreeSurfer_Discrete")
    
    return custom_cmap, max_id


def get_ventricle_center(seg_path):
    """
    Calculates coordinates centered specifically on the ventricles.
    """
    print(f"Calculating center from ventricles in: {seg_path}")
    img = nib.load(seg_path)
    
    # FreeSurfer Standard Labels for Ventricles:
    # 4: Left-Lateral-Ventricle
    # 14: 3rd-Ventricle
    # 43: Right-Lateral-Ventricle
    target_labels = [4, 14, 43]
    
    # Create binary mask of ventricles
    mask_data = np.isin(img.get_fdata(), target_labels).astype(float)
    
    new_img = nib.Nifti1Image(mask_data, img.affine, img.header)
    
    return plotting.find_xyz_cut_coords(new_img)


fs_labeled = ["synthseg", "gouhfi", "fastsurfer"]

def main():
    parser = argparse.ArgumentParser(
        description="Generate a grid of segmentation overlays in ortho view (Sagittal, Coronal, Axial)."
    )
    
    parser.add_argument("-i", "--image", required=True, help="Path to the T1w anatomical image (background).")
    parser.add_argument("-s", "--segs", required=True, nargs='+', help="List of segmentation NIfTI files.")
    parser.add_argument("-o", "--output", default="seg_comparison_ortho.png", help="Path to save the output image.")

    args = parser.parse_args()

    cmap, vmax = create_exact_colormap(lut_path, alpha=0.6)

    # 1. Setup Grid
    n_segs = len(args.segs)
    cols = 1
    rows = math.ceil(n_segs / cols)
    
    # Calculate figure size: Allocate more width (10 inches) per column for the ortho view
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(8 * cols, 2.45 * rows), 
        gridspec_kw={'wspace': 0, 'hspace': 0} 
    )    
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # Flatten axes for consistent indexing
    if n_segs > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # 2. Determine Cut Coordinates automatically
    cut_coords = get_ventricle_center(args.segs[0])
    
    print(f"Visualizing ortho slices at coordinates: {np.round(cut_coords,2)}")

    # 3. Loop through segmentations and plot
    for i, seg_path in enumerate(args.segs):
        print(f"Processing: {seg_path}")
        ax = axes[i]
        # Determine title from filename
        title = seg_path.split("/")[-1].replace(".nii.gz", "").replace(".nii", "").replace("_seg", "")
        
        plotting.plot_roi(
            roi_img=seg_path,
            bg_img=args.image,
            axes=ax,
            display_mode='ortho',  # Shows Sagittal, Coronal, and Axial cuts
            cut_coords=cut_coords, # Ensures all models show the exact same anatomical point
            cmap="tab20", #cmap,
            vmax=vmax if title in fs_labeled else None,
            vmin=0,
            #title=title,
            alpha=0.4,
            resampling_interpolation='nearest',
            annotate=False,
            draw_cross=True,
            black_bg=True,
            colorbar=False  
        )
        fig.text(0.01, 0.95, title,
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='top',
            color="white",
            fontsize=10,
            weight='bold',
            zorder=1000
        )

    plt.savefig(args.output, dpi=300, pad_inches=0.0, bbox_inches='tight')
    print(f"Saved comparison to: {args.output}")

if __name__ == "__main__":
    main()