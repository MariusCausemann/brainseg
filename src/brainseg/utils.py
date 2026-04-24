import numpy as np
import nibabel as nib
import sys
import shutil
from pathlib import Path
import subprocess

# Default container names (users can override with --container)
DEFAULT_IMAGES = {
    "synthseg": "brainseg_synthseg.sif",
    "gouhfi": "brainseg_gouhfi.sif",
    "fastsurfer": "brainseg_fastsurfer.sif",
    "simnibs": "brainseg_simnibs.sif",
    "synthstrip": "freesurfer_synthstrip.sif"
}

CONTAINER_URIS = {
    "synthseg": "docker://ghcr.io/mariuscausemann/brainseg:synthseg",
    "gouhfi": "docker://ghcr.io/mariuscausemann/brainseg:gouhfi",
    "fastsurfer": "docker://ghcr.io/mariuscausemann/brainseg:fastsurfer",
    "simnibs": "docker://ghcr.io/mariuscausemann/brainseg:simnibs",
    "synthstrip": "docker://freesurfer/synthstrip:latest"
}

def is_skull_stripped(image_path, brain_threshold_cc=1800):
    """
    Determines if an MRI is skull-stripped based on the physical volume 
    of non-zero (or non-noise) voxels.
    
    Parameters:
    - brain_threshold_cc: Maximum volume in cubic centimeters (cm^3) 
      expected for a stripped brain. Default is 1800cc.
    """
    img = nib.load(image_path)
    data = img.get_fdata()
    
    # Calculate volume of a single voxel in mm^3
    voxel_dims = img.header.get_zooms()[:3]
    voxel_volume_mm3 = np.prod(voxel_dims)
    
    # Use a small intensity threshold to avoid counting background noise
    # (Typical raw MRI noise is low, but not zero)
    nonzero_mask = (data > (np.max(data) * 0.02)) & (~np.isnan(data))
    nonzero_count = np.sum(nonzero_mask)
    
    # Convert mm^3 to cm^3 (cc)
    total_nonzero_volume_cc = (nonzero_count * voxel_volume_mm3) / 1000.0
    
    print(f"Non-zero Volume: {total_nonzero_volume_cc:.2f} cc")
    
    # If the total volume of non-zero tissue is within human brain range, 
    # it's likely stripped. If it's much larger (e.g. 2500cc+), it's a full head.
    return total_nonzero_volume_cc < brain_threshold_cc



def find_container(tool):
    """Finds the container locally, or builds it in ~/.brainseg_containers."""
    image_name = DEFAULT_IMAGES[tool]
    
    # 1. Check current directory and .containers
    if Path(image_name).exists():
        return Path(image_name).resolve()
    elif (Path(".containers") / image_name).exists():
        return (Path(".containers") / image_name).resolve()
    
    # 2. Check the global ~/.brainseg_containers directory
    global_container_dir = Path.home() / ".brainseg_containers"
    sif_path = global_container_dir / image_name
    
    if sif_path.exists():
        return sif_path.resolve()
        
    # 3. If missing entirely, attempt to build it from the Docker registry
    print(f"Container '{image_name}' not found locally.")
    uri = CONTAINER_URIS.get(tool)
    if not uri:
        sys.exit(f"Error: No download URI defined for tool '{tool}'.")
        
    print(f"Building from {uri} to {sif_path}...")
    global_container_dir.mkdir(parents=True, exist_ok=True)
    
    runtime = get_container_runtime()
    
    # Execute the apptainer/singularity build command
    build_cmd = [runtime, "build", str(sif_path), uri]
    run_command(build_cmd, f"Building SIF container for {tool}")
    
    if sif_path.exists():
        return sif_path.resolve()
    else:
        sys.exit(f"Error: Failed to build container to {sif_path}.")

def apply_brain_mask(image_path, mask_path, output_path):
    print(f"Applying brain mask {mask_path.name} to {image_path.name}...")
    img = nib.load(image_path)
    mask_img = nib.load(mask_path)
    
    # Ensure the mask is boolean
    mask_data = mask_img.get_fdata() > 0
    
    # Multiply the image data by the mask (background becomes 0)
    masked_data = img.get_fdata() * mask_data
    
    # Save the skull-stripped image
    nib.save(nib.Nifti1Image(masked_data, img.affine, img.header), output_path)


def get_container_runtime():
    """Returns the command for the available runtime or raises an error."""
    for tool in ["apptainer", "singularity"]:
        if shutil.which(tool):
            return tool
    raise RuntimeError(
        "No container runtime found! Please install Apptainer / Singularity"
        "If using Conda, try: 'conda install -c conda-forge apptainer'"
    )

def run_command(cmd, description):
    """Helper to run a subprocess command with error handling."""
    print(f"--- {description} ---")
    #print(f"running command: {str(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("Done.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error: {description} failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: Could not find the executable '{cmd[0]}'. Is Apptainer installed?")
        sys.exit(1)

