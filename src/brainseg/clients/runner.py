import argparse
import subprocess
import sys
from pathlib import Path
from brainseg.remap import remap_file
import brainseg.data
from importlib import resources
import shutil
import numpy as np
import nibabel as nib
from brainseg.clients import coregister_images, merge_csf_and_anatomy, extract_csf_mask
import tempfile
from pathlib import Path

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

def is_skull_stripped(image_path, zero_threshold=0.4):
    """
    Heuristic to determine if an MRI is already skull-stripped.
    If more than `zero_threshold` (e.g., 40%) of the voxels are exactly 0.0 (or nan),
    it assumes the background has been artificially masked out.
    """
    img = nib.load(image_path)
    data = img.get_fdata()
    zero_ratio = (np.sum(data <= 0) + np.isnan(data).sum())/ data.size
    print(f"zero_ratio: {zero_ratio}")
    return zero_ratio > zero_threshold


def run_synthseg(input_path, output_path, sif_path):
    """
    Runs SynthSeg.
    Logic: Input File -> Output File.
    """
    # 1. Prepare Bind Paths
    
    bind_args = [
        "--bind", f"{input_path.parent}:/data_in",
        "--bind", f"{ output_path.parent}:/data_out"
    ]

    # 2. Construct the Internal Command
    # SynthSeg takes specific file paths.
    internal_cmd = (
        "python /opt/synthseg/scripts/commands/SynthSeg_predict.py "
        f"--i /data_in/{input_path.name} "
        f"--o /data_out/{output_path.name} "
        "--cpu --threads 8"
    )

    # 3. Build Full Apptainer Command
    cmd = [
        get_container_runtime(), "exec",
        "--cleanenv",
        *bind_args,
        str(sif_path),
        "bash", "-c", internal_cmd
    ]

    run_command(cmd, f"Running SynthSeg on {input_path.name}")


def run_gouhfi(input_path, output_path, sif_path):
    """
    Runs GOUHFI.
    """
    # 1. Prepare Bind Paths
    # Input parent -> /data_in
    # Output folder -> /data_out
    bind_args = [
        "--bind", f"{input_path.parent}:/data_in",
        "--bind", f"{output_path.parent}:/data_out"
    ]

    stripped = is_skull_stripped(input_path)
    
    if stripped:
        print(f"Auto-detected skull-stripped input for {input_path.name}. Running GOUHFI conforming ...")
        # Skip run_preprocessing
        prep_cmd = "run_conforming -i /tmp/in -o /tmp/masked && "
    else:
        print(f"Auto-detected raw input for {input_path.name}. Running GOUHFI preprocessing...")
        prep_cmd = "run_preprocessing -i /tmp/in -o /tmp/masked && "

    # 2. Construct the internal command
    internal_cmd = (
        # A. Setup Staging
        "mkdir -p /tmp/in /tmp/masked /tmp/out && "
        f"cp /data_in/{input_path.name} /tmp/in/subject_0000.nii.gz && "
        
        # B. Run Preprocessing (Conditionally)
         + prep_cmd +
        
        # C. Run GOUHFI (Segmentation)
        "echo 'Running GOUHFI...' && "
        "run_gouhfi "
        "-i /tmp/masked "
        "-o /tmp/out "
        "--cpu "        # Force CPU mode
        "--np 8 "       # Use 8 cores
        "--skip_parc "  # Skip parcellation
        " && "
        "find /tmp/out/outputs_seg_postpro -name '*.nii.gz' "
        f" | sort | tail -n 1 | xargs -I {{}} mv {{}} /data_out/{output_path.name}"
    )

    # 3. Build Full Apptainer Command
    cmd = [
        get_container_runtime(), "exec",
        "--cleanenv",
        *bind_args,
        str(sif_path),
        "bash", "-c", internal_cmd
    ]

    run_command(cmd, f"Running GOUHFI on {input_path.name}")
    old_label_txt = resources.files(brainseg.data).joinpath("gouhfi-label-list-lut.txt")
    new_label_txt = resources.files(brainseg.data).joinpath("freesurfer-label-list-lut.txt")
    remap_file(output_path, old_label_txt, new_label_txt, output_path)


def run_fastsurfer(input_path, output_path, sif_path):
    """
    Runs fastsurfer.
    """
    # 1. Prepare Bind Paths
    
    bind_args = [
        "--bind", f"{input_path.parent}:/data_in",
        "--bind", f"{ output_path.parent}:/data_out"
    ]

    # 2. Construct the Internal Command
    internal_cmd = (
        "mkdir -p /tmp/out && "
        "/opt/FastSurfer/run_fastsurfer.sh "
        f"--t1 /data_in/{input_path.name} "
        "--sd  /tmp/out --sid sub1 --py python3 "
        "--seg_only --threads 8 --no_biasfield --no_cereb --no_hypothal --no_cereb --3T && "
        f"nib-convert /tmp/out/sub1/mri/aseg.auto.mgz /data_out/{output_path.name}"
    )

    # 3. Build Full Apptainer Command
    cmd = [
        get_container_runtime(), "exec",
        "--cleanenv",
        *bind_args,
        str(sif_path),
        "bash", "-c", internal_cmd
    ]

    run_command(cmd, f"Running FastSurfer on {input_path.name}")
    old_label_txt = resources.files(brainseg.data).joinpath("freesurfer-label-list-full-lut.txt")
    new_label_txt = resources.files(brainseg.data).joinpath("freesurfer-label-list-reduced-lut.txt")
    remap_file(output_path, old_label_txt, new_label_txt, output_path)

def run_simnibs(input_path, output_path, sif_path):
    """
    Runs simnibs.
    """
    # 1. Prepare Bind Paths
    
    bind_args = [
        "--bind", f"{input_path.parent}:/data_in",
        "--bind", f"{ output_path.parent}:/data_out"
    ]

    # 2. Construct the Internal Command
    internal_cmd = (
        "mkdir -p /tmp/out && cd /tmp/out && "
        "charm sub1 "
        f" /data_in/{input_path.name} "
        "--forcesform --forcerun && "
        "cp m2m_sub1/label_prep/tissue_labeling_upsampled.nii.gz "
        f"/data_out/{output_path.name}"

    )

    # 3. Build Full Apptainer Command
    cmd = [
        get_container_runtime(), "exec",
        "--cleanenv",
        *bind_args,
        str(sif_path),
        "bash", "-c", internal_cmd
    ]

    run_command(cmd, f"Running simnibs on {input_path.name}")

def run_synthstrip(input_path, output_path, sif_path, additional_cmds=None):
    """
    Runs SynthStrip for robust brain extraction.
    """
    bind_args = [
        "--bind", f"{input_path.parent}:/data_in",
        "--bind", f"{output_path.parent}:/data_out"
    ]

    # The freesurfer/synthstrip container's entrypoint takes the arguments directly
    cmd = [
        get_container_runtime(), "run",
        "--cleanenv",
        *bind_args,
        str(sif_path),
        "-i", f"/data_in/{input_path.name}",
        "-o", f"/data_out/{output_path.name}",
        "-m", f"/data_out/{output_path.name.replace('.nii.gz', '_mask.nii.gz')}",
    ]
    if not additional_cmds is None:
        cmd.append(additional_cmds)
    run_command(cmd, f"Running SynthStrip on {input_path.name}")

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

def run_hybrid_gouhfi_T2(t1_path, t2_path, output_path,
                         gouhfi_sif, synthstrip_sif):
    """
    Runs the hybrid T1+T2 pipeline for high-fidelity CFD meshing:
    1. Coregister T2 -> T1
    2. SynthStrip on T2
    3. apply SynthStrip mask from T2 to the T1
    3. Extract CSF mask from stripped T2
    4. GOUHFI on T1
    5. Merge GOUHFI anatomy with T2 CSF mask
    """
    print(f"\nStarting Hybrid T1+T2 GOUHFI Pipeline...")
    print(f"Target Output: {output_path}")

    # Use a temporary directory 
    with tempfile.TemporaryDirectory() as tmpdir:

        tmp_dir = Path(tmpdir)
        
        coreg_t2_path = tmp_dir / "T2_coreg_to_T1.nii.gz"
        stripped_t2_path = tmp_dir / "T2_stripped.nii.gz"
        csf_mask_path = tmp_dir / "T2_csf_mask.nii.gz"
        stripped_t1_path = tmp_dir / "T1_stripped.nii.gz"
        gouhfi_seg_path = tmp_dir / "gouhfi_anatomy.nii.gz"
        
        # Step 1: Coregister T2 to T1
        print("\n--- STEP 1: Coregistering T2 to T1 ---")
        coregister_images(t1_path, t2_path, coreg_t2_path)
        
        # Step 2: Run SynthStrip on the coregistered T2
        print("\n--- STEP 2: Running SynthStrip on Coregistered T2 ---")
        run_synthstrip(coreg_t2_path, stripped_t2_path, synthstrip_sif,
                       additional_cmds="-b 2")

        # Step 3: Apply the SynthStrip mask to the T1
        print("\n--- STEP 3: Skull-stripping T1 with SynthStrip Mask ---")
        #apply_brain_mask(t1_path, synthstrip_mask_path, stripped_t1_path)
        run_synthstrip(t1_path, stripped_t1_path , synthstrip_sif,
                       additional_cmds="--no-csf")

        # Step 4: Extract CSF mask using Li Thresholding
        print("\n--- STEP 4: Extracting CSF Mask ---")
        extract_csf_mask(stripped_t2_path, csf_mask_path)
        
        # Step 5: Run GOUHFI on the stripped T1
        print("\n--- STEP 5: Running GOUHFI on stripped T1 ---")
        run_gouhfi(stripped_t1_path, gouhfi_seg_path, gouhfi_sif)
        
        # Step 6: Merge CSF mask with GOUHFI seg
        print("\n--- STEP 6: Merging T2 CSF with T1 Anatomy ---")
        merge_csf_and_anatomy(gouhfi_seg_path, csf_mask_path, output_path)
        
        print(f"\nHybrid Pipeline Complete! Successfully generated {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="BrainSeg: Brain Segmentation Wrapper")
    parser.add_argument("-t", "--tool", required=True, help="Tool to run")

    # --- SynthSeg Parser ---
    parser.add_argument("-i", "--input", required=True, type=Path, 
                         help="Input NIfTI file")
    parser.add_argument("-o", "--output", required=True, type=Path, 
                         help="Output NIfTI filename (e.g. out.nii.gz)")
    parser.add_argument("--t2", type=Path, help="T2w image (Required for hybrid_gouhfi_T2)")
    parser.add_argument("--container", type=Path, help="Path to the container file")
    args = parser.parse_args()

    if args.container:
        sif_path = args.container
    elif "hybrid" not in args.tool:
        sif_path = find_container(args.tool)

    # Dispatch
    if args.tool == "synthseg":
        run_synthseg(args.input.resolve(), args.output.resolve(), sif_path)
    elif args.tool == "gouhfi":
        run_gouhfi(args.input.resolve(), args.output.resolve(), sif_path)
    elif args.tool == "fastsurfer":
        run_fastsurfer(args.input.resolve(), args.output.resolve(), sif_path)
    elif args.tool == "simnibs":
        run_simnibs(args.input.resolve(), args.output.resolve(), sif_path)
    elif args.tool == "synthstrip":
        run_synthstrip(args.input.resolve(), args.output.resolve(), sif_path)
    if args.tool == "hybrid_gouhfi_T2":
            if not args.t2:
                parser.error("--t2 is required when using the hybrid_gouhfi_T2 tool.")
                
            # We need both containers for the hybrid pipeline
            gouhfi_sif = find_container("gouhfi")
            synthstrip_sif = find_container("synthstrip")
            
            run_hybrid_gouhfi_T2(
                args.input.resolve(), 
                args.t2.resolve(), 
                args.output.resolve(), 
                gouhfi_sif, 
                synthstrip_sif
            )

if __name__ == "__main__":
    main()