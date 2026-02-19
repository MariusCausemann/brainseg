import argparse
import subprocess
import sys
from pathlib import Path
from brainseg.remap import remap_file
import brainseg.data
from importlib import resources
import shutil

# Default container names (users can override with --container)
DEFAULT_IMAGES = {
    "synthseg": "brainseg_synthseg.sif",
    "gouhfi": "brainseg_gouhfi.sif",
    "fastsurfer": "brainseg_fastsurfer.sif",
    "simnibs": "brainseg_simnibs.sif"
}

CONTAINER_URIS = {
    "synthseg": "docker://ghcr.io/mariuscausemann/brainseg:synthseg",
    "gouhfi": "docker://ghcr.io/mariuscausemann/brainseg:gouhfi",
    "fastsurfer": "docker://ghcr.io/mariuscausemann/brainseg:fastsurfer",
    "simnibs": "docker://ghcr.io/mariuscausemann/brainseg:simnibs"
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

    # 2. Construct the internal command
    internal_cmd = (
        # A. Setup Staging
        "mkdir -p /tmp/in /tmp/masked /tmp/out && "
        f"cp /data_in/{input_path.name} /tmp/in/subject_0000.nii.gz && "
        
        # B. Run Preprocessing (Skull Strip)
        "echo 'Running Preprocessing...' && "
        "run_preprocessing -i /tmp/in -o /tmp/masked && "
        
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

def main():
    parser = argparse.ArgumentParser(description="BrainSeg: Brain Segmentation Wrapper")
    parser.add_argument("-t", "--tool", required=True, help="Tool to run")

    # --- SynthSeg Parser ---
    parser.add_argument("-i", "--input", required=True, type=Path, 
                         help="Input NIfTI file")
    parser.add_argument("-o", "--output", required=True, type=Path, 
                         help="Output NIfTI filename (e.g. out.nii.gz)")
    parser.add_argument("--container", type=Path, help="Path to the container file")
    args = parser.parse_args()

    if args.container:
        sif_path = args.container
    else:
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

if __name__ == "__main__":
    main()