import argparse
import subprocess
import sys
from pathlib import Path

# Default container names (users can override with --container)
DEFAULT_IMAGES = {
    "synthseg": "brainseg_synthseg.sif",
    "gouhfi": "brainseg_gouhfi.sif",
    "fastsurfer": "brainseg_fastsurfer.sif",
    "simnibs": "brainseg_simnibs.sif"
}

def run_command(cmd, description):
    """Helper to run a subprocess command with error handling."""
    print(f"--- {description} ---")
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
        "singularity", "exec",
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
        "singularity", "exec",
        "--cleanenv",
        *bind_args,
        str(sif_path),
        "bash", "-c", internal_cmd
    ]

    run_command(cmd, f"Running GOUHFI on {input_path.name}")

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
        "singularity", "exec",
        "--cleanenv",
        *bind_args,
        str(sif_path),
        "bash", "-c", internal_cmd
    ]

    run_command(cmd, f"Running FastSurfer on {input_path.name}")

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
        "singularity", "exec",
        "--cleanenv",
        *bind_args,
        str(sif_path),
        "bash", "-c", internal_cmd
    ]

    run_command(cmd, f"Running simnibs on {input_path.name}")



def main():
    parser = argparse.ArgumentParser(description="BrainSeg: Brain Segmentation Wrapper")
    parser.add_argument("-t", "--tool", required=True, help="Tool to run")

    # --- SynthSeg Parser ---
    parser.add_argument("-i", "--input", required=True, type=Path, 
                         help="Input NIfTI file")
    parser.add_argument("-o", "--output", required=True, type=Path, 
                         help="Output NIfTI filename (e.g. out.nii.gz)")
    parser.add_argument("--container", type=Path, help="Path to brainseg_synthseg.sif")
    args = parser.parse_args()

    image_name = DEFAULT_IMAGES[args.tool]
    if args.container:
        sif_path = args.container
    elif Path(image_name).exists():
        sif_path = Path(image_name).resolve()
    elif (Path(".containers") / image_name).exists():
        sif_path = (Path("containers") / image_name).resolve()
    else:
        sys.exit("Error: Could not find container '{image_name}'." 
                 f"Please use --container to specify its location.")

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