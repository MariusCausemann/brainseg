from brainseg.utils import get_container_runtime, run_command

def run_synthseg(input_path, output_path, sif_path, do_parcellation=False):
    """
    Runs SynthSeg.
    Logic: Input File -> Output File.
    """
    # 1. Prepare Bind Paths
    parc_flag = "--parc" if do_parcellation else ""
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
        f"--cpu --threads 8 {parc_flag}"
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