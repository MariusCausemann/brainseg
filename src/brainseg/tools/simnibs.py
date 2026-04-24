from brainseg.utils import get_container_runtime, run_command

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