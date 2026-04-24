from brainseg.utils import get_container_runtime, run_command


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