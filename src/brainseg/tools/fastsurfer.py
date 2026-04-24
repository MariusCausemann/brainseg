from brainseg.utils import get_container_runtime, run_command
import brainseg.data
from importlib import resources
from brainseg.remap import remap_file

def run_fastsurfer(input_path, output_path, sif_path, do_parcellation=False):
    """
    Runs fastsurfer.
    """
    # 1. Prepare Bind Paths
    
    bind_args = [
        "--bind", f"{input_path.parent}:/data_in",
        "--bind", f"{ output_path.parent}:/data_out"
    ]

    fs_outfile = "aparc.DKTatlas+aseg.deep.mgz"
    # 2. Construct the Internal Command
    internal_cmd = (
        "mkdir -p /tmp/out && "
        "/opt/FastSurfer/run_fastsurfer.sh "
        f"--t1 /data_in/{input_path.name} "
        "--sd  /tmp/out --sid sub1 --py python3 "
        f"--seg_only --threads 8 --no_biasfield --no_cereb --no_hypothal --3T && "
        f"rm /data_out/{output_path.name} && "
        f"nib-convert /tmp/out/sub1/mri/{fs_outfile} /data_out/{output_path.name}"
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

    if not do_parcellation:
        old_label_txt = resources.files(brainseg.data).joinpath("freesurfer-label-list-full-lut.txt")
        new_label_txt = resources.files(brainseg.data).joinpath("freesurfer-label-list-reduced-lut.txt")
        remap_file(output_path, old_label_txt, new_label_txt, output_path)