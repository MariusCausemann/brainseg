from pathlib import Path

from brainseg.utils import get_container_runtime, run_command


def prob_output_path(output_path):
    """<output>_prob.nii.gz (or .nii) next to the mask output."""
    output_path = Path(output_path)
    name = output_path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return output_path.with_name(name[: -len(suffix)] + "_prob" + suffix)
    return output_path.with_name(name + "_prob.nii.gz")


def run_riscmi_arteries(
    input_path,
    output_path,
    sif_path,
    folds="0 1 2 3 4",
    tta=True,
    save_prob=False,
    gpu=False,
    workers=None,
    prob_path=None,
):
    """
    Runs the risc-mi M_S cerebral-artery model (binary vessel mask, 0/1).

    With save_prob, the vessel probability map is also written, to prob_path
    (default: <output>_prob.nii.gz). prob_path must be in the output directory,
    the only one bound writable into the container.
    """
    bind_args = [
        "--bind",
        f"{input_path.parent}:/data_in",
        "--bind",
        f"{output_path.parent}:/data_out",
    ]

    internal_cmd = [
        "python3",
        "/opt/riscmi/riscmi_predict.py",
        "-i",
        f"/data_in/{input_path.name}",
        "-o",
        f"/data_out/{output_path.name}",
        "--folds",
        folds,
    ]
    if save_prob:
        prob_path = prob_output_path(output_path) if prob_path is None else Path(prob_path)
        if prob_path.resolve().parent != output_path.resolve().parent:
            raise ValueError(f"prob_path {prob_path} is not in {output_path.parent}")
        internal_cmd += ["--prob", f"/data_out/{prob_path.name}"]
    if not tta:
        internal_cmd.append("--no_tta")
    if gpu:
        internal_cmd.append("--gpu")
    if workers is not None:
        internal_cmd += ["--workers", str(workers)]

    gpu_args = ["--nv"] if gpu else []
    cmd = [
        get_container_runtime(), "exec",
        "--cleanenv",
        # $HOME is bound by default: ignore ~/.local packages (e.g. a different torch)
        "--env", "PYTHONNOUSERSITE=1",
        *gpu_args,
        *bind_args,
        str(sif_path),
        *internal_cmd
    ]  # fmt: skip

    run_command(cmd, f"Running risc-mi artery segmentation on {input_path.name}")
