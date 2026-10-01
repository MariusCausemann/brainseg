"""Runs the risc-mi M_S cerebral-artery model (MIUA 2024) on one structural MRI.

The model is not orientation-agnostic: the input is reoriented to RAI (as recommended
upstream) and written as .nrrd (the dataset's file ending). The prediction is mapped
back onto the input grid.

nnU-Net averages the logits of every fold and mirror flip (TTA). On the GPU this runs
in nnU-Net itself. On the CPU, torch scales poorly beyond a few threads (and nnU-Net caps
it at 8), so the (fold, flip) passes are spread over worker processes with a few threads
each and averaged here, which gives the same result up to floating point rounding.
"""

import argparse
import contextlib
import io
import itertools
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

MODEL_DIR = os.environ.get(
    "MODEL_DIR", "/opt/riscmi/Dataset606_MRIVessels/nnUNetTrainerDA5__nnUNetPlans__3d_fullres"
)
THREADS_PER_WORKER = 2
GB_PER_WORKER = 4.0  # memory budget per CPU worker process


class PassPredictor(nnUNetPredictor):
    """Predicts one (fold, flip) pass: the network output for a single mirror flip."""

    flip = ()

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x):
        if not self.flip:
            return self.network(x)
        return torch.flip(self.network(torch.flip(x, self.flip)), self.flip)


def to_input_grid(img, ref, interp):
    orient = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(ref.GetDirection())
    img = sitk.DICOMOrient(img, orient)
    return sitk.Resample(img, ref, sitk.Transform(), interp, 0.0, img.GetPixelID())


def flips(predictor, tta):
    """All mirror flips nnU-Net averages over (as tensor dims), () meaning no flip."""
    axes = [a + 2 for a in predictor.allowed_mirroring_axes] if tta else []
    return [()] + [c for i in range(len(axes)) for c in itertools.combinations(axes, i + 1)]


def available_cpus():
    return len(os.sched_getaffinity(0))


def available_memory_gb():
    with open("/proc/meminfo") as f:
        mem = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}
    avail = mem["MemAvailable"]
    try:  # cgroup v2 limit (containers, Slurm with memory enforcement)
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            avail = min(avail, int(limit) - used)
    except (OSError, ValueError):
        pass
    return avail / 1e9


def load_fold(fold):
    ckpt = Path(MODEL_DIR) / f"fold_{fold}" / "checkpoint_final.pth"
    return torch.load(ckpt, map_location="cpu", weights_only=False)["network_weights"]


def worker(tasks, data, out, lock, threads, done, n_passes, network_file, init):
    torch.set_num_threads(threads)
    # The network comes from a file, not initialize_from_trained_model_folder: that imports every
    # nnU-Net trainer (and with it sklearn/joblib, whose semaphores break worker shutdown).
    with contextlib.redirect_stdout(io.StringIO()):  # nnU-Net's "only supported for cuda" notice
        predictor = PassPredictor(
            perform_everything_on_device=False, device=torch.device("cpu"), allow_tqdm=False
        )
    predictor.manual_initialization(torch.load(network_file, weights_only=False), *init)
    fold, acc = None, None
    while (task := tasks.get()) is not None:
        t0 = time.time()
        if task[0] != fold:
            fold = task[0]
            predictor.network.load_state_dict(load_fold(fold))
        predictor.flip = task[1]
        logits = predictor.predict_sliding_window_return_logits(data).float()
        acc = logits if acc is None else acc.add_(logits)
        with lock:
            done.value += 1
            print(
                f"pass {done.value}/{n_passes} done (fold {fold}, flip {task[1]}) "
                f"in {time.time() - t0:.0f} s",
                flush=True,
            )
    if acc is not None:
        with lock:
            out += acc


def predict_logits_cpu(predictor, data, folds, tta, workers, tmp):
    passes = [(f, fl) for f in folds for fl in flips(predictor, tta)]  # fold-major: few switches
    n_out = predictor.label_manager.num_segmentation_heads
    network_file = Path(tmp) / "network.pt"
    torch.save(predictor.network, network_file)
    init = (
        predictor.plans_manager,
        predictor.configuration_manager,
        None,
        predictor.dataset_json,
        predictor.trainer_name,
        predictor.allowed_mirroring_axes,
    )

    cpus = available_cpus()
    if workers is None:
        by_mem = int(available_memory_gb() // GB_PER_WORKER)
        workers = max(1, min(len(passes), cpus // THREADS_PER_WORKER, by_mem))
    workers = min(workers, len(passes))
    threads = max(1, cpus // workers)
    print(
        f"{len(passes)} passes ({len(folds)} folds x {len(passes) // len(folds)} flips) "
        f"on {workers} worker(s) x {threads} thread(s)",
        flush=True,
    )

    data = torch.from_numpy(data).share_memory_()
    out = torch.zeros((n_out, *data.shape[1:]), dtype=torch.float32).share_memory_()
    ctx = mp.get_context("spawn")
    tasks, lock, done = ctx.Queue(), ctx.Lock(), ctx.Value("i", 0)
    for p in passes + [None] * workers:
        tasks.put(p)
    procs = [
        ctx.Process(
            target=worker,
            args=(tasks, data, out, lock, threads, done, len(passes), network_file, init),
        )
        for _ in range(workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs) or done.value != len(passes):
        raise RuntimeError(
            f"a worker failed ({done.value}/{len(passes)} passes done); if it was killed for "
            "lack of memory, rerun with fewer --workers"
        )
    return out / len(passes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--prob", help="optional output path for the vessel probability map")
    p.add_argument("--folds", default="0 1 2 3 4")
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--gpu", action="store_true", help="run on the GPU (CUDA)")
    p.add_argument("--workers", type=int, help="CPU worker processes (default: automatic)")
    args = p.parse_args()
    folds = [int(f) for f in args.folds.split()]
    if args.gpu and not torch.cuda.is_available():
        raise SystemExit("--gpu given, but no CUDA device is visible")

    ref = sitk.ReadImage(args.input)
    with tempfile.TemporaryDirectory() as tmp:
        tin, tout = Path(tmp) / "in", Path(tmp) / "out"
        tin.mkdir()
        tout.mkdir()
        rai = sitk.DICOMOrient(sitk.Cast(ref, sitk.sitkFloat32), "RAI")
        sitk.WriteImage(rai, str(tin / "subject_0000.nrrd"))

        predictor = nnUNetPredictor(
            use_mirroring=not args.no_tta,
            device=torch.device("cuda" if args.gpu else "cpu"),
            perform_everything_on_device=args.gpu,
        )
        predictor.initialize_from_trained_model_folder(MODEL_DIR, folds, "checkpoint_final.pth")
        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
        data, _, props = preprocessor.run_case(
            [str(tin / "subject_0000.nrrd")],
            None,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
        print(f"preprocessed image: {data.shape[1:]}", flush=True)

        t0 = time.time()
        if args.gpu:
            logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data))
        else:
            logits = predict_logits_cpu(predictor, data, folds, not args.no_tta, args.workers, tmp)
        print(f"prediction took {time.time() - t0:.0f} s", flush=True)

        export_prediction_from_logits(
            logits.cpu(),
            props,
            predictor.configuration_manager,
            predictor.plans_manager,
            predictor.dataset_json,
            str(tout / "subject"),
            bool(args.prob),
            num_threads_torch=available_cpus(),
        )

        seg = sitk.ReadImage(str(tout / "subject.nrrd"))
        sitk.WriteImage(
            sitk.Cast(to_input_grid(seg, ref, sitk.sitkNearestNeighbor), sitk.sitkUInt8),
            args.output,
        )
        if args.prob:
            prob = np.load(tout / "subject.npz")["probabilities"][1].astype(np.float32)
            prob = sitk.GetImageFromArray(prob)
            prob.CopyInformation(seg)
            sitk.WriteImage(to_input_grid(prob, ref, sitk.sitkLinear), args.prob)


if __name__ == "__main__":
    main()
