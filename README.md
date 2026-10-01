# BrainSeg-container

This repository provides a streamlined Python wrapper and CLI tool to automatically download, run, and standardize outputs from state-of-the-art brain segmentation tools using Apptainer / Singularity containers.

Brain segmentation tools often have conflicting dependencies, complex installation steps, or require specific versions of system libraries. This package solves that problem by containerizing the tools and handling the execution, file binding, and label standardization for you.

![h:8cm](https://github.com/MariusCausemann/brainseg/raw/main/images/comparison_grid.png)
## The Tools

This pipeline currently supports the following deep-learning-based segmentation tools. We may add more in the future.

1. [**GOUHFI**](https://github.com/mafortin/GOUHFI)
   This tool was designed to handle the challenges of Ultra-High Field MRI (7T+). It utilizes "domain randomization" during training, which allows it to remain robust across different MRI contrasts and resolutions, including standard clinical scans.

   * **Resolution:** Native (preserves input resolution).

   * **CSF Availability:** Yes (segments ventricles and subarachnoid space CSF). Note that the quality of the SAS segmentation depends on the skull stripping (default antspynet). One can pass in a skull-stripped image (running e.g. synthstrip before), and might obtain better results. 

2. [**SynthSeg**](https://github.com/BBillot/SynthSeg)
   Developed by the FreeSurfer team, this tool is famous for working "out of the box" on almost any kind of MRI scan (different contrasts, resolutions, or messy clinical data).

   * **Resolution:** Fixed 1mm isotropic (always resamples input to 1mm).

   * **CSF Availability:** Yes.

3. [**FastSurfer**](https://github.com/Deep-MI/FastSurfer)
   A rapid deep-learning-based segmentation tool.

   * **Resolution:** Native (but experimental below 0.7mm).

   * **CSF Availability:** No (segments ventricles, but ignores subarachnoid space CSF).

4. [**SimNIBS (Charm)**](https://github.com/simnibs/simnibs)
   The "Complete Head Anatomy Reconstruction Method" from the SimNIBS suite. While designed for modeling brain stimulation (TMS/TES), it produces high-quality segmentation of extra-cerebral tissues (skull, scalp, etc.) in addition to the brain.

   * **Resolution:** Native (pipeline uses the upsampled output to match input).

   * **CSF Availability:** Yes.
   * **Segmented regions:** Charm provides the following segmentation labels: White-Matter, Gray-Matter, CSF, Bone, Scalp, Eye_bals, Compact_bone, Spongy_bone, Blood, Muscle, Cartilage, Fat, Electrode, Saline_or_gel

5. [**SynthStrip**](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)
   A robust, contrast-agnostic brain extraction tool from the FreeSurfer suite. While not a full segmentation tool, it is  optimized for creating accurate brain masks across diverse MRI contrasts and resolutions. 
   * **Resolution:** Native (preserves input resolution).
   * **Output:** Skull-stripped image and binary brain mask.

6. [**risc-mi cerebral arteries (M<sub>S</sub>)**](https://github.com/risc-mi/cerebral-artery-segmentation/tree/main/miua2024a) (`riscmi_arteries`)
   An nnU-Net model from RISC Software GmbH that segments the cerebral arteries from **structural** MRI (T1w, T2w, FLAIR, PD), without needing an angiographic scan. It was trained on IXI and TubeTK subjects with labels derived from the matching TOF-MRA.
   * **Resolution:** Predicted at 0.8 x 0.47 x 0.47 mm internally; the output is returned on the input grid.
   * **Output:** Binary vessel mask (0 = background, 1 = artery). `--save_prob` additionally writes the vessel probability map as `<output>_prob.nii.gz`.
   * **Orientation:** The model is not orientation-agnostic. The wrapper reorients the input to RAI (as recommended by the authors) and maps the prediction back.
   * **Runtime:** The default ensemble (5 folds, each with 8 mirrored test-time-augmentation passes) is expensive at the model's 0.47 mm resolution. Timings for a 0.5 mm whole-head T1w (512x512x368):
     * `--gpu` (NVIDIA A100): ~5 min in total. Requires an NVIDIA driver supporting CUDA 12.1; the container is run with `--nv`.
     * CPU, 128 cores (AMD EPYC 7601): ~80 min and ~110 GB RAM. On the CPU the 40 passes are spread over worker processes with a few threads each, which is several times faster than one multi-threaded process and gives the same result. The number of workers is chosen from the available CPUs and memory (~4 GB per worker); reduce it with `--workers N` if memory is short.
   * **Faster, lower-quality options:** `--no_tta` (8x fewer passes) and `--folds` (e.g. `--folds 0`). On our test scans, dropping TTA from the 5-fold ensemble changed the mask noticeably (Dice ~0.92 against the default, with more fragmented vessels), and a single fold without TTA much more (Dice ~0.72-0.81), so we recommend the default.
   * **Access:** The model weights are © RISC Software GmbH and not openly licensed, so this container is hosted in a **private** registry package and is available on request. Log in before first use with a GitHub token that has the `read:packages` scope: `apptainer registry login --username <github-user> docker://ghcr.io`.
   * **Citation:** Alshenoudy, A., Sabrowsky-Hirsch, B., Scharinger, J., Thumfart, S., Giretzlehner, M. (2024). Towards Segmenting Cerebral Arteries from Structural MRI. MIUA 2024. Springer Nature.

## Comparison Output

The pipeline can automatically generate a comparison grid so you can quickly inspect the differences between the tools.

## Getting Started

### Prerequisites

You must have Apptainer (or Singularity) installed on your system to run the containers.

* Ubuntu/Debian: `sudo apt install apptainer`
* Conda/Mamba: `conda install -c conda-forge apptainer `


### Installation

You can install the package directly via pip:

```bash
pip install brainseg-containers
```

*(Optional) If you want to use the plotting and comparison features, install with the `plot` extras:*
```bash
pip install brainseg-containers[plot]
```

---

## Usage

The package provides a simple command-line interface. The first time you run a specific tool, the wrapper will automatically download the corresponding container from the GitHub Container Registry and store it in `~/.brainseg_containers/`.

### Basic Command

```bash
brainseg -t <tool_name> -i <input_file.nii.gz> -o <output_file.nii.gz>
```

**Available Tools:** `synthseg`, `gouhfi`, `fastsurfer`, `simnibs`, `synthstrip`, `hybrid_gouhfi_T2`, `riscmi_arteries`

### Examples

**Run GOUHFI on a single subject:**
```bash
brainseg -t gouhfi -i inputs/sub-01_T1w.nii.gz -o results/sub-01_gouhfi.nii.gz
```

**Run SynthSeg on the same subject:**
```bash
brainseg -t synthseg -i inputs/sub-01_T1w.nii.gz -o results/sub-01_synthseg.nii.gz
```

*Note: You can optionally provide a custom path to a pre-downloaded `.sif` image using the `--container` flag.*

### Container Location

By default, containers are downloaded and cached in `~/.brainseg_containers/`. You can override this by setting the `BRAINSEG_CONTAINER_DIR` environment variable:

```bash
export BRAINSEG_CONTAINER_DIR=/path/to/shared/containers
brainseg -t gouhfi -i sub-01_T1w.nii.gz -o sub-01_gouhfi.nii.gz
```

This is useful on HPC clusters where containers should be stored on a shared filesystem. The lookup order is:

1. Current working directory
2. `.containers/` subdirectory of the current directory
3. `$BRAINSEG_CONTAINER_DIR` (if set)
4. `~/.brainseg_containers/` (default fallback)

If no container is found, it will be downloaded automatically to `$BRAINSEG_CONTAINER_DIR` (or `~/.brainseg_containers/` if unset).

### Parcellations

`synthseg`, `gouhfi`, `fastsurfer` and `hybrid_gouhfi_T2` also support the `--parc` flag, which enables the cortical segmentation. In this case, the resulting segmentation will contain both the cortical parcellation and the subcortical segmentation.


### Hybrid GOUHFI-CSF Segmentation (`hybrid_gouhfi_T2`)

Standard deep-learning segmentation tools (like GOUHFI or SynthSeg) often struggle to produce accurate and continuous segmentations of the Subarachnoid Space (SAS). For example, GOUHFI's CSF boundary is highly dependent on the initial skull-stripping tool used.

For computational modeling tasks that require highly accurate SAS labels, this package includes a fully automated **T1/T2 Hybrid Pipeline**. This method leverages the structural clarity of a T1w image for solid brain tissue alongside the superior fluid contrast of a T2w image.

**Under the hood, the pipeline automatically:**
1. **Co-registers** your T2 image to your T1 image using ANTsPy.
2. **Brain Extraction:** Runs SynthStrip on the T2 to generate a highly accurate brain mask.
3. **CSF Thresholding:** Extracts a continuous physical fluid mask from the stripped T2 using Li thresholding.
4. **Anatomical Segmentation:** Runs GOUHFI on the T1 image (after masking it with SynthStrip) to create an accurate tissue segmentation.
5. **Topological Merging:** Merges the T1 anatomy with the T2 fluid mask.

**Example Command:**
```bash
brainseg -t hybrid_gouhfi_T2 -i inputs/sub-01_T1w.nii.gz --t2 inputs/sub-01_T2w.nii.gz -o results/sub-01_hybrid_seg.nii.gz
```
### Note on Labels

Different tools use different numbers to represent brain regions. To make comparison easier, this pipeline automatically **remaps** the output labels of FastSurfer and GOUHFI to match the standard FreeSurfer lookup table.