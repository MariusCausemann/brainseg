# BrainSeg-container

This repository provides a streamlined Python wrapper and CLI tool to automatically download, run, and standardize outputs from state-of-the-art brain segmentation tools using Apptainer / Singularity containers.

Brain segmentation tools often have conflicting dependencies, complex installation steps, or require specific versions of system libraries. This package solves that problem by containerizing the tools and handling the execution, file binding, and label standardization for you.

![h:8cm](images/comparison_grid.png)
## The Tools

This pipeline currently supports the following deep-learning-based segmentation tools. We may add more in the future.

1. [**GOUHFI**](https://github.com/mafortin/GOUHFI)
   This tool was designed to handle the challenges of Ultra-High Field MRI (7T+). It utilizes "domain randomization" during training, which allows it to remain robust across different MRI contrasts and resolutions, including standard clinical scans.

   * **Resolution:** Native (preserves input resolution).

   * **CSF Availability:** Yes (segments ventricles and subarachnoid space CSF).

2. [**SynthSeg**](https://github.com/BBillot/SynthSeg)
   Developed by the FreeSurfer team, this tool is famous for working "out of the box" on almost any kind of MRI scan (different contrasts, resolutions, or messy clinical data) without needing retraining.

   * **Resolution:** Fixed 1mm isotropic (always resamples input to 1mm).

   * **CSF Availability:** Yes.

3. [**FastSurfer**](https://github.com/Deep-MI/FastSurfer)
   A rapid deep-learning-based segmentation tool that is optimized for speed, finishing in minutes rather than hours.

   * **Resolution:** Native (but experimental below 0.7mm).

   * **CSF Availability:** No (segments ventricles, but ignores subarachnoid space CSF).

4. [**SimNIBS (Charm)**](https://github.com/simnibs/simnibs)
   The "Complete Head Anatomy Reconstruction Method" from the SimNIBS suite. While designed for modeling brain stimulation (TMS/TES), it produces high-quality segmentation of extra-cerebral tissues (skull, scalp, etc.) in addition to the brain.

   * **Resolution:** Native (pipeline uses the upsampled output to match input).

   * **CSF Availability:** Yes.
   * **Segmented regions:** Charm provides the following segmentation labels: White-Matter, Gray-Matter, CSF, Bone, Scalp, Eye_bals, Compact_bone, Spongy_bone, Blood, Muscle, Cartilage, Fat, Electrode, Saline_or_gel

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
brainseg-containers -t <tool_name> -i <input_file.nii.gz> -o <output_file.nii.gz>
```

**Available Tools:** `synthseg`, `gouhfi`, `fastsurfer`, `simnibs`

### Examples

**Run GOUHFI on a single subject:**
```bash
brainseg-containers -t gouhfi -i inputs/sub-01_T1w.nii.gz -o results/sub-01_gouhfi.nii.gz
```

**Run SynthSeg on the same subject:**
```bash
brainseg-containers -t synthseg -i inputs/sub-01_T1w.nii.gz -o results/sub-01_synthseg.nii.gz
```

*Note: You can optionally provide a custom path to a pre-downloaded `.sif` image using the `--container` flag.*


### Note on Labels

Different tools use different numbers to represent brain regions. To make comparison easier, this pipeline automatically **remaps** the output labels of FastSurfer and GOUHFI to match the standard FreeSurfer lookup table.