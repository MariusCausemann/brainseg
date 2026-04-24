from brainseg.utils import get_container_runtime, run_command,is_skull_stripped
import brainseg.data
from importlib import resources
from brainseg.remap import remap, load_label_map
import numpy as np
import nibabel as nib
from brainseg.clients import coregister_images, merge_csf_and_anatomy, extract_csf_mask
import tempfile
from pathlib import Path
from brainseg.tools.synthstrip import run_synthstrip
import sys

def run_gouhfi(input_path, output_path, sif_path, do_parcellation=False, folds="0 1 2 3 4"):
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

    stripped = is_skull_stripped(input_path)
    
    if stripped:
        print(f"Auto-detected skull-stripped input for {input_path.name}. Running GOUHFI conforming ...")
        # Skip run_preprocessing
        prep_cmd = "run_conforming -i /tmp/in -o /tmp/masked && "
    else:
        print(f"Auto-detected raw input for {input_path.name}. Running GOUHFI preprocessing...")
        prep_cmd = "run_preprocessing -i /tmp/in -o /tmp/masked && "

    if do_parcellation:
        try:
            from nbmorph import dilate_labels_spherical as dilate
        except ImportError: 
            sys.exit("GOUHFI parcellation requires nbmorph. Please install with 'pip install nbmorph'")

        parc_flag = "" 
        output_handling_cmd = (
            f"cp /tmp/out/outputs_seg_postpro/subject.nii.gz /data_out/{output_path.name} && "
            f"cp /tmp/out/outputs_parc_postpro/subject.nii.gz /data_out/tmp_parc_{output_path.name}"
        )
    else: 
        parc_flag = "--skip_parc "
        output_handling_cmd = (
            f"cp /tmp/out/outputs_seg_postpro/subject.nii.gz /data_out/{output_path.name}"
        )

    # 2. Construct the internal command
    internal_cmd = (
        # A. Setup Staging
        "mkdir -p /tmp/in /tmp/masked /tmp/out && "
        f"cp /data_in/{input_path.name} /tmp/in/subject_0000.nii.gz && "
        
        # B. Run Preprocessing (Conditionally)
         + prep_cmd +
        
        # C. Run GOUHFI (Segmentation)
        "echo 'Running GOUHFI...' && "
        "run_gouhfi "
        "-i /tmp/masked "
        "-o /tmp/out "
        "--cpu "        # Force CPU mode
        f"--np 1 "
        f"--folds '{folds}' "
        f"{parc_flag}"
        " && ls -R /tmp/out"
        f" && {output_handling_cmd}"
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

    seg_img = nib.load(output_path)
    gouhfi_seg_labels = load_label_map(resources.files(brainseg.data).joinpath("gouhfi-label-list-lut.txt"))
    fs_labels = load_label_map(resources.files(brainseg.data).joinpath("freesurfer-label-list-lut.txt"))
    seg_relabeled = remap(seg_img, gouhfi_seg_labels, fs_labels)

    if not do_parcellation:
        nib.save(seg_relabeled, output_path)

    else:
        out_dir = output_path.parent
        tmp_parc_path = out_dir / f"tmp_parc_{output_path.name}"
        gouhfi_parc_labels = load_label_map(resources.files(brainseg.data).joinpath("gouhfi-label-list-cortex-lut.txt"))
        parc_img = nib.load(tmp_parc_path)
        parc_relabeled = remap(parc_img, gouhfi_parc_labels, fs_labels)

        seg_data = seg_relabeled.get_fdata().astype(np.int32)
        parc_data = parc_relabeled.get_fdata().astype(np.int32)
        
        is_parc_cortex = parc_data > 0
        is_seg_cortex = np.isin(seg_data, [3, 42])
        intersection = np.logical_and(is_parc_cortex, is_seg_cortex).sum()
        overlap_ratio = intersection / is_parc_cortex.sum()
        print(f"Cortical agreement: {overlap_ratio * 100:.2f}%")
        seg_data[is_seg_cortex] = dilate(parc_data, radius=2)[is_seg_cortex]
        assert np.isin(seg_data, [3, 42]).sum() == 0

        # Save the final combined file
        merged_img = nib.Nifti1Image(seg_data, seg_img.affine)
        nib.save(merged_img, output_path)




def run_hybrid_gouhfi_T2(t1_path, t2_path, output_path,
                         gouhfi_sif, synthstrip_sif,
                         do_parcellation=False):
    """
    Runs the hybrid T1+T2 pipeline for high-fidelity CFD meshing:
    1. Coregister T2 -> T1
    2. SynthStrip on T2
    3. apply SynthStrip mask from T2 to the T1
    3. Extract CSF mask from stripped T2
    4. GOUHFI on T1
    5. Merge GOUHFI anatomy with T2 CSF mask
    """
    print(f"\nStarting Hybrid T1+T2 GOUHFI Pipeline...")
    print(f"Target Output: {output_path}")

    # Use a temporary directory 
    with tempfile.TemporaryDirectory() as tmpdir:

        tmp_dir = Path(tmpdir)
        
        coreg_t2_path = tmp_dir / "T2_coreg_to_T1.nii.gz"
        stripped_t2_path = tmp_dir / "T2_stripped.nii.gz"
        csf_mask_path = tmp_dir / "T2_csf_mask.nii.gz"
        stripped_t1_path = tmp_dir / "T1_stripped.nii.gz"
        gouhfi_seg_path = tmp_dir / "gouhfi_anatomy.nii.gz"
        
        # Step 1: Coregister T2 to T1
        print("\n--- STEP 1: Coregistering T2 to T1 ---")
        coregister_images(t1_path, t2_path, coreg_t2_path)
        
        # Step 2: Run SynthStrip on the coregistered T2
        print("\n--- STEP 2: Running SynthStrip on Coregistered T2 ---")
        run_synthstrip(coreg_t2_path, stripped_t2_path, synthstrip_sif,
                       additional_cmds="-b 2")

        # Step 3: Apply the SynthStrip mask to the T1
        print("\n--- STEP 3: Skull-stripping T1 with SynthStrip Mask ---")
        #apply_brain_mask(t1_path, synthstrip_mask_path, stripped_t1_path)
        run_synthstrip(t1_path, stripped_t1_path , synthstrip_sif,
                       additional_cmds="--no-csf")

        # Step 4: Extract CSF mask using Li Thresholding
        print("\n--- STEP 4: Extracting CSF Mask ---")
        extract_csf_mask(stripped_t2_path, csf_mask_path)
        
        # Step 5: Run GOUHFI on the stripped T1
        print("\n--- STEP 5: Running GOUHFI on stripped T1 ---")
        run_gouhfi(stripped_t1_path, gouhfi_seg_path, gouhfi_sif,
                   do_parcellation=do_parcellation)
        
        # Step 6: Merge CSF mask with GOUHFI seg
        print("\n--- STEP 6: Merging T2 CSF with T1 Anatomy ---")
        merge_csf_and_anatomy(gouhfi_seg_path, csf_mask_path, output_path)
        
        print(f"\nHybrid Pipeline Complete! Successfully generated {output_path.name}")