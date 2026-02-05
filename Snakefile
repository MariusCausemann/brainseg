import os

configfile: "config.yml"

MODEL_CONFIG = {
    "v1": {
        "url": "https://zenodo.org/records/15255556/files/GOUHFI.zip",
        "zip": "GOUHFI.zip",
        "folder": "Dataset014_gouhfi",
        "flags": "--v1 --np 1"
    },
    "v2": {
        "url": "https://zenodo.org/records/17920473/files/gouhfi_2p0_brain_seg.zip",
        "zip": "gouhfi_2p0_brain_seg.zip",
        "folder": "Dataset020_gouhfi_2p0n2",
        "flags": "--np 8 --skip_parc"
    }
}

current_model = MODEL_CONFIG[config["gouhfi_version"]]

SUBJECTS, = glob_wildcards("inputs/{sub}_T1w.nii.gz")
segtools = ["synthseg", "gouhfi", "fastsurfer", "simnibs"]

rule all:
    input:
        expand("results/{sub}/{segtools}_seg.nii.gz", sub=SUBJECTS, segtools=segtools),


rule download_weights:
    output:
        directory(f".models/GOUHFI/{current_model['folder']}")
    params:
        url = current_model["url"],
        zip_path = f".models/GOUHFI/{current_model['zip']}",
        out_dir = ".models/GOUHFI"
    shell:
        """
        mkdir -p {params.out_dir}
        curl -fL {params.url} -o {params.zip_path}
        unzip -q -o {params.zip_path} -d {params.out_dir}
        rm {params.zip_path}
        """

rule install_antspynet_weights:
    output:
        ".models/.keras/ANTsXNet/brainExtractionRobustT1.h5"
    shell:
        """
        mkdir -p  .keras/ANTsXNet
        curl -L https://ndownloader.figshare.com/files/34821874 -o {output}
        """

rule run_gouhfi:
    input:
        t1 = "inputs/{sub}_T1w.nii.gz",
        weights = rules.download_weights.output,
        antspynet=rules.install_antspynet_weights.output
    output:
        seg = "results/{sub}/gouhfi_seg.nii.gz"
    conda:
        "envs/gouhfi.yml"
    params:
        model_path = lambda w: os.path.abspath(".models/GOUHFI"),
        flags = current_model["flags"]
    shell:
        """
        export GOUHFI_HOME=$(mktemp -d -t gouhfi_home_XXXXXXXX)
        export HOME=.models
        tmp_io=$(mktemp -d -t gouhfi_io_XXXXXXXX)
        trap "rm -rf $GOUHFI_HOME $tmp_io" EXIT

        ln -s {params.model_path} $GOUHFI_HOME/trained_model
        mkdir -p $tmp_io/input $tmp_io/segmented $tmp_io/output

        cp {input.t1} $tmp_io/input/{wildcards.sub}_0000.nii.gz

        run_preprocessing -i $tmp_io/input -o $tmp_io/masked

        run_gouhfi \
            -i $tmp_io/masked \
            -o $tmp_io/segmented \
            --cpu \
            {params.flags}
        
        run_labels_reordering \
        -i $tmp_io/segmented/outputs_seg_postpro -o $tmp_io/output \
        --old_labels_file ./resources/gouhfi-label-list-lut.txt \
        --new_labels_file ./resources/freesurfer-label-list-lut.txt

        find $tmp_io/output -name "*.nii.gz" | sort | tail -n 1 | xargs -I {{}} mv {{}} {output.seg}
        """


rule setup_synthseg:
    output:
        repo_dir = directory(".models/SYNTHSEG"),
        script   = ".models/SYNTHSEG/scripts/commands/SynthSeg_predict.py",
        weights  = ".models/SYNTHSEG/models/synthseg_2.0.h5"
    params:
        zip_url = "https://github.com/BBillot/SynthSeg/archive/refs/heads/master.tar.gz",
        weights_url = "https://github.com/MariusCausemann/brainseg/releases/download/synthseg2.0/synthseg_2.0.h5"
    shell:
        """
        mkdir -p {output.repo_dir} $(dirname {output.weights})
        curl -Ls {params.zip_url} | tar -xzf - --strip-components=1 -C {output.repo_dir}
        curl -L {params.weights_url} -o {output.weights}
        """

rule run_synthseg:
    input:
        t1 = "inputs/{sub}_T1w.nii.gz",
        script = rules.setup_synthseg.output.script,
    output:
        seg = "results/{sub}/synthseg_seg.nii.gz"
    conda:
        "envs/synthseg.yml"
    shell:
        """
        python {input.script} \
            --i {input.t1} \
            --o {output.seg} \
        """


rule setup_fastsurfer:
    output:
        repo_dir = directory(".models/fastsurfer"),
    params:
        zip_url = "https://github.com/Deep-MI/FastSurfer/archive/refs/tags/v2.4.2.tar.gz",
    shell:
        """
        mkdir -p {output.repo_dir} &&
        curl -Ls {params.zip_url} | tar -xzf - --strip-components=1 -C {output.repo_dir}
        """

rule run_fastsurfer:
    input:
        t1 = "inputs/{sub}_T1w.nii.gz",
        repo=rules.setup_fastsurfer.output.repo_dir
    output:
        seg = "results/{sub}/fastsurfer_seg.nii.gz"
    conda:
        "envs/fastsurfer.yml"
    shell:
        """
        tmp_io=$(mktemp -d -t FS_io_XXXXXXXX)
        trap "rm -rf $tmp_io" EXIT
        export FASTSURFER_HOME=.models/fastsurfer
        $FASTSURFER_HOME/run_fastsurfer.sh \
        --t1  $PWD/{input.t1} \
        --sd  $tmp_io --sid {wildcards.sub} --py python3 \
        --seg_only --threads 8 --no_biasfield --no_cereb --no_hypothal --no_cereb --3T 

        python3 scripts/remap_labels.py \
        -i $tmp_io/{wildcards.sub}/mri/aparc.DKTatlas+aseg.deep.mgz \
        -o results/{wildcards.sub}/fastsurfer_seg.nii.gz \
        --old-txt resources/freesurfer-label-list-full-lut.txt \
        --new-txt resources/freesurfer-label-list-reduced-lut.txt          
        """

rule run_simnibs:
    input:
        t1 = "inputs/{sub}_T1w.nii.gz",
    output:
        seg = "results/{sub}/simnibs_seg.nii.gz"
    conda:
        "envs/simnibs.yml"
    shell:
        """
        pip install git+https://github.com/simnibs/simnibs.git --no-deps
        tmp_io=$(mktemp -d -t SN_io_XXXXXXXX)
        trap "rm -rf $tmp_io" EXIT
        T1_ABS=$(realpath {input.t1})
        
        main_dir=$PWD
        cd "$tmp_io"
        charm {wildcards.sub} $T1_ABS --forcesform
        cp m2m_{wildcards.sub}/label_prep/tissue_labeling_upsampled.nii.gz $main_dir/{output.seg}
        """