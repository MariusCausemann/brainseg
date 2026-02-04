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
        "flags": "--folds 0 --skip_parc --np 1"
    }
}

current_model = MODEL_CONFIG[config["gouhfi_version"]]

SUBJECTS, = glob_wildcards("inputs/{sub}_T1w.nii.gz")

rule all:
    input:
        expand("results/{sub}/gouhfi_seg.nii.gz", sub=SUBJECTS)


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

rule run_gouhfi:
    input:
        t1 = "inputs/{sub}_T1w.nii.gz",
        weights = rules.download_weights.output
    output:
        seg = "results/{sub}/gouhfi_seg.nii.gz"
    conda:
        "envs/gouhfi.yml"
    params:
        model_path = lambda w: os.path.abspath(".models/GOUHFI"),
        flags = current_model["flags"]
    shell:
        """
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export GOUHFI_HOME=$(mktemp -d -t gouhfi_home_XXXXXXXX)
        tmp_io=$(mktemp -d -t gouhfi_io_XXXXXXXX)
        trap "rm -rf $GOUHFI_HOME $tmp_io" EXIT

        ln -s {params.model_path} $GOUHFI_HOME/trained_model
        mkdir -p $tmp_io/input $tmp_io/output

        cp {input.t1} $tmp_io/input/{wildcards.sub}_0000.nii.gz

        run_gouhfi \
            -i $tmp_io/input \
            -o $tmp_io/output \
            --cpu \
            {params.flags}

        mv $tmp_io/output/{wildcards.sub}*.nii.gz {output.seg}
        """