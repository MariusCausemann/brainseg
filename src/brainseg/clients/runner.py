from pathlib import Path
import argparse
import brainseg.tools


def main():
    from brainseg.utils import find_container

    parser = argparse.ArgumentParser(description="BrainSeg: Brain Segmentation Wrapper")
    subparsers = parser.add_subparsers(
        dest="tool", required=True, help="Segmentation tool to run"
    )

    # Reusable parent parsers to prevent rewriting the same arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-i", "--input", required=True, type=Path, help="Input NIfTI file"
    )
    common_parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output NIfTI filename (e.g. out.nii.gz)",
    )
    common_parser.add_argument(
        "--container", type=Path, help="Path to the container file"
    )

    parc_parser = argparse.ArgumentParser(add_help=False)
    parc_parser.add_argument(
        "--parc", action="store_true", help="Perform cortical parcellation"
    )

    # --- Tool Subcommands ---
    subparsers.add_parser(
        "synthseg", parents=[common_parser, parc_parser], help="Run SynthSeg"
    )
    subparsers.add_parser(
        "gouhfi", parents=[common_parser, parc_parser], help="Run GOUHFI"
    )
    subparsers.add_parser(
        "fastsurfer", parents=[common_parser, parc_parser], help="Run FastSurfer"
    )
    subparsers.add_parser("simnibs", parents=[common_parser], help="Run SimNIBS")
    subparsers.add_parser("synthstrip", parents=[common_parser], help="Run SynthStrip")

    hybrid_parser = subparsers.add_parser(
        "hybrid_gouhfi_T2",
        parents=[common_parser, parc_parser],
        help="Run Hybrid GOUHFI T1+T2",
    )
    hybrid_parser.add_argument(
        "--t2",
        type=Path,
        required=True,
        help="T2w image (Required for hybrid_gouhfi_T2)",
    )
    hybrid_parser.add_argument(
        "--save_tmp_files",
        action="store_true",
        help="Save intermediate files from the pipeline",
    )

    args = parser.parse_args()

    # Make sure output directory exists, if not create it
    try:
        # out_path.parent gets the directory containing the file
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Permission denied: Cannot create directory {args.output.resolve().parent}")
        raise
    except OSError as e:
        print(f"OS error occurred while creating directory {args.output.resolve().parent}: {e}")
        raise

    if args.container:
        sif_path = args.container
    elif "hybrid" not in args.tool:
        sif_path = find_container(args.tool)

    # Safely extract parcellation flag if it exists for the invoked subcommand
    do_parc = getattr(args, "parc", False)
    save_tmp_files = getattr(args, "save_tmp_files", False)

    # Dispatch
    if args.tool == "synthseg":
        brainseg.tools.run_synthseg(
            args.input.resolve(),
            args.output.resolve(),
            sif_path,
            do_parcellation=do_parc,
        )
    elif args.tool == "gouhfi":
        brainseg.tools.run_gouhfi(
            args.input.resolve(),
            args.output.resolve(),
            sif_path,
            do_parcellation=do_parc,
        )
    elif args.tool == "fastsurfer":
        brainseg.tools.run_fastsurfer(
            args.input.resolve(),
            args.output.resolve(),
            sif_path,
            do_parcellation=do_parc,
        )
    elif args.tool == "simnibs":
        brainseg.tools.run_simnibs(
            args.input.resolve(), args.output.resolve(), sif_path
        )
    elif args.tool == "synthstrip":
        brainseg.tools.run_synthstrip(
            args.input.resolve(), args.output.resolve(), sif_path
        )
    elif args.tool == "hybrid_gouhfi_T2":
        # We need both containers for the hybrid pipeline
        gouhfi_sif = find_container("gouhfi")
        synthstrip_sif = find_container("synthstrip")

        brainseg.tools.run_hybrid_gouhfi_T2(
            args.input.resolve(),
            args.t2.resolve(),
            args.output.resolve(),
            gouhfi_sif,
            synthstrip_sif,
            do_parcellation=do_parc,
            save_tmp_files=save_tmp_files,
        )


if __name__ == "__main__":
    main()
