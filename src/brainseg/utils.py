import numpy as np
import nibabel as nib
import sys
import shutil
import os
from pathlib import Path
import resource
import signal
import ssl
import subprocess
import tempfile
from importlib import resources

# Default container names (users can override with --container)
DEFAULT_IMAGES = {
    "synthseg": "brainseg_synthseg.sif",
    "gouhfi": "brainseg_gouhfi.sif",
    "fastsurfer": "brainseg_fastsurfer.sif",
    "simnibs": "brainseg_simnibs.sif",
    "synthstrip": "freesurfer_synthstrip.sif",
    "riscmi_arteries": "brainseg_riscmi.sif",
}

CONTAINER_URIS = {
    "synthseg": "docker://ghcr.io/mariuscausemann/brainseg:synthseg",
    "gouhfi": "docker://ghcr.io/mariuscausemann/brainseg:gouhfi",
    "fastsurfer": "docker://ghcr.io/mariuscausemann/brainseg:fastsurfer",
    "simnibs": "docker://ghcr.io/mariuscausemann/brainseg:simnibs",
    "synthstrip": "docker://freesurfer/synthstrip:latest",
    "riscmi_arteries": "docker://ghcr.io/mariuscausemann/brainseg-riscmi:latest",
}

# Images in private GHCR packages; pulling them requires a registry login
PRIVATE_TOOLS = {"riscmi_arteries"}

# Apptainer definition files (shipped with the package) used to build a container locally
# when it cannot be pulled, relative to brainseg/data
LOCAL_RECIPES = {"riscmi_arteries": "riscmi/riscmi.def"}

def is_skull_stripped(image_path, brain_threshold_cc=1800):
    """
    Determines if an MRI is skull-stripped based on the physical volume 
    of non-zero (or non-noise) voxels.
    
    Parameters:
    - brain_threshold_cc: Maximum volume in cubic centimeters (cm^3) 
      expected for a stripped brain. Default is 1800cc.
    """
    img = nib.load(image_path)
    data = img.get_fdata()
    
    # Calculate volume of a single voxel in mm^3
    voxel_dims = img.header.get_zooms()[:3]
    voxel_volume_mm3 = np.prod(voxel_dims)
    
    # Use a small intensity threshold to avoid counting background noise
    # (Typical raw MRI noise is low, but not zero)
    nonzero_mask = (data > (np.max(data) * 0.02)) & (~np.isnan(data))
    nonzero_count = np.sum(nonzero_mask)
    
    # Convert mm^3 to cm^3 (cc)
    total_nonzero_volume_cc = (nonzero_count * voxel_volume_mm3) / 1000.0
    
    print(f"Non-zero Volume: {total_nonzero_volume_cc:.2f} cc")
    
    # If the total volume of non-zero tissue is within human brain range, 
    # it's likely stripped. If it's much larger (e.g. 2500cc+), it's a full head.
    return total_nonzero_volume_cc < brain_threshold_cc



def find_container(tool, build=True):
    """Finds the container locally, or builds it in ~/.brainseg_containers.

    With build=False (e.g. on an air-gapped machine) a missing container raises
    FileNotFoundError instead of pulling from the registry.
    """
    image_name = DEFAULT_IMAGES[tool]

    # 1. Check current directory and .containers
    if Path(image_name).exists():
        return Path(image_name).resolve()
    elif (Path(".containers") / image_name).exists():
        return (Path(".containers") / image_name).resolve()

    # 2. Check BRAINSEG_CONTAINER_DIR environment variable
    env_container_dir = os.environ.get("BRAINSEG_CONTAINER_DIR")
    if env_container_dir and (Path(env_container_dir) / image_name).exists():
        return (Path(env_container_dir) / image_name).resolve()

    # 3. Check the global ~/.brainseg_containers directory
    global_container_dir = Path.home() / ".brainseg_containers"
    sif_path = global_container_dir / image_name
    
    if sif_path.exists():
        return sif_path.resolve()
        
    # 3. If missing entirely, attempt to build it from the Docker registry
    if not build:
        searched = [Path.cwd(), Path(".containers").resolve()]
        if env_container_dir:
            searched.append(Path(env_container_dir))
        searched.append(global_container_dir)
        raise FileNotFoundError(
            f"Container '{image_name}' not found in " + ", ".join(map(str, searched))
        )
    print(f"Container '{image_name}' not found locally.")
    uri = CONTAINER_URIS.get(tool)
    if not uri:
        sys.exit(f"Error: No download URI defined for tool '{tool}'.")
        
    if env_container_dir:
        global_container_dir = Path(env_container_dir)
        sif_path = global_container_dir / image_name
    print(f"Building from {uri} to {sif_path}...")
    if tool in PRIVATE_TOOLS:
        print(
            f"Note: '{tool}' is a private image. If the pull fails with an authentication "
            "error, ask for access and log in with a GitHub token (read:packages scope):\n"
            "  apptainer registry login --username <github-user> docker://ghcr.io\n"
            "or set APPTAINER_DOCKER_USERNAME and APPTAINER_DOCKER_PASSWORD."
        )
    global_container_dir.mkdir(parents=True, exist_ok=True)

    runtime = get_container_runtime()
    build_cmd = [runtime, "build", str(sif_path), uri]

    if tool in LOCAL_RECIPES:
        # try the registry first, then fall back to building from the shipped recipe
        if subprocess.run(build_cmd).returncode != 0:
            print(f"Could not pull {uri}; building '{tool}' locally instead.")
            build_container_locally(tool, sif_path, runtime)
    else:
        run_command(build_cmd, f"Building SIF container for {tool}")

    if sif_path.exists():
        return sif_path.resolve()
    else:
        sys.exit(f"Error: Failed to build container to {sif_path}.")


def _host_ca_bundle():
    """Path of the host's CA bundle, or None."""
    candidates = [
        os.environ.get("SSL_CERT_FILE"),
        ssl.get_default_verify_paths().cafile,
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/cert.pem",
    ]
    return next((Path(c) for c in candidates if c and Path(c).is_file()), None)


def build_container_locally(tool, sif_path, runtime=None):
    """Builds the container for `tool` from its Apptainer definition file in brainseg/data."""
    recipe = resources.files("brainseg.data") / LOCAL_RECIPES[tool]
    runtime = runtime or get_container_runtime()
    sif_path = Path(sif_path).resolve()
    print(
        f"--- Building {tool} from {recipe.name}: this downloads several GB "
        "(PyTorch and the model weights) and takes a while ---"
    )
    # Build in a scratch copy of the recipe directory: the definition file refers to its sibling
    # files relative to the build directory, and also gets the host CA bundle so that pip can
    # download behind networks that intercept TLS.
    with tempfile.TemporaryDirectory(prefix="brainseg_build_") as tmp:
        with resources.as_file(recipe.parent) as recipe_dir:
            shutil.copytree(recipe_dir, tmp, dirs_exist_ok=True)
        ca_bundle = _host_ca_bundle()
        if ca_bundle:
            shutil.copy(ca_bundle, Path(tmp) / "ca-bundle.crt")
        else:
            (Path(tmp) / "ca-bundle.crt").touch()
        try:
            subprocess.run([runtime, "build", str(sif_path), recipe.name], check=True, cwd=tmp)
        except subprocess.CalledProcessError as e:
            sif_path.unlink(missing_ok=True)
            sys.exit(f"Error: local build of '{tool}' failed with exit code {e.returncode}")


def apply_brain_mask(image_path, mask_path, output_path):
    print(f"Applying brain mask {mask_path.name} to {image_path.name}...")
    img = nib.load(image_path)
    mask_img = nib.load(mask_path)
    
    # Ensure the mask is boolean
    mask_data = mask_img.get_fdata() > 0
    
    # Multiply the image data by the mask (background becomes 0)
    masked_data = img.get_fdata() * mask_data
    
    # Save the skull-stripped image
    nib.save(nib.Nifti1Image(masked_data, img.affine, img.header), output_path)


def get_container_runtime():
    """Returns the command for the available runtime or raises an error."""
    for tool in ["apptainer", "singularity"]:
        if shutil.which(tool):
            return tool
    raise RuntimeError(
        "No container runtime found! Please install Apptainer / Singularity"
        "If using Conda, try: 'conda install -c conda-forge apptainer'"
    )

def run_command(cmd, description):
    """Helper to run a subprocess command with error handling."""
    print(f"--- {description} ---")
    #print(f"running command: {str(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("Done.\n")
    except subprocess.CalledProcessError as e:
        if e.returncode < 0:
            sig = signal.Signals(-e.returncode).name
            print(f"Error: {description} was killed by {sig}")
            limit, _ = resource.getrlimit(resource.RLIMIT_AS)
            if limit != resource.RLIM_INFINITY:
                print(
                    f"Hint: this shell limits virtual memory to {limit / 2**30:.0f} GiB "
                    "(ulimit -v). CUDA/PyTorch map much more address space than they use, "
                    "which can crash with SIGSEGV. Lift the limit, e.g. in Slurm jobs with "
                    "'srun --propagate=NONE ...'."
                )
            sys.exit(128 - e.returncode)
        print(f"Error: {description} failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: Could not find the executable '{cmd[0]}'. Is Apptainer installed?")
        sys.exit(1)

