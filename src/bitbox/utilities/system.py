import subprocess
import re
import warnings
import os

def select_gpu():
    try:
        # get GPU memory usage
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True
        )
        gpu_stats = result.stdout.strip().split('\n')
        usage = []
        for line in gpu_stats:
            idx, mem = map(int, re.findall(r'\d+', line))
            usage.append((mem, idx))
        # Return GPU index with least memory used
        return min(usage)[1]
    except Exception:
        # fallback to GPU 0 if anything goes wrong
        return 0

def detect_container_type(image):
    """
    Return:
      - "singularity" if image ends with .sandbox or .sif
      - "docker"      if image contains ':' and exists locally
      - None          otherwise
    """
    if not image:
        return None
    # if it is a path, that means it is not a Docker image but it can be a Singularity sandbox directory
    if bool(os.path.dirname(image)) and os.path.isdir(image) and image.endswith("sandbox"):
        return "singularity"
    elif image.endswith(".sif"):
        return "singularity"
    elif not os.path.isdir(image):
        # Not a local directory, so treat it as a Docker/Podman image reference.
        # Image refs may include a registry/namespace prefix (e.g. localhost/bitbox:openface),
        # so we must not reject names containing '/'.
        try:
            completed = subprocess.run(
                ["docker", "images", "-q", image.lower()],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            return None
        # `docker images -q` prints an image ID only when the image exists locally;
        # the return code is 0 even when nothing matches, so check the output.
        if completed.returncode == 0 and completed.stdout.strip():
            return "docker"

    return None