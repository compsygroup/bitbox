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
    elif not bool(os.path.dirname(image)):
        # check if docker image exists locally
        completed = subprocess.run(
            ["docker", "images", "-q", image.lower()],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if completed.returncode == 0:
            return "docker"

    return None