import subprocess
import re
import warnings

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
    if image and image.endswith(("sandbox", ".sif")):
        return "singularity"
    else:
        # check if docker image exists locally
        completed = subprocess.run(
            ["docker", "images", "-q", image],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if completed.returncode == 0:
            return "docker"
        
        warnings.warn(f"Docker image '{image}' not found.")

    return None