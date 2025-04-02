import subprocess
import re

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
