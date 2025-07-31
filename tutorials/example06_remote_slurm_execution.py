from bitbox.face_backend import FaceProcessor3DI as FP
from bitbox.utilities.slurm import SlurmClient

input_file = 'IS001_parent.mp4'
output_dir = 'output'

slurm_config = {
    'host': 'abc.xyz.edu',   # your login node
    'job_name': 'bitbox_process_test',
    'remote_input_dir': 'hpc/data/path_to_data_directory/',
    'remote_output_dir': 'hpc/data/path_to_output_directory/',
    'venv_path': '/hpc/mnt/path/to/bitbox_venv',
    'runtime': 'bitbox:latest', # docker image or singularity image path or 3di runtime path
    'parameters': {'fast': True, 'camera_model': 30}, # parameters for FaceProcessor3DI
}


sc = SlurmClient(slurm_config)

jobID = sc.slurm_submit(FP,input_file,output_dir) # submits slurm job to server and return jobID
sc.slurm_status(jobID)  # check job status by ID

