from bitbox.face_backend import FaceProcessor3DI as FP
from bitbox.utilities.slurm import slurm_submit

input_file = 'IS001_parent.mp4'
output_dir = 'output'

slurm_config = {
    'host': 'abc.xyz.edu',   # your login node
    'job_name': 'bitbox_process_test',
    'remote_input_dir': 'hpc/data/path_to_data_directory/',
    'remote_output_dir': 'hpc/data/path_to_output_directory/'
}
processor = FP(fast=True, verbose=True, slurm=True) 

# submits slurm job to server and return jobID
jobID = slurm_submit(processor, slurm_config, input_file=input_file, output_dir=output_dir,runtime='bitbox:latest') 