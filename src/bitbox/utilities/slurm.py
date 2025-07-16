import os
import paramiko

def connect_slurm(cfg, verbose=True):
    """
    Establish an SSHClient to a Slurm host, using key-based auth first
    then falling back to password. Returns a connected SSHClient.
    """
    
    host   = cfg['host']
    user   = cfg.get('user', os.getlogin())
    port   = cfg.get('port', 22)
    key    = cfg.get('key', os.path.expanduser('~/.ssh/id_rsa'))  # Use provided key or default

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    if os.path.exists(key):
        if verbose:
            print(f"Using SSH key {key} for {user}@{host}")
        ssh.connect(hostname=host, port=port, username=user,
                    key_filename=key, look_for_keys=True, allow_agent=True)
    else:
        raise RuntimeError(
            f"No SSH key found at {key}. Please create one for remote HPC connection. "
            "You can generate an SSH key using the command 'ssh-keygen -t rsa' or refer to "
            "the documentation at https://www.ssh.com/academy/ssh/keygen for detailed instructions."
        )

    return ssh

def stage_content_to_remote(
    ssh_client: paramiko.SSHClient,
    content: str,
    remote_path: str,
    verbose: bool = True
) -> None:
    """
    Write `content` directly to `remote_path` on the server via SFTP,
    creating parent dirs if needed, without a local temp file.
    """
    parent = os.path.dirname(remote_path)
    ssh_client.exec_command(f"mkdir -p {parent}")
    sftp = ssh_client.open_sftp()
    with sftp.file(remote_path, 'w') as remote_file:
        if verbose:
            print(f"Writing content to remote file {remote_path}")
        remote_file.write(content)
    sftp.close()

def upload_file_to_remote(
    ssh_client: paramiko.SSHClient,
    local_path: str,
    remote_path: str,
    verbose: bool = True
) -> str:
    """
    Upload a local file to `remote_path` on the server via SFTP,
    creating parent dirs if needed. If `remote_path` contains `/mnt/isilon`,
    skip the upload and return the `remote_path` as is.
    """
    if "/mnt/isilon" in remote_path:
        if verbose:
            print(f"Skipping upload for path {remote_path}: mounted on HPC under /mnt/isilon")
        return remote_path

    parent = os.path.dirname(remote_path)
    remote_path = os.path.expanduser(remote_path)
    ssh_client.exec_command(f"mkdir -p {parent}")
    sftp = ssh_client.open_sftp()
    if verbose:
        print(f"Uploading {local_path} → {remote_path}")

    sftp.put(local_path, remote_path)
    sftp.close()
    return remote_path



def write_sbatch_script(
    script_path: str = None,
    slurm_config: dict = None,
    ssh_client: paramiko.SSHClient = None,
    remote_path: str = None,
    runtime: str = None,
    output_dir: str = '/mnt/isilon/schultz_lab/tmp_output',
) -> str:
    """
    Generate an SBATCH script that:
    - builds or reuses a Singularity sandbox
    - ensures input/output dirs
    - runs the facial-processing example inside it.
    If ssh_client and remote_path are provided, send script to remote via SFTP.
    Returns the script_path (local or remote).
    """

    # Load config with defaults
    partition    = slurm_config.get('partition', 'gpuq')
    gres         = slurm_config.get('gres', 'gpu:1')
    cpus_per_task= slurm_config.get('cpus_per_task', 4)
    mem          = slurm_config.get('mem', '10G')
    job_name     = slurm_config.get('job_name', 'bitbox_job')
    output_dir   = output_dir


    lines = [
        "#!/bin/bash",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={job_name}_%j.log",
        f"#SBATCH --error={job_name}_%j.err",
        "",
        "# Define paths",
        f"RUNTIME=\"{runtime}\"",
        f"OUTPUT_DIR=\"{output_dir}\"",
        "",
        "# Step 1: If RUNTIME is a valid directory, check sandbox structure",
        "if [ -d \"$RUNTIME\" ]; then",
        "    echo \"Runtime found at $RUNTIME\"",
        "",
        "    # Ensure app/input and app/output directories exist",
        "    mkdir -p \"$RUNTIME/app/input\"",
        "    mkdir -p \"$RUNTIME/app/output\"",
        "else",
        "    echo \"[INFO] RUNTIME is not a valid directory. Skipping sandbox setup.\"",
        "fi",
        "",
        "# Step 2: Ensure host output directory exists",
        f"mkdir -p \"$OUTPUT_DIR\"",
        "",
        "# Run the processing script inside the sandbox",
        f"bash -c \"python3 run_face_processing.py\""
    ]

    script_content = "\n".join(lines)

    if ssh_client and remote_path:
        # send SBATCH script to remote host
        remote_path = os.path.expanduser(remote_path)
        parent = os.path.dirname(remote_path)
        ssh_client.exec_command(f"mkdir -p {parent}")
        stage_content_to_remote(ssh_client, script_content, remote_path)
        return remote_path
    else:
        # write locally
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        return script_path


def write_python_script(  
    input_file: str,
    output_dir: str,
    python_path: str = 'tmp/run_facial_processing.py',
    processor_class: str = 'FaceProcessor3DI',
    processor_kwargs: dict = None,
    ssh_client: paramiko.SSHClient = None,
    remote_path: str = None
) -> str:
    """
    Generate a Python launcher script that:
      - imports ProcessorClass as FP
      - sets input_file and output_dir
      - applies processor_kwargs
      - instantiates and runs the processor
    If ssh_client and remote_path are provided, writes directly to remote via SFTP.
    Otherwise, writes locally to python_path.
    Returns the path to the generated script (local or remote).
    """
    # build script lines
    lines = [
        "#!/usr/bin/env python3",
        f"from bitbox.face_backend import {processor_class} as FP",
        "",
        f"input_file = '{input_file}'",
        f"output_dir = '{output_dir}'",
        "",
    ]
    if processor_kwargs:
        for k, v in processor_kwargs.items():
            val = f"'{v}'" if isinstance(v, str) else v
            lines.append(f"{k} = {val}")
        lines.append("")
    args_list = [f"{k}={k}" for k in (processor_kwargs or {}).keys()]
    args_str = ", ".join(args_list)
    lines.extend([
        "# instantiate and run",
        f"processor = FP({args_str})" if args_str else "processor = FP()",
        "processor.io(input_file=input_file, output_dir=output_dir)",
        "rect, land, exp_global, pose, land_can, exp_local = processor.run_all(normalize=True)",
    ])
    content = "\n".join(lines)
    # decide local vs remote write
    if ssh_client and remote_path:
        remote_path = os.path.expanduser(remote_path)
        parent      = os.path.dirname(remote_path)
        ssh_client.exec_command(f'mkdir -p {parent}')
        stage_content_to_remote(ssh_client, content, remote_path)
        return remote_path
    else:
        os.makedirs(os.path.dirname(python_path), exist_ok=True)
        with open(python_path, 'w') as f:
            f.write(content)
        return python_path
    

def slurm_submit(processor, slurm_config, input_file=None, output_dir=None,runtime='bitbox:latest'):
    """
    Submit a job to Slurm using the provided processor and configuration.
    If input_file or output_dir are specified, set them on the processor.
    Returns the job ID.
    """
    remote_root = "/mnt/isilon/schultz_lab/cluster/isilon_usr/bitbox_compsy"
    base_remote = os.path.join(remote_root, os.getlogin()) # impute with the current user name
    remote_input_dir  = slurm_config.get('remote_input_dir') or os.path.join(base_remote, 'input')
    remote_output_dir = os.path.join(slurm_config.get('remote_output_dir'),output_dir) or os.path.join(base_remote, 'output')
    sandbox_dir = os.path.join(remote_root, 'bitbox', "bitbox_sandbox")
    ssh_client=connect_slurm(slurm_config, verbose=True)

    processor_kwargs = {
        k: v for k, v in processor.init_args.items()
        if k not in {'self', 'server', '__class__'}
    }

    processor_kwargs.update({f"arg_{i}": arg for i, arg in enumerate(processor_kwargs.pop('args', []))}, **processor_kwargs.pop('kwargs', {}))

    write_python_script(
        input_file   = os.path.join(remote_input_dir, input_file),
        output_dir   = remote_output_dir,
        processor_class = processor.__class__.__name__,
        processor_kwargs = processor_kwargs,     # ← use your filtered dict here
        ssh_client   = ssh_client,
        remote_path  = os.path.join(base_remote, "run_face_processing.py")
    )
        
    script_remote = write_sbatch_script(
                slurm_config=slurm_config,
                ssh_client= ssh_client,
                runtime = runtime,
                remote_path=os.path.join(base_remote, "run_bitbox_ssh.sh"),
                output_dir=remote_output_dir
            )
    
    stdin, stdout, stderr = ssh_client.exec_command(f"cd {base_remote} &&  source {remote_root}/env/bin/activate && sbatch {script_remote}")
    job_response = stdout.read().decode().strip()
    err = stderr.read().decode().strip()

    print("=== SBATCH SUBMISSION ===")
    print(job_response or "(no response)")
    if err:
        print("=== SBATCH ERROR ===")
        print(err)

    ssh_client.close()
    return job_response.split()[-1] if job_response else None