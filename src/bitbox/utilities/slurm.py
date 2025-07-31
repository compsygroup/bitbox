import os
import paramiko

class SlurmClient:
    def __init__(self, cfg, verbose=True):
        """
        Initialize SlurmClient with config and connect via SSH.
        """
        self.cfg = cfg
        self.verbose = verbose
        self.host = cfg['host']
        self.user = cfg.get('user', os.getlogin())
        self.port = cfg.get('port', 22)
        self.key = cfg.get('key', os.path.expanduser('~/.ssh/id_rsa'))
        self.ssh = None
        self._connect()

    def _connect(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if os.path.exists(self.key):
            if self.verbose:
                print(f"Using SSH key {self.key} for {self.user}@{self.host}")
            ssh.connect(hostname=self.host, port=self.port, username=self.user,
                        key_filename=self.key, look_for_keys=True, allow_agent=True)
        else:
            raise RuntimeError(
                f"No SSH key found at {self.key}. Please create one for remote HPC connection. "
                "Generate with 'ssh-keygen -t rsa' or see https://www.ssh.com/academy/ssh/keygen."
            )
        self.ssh = ssh

    def close(self):
        if self.ssh:
            self.ssh.close()
            self.ssh = None

    def stage_content_to_remote(self, content, remote_path):
        """
        Write content directly to remote_path on server via SFTP.
        """
        parent = os.path.dirname(remote_path)
        self.ssh.exec_command(f"mkdir -p {parent}")
        sftp = self.ssh.open_sftp()
        with sftp.file(remote_path, 'w') as remote_file:
            if self.verbose:
                print(f"Writing content to remote file {remote_path}")
            remote_file.write(content)
        sftp.close()

    def upload_file_to_remote(self, local_path, remote_path):
        """
        Upload a local file to remote_path on server via SFTP.
        """
        if "/mnt/isilon" in remote_path:
            if self.verbose:
                print(f"Skipping upload for path {remote_path}: mounted on HPC under /mnt/isilon")
            return remote_path

        parent = os.path.dirname(remote_path)
        remote_path = os.path.expanduser(remote_path)
        self.ssh.exec_command(f"mkdir -p {parent}")
        sftp = self.ssh.open_sftp()
        if self.verbose:
            print(f"Uploading {local_path} → {remote_path}")
        sftp.put(local_path, remote_path)
        sftp.close()
        return remote_path

    def write_sbatch_script(self, script_path=None, slurm_config=None, remote_path=None, runtime=None,
                           output_dir='/mnt/isilon/schultz_lab/tmp_output', python_path='tmp/run_face_processing.py'):
        """
        Generate and upload (or save locally) an SBATCH script.
        """
        slurm_config = slurm_config or self.cfg
        partition    = slurm_config.get('partition', 'gpuq')
        gres         = slurm_config.get('gres', 'gpu:1')
        cpus_per_task= slurm_config.get('cpus_per_task', 4)
        mem          = slurm_config.get('mem', '10G')
        job_name     = slurm_config.get('job_name', 'bitbox_job')
        tmp_dir      = os.path.join(output_dir, "tmp")

        lines = [
            "#!/bin/bash",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --gres={gres}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={tmp_dir}/{job_name}_%j.log",
            f"#SBATCH --error={tmp_dir}/{job_name}_%j.err",
            "",
            "# Define paths",
            f"RUNTIME=\"{runtime}\"",
            f"OUTPUT_DIR=\"{output_dir}\"",
            "",
            "if [ -d \"$RUNTIME\" ]; then",
            "    echo \"Runtime found at $RUNTIME\"",
            "    mkdir -p \"$RUNTIME/app/input\"",
            "    mkdir -p \"$RUNTIME/app/output\"",
            "else",
            "    echo \"[INFO] RUNTIME is not a valid directory. Skipping sandbox setup.\"",
            "fi",
            f"mkdir -p \"$OUTPUT_DIR\"",
            "",
            f"bash -c \"python3 {python_path}\"",
            f'find "{tmp_dir}" -type f -mtime +0 -exec rm -f {{}} \\;',
            "# Done"
        ]
        script_content = "\n".join(lines)

        if remote_path:
            remote_path = os.path.expanduser(remote_path)
            parent = os.path.dirname(remote_path)
            self.ssh.exec_command(f"mkdir -p {parent}")
            self.stage_content_to_remote(script_content, remote_path)
            return remote_path
        else:
            os.makedirs(os.path.dirname(script_path), exist_ok=True)
            with open(script_path, 'w') as f:
                f.write(script_content)
            return script_path

    def write_python_script(self, input_file, output_dir, parameters=None,
                           python_path='tmp/run_facial_processing.py', processor=None,
                           runtime='bitbox:latest', remote_path=None):
        """
        Generate a Python launcher script (local or remote).
        """
        processor_class = processor.__name__ if processor else 'FaceProcessor3DI'
        parameters = parameters or {}

        lines = [
            "#!/usr/bin/env python3",
            f"from bitbox.face_backend import {processor_class} as FP",
            "",
            f"input_file = '{input_file}'",
            f"output_dir = '{output_dir}'",
            "",
        ]
        for key, value in parameters.items():
            lines.append(f"{key} = {repr(value)}")
        if parameters:
            lines.append("")

        arg_pairs = [f"{key}={key}" for key in parameters]
        arg_pairs.append(f"runtime={repr(runtime)}")
        args_str = ", ".join(arg_pairs)

        lines.extend([
            "# instantiate and run",
            f"processor = FP({args_str})",
            "processor.io(input_file=input_file, output_dir=output_dir)",
            "rect, land, exp_global, pose, land_can, exp_local = processor.run_all(normalize=True)",
        ])
        content = "\n".join(lines)

        if remote_path:
            remote_path = os.path.expanduser(remote_path)
            parent = os.path.dirname(remote_path)
            self.ssh.exec_command(f'mkdir -p {parent}')
            self.stage_content_to_remote(content, remote_path)
            return remote_path
        else:
            os.makedirs(os.path.dirname(python_path), exist_ok=True)
            with open(python_path, 'w') as f:
                f.write(content)
            return python_path

    def slurm_submit(self, processor, input_file=None, output_dir=None):
        """
        Submit a job to Slurm using provided processor and config.
        Returns the job ID.
        """
        venv_path = self.cfg.get('venv_path') or self.cfg.get('remote_output_dir')
        base_remote = os.path.join(venv_path, os.getlogin())
        input_base_name, _ = os.path.splitext(input_file)
        remote_input_dir  = self.cfg.get('remote_input_dir') or os.path.join(base_remote, 'input')
        remote_output_dir = os.path.join(self.cfg.get('remote_output_dir'), output_dir) or os.path.join(base_remote, 'output')

        python_script_path = self.write_python_script(
            input_file=os.path.join(remote_input_dir, input_file),
            output_dir=remote_output_dir,
            processor=processor,
            parameters=self.cfg.get('parameters', {}),
            runtime=self.cfg.get('runtime', 'bitbox:latest'),
            remote_path=os.path.join(remote_output_dir, 'tmp', f"run_face_processing_{input_base_name}.py")
        )

        slurm_script_path = self.write_sbatch_script(
            slurm_config=self.cfg,
            runtime=self.cfg.get('runtime', 'bitbox:latest'),
            remote_path=os.path.join(remote_output_dir, 'tmp', f"run_bitbox_{input_base_name}.sh"),
            output_dir=remote_output_dir,
            python_path=python_script_path
        )

        stdin, stdout, stderr = self.ssh.exec_command(f"source {venv_path}/env/bin/activate && sbatch {slurm_script_path}")
        job_response = stdout.read().decode().strip()
        err = stderr.read().decode().strip()

        print("=== SBATCH SUBMISSION ===")
        print(job_response or "(no response)")
        if err:
            print("=== SBATCH ERROR ===")
            print(err)

        return job_response.split()[-1] if job_response else None

    def slurm_status(self, job_id):
        """
        Check a Slurm job’s state by ID and, if running, pull its stats.
        Returns dict: { job_id, state, stats }
        """
        cmd = f"squeue -h -j {job_id} -o %T"
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        state = stdout.read().decode().strip()

        if exit_code != 0:
            state = "COMPLETED"
        elif not state:
            state = "UNKNOWN"

        stats = None
        if state == "RUNNING":
            stats_cmd = f"scontrol show job {job_id} -o"
            _, stdout2, _ = self.ssh.exec_command(stats_cmd)
            stdout2.channel.recv_exit_status()
            raw = stdout2.read().decode().strip()
            stats = {}
            for pair in raw.split():
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    stats[k] = v

        memory = ""
        gpu = ""
        if stats and "AllocTRES" in stats:
            tres_parts = stats["AllocTRES"].split(",")
            tres = {kv.split("=",1)[0]: kv.split("=",1)[1] for kv in tres_parts if "=" in kv}
            memory = tres.get("mem", "")
            gpu = tres.get("gres/gpu", "")

        user      = stats.get("UserId") if stats else ""
        runtime   = stats.get("RunTime") if stats else ""
        partition = stats.get("Partition") if stats else ""
        cpus      = stats.get("NumCPUs") if stats else ""

        print(f"User     : {user}")
        print(f"State    : {state}")
        print(f"RunTime  : {runtime}")
        print(f"Partition: {partition}")
        print(f"CPUs     : {cpus}")
        print(f"Memory   : {memory}")
        print(f"GPUs     : {gpu}")

        return {"job_id": job_id, "state": state, "stats": stats}


