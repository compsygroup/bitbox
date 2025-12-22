import os
import warnings
import inspect
import requests
import json
import sys
import zipfile
import io
from typing import List, Optional, Sequence, Tuple
from time import time

import numpy as np

from ..utilities import (
    FileCache,
    generate_file_hash,
    select_gpu,
    detect_container_type,
    visualize_and_export,
    visualize_and_export_can_land,
    visualize_expressions_3d,
    visualize_and_export_pose,
    visualize_bfm_expression_pose,
    check_data_type,
)
from .reader3DI import read_expression, read_pose, read_pose_lite

class FaceProcessor:
    """Base class for GPU-backed face processing pipelines.

    The class encapsulates all shared logic for running either local binaries or
    remote API calls, configuring Docker/Singularity runtimes, and caching
    intermediate outputs. Subclasses only implement the backend-specific steps
    such as face detection, landmarking, and morphable-model fitting.
    """

    def __init__(self, runtime=None, return_output='dict', server=None, verbose=True, debug=False):
        """Initialize a processor instance and optionally attach to a remote API.

        Args:
            runtime: Path to a local backend install (e.g., `/opt/3DI`), or the
                container image identifier when running inside Docker/Singularity.
            return_output: Controls whether helper methods return file paths
                (`'file'`), parsed dictionaries (`'dict'`), or nothing.
            server: Optional mapping with ``host`` and ``port`` keys to execute
                jobs remotely via the Bitbox API service.
            verbose: When ``True`` logs progress for each executed step.
            debug: Enables verbose runtime logging without changing ``verbose``.
        """
        self.verbose = verbose
        self.debug = debug
        self.input_dir = None
        self.output_dir = None
        self.file_input = None
        self.output_ext = '.bit'
        
        self.runtime = runtime
        self.execDIR = None
         
        self.docker = None
        self.docker_input_dir = '/app/input'
        self.docker_output_dir = '/app/output'
        
        self.cache = FileCache()
        
        # whether to return the output or not
        self.return_output = return_output
        
        # prepare metadata
        self.base_metadata = {}
        
        # prepare data for API calls
        self.API_config = {}
        self.API = False
        # this is used to keep track of the active calls to the server and prevent redundant calls in _execute method
        self.API_callers = set()
        
        if server is not None: # Commands will be run on a remote server
            self.API = True
            # server address
            self.API_url = f"http://{server['host']}:{server['port']}/"
            self.API_session = None
            
            # get the configuration from the caller object (e.g., FaceProcessor3DI, FaceProcessor3DIlite)
            # we will send this information to the server
            frame = inspect.currentframe()
            outer = inspect.getouterframes(frame)
            if len(outer) > 1:
                caller_frame = outer[1].frame
                args = inspect.getargvalues(caller_frame).locals
                args.pop('__class__', None)
                args.pop('self', None)
                args.pop('server', None)
                self.API_config = dict(args)
                
            # test the connection to the server
            try:
                response = requests.get(self.API_url, timeout=3)
                if response.status_code == 200:
                    print(f"Connected to the server: {self.API_url}")
                else:
                    raise ValueError(f"Server returned status: {response.status_code}")
            except requests.RequestException:
                raise ValueError(f"Server is down or unreachable.")
            
    
    def _set_runtime(self, name='3DI', variable='BITBOX_3DI', executable='video_learn_identity', docker_path='/app/3DI'):
        """Resolve where the backend executables live.

        The method prefers an explicit ``runtime`` path, falls back to the
        ``BITBOX_DOCKER`` container, and finally inspects the relevant
        environment variable (``BITBOX_3DI`` by default).

        Args:
            name: Human readable name of the backend (only used in errors).
            variable: Environment variable that stores the install path.
            executable: File that must exist inside the install directory.
            docker_path: Path that executables live in inside Docker images.

        Raises:
            ValueError: If a valid backend install cannot be located.
        """
        # check if we are using a Docker image
        if self.runtime and detect_container_type(self.runtime):
            self.docker = self.runtime
        elif (not self.runtime) and os.environ.get('BITBOX_DOCKER') and detect_container_type(os.environ.get('BITBOX_DOCKER')):
            self.docker = os.environ.get('BITBOX_DOCKER')
        
        # if we are not using the docker container, we need to find out where the package is installed
        if self.docker is None:
            if self.runtime and bool(os.path.dirname(self.runtime)) and os.path.isdir(self.runtime):
                # check if it is a valid path including the executable
                if os.path.exists(os.path.join(self.runtime, executable)):
                    self.execDIR = self.runtime
                else:
                    raise ValueError(f"{name} package is not found. Please make sure runtime is set to a valid {name} path.")
            elif os.environ.get(variable): # we need to find the package in the system
                    # check if it is a valid path including the executable
                    if os.path.exists(os.path.join(os.environ.get(variable), executable)):
                        self.execDIR = os.environ.get(variable)
                    else:
                        raise ValueError(f"{name} package is not found. Please make sure {variable} system variable is set to a valid {name} path.")
        else:
            self.execDIR = docker_path       
    
    
    def _local_file(self, file_path):
        """Translate Docker paths back into host paths.

        Args:
            file_path: Absolute path that may point inside the container mount.

        Returns:
            str: Path that can be accessed from the host filesystem.
        """
        file_dir = os.path.dirname(file_path)
        
        if file_dir == self.docker_input_dir:
            return os.path.join(self.input_dir, os.path.basename(file_path))
        elif file_dir == self.docker_output_dir:
            return os.path.join(self.output_dir, os.path.basename(file_path))
        else:
            return file_path
          
          
    def io(self, input_file, output_dir):
        """Validate I/O paths and prime all intermediate filenames.

        Args:
            input_file: Video to process. Relative paths are resolved to abs paths.
            output_dir: Directory where backend artifacts will be written.

        Raises:
            ValueError: If the file is missing, unsupported, or output_dir cannot
                be created.
        """
        # supported video extensions
        supported_extensions = ['mp4', 'avi', 'mpeg']
        
        # Check if input & output are relative paths and make them absolute
        if not os.path.isabs(input_file):
            input_file = os.path.abspath(input_file)  # input file name with path
        if not os.path.isabs(output_dir):
            output_dir = os.path.abspath(output_dir) 
        
        self.input_dir = os.path.dirname(input_file)
        self.output_dir = output_dir
            
        # generate a hash for the input file
        self.base_metadata['input_hash'] = generate_file_hash(input_file)

        # check if input file exists
        if not os.path.exists(input_file):
            raise ValueError("Input file %s does not exist. Please check the path and permissions."%input_file)
        
        # check if input file extension is supported
        ext = input_file.split('.')[-1].lower()
        if not (ext in supported_extensions):
            raise ValueError("Input file extension is not supported. Please use one of the following extensions: %s" % supported_extensions)
        
        # if no exception is raised, set the input file and output directory
        if self.docker is None:
            _input_dir = self.input_dir
            _output_dir = self.output_dir
            # create output directory
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except:
                raise ValueError("Cannot create output directory. Please check the path and permissions.")
        else:
            _input_dir = self.docker_input_dir
            _output_dir = self.docker_output_dir  
            
        # set all the files
        self.file_input_base = '.'.join(os.path.basename(input_file).split('.')[:-1])
        self.file_input = os.path.join(_input_dir, self.file_input_base + '.' + ext) # input video file
        self.file_input_prep = os.path.join(_output_dir, self.file_input_base + '_preprocessed.' + ext) # preprocessed video file
        self.file_rectangles = os.path.join(_output_dir, self.file_input_base + '_rects' + self.output_ext) # face rectangles
        self.file_landmarks = os.path.join(_output_dir, self.file_input_base + '_landmarks' + self.output_ext) # landmarks
        self.file_shape_coeff  = os.path.join(_output_dir, self.file_input_base + '_shape_coeff' + self.output_ext) # shape coefficients
        self.file_texture_coeff  = os.path.join(_output_dir, self.file_input_base + '_texture_coeff' + self.output_ext) # texture coefficients
        self.file_shape  = os.path.join(_output_dir, self.file_input_base + '_shape' + self.output_ext) # shape model
        self.file_texture  = os.path.join(_output_dir, self.file_input_base + '_texture' + self.output_ext) # texture model
        self.file_expression  = os.path.join(_output_dir, self.file_input_base + '_expression' + self.output_ext) # expression coefficients
        self.file_pose  = os.path.join(_output_dir, self.file_input_base + '_pose' + self.output_ext) # pose info
        self.file_illumination  = os.path.join(_output_dir, self.file_input_base + '_illumination' + self.output_ext) # illumination coefficients
        self.file_expression_smooth = os.path.join(_output_dir, self.file_input_base + '_expression_smooth' + self.output_ext) # smoothed expression coefficients
        self.file_pose_smooth = os.path.join(_output_dir, self.file_input_base + '_pose_smooth' + self.output_ext) # smoothed pose info
        self.file_landmarks_canonicalized = os.path.join(_output_dir, self.file_input_base + '_landmarks_canonicalized' + self.output_ext) # canonicalized landmarks
        self.file_expression_localized = os.path.join(_output_dir, self.file_input_base + '_expression_localized' + self.output_ext) # localized expressions
        
        # if we are using the API, we need to send the input file to the server and get the session info
        if self.API:
            files = {'input_file': open(input_file, 'rb')}
            data = {
                'file_hash': self.base_metadata['input_hash']
            }
            
            # get session id
            output = self._remote_run_command('prepare_session', files=files, data=data)
            if output and ('session_id' in output):
                self.API_session = output["session_id"]
                print(f"Remote session with ID {self.API_session} was initiated.")
            else:
                raise ValueError("Failed to create a remote session at the server.")
            
        if self.debug:
            self.verbose = False
            print(f"Docker={self.docker}")
            print(f"Executables={self.execDIR}")


    def _remote_run_command(self, endpoint, files=None, data=None):
        """Invoke the Bitbox API and synchronize any outputs.

        Args:
            endpoint: REST path relative to the configured API URL.
            files: Optional multipart payload with file descriptors.
            data: Optional form data to submit alongside the files.

        Returns:
            dict | None: JSON response body or ``None`` if nothing was returned.

        Raises:
            ValueError: When the server returns invalid ZIP payloads.
        """
        url = self.API_url + endpoint
                
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')

            # extract the ZIP file that is returned from the server
            if 'application/zip' in content_type:
                try: 
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        # extract the file
                        z.extractall(self.output_dir)
                        
                        # change the remote file names to the local file names
                        for filename in os.listdir(self.output_dir):
                            if "input_file" in filename:
                                # rename the file
                                new_filename = os.path.join(self.output_dir, filename.replace("input_file", self.file_input_base))
                                os.rename(
                                    os.path.join(self.output_dir, filename),
                                    new_filename
                                )
                                # we will create their JSON files here to prevent multiple calls to the server
                                additional_metadata = {
                                    'cmd': 'API call',
                                    'input': self._local_file(self.file_input),
                                    'output': self.output_dir
                                }
                                metadata = {**self.base_metadata, **additional_metadata}              
                                self.cache.store_metadata(self._local_file(new_filename), metadata)
                
                except zipfile.BadZipFile:
                    raise ValueError("Invalid ZIP file received from the server.")
                except OSError as e:
                    raise ValueError(f"File renaming error: {e}")
            else:
                try:
                    output = response.json()
                    if output:
                        return output
                except requests.JSONDecodeError:
                    print("Invalid JSON response")
        else:
            print(f"Request failed with status {response.status_code}: {response.text}")

        return None
           
    
    def _run_command(self, executable, parameters, system_call):
        """Run a backend binary either locally or inside a container.

        Args:
            executable: Name of the binary/script to execute.
            parameters: Iterable of CLI parameters passed verbatim.
            system_call: When ``True`` executes via the system shell, otherwise
                the call is proxied through the remote API.

        Raises:
            ValueError: If required files are missing or the container type is
                unsupported.
        """
        # use a specific GPU with the least memory used
        gpu_id = select_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    
        if system_call: # if we are using system call          
            # executable
            # @TODO: use the minimally utilized GPU
            if self.docker:
                if detect_container_type(self.docker) =='singularity': 
                    cmd = f"singularity exec --nv --writable \
                        --bind {self.input_dir}:{self.docker_input_dir} \
                        --bind {self.output_dir}:{self.docker_output_dir} \
                        {self.docker} \
                        bash -c \"cd {self.execDIR} && ./{executable}"
                elif detect_container_type(self.docker) == 'docker':
                    cmd = f"\
                        docker run --rm --gpus device={gpu_id}\
                        -v {self.input_dir}:{self.docker_input_dir}\
                        -v {self.output_dir}:{self.docker_output_dir}\
                        -w {self.execDIR} {self.docker} ./{executable}"
                else:
                    raise ValueError("Unsupported container type. Please use Docker or Singularity.")
                        
                # collapse all runs of whitespace into single spaces (due to use of \ in the command)
                cmd = " ".join(cmd.split())
            else:
                cmd = os.path.join(self.execDIR, executable)
            
            # parameters (input & output files and config files)
            for p in parameters:
                if p is None:
                    raise ValueError("File names are not set correctly. Please use io() method prior to running any processing.")
                if self.docker and (self.docker.endswith("sandbox") or self.docker.endswith(".sif")):
                    cmd += f" {str(p)}"
                else:
                    cmd += ' ' + str(p)
                    
            if self.docker and (self.docker.endswith("sandbox") or self.docker.endswith(".sif")):
                cmd += "\""   # Close the double quotes for bash -c
                
            # suppress the output of the command. check whether we are on a Windows or Unix-like system
            if os.name == 'nt': # Windows
                cmd += ' > NUL'
            else: # Unix-like systems (Linux, macOS)
                cmd += ' > /dev/null'
            
            # @TODO: remove this part when the 3DI code is updated so that we don't need to change the working directory
            # set the working directory to the executable directory
            if self.docker is None:
                tmp = os.getcwd()
                os.chdir(self.execDIR)
            
            # run the command
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
            os.system(cmd)
            
            # @TODO: remove this part when the 3DI code is updated so that we don't need to change the working directory
            # set the working directory back
            if self.docker is None:
                os.chdir(tmp)
            
        else: # if we are using a python function
            cmd = "%s()" % executable
            # prepare the function
            func = getattr(self, executable)
            func(*parameters)
            
        return cmd
    
    
    def _execute(self, executable, parameters, name, output_file_idx=-1, system_call=True):
        """Run a backend step if its cached outputs are missing or stale.

        Args:
            executable: Command or function name to run.
            parameters: Argument list passed to the executable.
            name: Human-readable label for progress logging.
            output_file_idx: Index (or indices) inside ``parameters`` pointing to
                the expected output files so cache freshness can be checked.
            system_call: Forwarded to :meth:`_run_command` to decide between shell
                execution and Python function invocation.

        Raises:
            ValueError: If the command fails to generate the requested outputs.
        """
        # we will prevent redundant calls to the server from the same caller method
        # for example, fit() method makes multiple calls to _execute method, but we only need to call the server once
        caller_frame = inspect.stack()[1]
        caller_name = caller_frame.function
        if self.API and (caller_name in self.API_callers):
            verbose = False
        else:
            verbose = self.verbose
            # add the method name to the list of active calls so that we don't call the server again
            self.API_callers.add(caller_name)

            # set up a function to remove the caller from the active calls whent he caller method returns
            def remove_caller(frame, event, arg):
                if event == "return":
                    self.API_callers.remove(caller_name)
                return remove_caller
            caller_frame.frame.f_trace = remove_caller
            sys.settrace(lambda *args, **kwargs: None)
                
        status = False
        
        # get the output file name
        if not isinstance(output_file_idx, list):
            output_file_idx = [output_file_idx]
    
        # check if the output file already exists, if not run the executable
        file_exits = 0
        for idx in output_file_idx:
            tmp = self.cache.check_file(self._local_file(parameters[idx]), self.base_metadata, verbose=verbose)
            file_exits = max(file_exits, tmp)
        
        # run the executable if needed
        if file_exits > 0: # file does not exist, has different metadata, or it is older than the retention period
            # if needed, change the name of the output file
            # @TODO: when we change the file name, next time we run the code, we should be using the latest file generated, which is hard to track. We are rewriting for now.
            # @TODO: for the same reason above, we need to remove the old metadata file otherwise "file_generated" will be >0 and fail the check
            # @TODO: also we need to consider multiple output files
            if file_exits == 2:
                # delete this loop after resolving above @TODO
                for idx in output_file_idx:
                    self.cache.delete_old_file(self._local_file(parameters[idx]))
                #output_file = self.cache.get_new_file_name(output_file)  # uncomment after resolving above @TODO
                #parameters[output_file_idx] = output_file  # uncomment after resolving above @TODO
            
            if self.verbose:
                print("Running %s..." % name, end='', flush=True)
                t0 = time()
            
            # run the command
            if self.API: # Running on a remote server
                cmd = 'API call'
                # make the call to the server
                data = {
                    'session_id': self.API_session,
                    'processor_class': self.__class__.__name__,
                    'method': caller_name,
                    'config': json.dumps(self.API_config)
                }
                self._remote_run_command('execute', data=data)
            else: # Running locally
                cmd = self._run_command(executable, parameters, system_call)
            
            if self.verbose:
                print(" (Took %.2f secs)" % (time()-t0))
            
            # check if the command was successful
            file_generated = 0
            for idx in output_file_idx:
                tmp = self.cache.check_file(self._local_file(parameters[idx]), self.base_metadata, verbose=False, json_required=False, retention_period='5 minutes')
                # if tmp > 0:
                #     print(f"{self._local_file(parameters[idx])} was not generated.")
                file_generated = max(file_generated, tmp)
            
            if file_generated == 0: # file is generated (0 means the file is found)
                # store metadata
                additional_metadata = {
                    'cmd': cmd,
                    'input': self._local_file(self.file_input),
                    'output': self.output_dir
                }
                metadata = {**self.base_metadata, **additional_metadata}
                for idx in output_file_idx:                
                    self.cache.store_metadata(self._local_file(parameters[idx]), metadata)
                    
                status = True
            else:
                status = False
        else: # file is already present
            status = True
            
        if not status:
            raise ValueError("Failed running %s" % name)
        
    def plot(self, data: dict, random_seed: int = 42, num_frames: int = 5, overlay: Optional[list] = None, 
             rect_color: Optional[str] = None, landmark_color: Optional[str] = None,
             pose: Optional[dict] = None, video: bool = False, frames: Optional[List[int]] = None):
        """Visualize rectangles, landmarks, expressions, or pose on the source video/grid.

        Args:
            data: Dict from Bitbox readers (rectangle, landmark, landmark-can, expression, or pose).
            random_seed: Seed used when sampling representative frames.
            num_frames: Number of frames exported to display when video toggle if false.
            overlay: Optional rectangles/landmarks to draw alongside ``data``.
            rect_color: CSS color for rectangles (applies to grids and video overlay; default red).
            landmark_color: CSS color for landmarks/points (applies to grids and video overlay; default blue).
            pose: Optional pose dictionary used for overlays and to pick pose diverse frames when ``video`` is False.
            video: When ``True`` also writes an HTML/MP4 preview with overlays.
            frames: Optional explicit frame indices to export instead of sampling.

        Returns:
            list[str] | str: Paths to generated plots (list) or video (single path).
        """

        # helpers to parse overlay variants (list or dict) into landmark/rectangle pieces
        def _parse_overlay(ov) -> tuple:
            overlay_land = None
            overlay_rect = None
            if isinstance(ov, list):
                for value in ov:
                    if check_data_type(value, 'landmark'):
                        overlay_land = value
                    elif check_data_type(value, 'rectangle'):
                        overlay_rect = value
            elif isinstance(ov, dict):
                if check_data_type(ov, 'landmark'):
                    overlay_land = ov
                elif check_data_type(ov, 'rectangle'):
                    overlay_rect = ov
            return overlay_land, overlay_rect

        cushion_ratio = 0.35
        video_path = self._local_file(self.file_input)

        out_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(out_dir, exist_ok=True)
        if not check_data_type(data, ['rectangle', 'landmark', 'landmark-can', 'expression', 'pose']):
            raise ValueError("`data` must be a dict with a 'type' key in {'rectangle','landmark','landmark-can','expression','pose'}.")

        # Unpack overlay when provided as list/dict
        overlay_land, overlay_rect = _parse_overlay(overlay)

        start_time = time()
        if self.verbose:
            print("Running plot...", end='', flush=True)

        def _finish(result):
            if self.verbose:
                print(f" (Took {time() - start_time:.2f} secs)")
            return result

        if check_data_type(data, 'rectangle'):
            return _finish(visualize_and_export(
                rects=data,
                num_frames=num_frames,
                video_path=video_path,
                out_dir=out_dir,
                random_seed=random_seed,
                cushion_ratio=cushion_ratio,
                overlay=overlay_land,   # pass only landmark overlay here
                pose=pose,
                video=video,
                frames=frames,
                rect_color=rect_color,
                landmark_color=landmark_color,
                
            ))

        if check_data_type(data, 'landmark'):
            rects_src = overlay_rect if overlay_rect is not None else None
            return _finish(visualize_and_export(
                rects=rects_src,
                num_frames=num_frames,
                video_path=video_path,
                out_dir=out_dir,
                random_seed=random_seed,
                cushion_ratio=cushion_ratio,
                overlay=data,   # landmarks become the overlay
                pose=pose,
                video=video,
                frames=frames,
                rect_color=rect_color,
                landmark_color=landmark_color,
            ))
        
        if check_data_type(data, 'landmark-can'):
            return _finish(visualize_and_export_can_land(
                num_frames=num_frames,
                out_dir=out_dir,
                video_path=video_path,
                land_can=data,
                pose=pose,
                overlay=overlay,
                rect_color=rect_color,
                landmark_color=landmark_color,
                video=video,
                frames=frames,
            ))

        if check_data_type(data, 'expression'):
            return _finish(visualize_expressions_3d(
                expressions=data,
                out_dir=out_dir,
                video_path=video_path,
                smooth=0,
                downsample=1,
                play_fps=5,
                overlay=overlay,
            ))

        if check_data_type(data, 'pose'):
            return _finish(visualize_and_export_pose(
                pose=data,
                num_frames=num_frames,
                video_path=video_path,
                out_dir=out_dir,
                overlay=overlay,
                rect_color=rect_color,
                landmark_color=landmark_color,
                video=video,
                frames=frames,
            ))

        raise ValueError(f"Unknown data type: {getattr(data, 'get', lambda *_: None)('type')!r}. Expected 'rectangle', 'landmark', 'landmark-can', 'expression', or 'pose'.")
    
    def render_3d(
        self,
        expressions: Optional[dict] = None,
        pose: Optional[dict] = None,
        out_dir: Optional[str] = None,
        num_frames: int = 250,
        frames: Optional[List[int]] = None,
    ):
        """
        Generate BFM expression + pose visualizations without calling ``plot``.

        Args:
            expressions: Expression coefficients dict/DataFrame; falls back to the latest cached ``*_expression_smooth.3DI`` when ``None``.
            pose: Pose dict/DataFrame; falls back to the latest cached pose file when ``None``.
            out_dir: Output directory (defaults to ``<output_dir>/plots``).
            num_frames: Max frames to export when sampling automatically.
            frames: Optional explicit frame indices to render (overrides sampling).

        Returns:
            list[str] | str: Paths to generated plots/HTML.
        """

        start_time = time()
        if self.verbose:
            print("Running 3d face rendering ...", end='', flush=True)

        def _finish(result):
            if self.verbose:
                print(f" (Took {time() - start_time:.2f} secs)")
            return result

        def _normalize_payload(obj, expected_type: str) -> Optional[dict]:
            if obj is None:
                return None
            try:
                if check_data_type(obj, expected_type):
                    return obj
            except ValueError:
                pass
            if hasattr(obj, "iloc"):
                return {"type": expected_type, "data": obj}
            if isinstance(obj, dict):
                raise ValueError(f"`{expected_type}` payload is malformed.")
            return None

        def _load_expression_fallback() -> Optional[dict]:
            for candidate in (
                self._local_file(self.file_expression_smooth),
                self._local_file(self.file_expression),
            ):
                if candidate and os.path.exists(candidate):
                    try:
                        return read_expression(candidate)
                    except Exception:
                        continue
            return None

        def _load_pose_fallback() -> Optional[dict]:
            for candidate in (
                self._local_file(self.file_pose_smooth),
                self._local_file(self.file_pose),
            ):
                if not candidate or not os.path.exists(candidate):
                    continue
                try:
                    # Lite pose files need landmarks to recover full Rodrigues.
                    if str(candidate).lower().endswith("3dil"):
                        landmarks_path = (
                            self._local_file(self.file_landmarks)
                            if getattr(self, "file_landmarks", None)
                            else None
                        )
                        if landmarks_path and os.path.exists(landmarks_path):
                            return read_pose_lite(candidate, landmarks_path)
                    else:
                        return read_pose(candidate)
                except Exception:
                    continue
            return None

        expr_supplied = expressions is not None
        pose_supplied = pose is not None

        expressions_payload = _normalize_payload(expressions, 'expression')
        if expressions_payload is None and not expr_supplied:
            expressions_payload = _load_expression_fallback()
        if expressions_payload is None:
            raise ValueError(
                "`expressions` not provided and no fallback file found. "
                "Run `fit()` first or provide expression coefficients explicitly."
            )

        pose_payload = _normalize_payload(pose, 'pose')
        if pose_payload is None and not pose_supplied:
            pose_payload = _load_pose_fallback()
        if pose_payload is None:
            raise ValueError(
                "`pose` not provided and no fallback file found. "
                "Run `fit()` first or provide pose parameters explicitly."
            )

        def _payload_dataframe(payload: Optional[dict]):
            if payload is None:
                return None
            data = payload.get("data")
            return data if hasattr(data, "iloc") else None

        if out_dir is None:
            if not self.output_dir:
                raise ValueError("Output directory is not set. Call io(...) first or pass out_dir.")
            out_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(out_dir, exist_ok=True)

        target_num_frames = num_frames
        if frames is None:
            expr_df = _payload_dataframe(expressions_payload)
            pose_df = _payload_dataframe(pose_payload)
            if expr_df is not None and pose_df is not None:
                total = max(1, min(len(expr_df), len(pose_df)))
                target_num_frames = min(num_frames, total)

        load_identity_assets = not (expr_supplied and pose_supplied)
        shape_vertices = None
        texture_values = None
        illumination_values = None

        def _fallback_candidates(primary: Optional[str], fallback_ext: str) -> List[str]:
            if not primary:
                return []
            base, ext = os.path.splitext(primary)
            return [primary, base + fallback_ext] if ext.lower() != fallback_ext.lower() else [primary]

        if load_identity_assets:
            # Prefer backend-native assets; for 3DI-lite fall back to .3DI meshes when .3DIl is missing.
            shape_candidates = _fallback_candidates(getattr(self, "file_shape", None), ".3DI")
            for candidate in shape_candidates:
                shape_path = self._local_file(candidate)
                if shape_path and os.path.exists(shape_path):
                    try:
                        shape_vertices = np.loadtxt(shape_path, dtype=np.float32).reshape(-1, 3)
                        break
                    except Exception:
                        shape_vertices = None

            texture_candidates = _fallback_candidates(getattr(self, "file_texture", None), ".3DI")
            for candidate in texture_candidates:
                texture_path = self._local_file(candidate)
                if texture_path and os.path.exists(texture_path):
                    try:
                        texture_values = np.loadtxt(texture_path, dtype=np.float32)
                        break
                    except Exception:
                        texture_values = None

            illum_candidates = _fallback_candidates(getattr(self, "file_illumination", None), ".3DI")
            for candidate in illum_candidates:
                illum_path = self._local_file(candidate)
                if illum_path and os.path.exists(illum_path):
                    try:
                        illumination_values = np.loadtxt(illum_path, dtype=np.float32)
                        break
                    except Exception:
                        illumination_values = None

        return _finish(visualize_bfm_expression_pose(
            expressions=expressions_payload,
            pose=pose_payload,
            out_dir=out_dir,
            num_frames=target_num_frames,
            frames=frames,
            processor=self,
            identity_shape=shape_vertices,
            vertex_texture=texture_values,
            illumination=illumination_values,
        ))
        
        
    def preprocess(self):
        raise ValueError("Preprocess method is not implemented for the selected backend.")
            
            
    def detect_faces(self):
        raise ValueError("Face detection method is not implemented for the selected backend.")
            
            
    def detect_landmarks(self):
        raise ValueError("Landmark detection method is not implemented for the selected backend.")
        

    def fit(self):
        raise ValueError("Facial reconstruction method is not implemented for the selected backend.")
        

    def localized_expressions(self):
        raise ValueError("Localized expressions method is not implemented for the selected backend.")


    def run_all(self):
        raise ValueError("Run all method is not implemented for the selected backend.")
