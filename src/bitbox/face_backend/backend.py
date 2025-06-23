import os
import warnings
import inspect
import requests
import json
import sys
import zipfile
import io

from time import time

from ..utilities import FileCache, generate_file_hash, select_gpu

class FaceProcessor:
    def __init__(self, return_output='dict', server=None, verbose=True):
        self.verbose = verbose
        self.input_dir = None
        self.output_dir = None
        self.file_input = None
        self.output_ext = '.bit'
        
        self.execDIR = os.getcwd()
        
        self.docker = None
        self.docker_input_dir = '/app/input'
        self.docker_output_dir = '/app/output'
        self.docker_execDIR = '/app'
        
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
            
        else: # Commands will be run locally
            # check if we are running on a docker container
            docker_image = os.environ.get('DOCKER_BITBOX')
            if docker_image:
                if docker_image.endswith("sandbox") or docker_image.endswith(".sif"):
                    print(f"Using backend inside a Singularity container: {docker_image}")
                    self.docker = docker_image
                else:
                    result = os.system(f"docker images -q {docker_image} > /dev/null 2>&1")
                    if result != 0:
                        warnings.warn(f"The environment variable DOCKER_BITBOX is set to {docker_image}, but the image is not found.")
                    else:
                        print(f"Using backend inside a Docker container: {docker_image}")
                        self.docker = docker_image
    
    
    def _local_file(self, file_path):
        file_dir = os.path.dirname(file_path)
        
        if file_dir == self.docker_input_dir:
            return os.path.join(self.input_dir, os.path.basename(file_path))
        elif file_dir == self.docker_output_dir:
            return os.path.join(self.output_dir, os.path.basename(file_path))
        else:
            return file_path
          
          
    def io(self, input_file, output_dir):
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
    
    
    def _remote_run_command(self, endpoint, files=None, data=None):
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
           
    
    def _run_command(self, executable, parameters, output_file_idx, system_call):
        # use a specific GPU with the least memory used
        gpu_id = select_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    
        if system_call: # if we are using system call          
            # executable
            # @TODO: use the minimally utilized GPU
            if self.docker:
                if self.docker.endswith("sandbox") or self.docker.endswith(".sif"):
                    cmd = f"singularity exec --nv \
                        --bind {self.input_dir}:{self.docker_input_dir} \
                        --bind {self.output_dir}:{self.docker_output_dir} \
                        {self.docker} \
                        bash -c \"cd {self.docker_execDIR} && ./{executable}"
                else:
                    cmd = f"\
                        docker run --rm --gpus device={gpu_id}\
                        -v {self.input_dir}:{self.docker_input_dir}\
                        -v {self.output_dir}:{self.docker_output_dir}\
                        -w {self.docker_execDIR} {self.docker} ./{executable}"
                        
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
        # we will prevent redundant calls to the server from the same caller method
        # for example, fit() method makes multiple calls to _execute method, but we only need to call the server once
        caller_frame = inspect.stack()[1]
        caller_name = caller_frame.function
        if self.API and (caller_name in self.API_callers):
            verbose = False
        else:
            verbose = True
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
                cmd = self._run_command(executable, parameters, output_file_idx, system_call)
            
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
