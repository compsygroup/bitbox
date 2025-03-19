import os
import warnings

import numpy as np

from time import time

from ..utilities import FileCache

from .reader3DI import read_rectangles, read_landmarks
from .reader3DI import read_pose, read_pose_lite
from .reader3DI import read_expression, read_canonical_landmarks

class FaceProcessor3DI:
    def __init__(self, camera_model=30, landmark_model='global4', morphable_model='BFMmm-19830', basis_model='0.0.1.F591-cd-K32d', fast=False, lite=False, return_output=True):
        self.file_input = None
        self.dir_output = None
        self.file_input_org = None
        self.dir_output_org = None
        self.execDIR = None
        self.use_docker = False
        self.base_metadata = None
        
        self.model_camera = camera_model
        self.model_morphable = morphable_model
        self.model_landmark = landmark_model
        self.model_basis = basis_model
        self.fast = fast
        self.lite = lite
        self.liteDIR = None
        
        self.cache = FileCache()
        
        self.return_output = return_output
        
        # find out where 3DI package is installed
        if os.environ.get('DOCKER_3DI'):
            print("Using 3DI package inside the Docker container: %s" % os.environ.get('DOCKER_3DI'))
            self.use_docker = True
            self.docker = os.environ.get('DOCKER_3DI')
            self.execDIR = '/app/3DI'
            self.liteDIR = '/app/3DI_lite'
            self.docker_inDIR = '/app/input'
            self.docker_outDIR = '/app/output'
        else:
            if os.environ.get('PATH_3DI'):
                execDIRs = [os.environ.get('PATH_3DI')]
            else:
                warnings.warn("PATH_3DI environment variable is not set. Using default system PATH.")
                execDIRs = os.environ.get('PATH')
                if ';' in execDIRs:  # Windows
                    execDIRs = execDIRs.split(';')
                else:  # Unix-like systems (Linux, macOS)
                    execDIRs = execDIRs.split(':')
                    
            for d in execDIRs:
                if os.path.exists(os.path.join(d, 'video_learn_identity')):
                    self.execDIR = d
                    break
                
            if self.execDIR is None:
                raise ValueError("3DI package is not found. Please make sure you defined PATH_3DI system variable.")
            
            if self.lite:
                for d in execDIRs:
                    if os.path.exists(os.path.join(d, 'process_video.py')):
                        self.liteDIR = d
                        break
                
                if os.environ.get('PATH_3DI_LITE'):
                    if os.path.exists(os.path.join(os.environ.get('PATH_3DI_LITE'), 'process_video.py')):
                        self.liteDIR = os.environ.get('PATH_3DI_LITE')
                    
                if self.liteDIR is None:
                    raise ValueError("3DI-Lite package is not found. Please make sure you defined PATH_3DI_LITE system variable.")
            
    
        if not self.use_docker:
            # set the working directory
            # @TODO: remove this line when the 3DI code is updated by Vangelis
            os.chdir(self.execDIR)
        
        # prepare configuration files
        if self.fast:
            cfgid = 2
        else:
            cfgid = 1
        
        self.config_landmarks = os.path.join(self.execDIR, 'configs/%s.cfg%d.%s.txt' % (self.model_morphable, cfgid, self.model_landmark)) 
                
    
    def io(self, input_file, output_dir):
        # supported video extensions
        supported_extensions = ['mp4', 'avi', 'mpeg']

        # check if input file exists
        if not os.path.exists(input_file):
            raise ValueError("Input file does not exist. Please check the path and permissions.")
        
        # check if input file extension is supported
        ext = input_file.split('.')[-1].lower()
        if not (ext in supported_extensions):
            raise ValueError("Input file extension is not supported. Please use one of the following extensions: %s" % supported_extensions)
        
        # if no exception is raised, set the input file and output directory
        self.file_input_org = input_file
        self.dir_output_org = output_dir
        if self.use_docker: # check if we are using a docker container
            self.file_input = os.path.join(self.docker_inDIR, os.path.basename(input_file))
            self.dir_output = self.docker_outDIR
        else:
            self.file_input = input_file
            self.dir_output = output_dir
            # create output directory
            try:
                os.makedirs(output_dir, exist_ok=True)
            except:
                raise ValueError("Cannot create output directory. Please check the path and permissions.")  
            
        # set all the output files
        self.file_input_base = '.'.join(os.path.basename(input_file).split('.')[:-1])
        self.file_input_prep = os.path.join(self.dir_output, self.file_input_base + '_preprocessed.' + ext) # preprocessed video file
        self.file_rectangles = os.path.join(self.dir_output, self.file_input_base + '_rects.3DI') # face rectangles
        self.file_landmarks = os.path.join(self.dir_output, self.file_input_base + '_landmarks.3DI') # landmarks
        self.file_shape_coeff  = os.path.join(self.dir_output, self.file_input_base + '_shape_coeff.3DI') # shape coefficients
        self.file_texture_coeff  = os.path.join(self.dir_output, self.file_input_base + '_texture_coeff.3DI') # texture coefficients
        self.file_shape  = os.path.join(self.dir_output, self.file_input_base + '_shape.3DI') # shape model
        self.file_texture  = os.path.join(self.dir_output, self.file_input_base + '_texture.3DI') # texture model
        self.file_expression  = os.path.join(self.dir_output, self.file_input_base + '_expression.3DI') # expression coefficients
        self.file_pose  = os.path.join(self.dir_output, self.file_input_base + '_pose.3DI') # pose info
        self.file_illumination  = os.path.join(self.dir_output, self.file_input_base + '_illumination.3DI') # illumination coefficients
        self.file_expression_smooth = os.path.join(self.dir_output, self.file_input_base + '_expression_smooth.3DI') # smoothed expression coefficients
        self.file_pose_smooth = os.path.join(self.dir_output, self.file_input_base + '_pose_smooth.3DI') # smoothed pose info
        self.file_landmarks_canonicalized = os.path.join(self.dir_output, self.file_input_base + '_landmarks_canonicalized.3DI') # canonicalized landmarks
        self.file_expression_localized = os.path.join(self.dir_output, self.file_input_base + '_expression_localized.3DI') # localized expressions
    
    
    def _local_file(self, file_path):
        if self.use_docker:
            return os.path.join(self.dir_output_org, os.path.basename(file_path))
        else:
            return file_path
    
    def _run_command(self, executable, parameters, output_file_idx, system_call):                  
        if system_call: # if we are using system call          
            # executable
            if self.use_docker: # check if we are using a docker container
                input_dir = os.path.dirname(self.file_input_org)
                cmd = f"docker run --rm --gpus all -v {input_dir}:{self.docker_inDIR} -v {self.dir_output_org}:{self.docker_outDIR} -w {self.execDIR} {self.docker} ./{executable}"
            else:
                cmd = os.path.join(self.execDIR, executable)
            
            # parameters
            for p in parameters:
                if p is None:
                    raise ValueError("File names are not set correctly. Please use io() method prior to running any processing.")
                cmd += ' ' + str(p)
            # suppress the output of the command. check whether we are on a Windows or Unix-like system
            if os.name == 'nt': # Windows
                cmd += ' > NUL'
            else: # Unix-like systems (Linux, macOS)
                cmd += ' > /dev/null'          
            os.system(cmd)
        else: # if we are using a python function
            cmd = "%s()" % executable
            # prepare the function
            func = getattr(self, executable)
            func(*parameters)
            
        return cmd
    
    
    def _execute(self, executable, parameters, name, output_file_idx=-1, system_call=True):
        status = False
        
        # get the output file name
        if not isinstance(output_file_idx, list):
            output_file_idx = [output_file_idx]
       
        # check if the output file already exists, if not run the executable
        if self.lite:
            backend = '3DI-lite'
        else:
            backend = '3DI'
            
        self.base_metadata = {
            'backend' : backend,
            'morphable_model': self.model_morphable,
            'camera': self.model_camera,
            'landmark': self.model_landmark,
            'fast': self.fast
        }  
        
        file_exits = 0
        for idx in output_file_idx:
            tmp = self.cache.check_file(self._local_file(parameters[idx]), self.base_metadata, verbose=True)
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
            
            # run the command
            print("Running %s..." % name, end='', flush=True)
            t0 = time()
            cmd = self._run_command(executable, parameters, output_file_idx, system_call)
            print(" (Took %.2f secs)" % (time()-t0))
            
            # check if the command was successful
            file_generated = 0
            for idx in output_file_idx:
                tmp = self.cache.check_file(self._local_file(parameters[idx]), self.base_metadata, verbose=False, json_required=False, retention_period='5 minutes')
                file_generated = max(file_generated, tmp)
            
            if file_generated == 0: # file is generated (0 means the file is found)
                # store metadata
                additional_metadata = {
                    'cmd': cmd,
                    'input': self.file_input_org,
                    'output': self.dir_output_org
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
                 
        
    def preprocess(self, undistort=False):
        # run undistortion if needed
        if undistort==True:
            # check if proper camera parameters are provided
            # @TODO: check if self.model_camera is a valid file and includes undistortion parameters
            
            self._execute('video_undistort',
                                   [self.file_input, self.model_camera, self.file_input_prep],
                                   "video undistortion",
                                   output_file_idx=-1)
        
            self.file_input = self.file_input_prep
            
            
    def detect_faces(self):
        self._execute('video_detect_face',
                      [self.file_input, self.file_rectangles],
                      "face detection",
                      output_file_idx=-1)
               
        if self.return_output:
            return read_rectangles(self._local_file(self.file_rectangles))
        else:
            return None
            
            
    def detect_landmarks(self):
        # check if face detection was run and successful
        if self.cache.check_file(self._local_file(self.file_rectangles), self.base_metadata) > 0:
            raise ValueError("Face detection is not run or failed. Please run face detection first.")
        
        self._execute('video_detect_landmarks',
                      [self.file_input, self.file_rectangles, self.file_landmarks, self.config_landmarks],
                      "landmark detection",
                      output_file_idx=-2)
        
        if self.return_output:
            return read_landmarks(self._local_file(self.file_landmarks))
        else:
            return None
        

    def fit(self):
        # check if landmark detection was run and successful
        if self.cache.check_file(self._local_file(self.file_landmarks), self.base_metadata) > 0:
            raise ValueError("Landmark detection is not run or failed. Please run landmark detection first.")
        
        # Check if we are using the lite version
        if self.lite:
            tmp = self.execDIR
            self.execDIR = self.liteDIR
            if not self.use_docker:
                # set the working directory
                # @TODO: remove this line when the 3DI code is updated by Vangelis
                os.chdir(self.execDIR)
            
            # STEP 1-3: learn identity, shape and texture model, pose and expression
            self._execute('process_video.py',
                        [self.file_input, self.file_landmarks, self.file_expression, self.file_shape_coeff, self.file_texture_coeff, self.file_illumination, self.file_pose],
                        "expression and pose estimation",
                        output_file_idx=[-5, -4, -3, -2, -1])
            
            self.execDIR = tmp
            if not self.use_docker:
                # set the working directory
                # @TODO: remove this line when the 3DI code is updated by Vangelis
                os.chdir(self.execDIR)
            
            # STEP 4: Smooth expression and pose
            self.file_expression_smooth = self.file_expression
            self.file_pose_smooth = self.file_pose
        else:
            # STEP 1: learn identity   
            self._execute('video_learn_identity',
                        [self.file_input, self.file_landmarks, self.config_landmarks, self.model_camera, self.file_shape_coeff, self.file_texture_coeff],
                        "3D face model fitting",
                        output_file_idx=[-2, -1])
        
            # STEP 2: shape and texture model
            self._execute('scripts/save_identity_and_shape.py',
                        [self.file_shape_coeff, self.file_texture_coeff, '1', '0.4', self.file_shape, self.file_texture, self.model_morphable],
                        "shape and texture model",
                        output_file_idx=[-3, -2])

            # STEP 3: Pose and expression
            self._execute('video_from_saved_identity',
                        [self.file_input, self.file_landmarks, self.config_landmarks, self.model_camera, self.file_shape, self.file_texture, self.file_expression, self.file_pose, self.file_illumination],
                        "expression and pose estimation",
                        output_file_idx=[-3, -2, -1])

            # STEP 4: Smooth expression and pose
            self._execute('scripts/total_variance_rec.py',
                        [self.file_expression, self.file_expression_smooth, self.model_morphable],
                        "expression smoothing",
                        output_file_idx=-2)
                    
            self._execute('scripts/total_variance_rec_pose.py',
                        [self.file_pose, self.file_pose_smooth],
                        "pose smoothing",
                        output_file_idx=-1)
            
        # STEP 5: Canonicalized landmarks
        self._execute('scripts/produce_canonicalized_3Dlandmarks.py',
                    [self.file_expression_smooth, self.file_landmarks_canonicalized, self.model_morphable],
                    "canonicalized landmark estimation",
                    output_file_idx=-2)
        
        if self.return_output:
            out_exp = read_expression(self._local_file(self.file_expression_smooth))
            
            if self.lite:
                out_pose = read_pose_lite(self._local_file(self.file_pose_smooth))
            else:
                out_pose = read_pose(self._local_file(self.file_pose_smooth))
            
            out_land_can = read_canonical_landmarks(self._local_file(self.file_landmarks_canonicalized))
            
            return out_exp, out_pose, out_land_can
        else:
            return None, None, None
        

    def localized_expressions(self, normalize=True):
        # check if canonical landmark detection was run and successful
        if self.cache.check_file(self._local_file(self.file_expression_smooth), self.base_metadata) > 0:
            raise ValueError("Expression quantification is not run or failed. Please run fit() method first.")
        
        self._execute('scripts/compute_local_exp_coefficients.py',
                    [self.file_expression_smooth, self.file_expression_localized, self.model_morphable, self.model_basis, int(normalize)],
                    "localized expression estimation",
                    output_file_idx=-4)
        
        if self.return_output:
            return read_expression(self._local_file(self.file_expression_localized))
        else:
            return None


    def run_all(self, undistort=False, normalize=True):
        self.preprocess(undistort)
        rect = self.detect_faces()
        land = self.detect_landmarks()
        exp_glob, pose, land_can = self.fit()
        exp_loc = self.localized_expressions(normalize=normalize)
        
        if self.return_output:
            return rect, land, exp_glob, pose, land_can, exp_loc
        else:
            return None, None, None, None, None, None
    
    
class FaceProcessor3DITest(FaceProcessor3DI):
    def __init__(self):
        self.file_input = None
        self.dir_output = None
        self.file_input_org = None
        self.dir_output_org = None
        self.execDIR = None
        self.use_docker = False
        self.base_metadata = None
        
        self.model_camera = 30
        self.model_morphable = 'BFMmm-19830'
        self.model_landmark = 'global4'
        self.model_basis = '0.0.1.F591-cd-K32d'
        self.fast = False
        self.lite = False
        self.liteDIR = None
        
        self.cache = FileCache()
        
        self.return_output = False
        
        self.execDIR = './'
        
        # prepare configuration files
        if self.fast:
            cfgid = 2
        else:
            cfgid = 1
        
        self.config_landmarks = os.path.join(self.execDIR, 'configs/%s.cfg%d.%s.txt' % (self.model_morphable, cfgid, self.model_landmark))
        
        
    def _run_command(self, executable, parameters, output_file_idx, system_call):        
        for idx in output_file_idx:
            with open(parameters[idx], 'w') as file:
                file.write("This is an empty file for testing purposes. Well, it is not literally 'empty' but, you know, it is not what you expect.")
