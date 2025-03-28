import os
import warnings

from .backend import FaceProcessor

from ..utilities import FileCache

from .reader3DI import read_rectangles, read_landmarks
from .reader3DI import read_pose, read_pose_lite
from .reader3DI import read_expression, read_canonical_landmarks

class FaceProcessor3DI(FaceProcessor):
    def __init__(self, camera_model=30, landmark_model='global4', morphable_model='BFMmm-19830', basis_model='0.0.1.F591-cd-K32d', fast=False, return_output='dict', server=None, verbose=True):
        # Run the parent class init
        super().__init__(return_output=return_output, server=server, verbose=verbose)
        
        self.model_camera = camera_model
        self.model_morphable = morphable_model
        self.model_landmark = landmark_model
        self.model_basis = basis_model
        self.fast = fast
        
        self.execDIR = None
        self.docker_execDIR = '/app/3DI'
        
        # specific file extension for 3DI
        self.output_ext = '.3DI'
        
        # if we are not using the docker container, we need to find out where the 3DI package is installed
        if self.docker is None:
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
                raise ValueError("3DI package is not found. Please make sure you defined PATH_3DI system variable or use our Docker image.")
        
        # prepare configuration files
        if self.fast:
            cfgid = 2
        else:
            cfgid = 1
        
        if self.docker is None:
            self.config_landmarks = os.path.join(self.execDIR, 'configs/%s.cfg%d.%s.txt' % (self.model_morphable, cfgid, self.model_landmark))
        else:
            self.config_landmarks = os.path.join(self.docker_execDIR, 'configs/%s.cfg%d.%s.txt' % (self.model_morphable, cfgid, self.model_landmark))
        
        # prepare metadata
        self.base_metadata['backend'] = '3DI'
        self.base_metadata['morphable_model'] = self.model_morphable
        self.base_metadata['camera'] = self.model_camera
        self.base_metadata['landmark'] = self.model_landmark
        self.base_metadata['fast'] = self.fast
        self.base_metadata['local_bases'] = self.model_basis
            
                
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
               
        if self.return_output == 'file':
            return self._local_file(self.file_rectangles)
        elif self.return_output == 'dict':
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
        
        if self.return_output == 'file':
            return self._local_file(self.file_landmarks)
        elif self.return_output == 'dict':
            return read_landmarks(self._local_file(self.file_landmarks))
        else:
            return None
        

    def fit(self):
        # check if landmark detection was run and successful
        if self.cache.check_file(self._local_file(self.file_landmarks), self.base_metadata) > 0:
            raise ValueError("Landmark detection is not run or failed. Please run landmark detection first.")
        
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
    
        if self.return_output == 'file':
            files = (
                self._local_file(self.file_shape_coeff),
                self._local_file(self.file_texture_coeff),
                self._local_file(self.file_shape),
                self._local_file(self.file_texture),
                self._local_file(self.file_expression),
                self._local_file(self.file_pose),
                self._local_file(self.file_illumination),
                self._local_file(self.file_expression_smooth),
                self._local_file(self.file_pose_smooth),
                self._local_file(self.file_landmarks_canonicalized)
            )
            return files
        elif self.return_output == 'dict':
            out_exp = read_expression(self._local_file(self.file_expression_smooth))
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
        
        if self.return_output == 'file':
            return self._local_file(self.file_expression_localized)
        elif self.return_output == 'dict':
            return read_expression(self._local_file(self.file_expression_localized))
        else:
            return None


    def run_all(self, undistort=False, normalize=True):
        self.preprocess(undistort)
        rect = self.detect_faces()
        land = self.detect_landmarks()
        exp_glob, pose, land_can = self.fit()
        exp_loc = self.localized_expressions(normalize=normalize)
        
        return rect, land, exp_glob, pose, land_can, exp_loc


class FaceProcessor3DIlite(FaceProcessor3DI):
    def __init__(self, camera_model=30, landmark_model='global4', morphable_model='BFMmm-19830', basis_model='0.0.1.F591-cd-K32d', fast=False, return_output='dict', server=None, verbose=True):
        # Run the parent class init
        super().__init__(camera_model=camera_model,
                         landmark_model=landmark_model,
                         morphable_model=morphable_model,
                         basis_model=basis_model,
                         fast=fast,
                         return_output=return_output,
                         server=server,
                         verbose=verbose)
        
        self.docker_execDIR = '/app/3DI'
        
        # specific file extension for 3DI-lite
        self.output_ext = '.3DIl'
    
        self.liteDIR = None
        
        # if we are not using the docker container, we need to find out where the 3DI-lite package is installed
        if self.docker is None:
            if os.environ.get('PATH_3DI_LITE'):
                execDIRs = [os.environ.get('PATH_3DI_LITE')]
            else:
                warnings.warn("PATH_3DI_LITE environment variable is not set. Using default system PATH.")
                execDIRs = os.environ.get('PATH')
                if ';' in execDIRs:  # Windows
                    execDIRs = execDIRs.split(';')
                else:  # Unix-like systems (Linux, macOS)
                    execDIRs = execDIRs.split(':')
                    
            for d in execDIRs:
                if os.path.exists(os.path.join(d, 'process_video.py')):
                    self.liteDIR = d
                    break
                
            if self.liteDIR is None:
                raise ValueError("3DI-lite package is not found. Please make sure you defined PATH_3DI_LITE system variable or use our Docker image.")
        
        # prepare metadata
        self.base_metadata['backend'] = '3DI-lite'
        self.base_metadata['local_bases'] = self.model_basis   
        
                
    def fit(self):
        # check if landmark detection was run and successful
        if self.cache.check_file(self._local_file(self.file_landmarks), self.base_metadata) > 0:
            raise ValueError("Landmark detection is not run or failed. Please run landmark detection first.")
        
        # @TODO: remove this part when the 3DI code is updated so that we don't need to change the working directory
        # set the working directory
        tmp = self.execDIR
        self.execDIR = self.liteDIR
        self.docker_execDIR = '/app/3DI_lite'
        
        # STEP 1-4: learn identity, shape and texture model, pose and expression
        self._execute('process_video.py',
                    [self.file_input, self.file_landmarks, self.file_expression_smooth, self.file_shape_coeff, self.file_texture_coeff, self.file_illumination, self.file_pose_smooth],
                    "expression and pose estimation",
                    output_file_idx=[-5, -4, -3, -2, -1])
        
        # @TODO: remove this part when the 3DI code is updated so that we don't need to change the working directory
        self.execDIR = tmp
        self.docker_execDIR = '/app/3DI'
            
        # STEP 5: Canonicalized landmarks
        self._execute('scripts/produce_canonicalized_3Dlandmarks.py',
                    [self.file_expression_smooth, self.file_landmarks_canonicalized, self.model_morphable],
                    "canonicalized landmark estimation",
                    output_file_idx=-2)
        
        if self.return_output == 'file':
            files = (
                self._local_file(self.file_expression_smooth),
                self._local_file(self.file_shape_coeff),
                self._local_file(self.file_texture_coeff),
                self._local_file(self.file_illumination),
                self._local_file(self.file_pose_smooth),
                self._local_file(self.file_landmarks_canonicalized)
            )
            return files
        elif self.return_output == 'dict':
            out_exp = read_expression(self._local_file(self.file_expression_smooth))
            out_pose = read_pose_lite(self._local_file(self.file_pose_smooth))
            out_land_can = read_canonical_landmarks(self._local_file(self.file_landmarks_canonicalized))
            
            return out_exp, out_pose, out_land_can
        else:
            return None, None, None
        
    
class FaceProcessor3DITest(FaceProcessor3DI):
    def __init__(self):
        self.verbose = True
        self.server = None
        self.input_dir = None
        self.output_dir = None
        self.file_input = None
        self.output_ext = '.test'
        
        self.execDIR = None
        self.docker = None
        self.base_metadata = None
        
        self.model_camera = 30
        self.model_morphable = 'BFMmm-19830'
        self.model_landmark = 'global4'
        self.model_basis = '0.0.1.F591-cd-K32d'
        self.fast = False
        self.liteDIR = None
        
        self.cache = FileCache()
        
        self.return_output = None
        
        self.execDIR = os.getcwd()
        
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
