import os
import warnings
from importlib.metadata import version, PackageNotFoundError
from typing import Any, Dict, Optional, Tuple, Union

from .backend import FaceProcessor

from ..utilities import FileCache

from .reader3DI import read_rectangles, read_landmarks
from .reader3DI import read_pose, read_pose_lite
from .reader3DI import read_expression, read_canonical_landmarks

from .postProcess3DI import normalizeExpressions

class FaceProcessor3DI(FaceProcessor):
    """Full 3DI backend that drives the GPU pipeline end-to-end."""

    def __init__(
        self,
        *args: Any,
        camera_model: Union[int, float, str] = 30,
        landmark_model: str = 'global4',
        morphable_model: str = 'BFMmm-19830',
        basis_model: str = '0.0.1.F591-cd-K32d',
        fast: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure the 3DI processor with model variants and runtime tuning.

        Args:
            camera_model: Either a numeric field of view (degrees) or a path to a
                camera calibration file containing intrinsics/extrinsics.
            landmark_model: Name of the shipped 3DI landmark detector to use.
            morphable_model: Morphable model identifier provided by the backend.
            basis_model: Local expression basis applied inside
                :meth:`localized_expressions`.
            fast: When ``True`` switches to the lighter config preset.
            *args: Forwarded to :class:`FaceProcessor` (e.g., ``return_output``).
            **kwargs: Forwarded to :class:`FaceProcessor`.

        Raises:
            ValueError: If the runtime or config files cannot be located.
        """
        # Run the parent class init
        super().__init__(*args, **kwargs)

        self.model_camera = camera_model
        self.model_morphable = morphable_model
        self.model_landmark = landmark_model

        self.model_basis = basis_model
        self.fast = fast
        
        # run the following only if this is not called by a child class
        if self.__class__ is FaceProcessor3DI:
            # specific file extension for 3DI
            self.output_ext = '.3DI'
            
            if not self.API:
                self._set_runtime()
            
                if self.execDIR is None:
                    raise ValueError("3DI package is not found. Please make sure you defined BITBOX_3DI system variable or use our Docker image.")
                    
                # prepare configuration files
                if self.fast:
                    cfgid = 2
                else:
                    cfgid = 1
                    
                self.config_landmarks = os.path.join(self.execDIR, 'configs/%s.cfg%d.%s.txt' % (self.model_morphable, cfgid, self.model_landmark))
        
        # prepare metadata
        self.base_metadata['backend'] = '3DI'
        self.base_metadata['morphable_model'] = self.model_morphable
        self.base_metadata['camera'] = self.model_camera
        self.base_metadata['landmark'] = self.model_landmark
        self.base_metadata['fast'] = self.fast
        self.base_metadata['local_bases'] = self.model_basis
            
    
    def io(self, input_file: Optional[str] = None, output_dir: Optional[str] = None) -> None:
        """Extend the base :meth:`FaceProcessor.io` with optional undistortion.

        Args:
            input_file: Video that will be processed. Defaults to the previously
                configured file if ``None``.
            output_dir: Directory that will receive every intermediate file.

        Raises:
            ValueError: Propagated if the base method rejects the file or cannot
                create ``output_dir``.
        """
        # run the parent class io method
        super().io(input_file=input_file, output_dir=output_dir)
        
        # Auto‐undistort: if the camera model was provided as a string, run preprocess(undistort=True)
        if isinstance(self.model_camera, str):
            if self.verbose:
                print(f"Auto‐undistort: running video undistortion for camera_model='{self.model_camera}'")
            self.preprocess(undistort=True)
        

    def preprocess(self, undistort: bool = False) -> None:
        """Optionally undistort the video prior to face detection.

        Args:
            undistort: When ``True`` run the backend's undistortion executable
                using the configured ``camera_model`` parameters.

        Raises:
            ValueError: If required runtime files are missing or execution fails.
        """
        # run undistortion if needed
        if undistort==True:
            # check if proper camera parameters are provided
            # @TODO: check if self.model_camera is a valid file and includes undistortion parameters
            
            self._execute('video_undistort',
                          [self.file_input, self.model_camera, self.file_input_prep],
                          "video undistortion",
                          output_file_idx=-1)
        
        self.file_input = self.file_input_prep


    def detect_faces(self) -> Optional[Union[str, Tuple[str, ...], Dict[str, Any]]]:
        """Detect faces across the input video and cache the rectangle file.

        Returns:
            Optional[Union[str, Tuple[str, ...], Dict[str, Any]]]: Path(s) to the
            rectangles file when ``return_output == 'file'``, or the parsed
            rectangle dictionary when ``return_output == 'dict'``. ``None`` when
            outputs are suppressed.

        Raises:
            ValueError: If backend execution fails to generate the rectangles.
        """
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
            
            
    def detect_landmarks(self) -> Optional[Union[str, Tuple[str, ...], Dict[str, Any]]]:
        """Run the 3DI landmark detector on the cached rectangles file.

        Returns:
            Optional[Union[str, Tuple[str, ...], Dict[str, Any]]]: File path or
            parsed dictionary mirroring :meth:`detect_faces`.

        Raises:
            ValueError: If face detection has not been run or if landmark
                estimation fails to produce an output.
        """
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
        

    def fit(self, normalize: bool = False, k: int = 1) -> Optional[Any]:
        """Estimate identity, expression, and pose parameters for the video.

        Args:
            normalize: Whether to pass the smoothed expression coefficients through
                :func:`normalizeExpressions`.
            k: How many neighbours the normalizer should consider (only used when
                ``normalize`` is ``True``).

        Returns:
            Optional[Any]: Tuple of output paths (``'file'`` mode), tuple of
            dictionaries (``'dict'`` mode), or ``None`` when ``return_output`` is
            set to suppress results.

        Raises:
            ValueError: If landmarks have not been computed or if any backend step
                fails to produce its expected files.
        """
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
        try:
            self._execute('scripts/total_variance_rec.py',
                        [self.file_expression, self.file_expression_smooth, self.model_morphable],
                        "expression smoothing",
                        output_file_idx=-2)
        except:
            if self.verbose:
                print("Skipping expression smoothing as it failed for this input")
            self.file_expression_smooth = self.file_expression
        
        try:
            self._execute('scripts/total_variance_rec_pose.py',
                        [self.file_pose, self.file_pose_smooth],
                        "pose smoothing",
                        output_file_idx=-1)
        except:
            if self.verbose:
                print("Skipping pose smoothing as it failed for this input")
            self.file_pose_smooth = self.file_pose
        
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
            if normalize:
                out_exp = normalizeExpressions(out_exp, proc='3DI', k=k)
            out_pose = read_pose(self._local_file(self.file_pose_smooth))
            out_land_can = read_canonical_landmarks(self._local_file(self.file_landmarks_canonicalized))
            
            return out_exp, out_pose, out_land_can
        else:
            return None
        

    def localized_expressions(self, normalize: bool = True) -> Optional[Any]:
        """Project global expressions into localized coefficients.

        Args:
            normalize: Whether to normalize the localized coefficients inside the
                3DI post-processing script.

        Returns:
            Optional[Any]: Output path or dictionary depending on ``return_output``.

        Raises:
            ValueError: If :meth:`fit` has not been executed successfully.
        """
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


    def run_all(self, normalize: bool = True, k: int = 1) -> Optional[Any]:
        """Execute the entire 3DI pipeline in sequence.

        Args:
            normalize: Forwarded to :meth:`fit`.
            k: Forwarded to :meth:`fit`.

        Returns:
            Optional[Any]: Tuple of outputs mirroring the cumulative results of
            :meth:`detect_faces`, :meth:`detect_landmarks`, :meth:`fit`, and
            :meth:`localized_expressions`, or ``None`` when no outputs are
            requested.
        """
        rect = self.detect_faces()
        land = self.detect_landmarks()
        if self.return_output == 'file':
            exp = self.fit(normalize=normalize, k=k)
        elif self.return_output == 'dict':
            exp_glob, pose, land_can = self.fit(normalize=normalize, k=k)
        else:
            self.fit(normalize=normalize, k=k)
        exp_loc = self.localized_expressions(normalize=True)
        
        if self.return_output == 'file':
            files = (rect) + (land) + exp + (exp_loc)
            return files
        elif self.return_output == 'dict':
            return rect, land, exp_glob, pose, land_can, exp_loc
        else:
            return None


    def citation(self) -> None:
        """Print BibTeX-style references describing the 3DI processing chain."""
        try:
            bb_version = version("bitbox")
        except PackageNotFoundError:
            bb_version = "ERROR_IN_READING_VERSION_NUMBER"
            
        if isinstance(self.model_camera, str):
            camera_text = "intrinsic and extrinsic camera parameters were learned through camera calibration"
        else:
            camera_text = f"the camera field of view was set to {self.base_metadata['camera']} degrees"

        text = f""" 
        The dataset was processed using Bitbox [1] version {bb_version} for facial behavior analysis. Facial modeling was performed with the 3DI [2], configured with the Basel Face Model (BFM) 2009 [3]. For 3DI, {camera_text}, and the iBUG-51 landmark template [4] was used for landmark definition.
               
        [INCLUDE THE FOLLOWING IF YOU USED LOCAL EXPRESSION COEFFICIENTS]
        Localized expression coefficients were computed using Facial Basis [5]. 

        [1] TBN
        [2] Sariyanidi E, Zampella CJ, Schultz RT, Tunc B (2024). Inequality-Constrained 3D Morphable Face Model Fitting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(2), 1305-1318. https://doi.org/10.1109/TPAMI.2023.3334948
        [3] Paysan P, Knothe R, Amberg B, Romdhani S, Vetter T (2009). A 3D Face Model for Pose and Illumination Invariant Face Recognition. In Proceedings of the IEEE International Conference on Advanced Video and Signal based Surveillance (AVSS), 296-301. https://doi.org/10.1109/AVSS.2009.58
        [4] Sariyanidi E, Zampella CJ, Schultz RT, Tunc B (2020). Can facial pose and expression be separated with weak perspective camera? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 7173-7182. https://doi.org/10.1109/CVPR42600.2020.00720
        [5] Sariyanidi E, Yankowitz L, Schultz RT, Herrington JD, Tunc B, Cohn J (2025). Beyond FACS: Data-driven facial expression dictionaries, with application to predicting autism. In Proceedings of the IEEE International Conference on Automatic Face and Gesture Recognition (FG), 19, 1-10. https://doi.org/10.1109/fg61629.2025.11099288
        """
        print(text)
        
class FaceProcessor3DIlite(FaceProcessor3DI):
    """Lightweight wrapper around the 3DI-lite executables."""

    def __init__(
        self,
        *args: Any,
        morphable_model: str = 'BFMmm-23660',
        basis_model: str = 'local_basis_FacialBasis1.0',
        **kwargs: Any,
    ) -> None:
        """Configure the lite backend, which merges identity/pose estimation.

        Args:
            *args: Forwarded to :class:`FaceProcessor3DI`.
            morphable_model: Morphable model identifier bundled with 3DI-Lite.
            basis_model: Local basis used for :meth:`localized_expressions`.
            **kwargs: Forwarded to :class:`FaceProcessor3DI`.

        Raises:
            ValueError: If 3DI-Lite executable paths cannot be resolved.
        """
        # Run the parent class init
        super().__init__(*args, **kwargs)
        self.model_morphable = morphable_model
        self.model_basis = basis_model
        self.fast = False

        
        # specific file extension for 3DI-lite
        self.output_ext = '.3DIl'
    
        if not self.API:  
            self._set_runtime(name='3DI-lite', variable='BITBOX_3DI_LITE', executable='process_video.py', docker_path='/app/3DI_lite')
    
            if self.execDIR is None:
                raise ValueError("3DI-lite package is not found. Please make sure you defined BITBOX_3DI_LITE system variable or use our Docker image.")
            
            # prepare configuration files
            cfgid = 1
            self.config_landmarks = os.path.join(self.execDIR, 'configs/%s.cfg%d.%s.txt' % (self.model_morphable, cfgid, self.model_landmark))
        
        # prepare metadata
        self.base_metadata['backend'] = '3DI-lite'
        self.base_metadata['morphable_model'] = self.model_morphable
        self.base_metadata['local_bases'] = self.model_basis   
        
                
    def fit(self, normalize: bool = False, k: int = 1) -> Optional[Any]:
        """Run the single 3DI-Lite binary to estimate expressions and pose.

        Args:
            normalize: When ``True`` applies lite-specific normalization profiles.
            k: Neighborhood parameter for :func:`normalizeExpressions`.

        Returns:
            Optional[Any]: Tuple of output file paths (``'file'`` mode), tuple of
            parsed dictionaries (``'dict'`` mode), or ``None`` when suppressed.

        Raises:
            ValueError: If landmarks were not computed or backend execution fails.
        """
        # check if landmark detection was run and successful
        if self.cache.check_file(self._local_file(self.file_landmarks), self.base_metadata) > 0:
            raise ValueError("Landmark detection is not run or failed. Please run landmark detection first.")
             
        # STEP 1-4: learn identity, shape and texture model, pose and expression
        self._execute('process_video.py',
                    [self.file_input, self.file_landmarks, self.file_expression_smooth, self.file_pose_smooth, self.file_shape_coeff, self.file_texture_coeff, self.file_illumination],
                    "expression and pose estimation",
                    output_file_idx=[-5, -4, -3, -2, -1])
        
        # STEP 5: Canonicalized landmarks
        self._execute('produce_canonicalized_3Dlandmarks.py',
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
                self._local_file(self.file_landmarks_canonicalized),
                self._local_file(self.file_shape),
                self._local_file(self.file_texture)
            )
            return files
        elif self.return_output == 'dict':
            out_exp = read_expression(self._local_file(self.file_expression_smooth))
            if normalize:
                out_exp = normalizeExpressions(out_exp, proc='3DIl', k=k)
            out_pose = read_pose_lite(self._local_file(self.file_pose_smooth), self._local_file(self.file_landmarks))
            out_land_can = read_canonical_landmarks(self._local_file(self.file_landmarks_canonicalized))
            
            return out_exp, out_pose, out_land_can
        else:
            return None
        
    def localized_expressions(self, normalize: bool = True) -> Optional[Any]:
        """Compute localized expression coefficients for 3DI-Lite outputs.

        Args:
            normalize: Whether to request normalized coefficients from the helper
                script.

        Returns:
            Optional[Any]: Output location or dictionary mirroring ``return_output``.

        Raises:
            ValueError: If :meth:`fit` (and therefore expression smoothing) has not
                yet been executed.
        """
        # check if canonical landmark detection was run and successful
        if self.cache.check_file(self._local_file(self.file_expression_smooth), self.base_metadata) > 0:
            raise ValueError("Expression quantification is not run or failed. Please run fit() method first.")
        
        self._execute('compute_local_exp_coefficients.py',
                    [self.file_expression_smooth, self.file_expression_localized, int(normalize)],
                    "localized expression estimation",
                    output_file_idx=-2)
        
        if self.return_output == 'file':
            return self._local_file(self.file_expression_localized)
        elif self.return_output == 'dict':
            return read_expression(self._local_file(self.file_expression_localized))
        else:
            return None


    def citation(self) -> None:
        """Print citation text for studies processed with 3DI-Lite."""
        try:
            bb_version = version("bitbox")
        except PackageNotFoundError:
            bb_version = "ERROR_IN_READING_VERSION_NUMBER"
            
        text = f""" 
        The dataset was processed using Bitbox [1] version {bb_version} for facial behavior analysis. Facial modeling was performed with the 3DI-Lite [2], configured with the Basel Face Model (BFM) 2009 [3]. The iBUG-51 landmark template [4] was used for landmark definition.
        
        [INCLUDE THE FOLLOWING IF YOU USED LOCAL EXPRESSION COEFFICIENTS]
        Localized expression coefficients were computed using Facial Basis [2]. 

        [1] TBN
        [2] Sariyanidi E, Yankowitz L, Schultz RT, Herrington JD, Tunc B, Cohn J (2025). Beyond FACS: Data-driven facial expression dictionaries, with application to predicting autism. In Proceedings of the IEEE International Conference on Automatic Face and Gesture Recognition (FG), 19, 1-10. https://doi.org/10.1109/fg61629.2025.11099288
        [3] Paysan P, Knothe R, Amberg B, Romdhani S, Vetter T (2009). A 3D Face Model for Pose and Illumination Invariant Face Recognition. In Proceedings of the IEEE International Conference on Advanced Video and Signal based Surveillance (AVSS), 296-301. https://doi.org/10.1109/AVSS.2009.58
        [4] Sariyanidi E, Zampella CJ, Schultz RT, Tunc B (2020). Can facial pose and expression be separated with weak perspective camera? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 7173-7182. https://doi.org/10.1109/CVPR42600.2020.00720
        """
        print(text)
