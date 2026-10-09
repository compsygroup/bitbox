import json
import os
import shutil
from typing import Any, Optional

from .backend import FaceProcessor
from .readerOpenFace import (
    split_csv_to_of, read_confidence, read_rectangles, read_landmarks, read_canonical_landmarks,
    read_pose, read_gaze, read_eye_landmarks, read_expression, read_shape_params,
)

# order matches the of_names list in fit(); defines the 'dict'/'file' return order below
_OF_READERS = {
    'confidence': read_confidence, 'rects': read_rectangles, 'landmarks_2d': read_landmarks,
    'landmarks_3d': read_canonical_landmarks, 'pose': read_pose, 'gaze': read_gaze,
    'eye_landmarks': read_eye_landmarks, 'action_units': read_expression, 'shape_params': read_shape_params,
}


class FaceProcessorOpenFace(FaceProcessor):


    def __init__(self, *args: Any, **kwargs: Any) -> None:

        super().__init__(*args, **kwargs)

        self.output_ext = '.OF'
        self.file_openface_csv = None
        self.split_files = None

        if not self.API:
            self._set_runtime(
                name='OpenFace',
                variable='BITBOX_3DI',
                executable='FeatureExtraction',
                docker_path='/app/OpenFace/build/bin',
            )

            if self.execDIR is None:
                raise ValueError(
                    'OpenFace package is not found. Please make sure you defined '
                    'BITBOX_3DI system variable or use our Docker image.'
                )

        self.base_metadata['backend'] = 'OpenFace'

    def io(self, input_file: Optional[str] = None, output_dir: Optional[str] = None) -> None:
        """Validate I/O paths and register the expected OpenFace CSV output."""
        super().io(input_file=input_file, output_dir=output_dir)

        output_root = self.docker_output_dir if self.docker is not None else self.output_dir
        self.file_openface_csv = os.path.join(output_root, f'{self.file_input_base}.csv')

    def fit(self, split=True) -> Optional[Any]:
        """Run OpenFace feature extraction for the configured input video.

        Args:
            split: If True, automatically split the output CSV into separate
                files at runtime (default True). Split files are stored in
                self.split_files as a dict mapping type names to file paths.

        Returns:
            Optional[Any]: Tuple of output paths, in ``of_names`` order (``'file'`` mode);
            tuple of parsed dictionaries in the same order (``'dict'`` mode, matching how the
            3DI backends return several dicts from one ``fit()`` call); or ``None``.
            ``None`` in every mode when ``split=False``, since nothing was split to return.
        """
        of_names = ['confidence', 'rects', 'landmarks_2d', 'landmarks_3d',
                    'pose', 'gaze', 'eye_landmarks', 'action_units', 'shape_params']

        all_cached = False
        if split:
            of_paths = [os.path.join(self.output_dir, f'{self.file_input_base}_{name}.OF')
                        for name in of_names]
            all_cached = all(
                self.cache.check_file(p, self.base_metadata) == 0 for p in of_paths
            )
            if all_cached:
                self.split_files = {name: path for name, path in zip(of_names, of_paths)}

        if not all_cached:
            self._execute(
                'FeatureExtraction',
                [
                    '-f',
                    self.file_input,
                    '-out_dir',
                    self.docker_output_dir if self.docker is not None else self.output_dir,
                ],
                'feature extraction',
                expected_outputs=[self.file_openface_csv],
            )

            csv_path = self.file_openface_csv
            if self.docker is not None:
                csv_path = self._local_file(csv_path)

            if split and csv_path and os.path.isfile(csv_path):
                self.split_files = split_csv_to_of(csv_path, self.output_dir)
                metadata = self._build_split_metadata(csv_path)
                for path in self.split_files.values():
                    self.cache.store_metadata(path, metadata)

            # Clean up OpenFace artifacts (CSV, JSON, HOG, AVI, aligned frames)
            self._cleanup_openface_artifacts(csv_path)

        if not split:
            return None
        elif self.return_output == 'file':
            return tuple(self.split_files[name] for name in of_names)
        elif self.return_output == 'dict':
            results = []
            for name in of_names:
                path = self._local_file(self.split_files[name])
                out = _OF_READERS[name](path)
                if name == 'action_units':
                    # keep only the 17 continuous intensity columns (AU*_r); the binary presence
                    # flags (AU*_c) are dropped because they distort peak-based statistics in
                    # expressivity()/diversity()
                    out['data'] = out['data'][out['intensity_columns']]
                    out['format'] = 'for each frame (rows) Action Unit intensities (_r)'
                    out['presence_columns'] = []
                results.append(out)
            return tuple(results)
        return None

    def _build_split_metadata(self, csv_path):
        """Build metadata for .OF files, inheriting cmd from the CSV's JSON."""
        csv_json = csv_path + '.json'
        cmd = None
        if os.path.isfile(csv_json):
            with open(csv_json, 'r') as f:
                cmd = json.load(f).get('cmd')

        return {
            **self.base_metadata,
            'cmd': cmd,
            'input': self._local_file(self.file_input),
            'output': self.output_dir,
        }

    def _cleanup_openface_artifacts(self, csv_path):
        """Remove temporary OpenFace outputs (CSV, HOG, AVI, aligned frames)."""
        base = os.path.splitext(csv_path)[0]
        for ext in ['.csv', '.csv.json']:
            path = base + ext
            if os.path.isfile(path):
                os.remove(path)
