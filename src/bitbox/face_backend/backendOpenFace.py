import json
import os
import shutil
from typing import Any, Optional

from .backend import FaceProcessor
from .readerOpenFace import split_csv_to_of


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

    def fit(self, split=True) -> Optional[str]:
        """Run OpenFace feature extraction for the configured input video.

        Args:
            split: If True, automatically split the output CSV into separate
                files at runtime (default True). Split files are stored in
                self.split_files as a dict mapping type names to file paths.
        """
        if split:
            of_names = ['confidence', 'rects', 'landmarks_2d', 'landmarks_3d',
                        'pose', 'gaze', 'eye_landmarks', 'action_units', 'shape_params']
            of_paths = [os.path.join(self.output_dir, f'{self.file_input_base}_{name}.OF')
                        for name in of_names]
            all_cached = all(
                self.cache.check_file(p, self.base_metadata) == 0 for p in of_paths
            )
            if all_cached:
                self.split_files = {name: path for name, path in zip(of_names, of_paths)}
                if self.return_output == 'file':
                    return self.split_files
                return None

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

        if self.return_output == 'file':
            return self.split_files if split else None
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
