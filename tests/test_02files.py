from bitbox.face_backend import FaceProcessor3DI as FP3DI
from bitbox.face_backend import FaceProcessor3DIlite as FP3DIlite

import os
import csv
import pytest

def load_expected_files():
    path = os.path.join(os.path.dirname(__file__), "files.csv")
    cases = {0: [], 1: []}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['lite'])
            cases[idx].append(row['file'])
    return cases

expected_files = load_expected_files()

@pytest.mark.parametrize("lite", [False, True])
def test_creates_all_files(tmp_path, lite):
    # choose processor class
    cls = FP3DIlite if lite else FP3DI
    processor = cls(runtime='bitbox:cuda12', verbose=False, return_output=None)

    try:
        input = os.path.join(os.path.dirname(__file__), 'data', 'elaine.mp4')
        output = str(tmp_path)
        
        processor.io(input_file=input, output_dir=output)
        processor.detect_faces()
        processor.detect_landmarks()
        processor.fit()
        processor.localized_expressions()
        
        # verify existence of each expected file
        for file_name in expected_files[int(lite)]:
            file_path = os.path.join(output, file_name)
            assert (os.path.exists(file_path) and os.path.isfile(file_path)), f"Missing file: {file_path}"
    except ValueError:
        assert False