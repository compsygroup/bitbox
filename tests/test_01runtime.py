from bitbox.face_backend import FaceProcessor3DI as FP3DI
from bitbox.face_backend import FaceProcessor3DIlite as FP3DIlite

import os
import csv
import pytest

def load_test_cases():
    path = os.path.join(os.path.dirname(__file__), "conditions.csv")
    cases = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lite = row['lite'].strip() == '1'
            def parse(key):
                v = row[key].strip()
                return None if v == 'None' else v.strip("'")
            cases.append((
                lite,
                parse('BITBOX_3DI'),
                parse('BITBOX_3DI_LITE'),
                parse('BITBOX_DOCKER'),
                parse('runtime'),
                parse('expected_docker'),
                parse('expected_executables'),
            ))
    return cases

@pytest.mark.parametrize(
    "lite, BITBOX_3DI, BITBOX_3DI_LITE, BITBOX_DOCKER, runtime, expected_docker, expected_executables",
    load_test_cases()
)
def test_runtime_selection(capsys, lite, BITBOX_3DI, BITBOX_3DI_LITE, BITBOX_DOCKER, runtime, expected_docker, expected_executables):
    # set or unset env vars
    if BITBOX_3DI is not None:
        os.environ['BITBOX_3DI'] = BITBOX_3DI
    else:             
        os.environ.pop('BITBOX_3DI', None)
    
    if BITBOX_3DI_LITE is not None:
        os.environ['BITBOX_3DI_LITE'] = BITBOX_3DI_LITE
    else:                           
        os.environ.pop('BITBOX_3DI_LITE', None)

    if BITBOX_DOCKER is not None: 
        os.environ['BITBOX_DOCKER'] = BITBOX_DOCKER
    else:                         
        os.environ.pop('BITBOX_DOCKER', None)

    cls = FP3DIlite if lite else FP3DI
   
    # try to instantiate and run .io(); catch missing-runtime errors as None outputs
    try:
        processor = cls(runtime=runtime, debug=True)
        
        input = os.path.join(os.path.dirname(__file__), 'data', 'elaine.mp4')
        output = os.path.join(os.path.dirname(__file__), 'output')
        
        processor.io(input_file=input, output_dir=output)
        
        lines = capsys.readouterr().out.splitlines()
        raw_docker = lines[0].split('=', 1)[1]
        raw_exec = lines[1].split('=', 1)[1]
        docker_val = None if raw_docker == 'None' else raw_docker
        exec_val = None if raw_exec == 'None' else raw_exec
    except ValueError:
        docker_val = None
        exec_val = None

    assert docker_val == expected_docker
    assert exec_val == expected_executables
