# Behavioral Imaging Toolbox

Bitbox is a free and open-source Python library including a comprehensive set of tools for the computational analysis of nonverbal human behavior. The provided tools enable analysis of face, head, and body movements, expressions, and actions from videos and images. Included algorithms and metrics have been validated using clinically vetted datasets and published extensively, therefore, can be reliably used by behavioral, social, and medical scientists in their human subject research. As we closely follow state-of-the-art in computer vision and machine learning, provided methodologies can also be relied upon by computer vision researchers and other engineers as well.

Please refer to our [Wiki](https://github.com/compsygroup/bitbox/wiki) for further details.

## Installation

Bitbox itself has minimum requirements, but it relies on face/body backends to generate expression/movement signals. These backends usually have more requirements. We highly recommend using our Docker images to install these backends as installing them from source code may prove difficult for some. Unfortunately, for currently supported backends, you need NVIDIA GPUs. New backends with CPU support are coming soon. 

### Installing Face Backends 

The current version of Bitbox supports two face backends, namely 3DI and 3DI-lite. While 3DI provides more detailed outputs (e.g., full 3D model of the face), 3DI-lite is much faster and more robust to occlusions, etc. If your images don't have significant occlusions and you don't need a faster solution, we recommend using 3DI.

If you can install C++/CUDA codes from the source code, please go ahead and install 3DI from [here](https://github.com/compsygroup/3DI). The instructions are provided there. This approach will install the 3DI as a native application on your system and will be more convenient for using Bitbox.

Similarly, 3DI-lite can be installed from ... (COMING SOON)

The recommended way to install backends is to use our Docker images. Using Docker is usually very straightforward; however, 3DI requires downloading an external face model (you need to register individually and request access) and updating our image with this model.

We have a pre-compiled Docker image including both 3DI and 3DI-lite, but with a specific CUDA driver (i.e., 11.4.3). If your GPU can work with this version of CUDA, please use our image.

1. Download the [Dockerfile](https://raw.githubusercontent.com/compsygroup/bitbox/refs/heads/main/docker/3DI/Dockerfile)
2. Download the [3DMM model](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads)
3. Place the Dockerfile and the face model (`01_MorphableModel.mat`) in the same directory
4. Within this directory, run the following command to copy the face model
    ```bash
    docker build -t 3di:20250220-basel2009 . 
    ```
    The first parameter `3di:20250220-basel2009` is the name of the image to be created. You can change it if you wish. Please don't forget the `.` at the end. 
5. That's it! You will also need to set an environment variable `DOCKER_3DI`, which will be explained below.

### Installing Bitbox
To install Bitbox, follow these steps. **You will need to use python 3.8 or higher**. 

1. Create a virtual environment and activate it:
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```
    Note that this will create a virtual environment named `env` in the current directory. You can use any name, and you can install the virtual environment anywhere you like. Just don't forget where you installed it. For the following steps, we will assume you have activated the virtual environment.

2. Clone the Bitbox repository:
    ```bash
    git clone https://github.com/compsygroup/bitbox.git
    ```

3. Change to the Bitbox directory:
    ```bash
    cd bitbox
    ```

4. Install requirements:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

5. Install Bitbox using `python setup.py install`:
    ```bash
    python setup.py install
    ```

6. If you are not using Docker, set the environment variable `PATH_3DI` to indicate the directory in which 3DI was installed. We recommend setting it in .bahsrc (on Lunux/Mac) or in System's Environment Variables (on Windows).

    - **Linux**:
      ```bash
      export PATH_3DI=/path/to/3DI/directory
      ```

    - **Windows** (Command Prompt):
      ```bash
      set PATH_3DI=C:\path\to\3DI\directory
      ```

    - **Mac**:
      ```bash
      export PATH_3DI=/path/to/3DI/directory
      ```

7. If you are using Docker, set the environment variable `DOCKER_3DI` to indicate the 3DI image name/tag. Change the image name/tag if needed. We recommend setting it in .bahsrc (on Lunux/Mac) or in System's Environment Variables (on Windows).

    - **Linux**:
      ```bash
      export DOCKER_3DI=3di:20250220-basel2009
      ```

    - **Windows** (Command Prompt):
      ```bash
      set DOCKER_3DI=3di:20250220-basel2009
      ```

    - **Mac**:
      ```bash
      export DOCKER_3DI=3di:20250220-basel2009
      ```

Now you are ready to use Bitbox!

## Use

Once you are done with installation, you can use Bitbox by

1. Activate the virtual environment you created for Bitbox:
    ```bash
    source env/bin/activate
    ```
2. Set the environment variable `PATH_3DI` or `DOCKER_3DI` if you have not set them already in .bahsrc (on Lunux/Mac) or in System's Environment Variables (on Windows). If you did that you can skip this step.

3. Import the library in your Python code:
 ```python
from bitbox.face_backend import FaceProcessor3DI
 ```

### Example Usage

 ```python
from bitbox.face_backend import FaceProcessor3DI
import os

# Please make sure you give the correct full (not relative) path
DIR = '/path/to/data'
input_file = os.path.join(DIR, 'video.mp4') 
output_dir = os.path.join(DIR, 'output')

# define a face processor
processor = FaceProcessor3DI()

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

# detect faces
rects = processor.detect_faces()

# detect landmarks
lands = processor.detect_landmarks()

# compute global expressions
exp_global, pose, lands_can = processor.fit()

# compute localized expressions
exp_local = processor.localized_expressions()
 ```

