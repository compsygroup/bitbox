---
layout:
  width: default
  title:
    visible: false
  description:
    visible: false
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
  metadata:
    visible: false
---

# Getting Started

<h2 align="center"><strong>Getting Started</strong></h2>

Bitbox is a free and open-source Python library that includes a comprehensive set of tools for the computational analysis of nonverbal human behavior. The provided tools enable the analysis of face, head, and body movements, expressions, and actions from videos and images. The included algorithms and metrics have been validated using clinically vetted datasets and extensively published, making them reliably usable by behavioral, social, and medical scientists in their human subject research.

In this example, we will walk through the process of analyzing facial expressions. Let's begin by importing the necessary backend. We will use 3DI as the backend.

```python
from bitbox.face_backend import FaceProcessor3DI as FP
```

Define the input video file and output directory. We will use the example video provided in the `tutorial` directory in the GitHub repository.

```python
# define input file and output directory
input_file = 'data/elaine.mp4'
output_dir = 'output'
```

```python
# define a face processor
processor = FP(runtime='bitbox:latest')
```

Set `runtime` to the Docker image name. If 3DI is installed natively, set `runtime` to the path where the 3DI executables are located. To avoid setting `runtime` every time you write your code, you can set the system variable `BITBOX_DOCKER`. See more details [here](../running-bitbox/standalone-mode.md#setting-up-backend-processors).

```python
# set input and output
processor.io(input_file=input_file, output_dir=output_dir)
```

3DI and 3DI-lite allow you to run individual steps separately, as shown below, in case you need only some of them.

```python
# detect faces
rects = processor.detect_faces()
```

This step detects the location of the face in each video frame as a rectangle. See details on output formats [here](outputs.md#standard-output-formats).

```python
# detect landmarks
lands = processor.detect_landmarks()
```

This step will detect facial landmarks (51 points) within each rectangle, and output their 2D x, y coordinates.

```python
# compute global expressions
exp_global, pose, lands_can = processor.fit()
```

This step produces the main outcomes of 3DI:&#x20;

1. 3D face mesh. The output is written to a file inside the `output` directory, but not returned as a Python object
2. Pose of the head. The output, `pose`, stores pitch, yaw, and roll angles of the head for each frame.
3. Canonicalized 3D landmarks. The output, `land_can`, stores 3D x, y, z coordinates of the same 51 landmark points for each frame, corrected for pose and identity, thus capturing only expression-related variation.
4. Deformation of the facial mesh caused by facial expressions. The output, `exp_global`, stores coefficients (magnitude) for 79 expression bases (defined by a PCA model of expressions) for each frame. Note that each expression basis is a global deformation of the facial mesh, thus difficult to interpret, unlike Action Units of FACS.

The 3DI and 3DI-lite also provide functionality to estimate localized expression variations, creating interpretable facial expression coefficients similar to the Action Units of FACS.&#x20;

```python
# compute localized expressions
exp_local = processor.localized_expressions()
```

The output, `exp_local`, stores coefficients for localized expressions. More details are provided [here](../affective-expressions/localized-expression-units.md).

Please follow these steps in the specified order. Each step relies on the output (files stored in `output`) from the previous step. Alternatively, you can run all steps at once.

```python
rects, lands, exp_global, pose, lands_can, exp_local = processor.run_all()
```
