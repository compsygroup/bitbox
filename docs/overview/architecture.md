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

# Architecture

<h2 align="center"><strong>Architecture</strong></h2>

Bitbox is composed of two main components:

1. **Backend Processors**: A set of backend processors for analyzing face, body, and speech behavior. These processors generate _raw_ signals that encode how the face deforms, the body moves, or speech occurs. They are included in our Docker image, so you don't need to install and set them up individually.
2. **Python Library**: A Python library that includes wrapper functions for running the backend processors and additional analysis functions that use the raw signals to produce measurements of psychomotor behavior, affective expressions, and interpersonal dynamics.

<figure><img src="../.gitbook/assets/architecture (1).png" alt="" width="563"><figcaption></figcaption></figure>

### Backend Processors

Bitbox includes open-source processors (3DI, 3DI-Lite) created by the group behind Bitbox, along with other open-source processors available from different groups for research purposes. All processors can be used as standalone executables, either as actual executable files (e.g., exe) or Python scripts. When using our Docker image, you don't need to worry about the differences between executable types, as we handle the installation and setup for you.&#x20;

Each processor generates specific output files. For example, 3DI generates text files containing time series data for face bounding box coordinates, landmark coordinates, head pose angles, facial expression coefficients, and more. Bitbox utilizes wrapper functions to run these executables and translates their output files into a standard Python dictionary format. This ensures consistent output for each element of analysis (_e.g._, landmarks or facial expressions) regardless of the processor used.

### Wrapper Functions

Bitbox provides a specialized wrapper class for each backend processor.

```python
# to use 3DI backend
from bitbox.face_backend import FaceProcessor3DI
# to use 3DI-Lite backend
from bitbox.face_backend import FaceProcessor3DIlite
```

Bitbox wrappers define a unified usage pattern and output format across processors. For example, the code segment below will be the same whether you use 3DI, 3DI-Lite, or OpenFace (coming soon).

```python
# define input file and output directory
input_file = 'data/elaine.mp4'
output_dir = 'output'

# define a face processor
processor = FP()

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)

# when using OpenFace you will skip exp_global and lands_can,
# as it does not generate these variables
rects, lands, exp_global, poses, lands_can, exp_local = processor.run_all()
```

### Analysis Functions

Bitbox offers functionalities to study three core aspects of social-emotional behavior: psychomotor behavior, affective expressions, and interpersonal dynamics. Each function provides a set of measurements that can be used in statistical analysis pipelines or as a collection in machine learning/AI pipelines.

For example, once the local expression coefficients are computed, overall expressivity of the face can be measured using the following function.

```python
# Quantify the overall expressivity (and its stats) of the face
expressivity_stats = expressivity(exp_local, use_negatives=0, num_scales=6, robust=True, fps=30)
```

More details on analysis functions are provided in [Psychomotor Behavior](broken-reference), [Affective Expressions](../affective-expressions/), and [Interpersonal Dynamics](../social-dynamics/) sections.
