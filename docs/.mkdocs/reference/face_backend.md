# Face Backend API

GPU-ready wrappers around the 3DI face backends with built-in caching, Docker/runtime
resolution, and optional remote execution.

<div class="grid cards" markdown>

-   :material-face-recognition: **Unified face stack**
    
    ---
    
    Detect faces, landmarks, expressions, and pose through one coherent API.

-   :material-chip: **GPU + container aware**
    
    ---
    
    Run directly on your local GPU or via Docker / Singularity on servers.

-   :material-database-sync: **Cache-first design**
    
    ---
    
    Every intermediate artifact is cached to disk and can be reused across runs.

-   :material-code-json: **Flexible outputs**
    
    ---
    
    Work with Python dicts during exploration or file paths for large-scale jobs.

</div>


## Usage overview

The face backend wraps 3DI binaries into a single, stateful `FaceProcessor3DI` or `FaceProcessor3DIlite`  objects:

1. Configure how you want outputs returned (`dict` vs `file`).
2. Point the processor to your input (`video`, `image sequence`, etc.).
3. Call stepwise methods (`detect_faces`, `detect_landmarks`, `fit`, …)  
   or the all-in-one `run_all()` pipeline.


## Prerequisites

!!! info ":material-clipboard-list: Setup checklist"

    - Install the 3DI backend (full or lite) or pull the container image, then either:
      - pass a `runtime` when you instantiate the processor, **or**
      - set env vars: `BITBOX_3DI`, `BITBOX_3DI_LITE`, `BITBOX_DOCKER`.
    - Point `output_dir` to a writable location; every intermediate artifact is cached there.
    - Verify the GPU is visible if you want acceleration (e.g., `nvidia-smi` works in your runtime/container).


## Core usage patterns

=== "Local GPU"

    ```python
    from bitbox.face_backend.backend3DI import FaceProcessor3DI

    # Return rich Python dicts for interactive work
    processor = FaceProcessor3DI()

    # Attach inputs and outputs
    processor.io("data/video.mp4", output_dir="output/")

    # Stepwise processing
    rectangles = processor.detect_faces()
    landmarks = processor.detect_landmarks()
    expressions, pose, canonical_landmarks = processor.fit(normalize=True, k=1)
    localized = processor.localized_expressions()
    ```

=== "Docker/Singularity Execution"

    ```python
    from bitbox.face_backend.backend3DI import FaceProcessor3DI

    processor = FaceProcessor3DI(runtime='bitbox:latest')
    processor.io("data/video.mp4", output_dir="output/")
    results = processor.run_all()
    ```

=== "Remote API Server"

    ```python
    from bitbox.face_backend.backend3DI import FaceProcessor3DI

    # Server definition matches your Bitbox API deployment
    server = {"host": "api.bitbox.local", "port": 8000}

    processor = FaceProcessor3DI(
        server=server,
        return_output="dict",
    )
    processor.io("data/video.mp4", output_dir="output/remote")

    results = processor.run_all()

    ```


## Return modes

!!! note "Choosing between `dict` and `file`"

    - `return_output="dict"` (default):  
      Ideal for notebooks, prototyping, and analysis scripts.
      - You get Python objects: arrays, dicts keyed by frame, etc.
      - Intermediate disk artifacts still exist for caching, but you work in memory.

    - `return_output="file"`:  
      Best for large-scale or distributed pipelines.
      - Methods return file paths to cached artifacts.
      - Downstream jobs can pick up these paths without loading everything into RAM.


### Outputs at a glance

| Step                       | `return_output="dict"`                        | `return_output="file"`                     |
| -------------------------- | --------------------------------------------- | ------------------------------------------ |
| `detect_faces()`           | Rectangle dict keyed by frame                 | `.rect` file                               |
| `detect_landmarks()`      | Landmark dict keyed by frame                  | `.land` file                               |
| `fit(normalize=True, k=1)`| Expressions, pose, canonical landmarks        | Coefficient / expression / pose files      |
| `localized_expressions()` | Localized expression coefficients             | Localized coefficient file                 |
| `run_all()`               | Tuple of all outputs above                    | Matching tuple of file paths               |

## API reference

The full Python API is documented via `mkdocstrings`:

::: bitbox.face_backend.backend.FaceProcessor
    options:
      show_root_heading: true
      show_source: true
      filters:
        - "!^_"
        - "!^__"

::: bitbox.face_backend.backend3DI.FaceProcessor3DI
    options:
      show_root_heading: true
      show_source: true
      filters:
        - "!^_"
        - "!^__"

::: bitbox.face_backend.backend3DI.FaceProcessor3DIlite
    options:
      show_root_heading: true
      show_source: true
      filters:
        - "!^_"
        - "!^__"
