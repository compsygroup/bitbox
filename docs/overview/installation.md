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

# Installation

<h2 align="center"><strong>Installation</strong></h2>

Bitbox itself has minimum requirements, but it relies on face/body backends to generate expression and movement signals. These backends usually have more requirements. We highly recommend using our Docker images to install these backends, as installing them from source code may prove difficult for some. **For currently supported backends, you need NVIDIA GPUs.** New backends with CPU support are coming soon.

### **Installing Backends**

The current version of Bitbox supports two face backends, namely 3DI and 3DI-lite. While 3DI provides more detailed outputs (_e.g._, a full 3D model of the face), 3DI-lite is much faster and more robust to occlusions. If your images don't have significant occlusions and you don't need a faster solution, we recommend using 3DI.&#x20;

**We recommend using Docker** for installing all backends at once with no extra steps.

{% tabs %}
{% tab title="Docker (Recommended)" %}
Using Docker is usually straightforward. However, 3DI requires downloading an external face model, which involves individually registering and requesting access, and updating our image with this model.

1. Download the Dockerfile. We have two options for [CUDA 11.8.0](https://raw.githubusercontent.com/compsygroup/bitbox/refs/heads/main/docker/cuda11.8_cv4.5/Dockerfile) or [CUDA 12.2.2](https://raw.githubusercontent.com/compsygroup/bitbox/refs/heads/main/docker/cuda12.2_cv4.8/Dockerfile). Select the one that is most compatible with your NVIDIA GPU.
2. Register and download the [face model](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2\&id=downloads). This is an automated process, and you should immediately get the download link in your email. Extract the model you downloaded and locate the file `01_MorphableModel.mat` .&#x20;
3. Place the Dockerfile and the face model (`01_MorphableModel.mat`) in the same directory
4.  Within this directory, run the following command to build the image

    ```bash
    docker build -t bitbox:latest . 
    ```

    The parameter `bitbox:latest` is the name of the image to be created. You can change it if you wish. Please don't forget the `.` at the end.
{% endtab %}

{% tab title="Native (Not Recommended)" %}
If you can install C++/CUDA codes from the source code, however, please go ahead and install 3DI from [here](https://github.com/compsygroup/3DI/tree/v0.3.1). The instructions are provided there. This approach will install the 3DI as a native application on your system and can be slightly faster than using Docker.

Similarly, 3DI-lite can be installed from ... (COMING SOON)
{% endtab %}
{% endtabs %}

### Installing Python Library

Once you installed the backends, you can install the stable version of the Bitbox Python library using `pip` or the latest version directly from GitHub.

{% tabs %}
{% tab title="pip" %}
```bash
pip3 install bitbox
```
{% endtab %}

{% tab title="GitHub" %}
We recommend using a virtual environment. **You will need to use python 3.8 or higher**.&#x20;

```bash
python3 -m venv env
source env/bin/activate
```

For the following steps, we will assume you have activated the virtual environment.

1.  Clone the Bitbox repository:

    ```bash
    git clone https://github.com/compsygroup/bitbox.git
    ```
2.  Change to the Bitbox directory:

    ```bash
    cd bitbox
    ```
3.  Install requirements:

    ```bash
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    ```
4.  Install Bitbox:

    ```bash
    python3 setup.py install
    ```
{% endtab %}
{% endtabs %}
