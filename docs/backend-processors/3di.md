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

# 3DI

<h2 align="center">3DI</h2>

3DI is an algorithm designed to create a detailed 3D mesh (\~20k points) of a face from 2D images or videos. This 3D mesh generation enables the separation of pose, expression, and person-specific features—key challenge for improving the accuracy and reliability of expression analysis. The mesh has a universal topology, meaning each point represents the same facial part across reconstructions. For instance, the 10665th point always aligns with the left lip corner, enabling consistent expression analysis across different individuals and situations.

### Model Architecture

The 3DI processes a video through the following five steps:

1. **Face Detection**: Identify faces in the entire video.
2. **Facial Landmark Detection**: Detect facial landmarks throughout the video.
3. **3D Identity Estimation**: Estimate neutral 3D identity parameters from a subset of video frames.
4. **3D Reconstruction**: Perform frame-by-frame 3D reconstruction using fixed identity parameters obtained from the previous step.
5. **Temporal Smoothing**: Smooth pose and expression parameters over time for consistency.

<figure><img src="../.gitbook/assets/3di (1).png" alt=""><figcaption></figcaption></figure>

### Separation of Pose, Expression, and Identity

Analyzing facial behavior in video involves quantifying pose and expression from 2D frames, which can be challenging. Facial pose, expressions, and individual identity are intertwined in 2D space, complicating analysis. For instance, a person viewed frontally may appear to frown if their head tilts downward, altering the distance between facial features. Furthermore, each person’s unique facial structure affects analysis, as variations in feature shapes and distances can skew expression detection. For example, naturally low eyebrows might falsely suggest a frown.

The use of 3D modeling offers a solution by accurately fitting a 3D morphable model (3DMM) to the subject's face. This approach effectively disentangles pose, expression, and identity, providing expression coefficients that represent expressions free from interference by pose or identity traits.

### Head Pose

The 3DI measures head pose using three angular rotations and the three Euclidean coordinates (x, y, z) capturing the head's location relative to the camera. Analyzing head pose throughout a video is crucial for understanding facial behavior, as head movements are key indicators of social communication, such as backchanneling. Additionally, static head pose is significant for assessing social orientation and attention, with some studies using them as proxies for eye contact and social gaze.

### Facial Expressions

3DI provides per-frame expression vectors via 3DMM fitting, consisting of 79 coefficients that capture facial expressions by describing deformation across the entire face. These vectors effectively isolate expressions from pose and person-specific facial features, the primary nuisances in expression analysis. With a detailed mesh of approximately 20,000 points, they ensure precise movement capture.

### Canonicalized 3D Landmarks

Facial landmarks are points on the face representing features like brows, eyes, nose, and mouth. By tracking these points in a video, we can measure facial expressions and movements. 3DI offers both 2D and 3D facial landmarks, including canonicalized 3D landmarks, which are ideal for analyzing expressions as they remove the effect of head or body movements and individual facial features. The standard landmark template in 3DI is the iBUG-51, which uses 51 points to track the brows, eyes, nose, and mouth. Unlike facial expression coefficients, landmarks offer movement in millimeters, making them valuable in applications requiring precise geometric measurements. This also allows for detailed analysis of specific points, such as lip corners.
