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

# 3DI-Lite

<h2 align="center">3DI-Lite</h2>

3DI-Lite has been developed to make 3DI more robust to occlusions, frame cuts, and other issues that occur frequently in natural videos, as well as to improve speed. 3DI-Lite is a deep learning model that takes video and facial landmarks as input and outputs head pose and expression coefficients as they would be computed by 3DI. Specifically, it was trained to predict 3DI outputs from video without explicitly fitting a 3DMM. This approach dramatically improves speed.

3DI-Lite incorporates the same components as 3DI for face rectangle detection, 2D landmark identification, and computing canonicalized 3D landmarks from expression coefficients.
