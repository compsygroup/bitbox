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

# Symmetry

<h2 align="center">Symmetry of Expressions</h2>

Bitbox analyzes the asymmetry of facial expressions, with body expression analysis coming soon. It measures the disparity between the left and right sides by calculating the Euclidean distance between the coordinates of key landmarks on facial components such as the eyes, brows, nose, and mouth in each frame. The left and right eyes are compared, the left and right sides of the mouth are compared, and so on.

This function supports only facial landmarks as input. You can use either 2D or 3D canonicalized landmarks. However, we strongly recommend using 3D canonicalized landmarks, especially if the face is not strictly frontal. This is because pose differences can significantly and variably affect 2D landmark coordinates on the left and right sides.
