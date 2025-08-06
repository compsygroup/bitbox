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

# Kinematics

<h2 align="center">Motion Kinematics</h2>

Bitbox computes fundamental kinematic features of motion, including range of motion, total path length, average speed, and average acceleration.

{% hint style="info" %}
**Range of motion**: Euclidean distance between maximal and minimal coordinates of the object, quantifying the overall range of motion.

**Total path length**: Cumulative Euclidean distance traveled by the object, summing frame-to-frame displacements for the entire video.

**Average speed**: Mean magnitude of the object’s frame-to-frame velocity (displacement × fps), giving its average speed.

**Average acceleration**: Mean magnitude of the second-order velocity differences, providing the object’s average acceleration.
{% endhint %}

You can compute these variables for face bounding boxes, head pose, facial landmarks, or body joints (coming soon).

```python
# run the processor
rects, lands, exp_global, pose, lands_can, exp_local = processor.run_all()

from bitbox.biomechanics import motion_kinematics

# compute motion kinematics for face rectangles
mrange, path, speed, accelaration = motion_kinematics(rects)

# compute motion kinematics for head pose
mrange, path, speed, accelaration = motion_kinematics(pose)

# compute motion kinematics for facial landmarks
mrange, path, speed, accelaration = motion_kinematics(lands)
```

When using face bounding boxes (rectangles), the center coordinates of the box are computed and used. With head pose, you can either use translation coordinates or rotation angles (yaw, pitch, roll).

```python
# using translation coordinates
mrange, path, speed, accelaration = motion_kinematics(pose)

# using rotation angles
mrange, path, speed, accelaration = motion_kinematics(pose, angular=True)
```

When using landmarks, variables are calculated for each landmark separately.
