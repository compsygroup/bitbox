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

# Imitation

<h2 align="center">Imitation Quality</h2>

Bitbox can quantify how closely one set of signals follows a reference set of signals. This functionality is useful for measuring imitation quality in facial expressions, head movements, or body actions (coming soon).

This function accepts only landmarks ([2D ](../overview/outputs.md#id-2d-face-landmarks)or [3D](../overview/outputs.md#id-3d-face-landmarks)), head pose, or facial expressions ([global](../affective-expressions/facial-expressions.md#expression-related-global-deformations) or [local](../affective-expressions/facial-expressions.md#localized-expression-units)). It computes a windowed, lagged cross-correlation between all pairs of signals.

```python
from bitbox.social import imitation

# output directory
output_dir = 'output'

# define a face processor
processor = FP(runtime='bitbox:latest')

# get all facial signals for a participant
processor.io(input_file='participant.mp4', output_dir=output_dir)
rect_p, land_p, exp_glob_p, pose_p, land_can_p, exp_loc_p = processor.run_all(normalize=True)

# get all facial signals for the reference model
processor.io(input_file='model.mp4', output_dir=output_dir)
rect_r, land_r, exp_glob_r, pose_r, land_can_r, exp_loc_r = processor.run_all(normalize=True)

# quantify imitation performance
corr_mean, corr_std, corr_lag = imitation(exp_glob_p, exp_glob_r, width=1.1, step=0.5, fps=30)
```

{% hint style="success" %}
`corr_mean`: the average correlation between two signals across all time windows, \
`corr_std`: the standard deviation of correlations, \
`corr_lag`: the optimal lag that maximizes the correlation.
{% endhint %}

All three outputs are matrices where rows correspond to the first set of signals (_e.g._, 79 global expression units for the participant) and columns to the second (_e.g._, 79 global expression units for the model).

Bitbox automatically learns the optimal lag between signals. In imitation mode, lag is restricted to one direction — a time point in the first signal can match the same or an earlier point in the reference signal (you can imitate a model slightly behind, but not ahead). To disable this causality constraint and allow lags in both directions, set `casuality=False`. Note that lags are defined relative to the second signal; if you need bidirectional lags, use the [coordination](coordination.md) function instead.

```python
# allow lags both in past and future
corr_mean, corr_std, corr_lag = imitation(exp_glob_p, exp_glob_r, casuality=False)
```

{% hint style="warning" %}
The window width (`width`) and step size (`step`) are defined in seconds, so ensure that the frame rate (`fps`) matches your signal’s sampling rate.
{% endhint %}

By default, positive and negative correlations are treated the same (the maximum absolute correlation is used per window). To ignore negative correlations and consider only positive ones, set `polarity=False`.

```python
# Only positive correlations are preferred
corr_mean, corr_std, corr_lag = imitation(exp_glob_p, exp_glob_r, polarity=False)
```
