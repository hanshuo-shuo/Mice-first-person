# Hypothesis

Passing through the frozen SAC policy's deterministic `base_head_action`
unchanged will improve clean success over the legal symmetric scan on at least
two of 128 paired development episodes without increasing capture episodes.

## Predicted effect

The learned image-conditioned head command should preserve the historical
active-gaze advantage while leaving the model's forward velocity, body yaw,
observation count, inference count, physics, reward, and termination unchanged.
The expected direction is positive paired clean-success delta and lower or
equal capture count; gaze travel may change but is secondary.
