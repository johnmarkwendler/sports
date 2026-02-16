"""
Pool keypoint indices for homography (frame → pool).

Keypoint template corners: 00 (top-left), 05 (bottom-left), 13 (bottom-right), 18 (top-right).
config.vertices order: bottom-left, top-left, top-right, bottom-right.
"""

import numpy as np

# Keypoint indices for the four pool corners, in the same order as config.vertices:
# [bottom-left, top-left, top-right, bottom-right]
POOL_CORNER_KEYPOINT_INDICES = np.array([5, 0, 18, 13])
