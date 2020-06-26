import numpy as np
import cv2

def np2Depth(input_tensor, invaild_mask):
    normalized = (input_tensor - 0.02) / (2.0 - 0.02) * 255.0
    normalized = normalized.astype(np.uint8)
    normalized = cv2.applyColorMap(normalized, cv2.COLORMAP_RAINBOW)
    normalized[invaild_mask] = 0
    return normalized