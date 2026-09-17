# count_pixels_interactive.py
#
# Usage:
#   python count_pixels_interactive.py image.png
#
# Zoom/pan first using matplotlib toolbar.
# When ready, press ENTER in terminal.
# Then click two points.

import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = np.asarray(Image.open(sys.argv[1]))

fig, ax = plt.subplots(figsize=(10,8))
ax.imshow(img, cmap="gray")
ax.set_title("Zoom/pan first, then press ENTER in terminal")
ax.axis("on")

plt.show(block=False)

input("Zoom/pan as desired, then press ENTER to select points... ")

ax.set_title("Click two points")
pts = plt.ginput(2, timeout=-1)

plt.close(fig)

(x1, y1), (x2, y2) = pts

dx = x2 - x1
dy = y2 - y1
dist = np.sqrt(dx**2 + dy**2)

beamspot = 1e-2
pixel_pitch = 1.55e-6
print(f"Point 1: ({x1:.2f}, {y1:.2f})")
print(f"Point 2: ({x2:.2f}, {y2:.2f})")
print(f"dx = {abs(dx):.2f} px")
print(f"dy = {abs(dy):.2f} px")
print(f"distance = {dist:.2f} px")
print(f"Magnification: {pixel_pitch * dist / beamspot}")