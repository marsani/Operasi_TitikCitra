import cv2
import numpy as np

# 1. 8x8 Educational Matrix (Grayscale & Basic Colors)
img_8x8 = np.array([
    [[255, 0, 0],   [255, 0, 0],   [0, 255, 0],   [0, 255, 0],   [0, 0, 255],   [0, 0, 255],   [255, 255, 0], [255, 255, 0]],
    [[255, 0, 0],   [200, 50, 50], [0, 255, 0],   [50, 200, 50], [0, 0, 255],   [50, 50, 200], [255, 255, 0], [200, 200, 0]],
    [[0, 255, 255], [0, 255, 255], [255, 0, 255], [255, 0, 255], [255, 255, 255],[255, 255, 255],[128, 128, 128],[128, 128, 128]],
    [[0, 200, 200], [0, 150, 150], [200, 0, 200], [150, 0, 150], [220, 220, 220],[180, 180, 180],[100, 100, 100],[60, 60, 60]],
    [[0, 0, 0],     [30, 30, 30],  [60, 60, 60],  [90, 90, 90],  [120, 120, 120],[150, 150, 150],[180, 180, 180],[210, 210, 210]],
    [[255, 128, 0], [255, 165, 0], [128, 0, 128], [75, 0, 130],  [0, 128, 128], [0, 255, 128], [255, 192, 203],[255, 20, 147]],
    [[10, 20, 30],  [40, 50, 60],  [70, 80, 90],  [100, 110, 120],[130, 140, 150],[160, 170, 180],[190, 200, 210],[220, 230, 240]],
    [[255, 255, 255],[0, 0, 0],    [255, 255, 255],[0, 0, 0],    [255, 255, 255],[0, 0, 0],    [255, 255, 255],[0, 0, 0]]
], dtype=np.uint8)
# OpenCV uses BGR
img_8x8_bgr = cv2.cvtColor(img_8x8, cv2.COLOR_RGB2BGR)
cv2.imwrite("sample_images/contoh_8x8_pola_matriks.png", img_8x8_bgr)

# 2. 16x16 Color Grid
img_16x16 = np.zeros((16, 16, 3), dtype=np.uint8)
for r in range(16):
    for c in range(16):
        img_16x16[r, c] = [int(r * 16), int(c * 16), int((r + c) * 8)]
cv2.imwrite("sample_images/contoh_16x16_gradien_warna.png", cv2.cvtColor(img_16x16, cv2.COLOR_RGB2BGR))

# 3. 128x128 High Resolution Test Pattern
img_128 = np.zeros((128, 128, 3), dtype=np.uint8)
# Gradient background
for y in range(128):
    for x in range(128):
        img_128[y, x] = [int(x * 2), int(y * 2), int((x + y))]
# Draw circles and rectangles
cv2.circle(img_128, (64, 64), 36, (255, 255, 255), -1)
cv2.circle(img_128, (64, 64), 24, (255, 50, 50), -1)
cv2.rectangle(img_128, (20, 20), (50, 50), (50, 255, 50), 2)
cv2.rectangle(img_128, (78, 78), (108, 108), (50, 50, 255), -1)
cv2.line(img_128, (10, 118), (118, 10), (255, 255, 0), 2)
cv2.imwrite("sample_images/contoh_128x128_geometris.png", img_128)

print("Generated sample images successfully!")
