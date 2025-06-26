import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

cell_size = 8
orientations = 9
frame_size = 3

image = cv2.imread("zad8.png", cv2.IMREAD_GRAYSCALE)
height, width = image.shape

features, _ = hog(image,
                  orientations=orientations,
                  pixels_per_cell=(cell_size, cell_size),
                  cells_per_block=(2, 2),
                  block_norm='L2-Hys',
                  visualize=True,
                  feature_vector=True)

cells_x = width // cell_size
cells_y = height // cell_size
hog_array = features.reshape((cells_y - 1, cells_x - 1, 4, orientations)).sum(axis=2)

n_x = (width // 2) // cell_size
n_y = (height // 2) // cell_size

def get_hist(y, x):
    if 0 <= y < hog_array.shape[0] and 0 <= x < hog_array.shape[1]:
        hist = hog_array[y, x]
        return hist if np.sum(hist) > 0 else np.ones(orientations) * 1e-6
    return np.ones(orientations) * 1e-6

def get_area_hist(y, x, size=3):
    offset = size // 2
    hist = np.zeros(orientations)
    for dy in range(-offset, offset + 1):
        for dx in range(-offset, offset + 1):
            hist += get_hist(y + dy, x + dx)
    return hist

hist_prev = get_area_hist(n_y - 1, n_x, frame_size)
hist_curr = get_area_hist(n_y, n_x, frame_size)
hist_next = get_area_hist(n_y + 1, n_x, frame_size)

image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
half = (frame_size // 2) * cell_size
x_start = max(0, (n_x - 1) * cell_size)
y_start = max(0, (n_y - 1) * cell_size)
x_end = min(width, (n_x + 2) * cell_size)
y_end = min(height, (n_y + 2) * cell_size)
cv2.rectangle(image_color, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

cv2.imshow("Twarz środkowego mężczyzny", image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

def plot_hog_histogram(hist, ax, title):
    ax.bar(np.arange(orientations), hist, color='blue')
    ax.set_title(title)
    ax.set_xlabel("HOG feature")
    ax.set_ylabel("vector height")
    ax.set_xticks(np.arange(orientations))

def plot_orientation_arrows(hist, ax):
    ax.axis('equal')
    ax.axis('off')
    max_len = max(hist)
    for i in range(orientations):
        angle = 2 * np.pi * i / orientations
        dx = np.cos(angle)
        dy = np.sin(angle)
        ax.arrow(0, 0, dx * hist[i] / max_len, dy * hist[i] / max_len, head_width=0.05, head_length=0.1, fc='blue', ec='blue')

fig, axes = plt.subplots(3, 2, figsize=(10, 10))

plot_hog_histogram(hist_prev, axes[0, 0], "Histogram n-1")
plot_orientation_arrows(hist_prev, axes[0, 1])

plot_hog_histogram(hist_curr, axes[1, 0], "Histogram n (środek twarzy)")
plot_orientation_arrows(hist_curr, axes[1, 1])

plot_hog_histogram(hist_next, axes[2, 0], "Histogram n+1")
plot_orientation_arrows(hist_next, axes[2, 1])

plt.tight_layout()
plt.show()

