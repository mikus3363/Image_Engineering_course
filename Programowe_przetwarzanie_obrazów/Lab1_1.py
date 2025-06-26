import cv2
import numpy as np
from matplotlib import pyplot as plt

# wczytanie obrazu
plt.rcParams["figure.figsize"] = (10, 5) # ustalenie rozmiaru
image = cv2.cvtColor(cv2.imread("obrazek.jpg"), cv2.COLOR_BGR2RGB) # wczytanie obrazka z zamianą z BGR na RBG

# maski Górnoprzepustowa
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

kernel = np.asarray(kernel)
# filtracja
filtered_image = cv2.filter2D(image, -1, kernel) # przekształcenie obrazu za pomocą maski

# wyświetlenie obrazu
plt.subplot(1, 2, 1) # tworzenie pierszwej sekcji na obraz
plt.title("Oryginalny obraz") # nadanie tytułu obrazkowi
plt.imshow(image) # wyświelenie obrazu przed filtracją
plt.axis("off") # ukrycie osi

plt.subplot(1, 2, 2) # tworzenie drugiej sekcji na obraz
plt.title("Filtr górnoprzepustowy") # nadanie tytułu obrazkowi
plt.imshow(filtered_image) # wyświelenie obrazu po filtracji
plt.axis("off") # ukrycie osi

plt.show() # wyświetlenie całości