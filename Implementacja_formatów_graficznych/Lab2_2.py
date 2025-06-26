import numpy as np
import cv2
from matplotlib import pyplot as plt

# Nagłówek pliku PPM w formacie P3
ppm_header = 'P3\n120 8\n255\n'  # Format PPM: szerokość=120, wysokość=8, maksymalna wartość koloru=255

# Inicjalizacja pierwszego piksela (czarny kolor: [0, 0, 0])
image = np.array([0, 0, 0], dtype=np.uint8)  # Tablica przechowująca dane obrazu
dummy = np.array([0, 0, 0], dtype=np.uint8)  # Tymczasowa tablica do modyfikacji wartości RGB

step = 15  # Krok zmiany koloru (255 / 17), aby uzyskać płynne przejście w spektrum

# generowanie spektrum
# przejście od czarnego do niebieskiego
for i in range(0, 17):
    dummy[2] += step  # Zwiększ wartość niebieskiego (B)
    image = np.append(image, dummy)  # Dodaj nowy piksel do obrazu

# przejście od niebieskiego do zielonego
for i in range(0, 17):
    dummy[1] += step  # Zwiększ wartość zielonego (G)
    image = np.append(image, dummy)

# przejście od zielonego do cyjanowego
for i in range(0, 17):
    dummy[2] -= step  # Zmniejsz wartość niebieskiego (B)
    image = np.append(image, dummy)

# przejście od cyjanowego do czerwonego
for i in range(0, 17):
    dummy[0] += step  # Zwiększ wartość czerwonego (R)
    image = np.append(image, dummy)

# przejście od czerwonego do żółtego
for i in range(0, 17):
    dummy[1] -= step  # Zmniejsz wartość zielonego (G)
    image = np.append(image, dummy)

# przejście od żółtego do magenty
for i in range(0, 17):
    dummy[2] += step  # Zwiększ wartość niebieskiego (B)
    image = np.append(image, dummy)

# przejście od magenty do białego
for i in range(0, 17):
    dummy[1] += step  # Zwiększ wartość zielonego (G)
    image = np.append(image, dummy)

line = np.copy(image)  # Kopia jednej linii spektrum
image = np.tile(line, (8,))  # Powielamy linię spektrum tak, aby uzyskać obraz o wysokości=8

image_array = image.reshape(8, 120, 3)

# Zapis obrazu w formacie PPM (tekstowy - P3)
with open('transition_spectrum.ppm', 'w') as fh:
    fh.write(ppm_header)  # Zapis nagłówka pliku PPM
    for row in image_array:
        for pixel in row:
            fh.write(f"{pixel[0]} {pixel[1]} {pixel[2]} ")
        fh.write("\n")

print("Plik 'transition_spectrum.ppm' został zapisany.")

# Wyświetlenie obrazu za pomocą Matplotlib
plt.imshow(image_array.astype(np.uint8))
plt.title("Spektrum RGB")
plt.axis("off")
plt.show()