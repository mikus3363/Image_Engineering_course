import cv2  # import biblioteki opencv do przetwarzania obrazów
import numpy as np  # import biblioteki numpy do operacji na macierzach
from matplotlib import pyplot as plt  # import modułu pyplot do wizualizacji

# wczytanie obrazu
image_path = "../lab3/obrazek.jpg"  # ścieżka do pliku z obrazem
image = cv2.imread(image_path)  # wczytanie obrazu w formacie bgr (domyślny)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # konwersja kolorów z bgr na rgb

float_image = np.float32(image.reshape(-1,3)) / 255.0  # spłaszczenie obrazu do listy pikseli i normalizacja do zakresu [0.0, 1.0]

transform = [  # macierz transformacji kolorów
    [0.393, 0.769, 0.189],
    [0.349, 0.689, 0.164],
    [0.272, 0.534, 0.131],
]

transform = np.asarray(transform)  # konwersja listy na tablicę numpy
transformed_image = []  # inicjalizacja listy na przekształcone piksele

for row in float_image:  # iteracja po każdym pikselu (w formacie [r, g, b])
    column = np.asarray(row.reshape(3,1))  # przekształcenie piksela w kolumnę (3x1) np. R1, G1, B1
    sums = np.matmul(transform, column)  # mnożenie macierzy transformacji przez piksel np. 0.393 * R1 + 0.769 * G1+ 0.189 * B1
    sums = sums.reshape(1,3)  # zmiana kształtu wyniku na (1x3) np. R1', G1', B1'
    transformed_image.append(sums)  # dodanie przekształconego piksela do listy

transformed_image = np.clip(transformed_image, 0.0, 1.0)  # obcięcie wartości do zakresu [0.0, 1.0]

transformed_image = np.asarray(transformed_image).reshape(image.shape)  # przekształcenie listy z powrotem w kształt obrazu

# wyświetlenie wyników
plt.figure(figsize=(10, 5))  # utworzenie figury o rozmiarze 10x5 cali
plt.subplot(1, 2, 1)  # utworzenie lewego panela na oryginalny obraz
plt.title("oryginalny obraz")  # tytuł dla oryginalnego obrazu
plt.imshow(image)  # wyświetlenie oryginalnego obrazu
plt.axis("off")  # ukrycie osi

plt.subplot(1, 2, 2)  # utworzenie prawego panela na obraz po transformacji
plt.title("po transformacji")  # tytuł dla przekształconego obrazu
plt.imshow(transformed_image)  # wyświetlenie przekształconego obrazu
plt.axis("off")  # ukrycie osi

plt.show()  # renderowanie i pokazanie całej figury