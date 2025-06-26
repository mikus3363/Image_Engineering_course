import cv2  # importowanie biblioteki opencv do pracy z obrazami
import numpy as np  # importowanie numpy do operacji na tablicach
from matplotlib import pyplot as plt  # importowanie pyplot z matplotlib do wizualizacji

image_path = "../lab3/obrazek.jpg"  # ścieżka do pliku z obrazem
image = cv2.imread(image_path)  # wczytanie obrazu z podanej ścieżki
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # konwersja kolorów z formatu BGR (domyślny w opencv) na RGB

float_image = np.asarray(image.reshape(-1,3))  # przekształcenie obrazu do tablicy 2D gdzie każdy wiersz to piksel [R,G,B]
float_image = float_image.astype(np.uint8)  # konwersja typu danych na uint8 (bez znaku 8-bitowy)

convert = [  # macierz współczynników do konwersji RGB -> YCrCb
    [0.229, 0.587, 0.114],  # współczynniki dla składowej Y
    [0.500, -0.418, -0.082],  # współczynniki dla składowej Cr
    [-0.168, -0.331, 0.500],  # współczynniki dla składowej Cb
]
convert = np.asarray(convert)  # zamiana listy na tablicę numpy

add = [  # wektor przesunięcia wartości w przestrzeni YCrCb
    [0],  # brak przesunięcia dla Y
    [128],  # przesunięcie dla Cr
    [128],  # przesunięcie dla Cb
]
add = np.asarray(add)  # zamiana listy na tablicę numpy

ycrb = []  # pusta lista na wyniki konwersji

for row in float_image:  # iteracja po każdym pikselu w obrazie
    column = np.asarray(row.reshape(3,1))  # przekształcenie piksela do postaci kolumnowej 3x1
    sums = np.matmul(convert,column)  # wykonanie mnożenia macierzy konwersji przez wektor piksela
    sums += add  # dodanie wektora przesunięcia do wyniku
    sums = sums.reshape(1,3)  # spłaszczenie wyniku do postaci [Y, Cr, Cb]
    ycrb.append(sums)  # dodanie przetworzonego piksela do listy

np.clip(ycrb, 0, 255)  # przycięcie wartości do zakresu 0-255 (zabezpieczenie przed przepełnieniem)
ycrb = np.asarray(ycrb).reshape(image.shape)  # przekształcenie listy z powrotem w kształt oryginalnego obrazu
ycrb = ycrb.astype(np.uint8)  # konwersja typu danych na uint8



ycrb_flat = ycrb.reshape(-1, 3)  # spłaszczenie obrazu YCrCb do tablicy 2D pikseli
convert_inv = np.linalg.inv(convert)  # obliczenie macierzy odwrotnej do macierzy konwersji

rgb_inv = []  # lista na wyniki konwersji odwrotnej
for row in ycrb_flat:  # iteracja po każdym pikselu w przestrzeni YCrCb
    pixel = row.reshape(3, 1) - add.reshape(3, 1)  # odjęcie przesunięcia i przekształcenie do postaci kolumnowej
    inv_pixel = np.matmul(convert_inv, pixel)  # wykonanie mnożenia przez macierz odwrotną (YCrCb -> RGB)
    rgb_inv.append(inv_pixel.ravel())  # dodanie spłaszczonego wyniku do listy

rgb_inv = np.array(rgb_inv, dtype=np.float32)  # konwersja listy na tablicę numpy i obcięcie wartości do zakresu 0-255
rgb_inv = np.clip(rgb_inv, 0, 255).astype(np.uint8)  # ograniczenie wartości do zakresu 0-255 i konwersja na uint8
rgb_inv_image = rgb_inv.reshape(image.shape)  # przekształcenie wyniku do kształtu oryginalnego obrazu


# wyświetlenie wyników
plt.figure(figsize=(15, 10))  # utworzenie figury o rozmiarze 15x10 cali

plt.subplot(2, 3, 1)  # wyświetlenie oryginalnego obrazu
plt.title("Oryginalny")  # tytuł wykresu
plt.imshow(image)  # wyświetlenie obrazu
plt.axis('off')  # ukrycie osi

plt.subplot(2, 3, 2)  # wyświetlenie składowej Y w odcieniach szarości
plt.title("Y")  # tytuł wykresu
plt.imshow(ycrb[:,:,0],cmap = "Greys_r")  # wyświetlenie obrazu w skali szarości
plt.axis('off')  # ukrycie osi

plt.subplot(2, 3, 3)  # wyświetlenie składowej Cr w odcieniach szarości
plt.title("Cr")  # tytuł wykresu
plt.imshow(ycrb[:,:,2],cmap = "Greys_r")  # wyświetlenie obrazu w skali szarości
plt.axis('off')  # ukrycie osi

plt.subplot(2, 3, 4)  # wyświetlenie składowej Cb w odcieniach szarości
plt.title("Cb")  # tytuł wykresu
plt.imshow(ycrb[:,:,1],cmap = "Greys_r")  # wyświetlenie obrazu w skali szarości
plt.axis('off')  # ukrycie osi

plt.subplot(2, 3, 5)  # wyświetlenie obrazu po konwersji odwrotnej
plt.title("Konwersja odwrotna")  # tytuł wykresu
plt.imshow(rgb_inv_image)  # wyświetlenie obrazu
plt.axis('off')  # ukrycie osi

plt.tight_layout()  # automatyczne dopasowanie układu wykresów
plt.show()  # wyświetlenie wszystkich wykresów