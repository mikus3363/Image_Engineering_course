import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie i konwersja obrazu
image_path = "../lab3/obrazek.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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


# Subsampling
H, W = ycrb.shape[:2]  # pobieranie wysokości i szerokości obrazu
Y = ycrb[..., 0].ravel().tolist()  # płaska lista składowej Y

Cb = []  # lista do przechowywania składowej Cb
Cr = []  # lista do przechowywania składowej Cr

for x in range(0, H, 2):  # pętla po wierszach z krokiem 2
    for y in range(0, W, 2):  # pętla po kolumnach z krokiem 2
        Cr.append(ycrb[x, y, 1])  # dodawanie składowej Cr do listy
        Cb.append(ycrb[x, y, 2])  # dodawanie składowej Cb do listy

subsampled_H = (H + 1) // 2  # obliczanie subsamplowanej wysokości
subsampled_W = (W + 1) // 2  # obliczanie subsamplowanej szerokości

# Upsampling
CbUp = []  # lista do przechowywania upsampled składowej Cb
CrUp = []  # lista do przechowywania upsampled składowej Cr

for x in range(H):  # pętla po wierszach
    for y in range(W):  # pętla po kolumnach
        idx = (x//2) * subsampled_W + (y//2)  # obliczanie indeksu
        CbUp.append(Cb[idx])  # dodawanie upsampled składowej Cb do listy
        CrUp.append(Cr[idx])  # dodawanie upsampled składowej Cr do listy

# Rekonstrukcja obrazu
tab = np.stack([Y, CrUp, CbUp], axis=1).reshape(ycrb.shape).astype(np.uint8)  # łączenie składowych i rekonstrukcja obrazu
newImage = cv2.cvtColor(tab, cv2.COLOR_YCrCb2RGB)  # konwersja z YCrCb do RGB


# Przygotowanie danych do wizualizacji
Y_img = np.array(Y).reshape(H, W)  # tworzenie obrazu składowej Y
Cb_subsampled = np.array(Cb).reshape(subsampled_H, subsampled_W)  # tworzenie subsamplowanego obrazu Cb
Cr_subsampled = np.array(Cr).reshape(subsampled_H, subsampled_W)  # tworzenie subsamplowanego obrazu Cr


# wyświetlenie wyników
plt.figure(figsize=(15, 10))  # utworzenie figury o rozmiarze 15x10 cali

plt.subplot(2, 3, 1)  # wyświetlenie oryginalnego obrazu
plt.title("Oryginalny")  # tytuł wykresu
plt.imshow(image)  # wyświetlenie obrazu
plt.axis('off')  # ukrycie osi

plt.subplot(2, 3, 2)  # wyświetlenie składowej Y w odcieniach szarości
plt.title("Składowa Y")  # tytuł wykresu
plt.imshow(Y_img,cmap = "Greys_r")  # wyświetlenie obrazu w skali szarości
plt.axis('off')  # ukrycie osi

plt.subplot(2, 3, 3)  # wyświetlenie składowej Cr w odcieniach szarości
plt.title("Cr po subsamplingu")  # tytuł wykresu
plt.imshow(Cr_subsampled,cmap = "Greys_r")  # wyświetlenie obrazu w skali szarości
plt.axis('off')  # ukrycie osi

plt.subplot(2, 3, 4)  # wyświetlenie składowej Cb w odcieniach szarości
plt.title("Cb po subsamplingu")  # tytuł wykresu
plt.imshow(Cb_subsampled,cmap = "Greys_r")  # wyświetlenie obrazu w skali szarości
plt.axis('off')  # ukrycie osi

plt.subplot(2, 3, 5)  # wyświetlenie obrazu po konwersji odwrotnej
plt.title("Cr po upsamplingu")  # tytuł wykresu
plt.imshow(tab[...,1])  # wyświetlenie obrazu
plt.axis('off')  # ukrycie osi

plt.subplot(2, 3, 6)  # wyświetlenie obrazu po konwersji odwrotnej
plt.title("Obraz zrekonstruowany")  # tytuł wykresu
plt.imshow(newImage)  # wyświetlenie obrazu
plt.axis('off')  # ukrycie osi

plt.tight_layout()  # automatyczne dopasowanie układu wykresów
plt.show()  # wyświetlenie wszystkich wykresów