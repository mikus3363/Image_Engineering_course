import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

# Funkcja do pobierania ścieżki do zasobów
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(".."), relative_path)

def zad1_1():
    # Wczytanie obrazu
    image_path = resource_path("../lab3/obrazek.jpg")
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Maski Górnoprzepustowa
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    kernel = np.asarray(kernel)
    # Filtracja
    filtered_image = cv2.filter2D(image, -1, kernel)  # przekształcenie obrazu za pomocą maski

    # Wyświetlenie obrazu
    plt.subplot(1, 2, 1)  # tworzenie pierwszej sekcji na obraz
    plt.title("Oryginalny obraz")  # tytuł wykresu
    plt.imshow(image)  # wyświetlenie obrazu przed filtracją
    plt.axis("off")  # ukrycie osi

    plt.subplot(1, 2, 2)  # tworzenie drugiej sekcji na obraz
    plt.title("Filtr górnoprzepustowy")  # tytuł wykresu
    plt.imshow(filtered_image)  # wyświetlenie obrazu po filtracji
    plt.axis("off")  # ukrycie osi

    plt.show()  # wyświetlenie całości

def zad1_2():
    # Wczytanie obrazu
    image_path = resource_path("../lab3/obrazek.jpg")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # konwersja kolorów z BGR na RGB

    float_image = np.float32(
        image.reshape(-1, 3)) / 255.0  # spłaszczenie obrazu do listy pikseli i normalizacja do zakresu [0.0, 1.0]

    transform = [  # macierz transformacji kolorów
        [0.393, 0.769, 0.189],
        [0.349, 0.689, 0.164],
        [0.272, 0.534, 0.131],
    ]

    transform = np.asarray(transform)  # konwersja listy na tablicę numpy
    transformed_image = []  # inicjalizacja listy na przekształcone piksele

    for row in float_image:  # iteracja po każdym pikselu (w formacie [r, g, b])
        column = np.asarray(row.reshape(3, 1))  # przekształcenie piksela w kolumnę (3x1) np. R1, G1, B1
        sums = np.matmul(transform, column)  # mnożenie macierzy transformacji przez piksel np. 0.393 * R1 + 0.769 * G1+ 0.189 * B1
        sums = sums.reshape(1, 3)  # zmiana kształtu wyniku na (1x3) np. R1', G1', B1'
        transformed_image.append(sums)  # dodanie przekształconego piksela do listy

    transformed_image = np.clip(transformed_image, 0.0, 1.0)  # obcięcie wartości do zakresu [0.0, 1.0]

    transformed_image = np.asarray(transformed_image).reshape(
        image.shape)  # przekształcenie listy z powrotem w kształt oryginalnego obrazu

    # Wyświetlenie wyników
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

def zad1_3():
    # Wczytanie obrazu
    image_path = resource_path("../lab3/obrazek.jpg")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # konwersja kolorów z BGR na RGB

    float_image = np.asarray(
        image.reshape(-1, 3))  # przekształcenie obrazu do tablicy 2D gdzie każdy wiersz to piksel [R,G,B]
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
        column = np.asarray(row.reshape(3, 1))  # przekształcenie piksela do postaci kolumnowej 3x1
        sums = np.matmul(convert, column)  # wykonanie mnożenia macierzy konwersji przez wektor piksela
        sums += add  # dodanie wektora przesunięcia do wyniku
        sums = sums.reshape(1, 3)  # spłaszczenie wyniku do postaci [Y, Cr, Cb]
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

    # Wyświetlenie wyników
    plt.figure(figsize=(15, 10))  # utworzenie figury o rozmiarze 15x10 cali

    plt.subplot(2, 3, 1)  # wyświetlenie oryginalnego obrazu
    plt.title("Oryginalny")  # tytuł wykresu
    plt.imshow(image)  # wyświetlenie obrazu
    plt.axis('off')  # ukrycie osi

    plt.subplot(2, 3, 2)  # wyświetlenie składowej Y w odcieniach szarości
    plt.title("Y")  # tytuł wykresu
    plt.imshow(ycrb[:, :, 0], cmap="Greys_r")  # wyświetlenie obrazu w skali szarości
    plt.axis('off')  # ukrycie osi

    plt.subplot(2, 3, 3)  # wyświetlenie składowej Cr w odcieniach szarości
    plt.title("Cr")  # tytuł wykresu
    plt.imshow(ycrb[:, :, 2], cmap="Greys_r")  # wyświetlenie obrazu w skali szarości
    plt.axis('off')  # ukrycie osi

    plt.subplot(2, 3, 4)  # wyświetlenie składowej Cb w odcieniach szarości
    plt.title("Cb")  # tytuł wykresu
    plt.imshow(ycrb[:, :, 1], cmap="Greys_r")  # wyświetlenie obrazu w skali szarości
    plt.axis('off')  # ukrycie osi

    plt.subplot(2, 3, 5)  # wyświetlenie obrazu po konwersji odwrotnej
    plt.title("Konwersja odwrotna")  # tytuł wykresu
    plt.imshow(rgb_inv_image)  # wyświetlenie obrazu
    plt.axis('off')  # ukrycie osi

    plt.tight_layout()  # automatyczne dopasowanie układu wykresów
    plt.show()  # wyświetlenie wszystkich wykresów


def zad1_4():
    # Wczytanie i konwersja obrazu
    image_path = resource_path("../lab3/obrazek.jpg")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    float_image = np.asarray(
        image.reshape(-1, 3))  # przekształcenie obrazu do tablicy 2D gdzie każdy wiersz to piksel [R,G,B]
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
        column = np.asarray(row.reshape(3, 1))  # przekształcenie piksela do postaci kolumnowej 3x1
        sums = np.matmul(convert, column)  # wykonanie mnożenia macierzy konwersji przez wektor piksela
        sums += add  # dodanie wektora przesunięcia do wyniku
        sums = sums.reshape(1, 3)  # spłaszczenie wyniku do postaci [Y, Cr, Cb]
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
            idx = (x // 2) * subsampled_W + (y // 2)  # obliczanie indeksu
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
    plt.imshow(Y_img, cmap="Greys_r")  # wyświetlenie obrazu w skali szarości
    plt.axis('off')  # ukrycie osi

    plt.subplot(2, 3, 3)  # wyświetlenie składowej Cr w odcieniach szarości
    plt.title("Cr po subsamplingu")  # tytuł wykresu
    plt.imshow(Cr_subsampled, cmap="Greys_r")  # wyświetlenie obrazu w skali szarości
    plt.axis('off')  # ukrycie osi

    plt.subplot(2, 3, 4)  # wyświetlenie składowej Cb w odcieniach szarości
    plt.title("Cb po subsamplingu")  # tytuł wykresu
    plt.imshow(Cb_subsampled, cmap="Greys_r")  # wyświetlenie obrazu w skali szarości
    plt.axis('off')  # ukrycie osi

    plt.subplot(2, 3, 5)  # wyświetlenie obrazu po konwersji odwrotnej
    plt.title("Cr po upsamplingu")  # tytuł wykresu
    plt.imshow(tab[..., 1])  # wyświetlenie obrazu
    plt.axis('off')  # ukrycie osi

    plt.subplot(2, 3, 6)  # wyświetlenie obrazu po konwersji odwrotnej
    plt.title("Obraz zrekonstruowany")  # tytuł wykresu
    plt.imshow(newImage)  # wyświetlenie obrazu
    plt.axis('off')  # ukrycie osi

    plt.tight_layout()  # automatyczne dopasowanie układu wykresów
    plt.show()  # wyświetlenie wszystkich wykresów

def zad1_5():
    # Wczytanie i konwersja obrazu
    image_path = resource_path("../lab3/obrazek.jpg")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    float_image = np.asarray(
        image.reshape(-1, 3))  # przekształcenie obrazu do tablicy 2D gdzie każdy wiersz to piksel [R,G,B]
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
        column = np.asarray(row.reshape(3, 1))  # przekształcenie piksela do postaci kolumnowej 3x1
        sums = np.matmul(convert, column)  # wykonanie mnożenia macierzy konwersji przez wektor piksela
        sums += add  # dodanie wektora przesunięcia do wyniku
        sums = sums.reshape(1, 3)  # spłaszczenie wyniku do postaci [Y, Cr, Cb]
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
            idx = (x // 2) * subsampled_W + (y // 2)  # obliczanie indeksu
            CbUp.append(Cb[idx])  # dodawanie upsampled składowej Cb do listy
            CrUp.append(Cr[idx])  # dodawanie upsampled składowej Cr do listy

    # Rekonstrukcja obrazu
    tab = np.stack([Y, CrUp, CbUp], axis=1).reshape(ycrb.shape).astype(
        np.uint8)  # łączenie składowych i rekonstrukcja obrazu
    newImage = cv2.cvtColor(tab, cv2.COLOR_YCrCb2RGB)  # konwersja z YCrCb do RGB

    # Przygotowanie danych do wizualizacji
    Y_img = np.array(Y).reshape(H, W)  # tworzenie obrazu składowej Y
    Cb_subsampled = np.array(Cb).reshape(subsampled_H, subsampled_W)  # tworzenie subsamplowanego obrazu Cb
    Cr_subsampled = np.array(Cr).reshape(subsampled_H, subsampled_W)  # tworzenie subsamplowanego obrazu Cr


    MSE_RGB = 0  # inicjacja zmiennej do przechowywania błędu średniokwadratowego dla kanałów RGB
    MSE_Y = 0  # inicjacja zmiennej do przechowywania błędu średniokwadratowego dla kanału Y
    MSE_Cr = 0  # inicjacja zmiennej do przechowywania błędu średniokwadratowego dla kanału Cr
    MSE_Cb = 0  # inicjacja zmiennej do przechowywania błędu średniokwadratowego dla kanału Cb

    for x in range(H):  # pętla po wierszach obrazu
        for y in range(W):  # pętla po kolumnach obrazu
            MSE_RGB += (float(image[x][y][0]) - float(newImage[x][y][0])) ** 2  # dodawanie do błędu kwadratu różnicy między kanałem R oryginalnego i zrekonstruowanego obrazu
            MSE_RGB += (float(image[x][y][1]) - float(newImage[x][y][1])) ** 2  # dodawanie do błędu kwadratu różnicy między kanałem G oryginalnego i zrekonstruowanego obrazu
            MSE_RGB += (float(image[x][y][2]) - float(newImage[x][y][2])) ** 2  # dodawanie do błędu kwadratu różnicy między kanałem B oryginalnego i zrekonstruowanego obrazu
            MSE_Y += (float(ycrb[x][y][0]) - float(tab[x][y][0])) ** 2  # dodawanie do błędu kwadratu różnicy między kanałem Y oryginalnego i zrekonstruowanego obrazu
            MSE_Cr += (float(ycrb[x][y][1]) - float(tab[x][y][1])) ** 2  # dodawanie do błędu kwadratu różnicy między kanałem Cr oryginalnego i zrekonstruowanego obrazu
            MSE_Cb += (float(ycrb[x][y][2]) - float(tab[x][y][2])) ** 2  # dodawanie do błędu kwadratu różnicy między kanałem Cb oryginalnego i zrekonstruowanego obrazu

    pixels = H * W  # obliczanie łącznej liczby pikseli w obrazie
    MSE_RGB /= pixels * 3  # normalizacja błędu średniokwadratowego dla kanałów RGB przez liczbę pikseli i kanałów
    MSE_Y /= pixels  # normalizacja błędu średniokwadratowego dla kanału Y przez liczbę pikseli
    MSE_Cr /= pixels  # normalizacja błędu średniokwadratowego dla kanału Cr przez liczbę pikseli
    MSE_Cb /= pixels  # normalizacja błędu średniokwadratowego dla kanału Cb przez liczbę pikseli

    print("Błąd średniokwadratowy między obrazami wynosi: " + str(MSE_RGB))  # wyświetlenie błędu średniokwadratowego dla kanałów RGB
    print("Błąd średniokwadratowy między kanałami Y wynosi: " + str(MSE_Y))  # wyświetlenie błędu średniokwadratowego dla kanału Y
    print("Błąd średniokwadratowy między kanałami Cb wynosi: " + str(MSE_Cb))  # wyświetlenie błędu średniokwadratowego dla kanału Cb
    print("Błąd średniokwadratowy między kanałami Cr wynosi: " + str(MSE_Cr))  # wyświetlenie błędu średniokwadratowego dla kanału Cr


    # wyświetlenie wyników
    plt.figure(figsize=(15, 10))  # utworzenie figury o rozmiarze 15x10 cali

    plt.subplot(2, 3, 1)  # wyświetlenie oryginalnego obrazu
    plt.title("Oryginalny")  # tytuł wykresu
    plt.imshow(image)  # wyświetlenie obrazu
    plt.axis('off')  # ukrycie osi

    plt.subplot(2, 3, 2)  # wyświetlenie składowej Y w odcieniach szarości
    plt.title("Składowa Y")  # tytuł wykresu
    plt.imshow(Y_img, cmap="Greys_r")  # wyświetlenie obrazu w skali szarości
    plt.axis('off')  # ukrycie osi

    plt.subplot(2, 3, 3)  # wyświetlenie składowej Cr w odcieniach szarości
    plt.title("Cr po subsamplingu")  # tytuł wykresu
    plt.imshow(Cr_subsampled, cmap="Greys_r")  # wyświetlenie obrazu w skali szarości
    plt.axis('off')  # ukrycie osi

    plt.subplot(2, 3, 4)  # wyświetlenie składowej Cb w odcieniach szarości
    plt.title("Cb po subsamplingu")  # tytuł wykresu
    plt.imshow(Cb_subsampled, cmap="Greys_r")  # wyświetlenie obrazu w skali szarości
    plt.axis('off')  # ukrycie osi

    plt.subplot(2, 3, 5)  # wyświetlenie obrazu po konwersji odwrotnej
    plt.title("Cr po upsamplingu")  # tytuł wykresu
    plt.imshow(tab[..., 1])  # wyświetlenie obrazu
    plt.axis('off')  # ukrycie osi

    plt.subplot(2, 3, 6)  # wyświetlenie obrazu po konwersji odwrotnej
    plt.title("Obraz zrekonstruowany")  # tytuł wykresu
    plt.imshow(newImage)  # wyświetlenie obrazu
    plt.axis('off')  # ukrycie osi

    plt.tight_layout()  # automatyczne dopasowanie układu wykresów
    plt.show()  # wyświetlenie wszystkich wykresów


def menu():
    print("\nAutor: Mikołaj Lipiński")
    while True:
        print("\nMenu:")
        print("1. Zadanie 1.1 - filtr gornoprzepustowy")
        print("2. Zadanie 1.2 - przeksztalcenie kolorow RGB")
        print("3. Zadanie 1.3 - konwersja RGB na YCbCr")
        print("4. Zadanie 1.4 - downsampling i upsampling")
        print("5. Zadanie 1.5 - blad sredniokwadratowy")
        print("6. Wyjście z programu")

        choice = input("Wybierz opcję (1-6): ")

        if choice == "1":
            print("Wybrano Zadanie 1")
            zad1_1()
        elif choice == "2":
            print("Wybrano Zadanie 2")
            zad1_2()
        elif choice == "3":
            print("Wybrano Zadanie 3")
            zad1_3()
        elif choice == "4":
            print("Wybrano Zadanie 4")
            zad1_4()
        elif choice == "5":
            print("Wybrano Zadanie 5")
            zad1_5()
        elif choice == "6":
            print("Wyjście z menu. Żegnaj!")
            break
        else:
            print("Nieprawidłowy wybór. Proszę spróbuj ponownie.")

menu()