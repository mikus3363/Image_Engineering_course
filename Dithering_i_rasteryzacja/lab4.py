import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def lab4_zad1():
    def find_closest_palette_color(value):  # znajduje najbliższy kolor z palety (0 lub 255)
        return round(value / 255) * 255  # zaokrąglamy wartość do 0 lub 255

    def floyd_steinberg_dithering(image):

        result = image.copy().astype(np.float64)  # kopiujemy obraz i konwertujemy na float64
        height, width = result.shape  # pobieramy wymiary

        for y in range(height):  # iterujemy po wierszach
            for x in range(width):  # iterujemy po kolumnach

                old_value = result[y, x]  # pobieramy starą wartość piksela

                new_value = find_closest_palette_color(old_value)  # znajdujemy najbliższy kolor z palety

                result[y, x] = new_value  # ustawiamy nową wartość piksela

                error = old_value - new_value  # obliczamy błąd kwantyzacji

                if x < width - 1:  # jeśli nie na prawej krawędzi
                    result[y, x + 1] += error * 7 / 16  # rozprowadzamy błąd

                if y < height - 1:  # jeśli nie na dolnej krawędzi
                    if x > 0:  # jeśli nie na lewej krawędzi
                        result[y + 1, x - 1] += error * 3 / 16  # rozprowadzamy błąd na piksel lewo-dół

                    result[y + 1, x] += error * 5 / 16  # rozprowadzamy błąd na piksel pod spodem

                    if x < width - 1:  # jeśli nie na prawej krawędzi
                        result[y + 1, x + 1] += error * 1 / 16  # rozprowadzamy błąd na piksel prawo-dół

        np.clip(result, 0, 255, out=result)  # przycinamy wartości

        return result.astype(np.uint8)  # konwertujemy wynik na uint8

    image_path = resource_path('obrazek.jpg')
    image = np.array(Image.open(image_path).convert('L'))

    dithered_image = floyd_steinberg_dithering(image)  # wykonujemy dithering

    Image.fromarray(dithered_image).save('dithered_image.jpg')  # zapisujemy

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Oryginalny obraz (skala szarości)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(dithered_image, cmap='gray')
    plt.title("Po ditheringu")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def lab4_zad2():
    def find_closest_palette_color(value, k=2):
        return round((k - 1) * value / 255) * 255 / (k - 1)  # zaokrąglamy wartość do najbliższego poziomu palety

    def color_reduction(image, k_values):

        result = image.copy().astype(np.float64)

        for channel in range(3):  # redukcja kolorów dla każdego kanału RGB
            for y in range(result.shape[0]):
                for x in range(result.shape[1]):
                    result[y, x, channel] = find_closest_palette_color(result[y, x, channel], k_values[channel])

        np.clip(result, 0, 255, out=result)  # przycinamy wartości

        return result.astype(np.uint8)

    def floyd_steinberg_dithering(image, k_values):

        result = image.copy().astype(np.float64)

        height, width, channels = result.shape

        for channel in range(channels):  # dla każdego kanału RGB
            k = k_values[channel]

            for y in range(height):  # dla każdego piksela w obrazie
                for x in range(width):
                    old_value = result[y, x, channel]  # stara wartość piksela

                    new_value = find_closest_palette_color(old_value,
                                                           k)  # najbliższy kolor z palety dla danej składowej

                    result[y, x, channel] = new_value  # nowa wartość piksela

                    error = old_value - new_value  # oblicz błąd kwantyzacji

                    # rozprowadź błąd do sąsiednich pikseli
                    if x < width - 1:
                        result[y, x + 1, channel] += error * 7 / 16

                    if y < height - 1:
                        if x > 0:
                            result[y + 1, x - 1, channel] += error * 3 / 16

                        result[y + 1, x, channel] += error * 5 / 16

                        if x < width - 1:
                            result[y + 1, x + 1, channel] += error * 1 / 16

        np.clip(result, 0, 255, out=result)  # przycinamy wartości

        return result.astype(np.uint8)

    def create_color_histogram(image, title="Color Histogram"):

        colors = ("red", "green", "blue")  # kolory dla każdego kanału

        fig, ax = plt.subplots(figsize=(10, 6))  # wykres histogramu
        ax.set_xlim([0, 256])

        for channel_id, color in enumerate(colors):
            histogram, bin_edges = np.histogram(image[:, :, channel_id], bins=256,
                                                range=(0, 256))  # obliczamy histogram dla danego kanału
            ax.plot(bin_edges[0:-1], histogram, color=color)  # rysujemy linię histogramu w odpowiednim kolorze

        ax.set_title(title)
        ax.set_xlabel("Wartość koloru")
        ax.set_ylabel("Liczba pikseli")

        return fig

    image_path = resource_path('obrazek.jpg')
    image = np.array(Image.open(image_path))

    k_values = (9, 9, 9)

    reduced_image = color_reduction(image, k_values)  # redukcja palety kolorów bez dithering

    dithered_image = floyd_steinberg_dithering(image, k_values)  # z dithering

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Oryginalny obraz')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(reduced_image)
    plt.title('Redukcja kolorów (bez dithering)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(dithered_image)
    plt.title('Floyd-Steinberg Dithering')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    create_color_histogram(image, "Histogram oryginalnego obrazu")
    plt.show()
    create_color_histogram(reduced_image, "Histogram obrazu po redukcji kolorów")
    plt.show()
    create_color_histogram(dithered_image, "Histogram obrazu po dithering")
    plt.show()


def lab4_zad3():
    def draw_point(x, y, color, image):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # sprawdź, czy punkt jest w granicach obrazu
            image[y, x] = color

    def draw_line(x0, y0, x1, y1, draw_point, color, image):  # linia między dwoma punktami

        steep = abs(y1 - y0) > abs(x1 - x0)  # sprawdź, czy linia jest stroma
        if steep:  # jeśli linia jest stroma, zamieniamy x z y
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:  # jeśli początek jest dalej niż koniec, zamieniamy miejscami
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = x1 - x0  # różnica współrzędnych x
        dy = abs(y1 - y0)  # bezwzględna różnica współrzędnych y
        error = dx / 2  # początkowy błąd
        y = int(y0)  # bieżąca współrzędna y
        y_step = 1 if y0 < y1 else -1  # kierunek zmiany y

        for x in range(int(x0), int(x1) + 1):  # iterujemy po x od początku do końca

            if steep:  # jeśli linia była stroma, zamieniamy x z y przy rysowaniu punktu
                draw_point(y, x, color, image)
            else:
                draw_point(x, y, color, image)

            error -= dy  # aktualizujemy błąd

            if error < 0:  # jeśli błąd jest ujemny, przesuwamy y
                y += y_step
                error += dx

    def is_point_in_triangle(px, py, x0, y0, x1, y1, x2, y2):  # sprawdza czy punkt jest w trójkącie

        # iloczyny wektorowe
        d1 = (px - x1) * (y0 - y1) - (x0 - x1) * (py - y1)
        d2 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
        d3 = (px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)  # czy iloczyn jest ujemny
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)  # czy iloczyn jest dodatni

        return not (has_neg and has_pos)  # punkt jest w trójkącie jeśli nie ma zarówno dodatnich jak i ujemnych

    def draw_triangle(x0, y0, x1, y1, x2, y2, draw_point, color, image):  # wypełniony trójkąt

        xmin = int(min(x0, x1, x2))  # min wartość x wierzchołków
        xmax = int(max(x0, x1, x2))  # max wartość x wierzchołków

        ymin = int(min(y0, y1, y2))  # min wartość y wierzchołków
        ymax = int(max(y0, y1, y2))  # max wartość y wierzchołków

        for x in range(xmin, xmax + 1):  # iterujemy po wszystkich x w zakresie
            for y in range(ymin, ymax + 1):  # iterujemy po wszystkich y w zakresie

                if is_point_in_triangle(x, y, x0, y0, x1, y1, x2, y2):  # sprawdzamy czy punkt leży w trójkącie
                    draw_point(x, y, color, image)  # rysujemy punkt

    image = np.zeros((200, 200, 3), dtype=np.uint8)  # pusty obraz RGB 200x200, czarne tło

    # kolory RGB
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)

    # kilka linii
    draw_line(10, 10, 190, 10, draw_point, red, image)
    draw_line(10, 10, 10, 190, draw_point, green, image)
    draw_line(10, 190, 190, 190, draw_point, blue, image)
    draw_line(190, 10, 190, 190, draw_point, yellow, image)
    draw_line(10, 10, 190, 190, draw_point, red, image)
    draw_line(190, 10, 10, 190, draw_point, blue, image)

    # wypełniony trójkąt
    draw_triangle(50, 50, 150, 70, 100, 180, draw_point, (0, 255, 255), image)

    img = Image.fromarray(image)
    img.save("wynik.png")
    print("Obraz zapisany jako wynik.png")

    plt.imshow(image)
    plt.title("Rasteryzacja linii i trójkąta")
    plt.axis('off')
    plt.show()


def lab4_zad4():
    def draw_point(x, y, color, image):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # sprawdzamy, czy punkt jest w granicach obrazu
            image[y, x] = np.clip(color, 0, 255)  # ustawiamy kolor piksela, przycinając wartość

    def interpolate_color(c0, c1, t):
        return (1 - t) * np.array(c0) + t * np.array(c1)  # zwraca interpolowany kolor jako tablicę numpy

    def draw_line(x0, y0, x1, y1, c0, c1, draw_point, image):
        steep = abs(y1 - y0) > abs(x1 - x0)  # sprawdzamy, czy linia jest stroma

        if steep:  # jeśli tak, zamieniamy x z y
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:  # jeśli początek jest dalej niż koniec, zamieniamy miejscami
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            c0, c1 = c1, c0

        dx = x1 - x0  # różnica współrzędnych x
        dy = abs(y1 - y0)  # bezwzględna różnica współrzędnych y
        error = dx / 2  # początkowy błąd
        y = int(y0)  # bieżąca współrzędna y
        y_step = 1 if y0 < y1 else -1  # kierunek zmiany y
        length = int(x1 - x0) if dx != 0 else 1  # długość linii

        for i, x in enumerate(range(int(x0), int(x1) + 1)):  # iterujemy po x od początku do końca
            t = i / length
            color = interpolate_color(c0, c1, t)  # interpolujemy kolor

            if steep:  # jeśli linia była stroma, zamieniamy x z y przy rysowaniu punktu
                draw_point(y, x, color, image)
            else:
                draw_point(x, y, color, image)

            error -= dy  # aktualizujemy błąd

            if error < 0:  # jeśli błąd jest ujemny, przesuwamy y
                y += y_step
                error += dx

    def barycentric_coords(px, py, x0, y0, x1, y1, x2, y2):

        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)  # wyznacznik macierzy

        if det == 0:  # jeśli wyznacznik jest zerowy, trójkąt jest zdegenerowany
            return -1, -1, -1  # zwracamy wartości ujemne

        l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / det  # współrzędna l0
        l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / det  # współrzędna l1
        l2 = 1 - l0 - l1  # współrzędna l2

        return l0, l1, l2  # zwracamy współrzędne barycentryczne

    def draw_triangle(x0, y0, x1, y1, x2, y2, c0, c1, c2, draw_point, image):

        xmin = int(min(x0, x1, x2))  # min wartość x wierzchołków
        xmax = int(max(x0, x1, x2))  # max wartość x wierzchołków
        ymin = int(min(y0, y1, y2))  # min wartość y wierzchołków
        ymax = int(max(y0, y1, y2))  # max wartość y wierzchołków

        for x in range(xmin, xmax + 1):  # iterujemy po wszystkich x w zakresie
            for y in range(ymin, ymax + 1):  # iterujemy po wszystkich y w zakresie

                l0, l1, l2 = barycentric_coords(x, y, x0, y0, x1, y1, x2, y2)  # obliczamy współrzędne

                if min(l0, l1, l2) >= 0:  # jeśli wszystkie współrzędne są nieujemne, punkt leży w trójkącie
                    color = l0 * np.array(c0) + l1 * np.array(c1) + l2 * np.array(
                        c2)  # interpolujemy kolor z wierzchołków
                    draw_point(x, y, color, image)  # rysujemy piksel

    image = np.zeros((200, 200, 3), dtype=np.uint8)  # tworzymy pusty obraz RGB 200x200, czarne tło

    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)

    # linie z interpolacją kolorów
    draw_line(10, 10, 190, 10, red, blue, draw_point,
              image)  # linia pozioma u góry, kolor od czerwonego do niebieskiego
    draw_line(10, 10, 10, 190, green, yellow, draw_point,
              image)  # linia pionowa po lewej, kolor od zielonego do żółtego
    draw_line(10, 190, 190, 190, blue, red, draw_point,
              image)  # linia pozioma na dole, kolor od niebieskiego do czerwonego
    draw_line(190, 10, 190, 190, yellow, green, draw_point,
              image)  # linia pionowa po prawej, kolor od żółtego do zielonego
    draw_line(10, 10, 190, 190, red, green, draw_point,
              image)  # linia ukośna z lewego górnego do prawego dolnego, kolor od czerwonego do zielonego
    draw_line(190, 10, 10, 190, blue, magenta, draw_point,
              image)  # linia ukośna z prawego górnego do lewego dolnego, kolor od niebieskiego do magenty

    # trójkąt z interpolacją kolorów wierzchołków
    draw_triangle(50, 50, 150, 70, 100, 180, red, green, blue, draw_point, image)  # trójkąt z kolorami w wierzchołkach

    img = Image.fromarray(image)
    img.save("wynik_interpolacja.png")  # zapis
    print("Obraz zapisany jako wynik_interpolacja.png")

    plt.imshow(image)
    plt.title("Interpolacja kolorów")
    plt.axis('off')
    plt.show()

def lab4_zad5():
    def draw_point(x, y, color, image):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            image[y, x] = np.clip(color, 0, 255)  # ustawiamy kolor piksela

    def interpolate_color(c0, c1, t):
        return (1 - t) * np.array(c0) + t * np.array(c1)  # zwraca interpolowany kolor jako tablicę numpy

    def draw_line(x0, y0, x1, y1, c0, c1, draw_point, image):
        steep = abs(y1 - y0) > abs(x1 - x0)  # sprawdzamy, czy linia jest stroma

        if steep:  # jeśli tak, zamieniamy x z y
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:  # jeśli początek jest dalej niż koniec, zamieniamy miejscami
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            c0, c1 = c1, c0

        dx = x1 - x0  # różnica współrzędnych x
        dy = abs(y1 - y0)  # bezwzględna różnica współrzędnych y
        error = dx / 2  # początkowy błąd
        y = int(y0)  # bieżąca współrzędna y
        y_step = 1 if y0 < y1 else -1  # kierunek zmiany y
        length = int(x1 - x0) if dx != 0 else 1  # długość linii, zabezpieczenie przed dzieleniem przez zero

        for i, x in enumerate(range(int(x0), int(x1) + 1)):  # iterujemy od początku do końca
            t = i / length
            color = interpolate_color(c0, c1, t)  # interpolujemy kolor

            if steep:  # jeśli linia była stroma, zamieniamy x z y przy rysowaniu punktu
                draw_point(y, x, color, image)
            else:
                draw_point(x, y, color, image)

            error -= dy  # aktualizujemy błąd

            if error < 0:  # jeśli błąd jest ujemny, przesuwamy y
                y += y_step
                error += dx

    def barycentric_coords(px, py, x0, y0, x1, y1, x2, y2):  # funkcja oblicza współrzędne punktu względem trójkąta
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)  # wyznacznik macierzy

        if det == 0:  # jeśli wyznacznik jest zerowy, trójkąt jest zdegenerowany
            return -1, -1, -1  # zwracamy wartości ujemne

        l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / det  # współrzędna l0
        l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / det  # współrzędna l1
        l2 = 1 - l0 - l1  # współrzędna l2

        return l0, l1, l2  # zwracamy współrzędne barycentryczne

    def draw_triangle(x0, y0, x1, y1, x2, y2, c0, c1, c2, draw_point,
                      image):  # funkcja rysuje trójkąt z interpolacją kolorów\

        xmin = int(min(x0, x1, x2))  # min wartość x wierzchołków
        xmax = int(max(x0, x1, x2))  # max wartość x wierzchołków
        ymin = int(min(y0, y1, y2))  # min wartość y wierzchołków
        ymax = int(max(y0, y1, y2))  # max wartość y wierzchołków

        for x in range(xmin, xmax + 1):  # iterujemy po wszystkich x w zakresie
            for y in range(ymin, ymax + 1):  # iterujemy po wszystkich y w zakresie

                l0, l1, l2 = barycentric_coords(x, y, x0, y0, x1, y1, x2, y2)  # obliczamy współrzędne

                if min(l0, l1, l2) >= 0:  # jeśli wszystkie współrzędne są nieujemne, punkt leży w trójkącie
                    color = l0 * np.array(c0) + l1 * np.array(c1) + l2 * np.array(
                        c2)  # interpolujemy kolor z wierzchołków
                    draw_point(x, y, color, image)  # rysujemy piksel

    def downsample(image, scale):

        h, w, c = image.shape  # pobieramy wymiary
        new_h, new_w = h // scale, w // scale  # nowe wymiary po skalowaniu
        small = np.zeros((new_h, new_w, c), dtype=np.uint8)  # tworzymy pustą tablicę na zmniejszony obraz

        for y in range(new_h):
            for x in range(new_w):
                block = image[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale]  # wybieramy blok scale x scale
                avg = block.mean(axis=(0, 1))  # obliczamy średnią wartość koloru w bloku
                small[y, x] = np.clip(avg, 0, 255)  # ustawiamy piksel w zmniejszonym obrazie

        return small  # zwracamy zmniejszony obraz

    scale = 2  # SSAA x2
    width, height = 200, 200  # wymiary końcowego obrazu
    big_image = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)  # tworzymy duży obraz do rysowania

    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)

    # linie z interpolacją kolorów (współrzędne * scale)
    draw_line(10 * scale, 10 * scale, 190 * scale, 10 * scale, red, blue, draw_point, big_image)  # linia pozioma u góry
    draw_line(10 * scale, 10 * scale, 10 * scale, 190 * scale, green, yellow, draw_point,
              big_image)  # linia pionowa po lewej
    draw_line(10 * scale, 190 * scale, 190 * scale, 190 * scale, blue, red, draw_point,
              big_image)  # linia pozioma na dole
    draw_line(190 * scale, 10 * scale, 190 * scale, 190 * scale, yellow, green, draw_point,
              big_image)  # linia pionowa po prawej
    draw_line(10 * scale, 10 * scale, 190 * scale, 190 * scale, red, green, draw_point,
              big_image)  # linia ukośna z lewego górnego do prawego dolnego
    draw_line(190 * scale, 10 * scale, 10 * scale, 190 * scale, blue, magenta, draw_point,
              big_image)  # linia ukośna z prawego górnego do lewego dolnego

    # trójkąt z interpolacją kolorów wierzchołków
    draw_triangle(50 * scale, 50 * scale, 150 * scale, 70 * scale, 100 * scale, 180 * scale, red, green, blue,
                  draw_point, big_image)  # rysujemy trójkąt z interpolacją kolorów

    # downsampling przez uśrednianie bloków 2x2
    image = downsample(big_image, scale)  # zmniejszamy obraz do docelowego rozmiaru

    img = Image.fromarray(image)  # konwertujemy tablicę numpy na obraz PIL
    img.save("wynik_ssaa.png")  # zapis
    print("Obraz zapisany jako wynik_ssaa.png")

    plt.imshow(image)
    plt.title("Antyaliasing SSAA")
    plt.axis('off')
    plt.show()

# Aktualizacja menu
def menu():
    while True:
        print("\n=== LABORATORIUM 4 ===")
        print("1. Zadanie 1 - Dithering (skala szarości)")
        print("2. Zadanie 2 - Dithering RGB + histogramy")
        print("3. Zadanie 3 - Rasteryzacja podstawowych kształtów")
        print("4. Zadanie 4 - Interpolacja kolorów")
        print("5. Zadanie 5 - SSAA")
        print("6. Wyjście")

        choice = input("Wybierz opcję (1-6): ")

        if choice == "1":
            lab4_zad1()
        elif choice == "2":
            lab4_zad2()
        elif choice == "3":
            lab4_zad3()
        elif choice == "4":
            lab4_zad4()
        elif choice == "5":
            lab4_zad5()
        elif choice == "6":
            print("Wyjście z programu.")
            break
        else:
            print("Nieprawidłowa opcja. Spróbuj ponownie.")

if __name__ == "__main__":
    menu()