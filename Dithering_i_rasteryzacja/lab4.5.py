import numpy as np
from PIL import Image

def draw_point(x, y, color, image):
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        image[y, x] = np.clip(color, 0, 255)  # ustawiamy kolor piksela


def interpolate_color(c0, c1, t):
    return (1-t) * np.array(c0) + t * np.array(c1)  # zwraca interpolowany kolor jako tablicę numpy


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
    det = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)  # wyznacznik macierzy

    if det == 0:  # jeśli wyznacznik jest zerowy, trójkąt jest zdegenerowany
        return -1, -1, -1  # zwracamy wartości ujemne

    l0 = ((y1 - y2)*(px - x2) + (x2 - x1)*(py - y2)) / det  # współrzędna l0
    l1 = ((y2 - y0)*(px - x2) + (x0 - x2)*(py - y2)) / det  # współrzędna l1
    l2 = 1 - l0 - l1  # współrzędna l2

    return l0, l1, l2  # zwracamy współrzędne barycentryczne


def draw_triangle(x0, y0, x1, y1, x2, y2, c0, c1, c2, draw_point, image):  # funkcja rysuje trójkąt z interpolacją kolorów

    xmin = int(min(x0, x1, x2))  # min wartość x wierzchołków
    xmax = int(max(x0, x1, x2))  # max wartość x wierzchołków
    ymin = int(min(y0, y1, y2))  # min wartość y wierzchołków
    ymax = int(max(y0, y1, y2))  # max wartość y wierzchołków

    for x in range(xmin, xmax + 1):  # iterujemy po wszystkich x w zakresie
        for y in range(ymin, ymax + 1):  # iterujemy po wszystkich y w zakresie

            l0, l1, l2 = barycentric_coords(x, y, x0, y0, x1, y1, x2, y2)  # obliczamy współrzędne

            if min(l0, l1, l2) >= 0:  # jeśli wszystkie współrzędne są nieujemne, punkt leży w trójkącie
                color = l0 * np.array(c0) + l1 * np.array(c1) + l2 * np.array(c2)  # interpolujemy kolor z wierzchołków
                draw_point(x, y, color, image)  # rysujemy piksel


def downsample(image, scale):

    h, w, c = image.shape  # pobieramy wymiary
    new_h, new_w = h // scale, w // scale  # nowe wymiary po skalowaniu
    small = np.zeros((new_h, new_w, c), dtype=np.uint8)  # tworzymy pustą tablicę na zmniejszony obraz

    for y in range(new_h):
        for x in range(new_w):

            block = image[y*scale:(y+1)*scale, x*scale:(x+1)*scale]  # wybieramy blok scale x scale
            avg = block.mean(axis=(0,1))  # obliczamy średnią wartość koloru w bloku
            small[y, x] = np.clip(avg, 0, 255)  # ustawiamy piksel w zmniejszonym obrazie

    return small  # zwracamy zmniejszony obraz


def main():
    scale = 2  # SSAA x2
    width, height = 200, 200  # wymiary końcowego obrazu
    big_image = np.zeros((height*scale, width*scale, 3), dtype=np.uint8)  # tworzymy duży obraz do rysowania

    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)

    # linie z interpolacją kolorów (współrzędne * scale)
    draw_line(10*scale, 10*scale, 190*scale, 10*scale, red, blue, draw_point, big_image)  # linia pozioma u góry
    draw_line(10*scale, 10*scale, 10*scale, 190*scale, green, yellow, draw_point, big_image)  # linia pionowa po lewej
    draw_line(10*scale, 190*scale, 190*scale, 190*scale, blue, red, draw_point, big_image)  # linia pozioma na dole
    draw_line(190*scale, 10*scale, 190*scale, 190*scale, yellow, green, draw_point, big_image)  # linia pionowa po prawej
    draw_line(10*scale, 10*scale, 190*scale, 190*scale, red, green, draw_point, big_image)  # linia ukośna z lewego górnego do prawego dolnego
    draw_line(190*scale, 10*scale, 10*scale, 190*scale, blue, magenta, draw_point, big_image)  # linia ukośna z prawego górnego do lewego dolnego

    # trójkąt z interpolacją kolorów wierzchołków
    draw_triangle(50*scale, 50*scale, 150*scale, 70*scale, 100*scale, 180*scale,red, green, blue, draw_point, big_image)  # rysujemy trójkąt z interpolacją kolorów

    # downsampling przez uśrednianie bloków 2x2
    image = downsample(big_image, scale)  # zmniejszamy obraz do docelowego rozmiaru

    img = Image.fromarray(image)  # konwertujemy tablicę numpy na obraz PIL
    img.save("wynik_ssaa.png")  # zapis
    print("Obraz zapisany jako wynik_ssaa.png")


if __name__ == "__main__":
    main()
