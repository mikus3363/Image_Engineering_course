import numpy as np
from PIL import Image

def draw_point(x, y, color, image):
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # sprawdzamy, czy punkt jest w granicach obrazu
        image[y, x] = np.clip(color, 0, 255)  # ustawiamy kolor piksela, przycinając wartość

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

    det = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)  # wyznacznik macierzy

    if det == 0:  # jeśli wyznacznik jest zerowy, trójkąt jest zdegenerowany
        return -1, -1, -1  # zwracamy wartości ujemne

    l0 = ((y1 - y2)*(px - x2) + (x2 - x1)*(py - y2)) / det  # współrzędna l0
    l1 = ((y2 - y0)*(px - x2) + (x0 - x2)*(py - y2)) / det  # współrzędna l1
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
                color = l0 * np.array(c0) + l1 * np.array(c1) + l2 * np.array(c2)  # interpolujemy kolor z wierzchołków
                draw_point(x, y, color, image)  # rysujemy piksel

def main():
    image = np.zeros((200, 200, 3), dtype=np.uint8)  # tworzymy pusty obraz RGB 200x200, czarne tło

    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)

    # linie z interpolacją kolorów
    draw_line(10, 10, 190, 10, red, blue, draw_point, image)  # linia pozioma u góry, kolor od czerwonego do niebieskiego
    draw_line(10, 10, 10, 190, green, yellow, draw_point, image)  # linia pionowa po lewej, kolor od zielonego do żółtego
    draw_line(10, 190, 190, 190, blue, red, draw_point, image)  # linia pozioma na dole, kolor od niebieskiego do czerwonego
    draw_line(190, 10, 190, 190, yellow, green, draw_point, image)  # linia pionowa po prawej, kolor od żółtego do zielonego
    draw_line(10, 10, 190, 190, red, green, draw_point, image)  # linia ukośna z lewego górnego do prawego dolnego, kolor od czerwonego do zielonego
    draw_line(190, 10, 10, 190, blue, magenta, draw_point, image)  # linia ukośna z prawego górnego do lewego dolnego, kolor od niebieskiego do magenty

    # trójkąt z interpolacją kolorów wierzchołków
    draw_triangle(50, 50, 150, 70, 100, 180, red, green, blue, draw_point, image)  # trójkąt z kolorami w wierzchołkach

    img = Image.fromarray(image)
    img.save("wynik_interpolacja.png")  # zapis
    print("Obraz zapisany jako wynik_interpolacja.png")

if __name__ == "__main__":
    main()
