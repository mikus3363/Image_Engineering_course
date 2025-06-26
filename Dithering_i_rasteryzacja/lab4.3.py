import numpy as np
from PIL import Image

def draw_point(x, y, color, image):
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]: # sprawdź, czy punkt jest w granicach obrazu
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
    d1 = (px - x1)*(y0 - y1) - (x0 - x1)*(py - y1)
    d2 = (px - x2)*(y1 - y2) - (x1 - x2)*(py - y2)
    d3 = (px - x0)*(y2 - y0) - (x2 - x0)*(py - y0)

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


def main():
    image = np.zeros((200, 200, 3), dtype=np.uint8) # pusty obraz RGB 200x200, czarne tło

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

if __name__ == "__main__":
    main()
