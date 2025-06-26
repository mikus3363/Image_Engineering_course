import numpy as np
from PIL import Image

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

def main():  # główna funkcja programu
    image = np.array(Image.open('obrazek.jpg').convert('L'))  # otwieramy obraz, konwertujemy do odcieni szarości i zamieniamy na tablicę numpy

    dithered_image = floyd_steinberg_dithering(image)  # wykonujemy dithering

    Image.fromarray(dithered_image).save('dithered_image.jpg')  # zapisujemy

if __name__ == "__main__":
    main()
