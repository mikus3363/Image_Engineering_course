import numpy as np
import struct
import zlib
import cv2
from matplotlib import pyplot
from PIL import Image


def generate_rainbow_line():
    step = 15  # Krok zmiany koloru (255 / 17)
    image = np.array([0, 0, 0], dtype=np.uint8)  # Inicjalizacja pierwszego piksela (czarny kolor)
    dummy = np.array([0, 0, 0], dtype=np.uint8)  # Tymczasowa tablica do modyfikacji wartości RGB

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

    return image.reshape(1, -1, 3)  # Zwraca linię tęczy o wysokości 1


def create_png_rainbow():
    rainbow_line = generate_rainbow_line()  # Generowanie linii tęczy
    height = 8  # Wysokość obrazu (dowolna wartość)
    width = rainbow_line.shape[1]  # Szerokość obrazu na podstawie długości linii tęczy

    # Powielanie linii tęczy na całą wysokość obrazu
    rainbow_image = np.tile(rainbow_line, (height, 1, 1)).astype(np.uint8)

    # Przygotowanie danych obrazu w formacie skanlinii (z filtrem None)
    raw_data = b''.join(b'\x00' + rainbow_image[y].tobytes() for y in range(height))

    compressed_data = zlib.compress(raw_data)  # Kompresja danych obrazu

    # Nagłówek PNG (IHDR)
    png_signature = b'\x89PNG\r\n\x1a\n'

    width_bytes = struct.pack('!I', width)
    height_bytes = struct.pack('!I', height)

    ihdr_content = width_bytes + height_bytes + b'\x08\x02\x00\x00\x00'  # Głębia: 8 bitów, RGB bez kompresji i filtrów
    ihdr_crc = struct.pack('!I', zlib.crc32(b'IHDR' + ihdr_content))
    ihdr_chunk = struct.pack('!I', len(ihdr_content)) + b'IHDR' + ihdr_content + ihdr_crc

    # Dane obrazu (IDAT)
    idat_crc = struct.pack('!I', zlib.crc32(b'IDAT' + compressed_data))
    idat_chunk = struct.pack('!I', len(compressed_data)) + b'IDAT' + compressed_data + idat_crc

    # Koniec pliku PNG (IEND)
    iend_content = b''
    iend_crc = struct.pack('!I', zlib.crc32(b'IEND' + iend_content))
    iend_chunk = struct.pack('!I', len(iend_content)) + b'IEND' + iend_content + iend_crc

    # Zapis pliku PNG
    with open('rainbow.png', 'wb') as f:
        f.write(png_signature)
        f.write(ihdr_chunk)
        f.write(idat_chunk)
        f.write(iend_chunk)


# Generowanie pliku PNG z tęczą
create_png_rainbow()
print("Plik 'rainbow.png' został zapisany.")

# Odczyt pliku PNG
image = Image.open('rainbow.png')

# Wyświetlenie obrazu za pomocą Matplotlib
pyplot.imshow(image)
pyplot.title("Tęcza (odczyt z pliku PNG)")
pyplot.axis("off")
pyplot.show()
