import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import struct
import zlib
from PIL import Image

def resource_path(relative_path):
    try:
        # Jeśli uruchamiane z pliku .exe
        base_path = sys._MEIPASS
    except AttributeError:
        # Jeśli uruchamiane jako skrypt .py
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def zad1_1():
    # Funkcja do generowania szkicu RGB
    def generate_sketch(width, height):
        sketch_rgb = np.zeros((height, width, 3), dtype=np.uint8)  # inicjalizacja obrazu wypełnionego zerami
        for i in range(height):
            for j in range(width):
                sketch_rgb[i, j] = [i * 50 % 256, j * 50 % 256,
                                    (i + j) * 50 % 256]  # generowanie wartości RGB dla każdego piksela
        return sketch_rgb  # zwrócenie szkicu

    # Funkcja do zapisu obrazu w formacie P3 - tekstowy
    def save_ppm_p3(filename, image_array):
        height, width, _ = image_array.shape  # Pobierz wymiary obrazu
        max_val = 255  # Maksymalna wartość koloru dla RGB

        with open(filename, 'w') as f:
            f.write(f"P3\n{width} {height}\n{max_val}\n")  # Zapis nagłówka PPM
            for row in image_array:
                for pixel in row:
                    f.write(f"{pixel[0]} {pixel[1]} {pixel[2]} ")
                f.write("\n")  # Dodaj nową linię po każdej linii obrazu

    # Funkcja do zapisu obrazu w formacie P6 - binarny
    def save_ppm_p6(filename, image_array):
        height, width, _ = image_array.shape  # pobranie wymiarów obrazu (wysokość, szerokość)
        max_val = 255  # maksymalna wartość koloru dla RGB

        with open(filename, 'wb') as f:  # otwarcie pliku w trybie zapisu binarnego
            f.write(bytearray(f"P6\n{width} {height}\n{max_val}\n", 'ascii'))  # zapis nagłówka PPM jako bajty
            f.write(image_array.tobytes())  # zapis danych pikseli jako ciąg bajtów

        print(f"Struktura pliku P6 ({filename}):")  # wyświetlenie struktury pliku P6
        print(f"P6\n{width} {height}\n{max_val}")  # wyświetlenie nagłówka pliku P6

    # Funkcja do odczytu obrazu z formatu P3 - tekstowy
    def read_ppm_p3(filename):
        with open(filename, 'r', encoding='utf-8') as f:  # Otwórz plik z wymuszonym kodowaniem UTF-8
            lines = f.readlines()  # Odczytaj wszystkie linie

            # Usuń komentarze z nagłówka
            lines = [line for line in lines if not line.startswith('#')]

            # Sprawdź, czy format to "P3"
            if lines[0].strip() != "P3":
                raise ValueError(f"Plik {filename} nie jest w formacie P3!")

            width, height = map(int, lines[1].split())  # Pobierz szerokość i wysokość obrazu
            max_val = int(lines[2])  # Pobierz maksymalną wartość koloru

            # Pobierz wartości pikseli z pozostałych linii
            pixels = [int(val) for val in ' '.join(lines[3:]).split()]

            # Sprawdź, czy liczba pikseli odpowiada wymiarom obrazu
            if len(pixels) != width * height * 3:
                raise ValueError(f"Nieprawidłowa liczba pikseli w pliku {filename}!")

            return np.array(pixels).reshape((height, width, 3)).astype(np.uint8)

    # Funkcja do odczytu obrazu z formatu P6 - binarny
    def read_ppm_p6(filename):
        with open(filename, 'rb') as f:
            header = []
            while len(header) < 3:
                line = f.readline().strip()
                if not line.startswith(b'#'):  # Pomijaj komentarze
                    header.append(line.decode())

            if header[0] != "P6":
                raise ValueError(f"Plik {filename} nie jest w formacie P6!")

            width, height = map(int, header[1].split())
            max_val = int(header[2])

            # Odczytaj dane pikseli
            pixel_data = np.frombuffer(f.read(), dtype=np.uint8)

            # Sprawdź poprawność liczby pikseli
            if len(pixel_data) != width * height * 3:
                raise ValueError(
                    f"Nieprawidłowa liczba pikseli w pliku {filename}! Oczekiwano {width * height * 3}, otrzymano {len(pixel_data)}.")

            return pixel_data.reshape((height, width, 3)).astype(np.uint8)

    def read_ppm(filename):
        format_type = detect_ppm_format(filename)
        if format_type == "P3":
            return read_ppm_p3(filename)
        elif format_type == "P6":
            return read_ppm_p6(filename)

    def detect_ppm_format(filename):
        with open(filename, 'rb') as f:  # Otwórz plik w trybie binarnym
            header = f.readline().strip()
            if header == b'P3':
                return "P3"
            elif header == b'P6':
                return "P6"
            else:
                raise ValueError(f"Nieznany format pliku: {header.decode()}")

    # Główna część programu
    if __name__ == "__main__":
        width, height = 5, 5  # ustawienie wymiarów szkicu (szerokość i wysokość)
        sketch_rgb = generate_sketch(width, height)  # generowanie szkicu RGB za pomocą funkcji

        save_ppm_p3("sketch_p3.ppm", sketch_rgb)  # zapis szkicu w formacie tekstowym P3
        save_ppm_p6("sketch_p6.ppm", sketch_rgb)  # zapis szkicu w formacie binarnym P6

        sketch_p3_read = read_ppm_p3("sketch_p3.ppm")  # odczyt pliku tekstowego P3 ze szkicem
        sketch_p6_read = read_ppm_p6("sketch_p6.ppm")  # odczyt pliku binarnego P6 ze szkicem

        p3_size = len(open("sketch_p3.ppm", "rb").read())  # obliczenie rozmiaru pliku tekstowego P3 w bajtach
        p6_size = len(open("sketch_p6.ppm", "rb").read())  # obliczenie rozmiaru pliku binarnego P6 w bajtach

        print(f"\nRozmiar pliku P3: {p3_size} bajtów")
        print(f"Rozmiar pliku P6: {p6_size} bajtów\n")

        try:
            # Pobranie ścieżki do pliku 'obrazek.ppm'
            image_path = resource_path("obrazek.ppm")

            # Automatyczny odczyt zdjęcia RGB z pliku PPM
            photo_rgb = read_ppm(image_path)[:, :, ::-1]  # Odwrócenie kanałów kolorów (RGB -> BGR)

            save_ppm_p3("photo_p3.ppm", photo_rgb)  # Zapis zdjęcia w formacie tekstowym P3
            save_ppm_p6("photo_p6.ppm", photo_rgb)  # Zapis zdjęcia w formacie binarnym P6

            photo_p3_read = read_ppm("photo_p3.ppm")  # Odczyt pliku tekstowego P3
            photo_p6_read = read_ppm("photo_p6.ppm")  # Odczyt pliku binarnego P6

            photo_p3_size = len(open("photo_p3.ppm", "rb").read())
            photo_p6_size = len(open("photo_p6.ppm", "rb").read())

            print(f"\nRozmiar pliku zdjęcia P3: {photo_p3_size} bajtów")
            print(f"Rozmiar pliku zdjęcia P6: {photo_p6_size} bajtów")

        except FileNotFoundError:
            print("\nPlik 'obrazek.ppm' nie istnieje")
        except ValueError as e:
            print(f"\nBłąd podczas odczytu pliku 'obrazek.ppm': {e}")

    # Wyświetlenie szkicu za pomocą Matplotlib
    plt.imshow(sketch_rgb)
    plt.title("Szkic RGB")
    plt.axis("off")
    plt.show()

def zad1_2():
    # Nagłówek pliku PPM w formacie P3
    ppm_header = 'P3\n120 8\n255\n'  # Format PPM: szerokość=120, wysokość=8, maksymalna wartość koloru=255

    # Inicjalizacja pierwszego piksela (czarny kolor: [0, 0, 0])
    image = np.array([0, 0, 0], dtype=np.uint8)  # Tablica przechowująca dane obrazu
    dummy = np.array([0, 0, 0], dtype=np.uint8)  # Tymczasowa tablica do modyfikacji wartości RGB

    step = 15  # Krok zmiany koloru (255 / 17), aby uzyskać płynne przejście w spektrum

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

    line = np.copy(image)  # Kopia jednej linii spektrum
    image = np.tile(line, (8,))  # Powielamy linię spektrum tak, aby uzyskać obraz o wysokości=8

    image_array = image.reshape(8, 120, 3)

    # Zapis obrazu w formacie PPM (tekstowy - P3)
    with open('transition_spectrum.ppm', 'w') as fh:
        fh.write(ppm_header)  # Zapis nagłówka pliku PPM
        for row in image_array:
            for pixel in row:
                fh.write(f"{pixel[0]} {pixel[1]} {pixel[2]} ")
            fh.write("\n")

    print("Plik 'transition_spectrum.ppm' został zapisany.")

    # Wyświetlenie obrazu za pomocą Matplotlib
    plt.imshow(image_array.astype(np.uint8))
    plt.title("Spektrum RGB")
    plt.axis("off")
    plt.show()

def zad1_3():
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
    plt.imshow(image)
    plt.title("Tęcza (odczyt z pliku PNG)")
    plt.axis("off")
    plt.show()


def zad1_4():
    def rgb_to_ycbcr(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)  # konwertuje obraz z RGB na YCbCr

    def ycbcr_to_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)  # konwertuje obraz z YCbCr na RGB

    # Funkcja do downsamplingu
    def downsample(channel, factor):
        if factor == 1:  # Bez próbkowania
            return channel
        elif factor == 2:  # Co drugi element
            return channel[::2, ::2]  # zwraca kanał próbkowany co drugi piksel w obu wymiarach
        elif factor == 4:  # Co czwarty element
            return channel[::4, ::4]  # zwraca kanał próbkowany co czwarty piksel w obu wymiarach
        else:
            raise ValueError("Nieprawidłowy współczynnik próbkowania.")

    # Funkcja do upsamplingu
    def upsample(channel, original_shape):
        return cv2.resize(channel, (original_shape[1], original_shape[0]),
                          interpolation=cv2.INTER_NEAREST)  # przywraca kanał do pierwotnego rozmiaru za pomocą interpolacji najbliższego sąsiada

    # Funkcja do podziału na bloki 8x8
    def split_into_blocks(channel):
        h, w = channel.shape  # pobiera wysokość i szerokość kanału
        blocks = []  # inicjalizuje listę bloków

        for i in range(0, h, 8):
            for j in range(0, w, 8):
                blocks.append(channel[i:i + 8, j:j + 8])  # dodaje blok 8x8 do listy
        return blocks

    # Funkcja do kodowania współczynników
    def zigzag(block):
        rows, cols = block.shape  # pobiera rozmiar bloku
        result = []  # inicjalizuje listę wynikową

        for sum_idx in range(rows + cols - 1):  # iteruje po przekątnych bloku
            if sum_idx % 2 == 0:
                # dla parzystych przekątnych iteruje od góry do dołu i od lewej do prawej
                for i in range(rows):
                    for j in range(cols):
                        if i + j == sum_idx:
                            result.append(block[i][j])
            else:
                # dla nieparzystych przekątnych iteruje od dołu do góry i od prawej do lewej
                for i in range(rows - 1, -1, -1):
                    for j in range(cols - 1, -1, -1):
                        if i + j == sum_idx:
                            result.append(block[i][j])

        return np.array(result)  # zwraca zakodowaną tablicę

    # Funkcja odwrotna
    def inverse_zigzag(array, rows=8, cols=8):
        block = np.zeros((rows, cols))  # inicjalizuje pusty blok
        idx = 0  # indeks dla elementów tablicy wejściowej

        for sum_idx in range(rows + cols - 1):
            if sum_idx % 2 == 0:
                for i in range(rows):
                    for j in range(cols):
                        if i + j == sum_idx:
                            block[i][j] = array[idx]
                            idx += 1
            else:
                for i in range(rows - 1, -1, -1):
                    for j in range(cols - 1, -1, -1):
                        if i + j == sum_idx:
                            block[i][j] = array[idx]
                            idx += 1
        return block

    # Funkcja kompresji danych
    def compress_data(data):
        return zlib.compress(data.tobytes())  # kompresuje dane wejściowe za pomocą zlib

    def jpeg_algorithm(image_path, sampling_factor):

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ycbcr_image = rgb_to_ycbcr(image)

        # rozdzielenie obraz YCbCr na kanały: Y, Cb, Cr
        Y_channel = ycbcr_image[:, :, 0]
        Cb_channel = ycbcr_image[:, :, 1]
        Cr_channel = ycbcr_image[:, :, 2]

        # wykonanie próbkowanie kanałów Cb i Cr zgodnie z podanym współczynnikiem
        Cb_downsampled = downsample(Cb_channel, sampling_factor)
        Cr_downsampled = downsample(Cr_channel, sampling_factor)

        # przywrócenie kanałów Cb i Cr do pierwotnego rozmiaru obrazu
        Cb_upsampled = upsample(Cb_downsampled, Y_channel.shape)
        Cr_upsampled = upsample(Cr_downsampled, Y_channel.shape)

        # tworzenie nowego obrazu YCbCr po próbkowaniu przez połączenie kanałów
        ycbcr_downsampled_image = np.stack((Y_channel, Cb_upsampled, Cr_upsampled), axis=-1)

        # konwertowanie obrazu YCbCr z powrotem na RGB - do wizualizacji
        rgb_downsampled_image = ycbcr_to_rgb(ycbcr_downsampled_image)

        # kompresowanie danych dla każdego kanału za pomocą algorytmu zlib
        compressed_Y = compress_data(Y_channel)
        compressed_Cb = compress_data(Cb_downsampled)
        compressed_Cr = compress_data(Cr_downsampled)

        # obliczanie całkowitego rozmiaru skompresowanego obrazu jako sumę rozmiarów skompresowanych kanałów
        total_size = len(compressed_Y) + len(compressed_Cb) + len(compressed_Cr)

        print(f"Rozmiar skompresowanego obrazu dla próbkowania {sampling_factor}: {total_size} bajtów")

        return rgb_downsampled_image, total_size

    image_path = "rainbow.png"

    # generowanie obrazów
    image_no_sampling, size_no_sampling = jpeg_algorithm(image_path, sampling_factor=1)
    image_half_sampling, size_half_sampling = jpeg_algorithm(image_path, sampling_factor=2)
    image_quarter_sampling, size_quarter_sampling = jpeg_algorithm(image_path, sampling_factor=4)

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.imshow(image_no_sampling)
    plt.title(f"No sampling (Rozmiar: {size_no_sampling} bajtów)")

    plt.subplot(3, 1, 2)
    plt.imshow(image_half_sampling)
    plt.title(f"Sampling: every 2nd (Rozmiar: {size_half_sampling} bajtów)")

    plt.subplot(3, 1, 3)
    plt.imshow(image_quarter_sampling)
    plt.title(f"Sampling: every 4th (Rozmiar: {size_quarter_sampling} bajtów)")

    plt.tight_layout()
    plt.show()


def menu():
    print("\nAutor: Mikołaj Lipiński")
    while True:
        print("\nMenu:")
        print("1. Zadanie 2.1 - obsluga formatu PPM")
        print("2. Zadanie 2.2 - przestrzen barw RGB")
        print("3. Zadanie 2.3 - tworzenie zbioru w formacie PNG")
        print("4. Zadanie 2.4 - algorytm JPEG")
        print("5. Wyjście z programu")

        choice = input("Wybierz opcję (1-5): ")

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
            print("Wyjście z menu. Żegnaj!")
            break
        else:
            print("Nieprawidłowy wybór. Proszę spróbuj ponownie.")

menu()