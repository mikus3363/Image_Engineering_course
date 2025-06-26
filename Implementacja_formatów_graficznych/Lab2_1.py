import numpy as np
import cv2
from matplotlib import pyplot as plt

# Funkcja do generowania szkicu RGB
def generate_sketch(width, height):
    sketch_rgb = np.zeros((height, width, 3), dtype=np.uint8)  # inicjalizacja obrazu wypełnionego zerami
    for i in range(height):
        for j in range(width):
            sketch_rgb[i, j] = [i * 50 % 256, j * 50 % 256, (i + j) * 50 % 256]  # generowanie wartości RGB dla każdego piksela
    return sketch_rgb  # zwrócenie szkicu

# Funkcja do zapisu obrazu w formacie P3 - tekstowy
def save_ppm_p3(filename, image_array):
    height, width, _ = image_array.shape  # pobranie wymiarów obrazu (wysokość, szerokość)
    max_val = 255  # maksymalna wartość koloru dla RGB

    with open(filename, 'w') as f:  # otwarcie pliku w trybie zapisu tekstowego
        f.write(f"P3\n{width} {height}\n{max_val}\n")  # zapis nagłówka PPM
        image_array.tofile(f, sep=' ')  # zapis danych pikseli jako ciąg liczb oddzielonych spacjami
        f.write('\n')  # dodanie nowej linii na końcu pliku

    print(f"Struktura pliku P3 ({filename}):")  # wyświetlenie struktury pliku P3
    print(f"P3\n{width} {height}\n{max_val}")  # wyświetlenie nagłówka pliku P3

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
    with open(filename, 'r') as f:  # otwarcie pliku w trybie odczytu tekstowego
        lines = f.readlines()  # odczyt wszystkich linii z pliku

        assert lines[0].strip() == "P3"  # sprawdzenie, czy format to "P3"

        width, height = map(int, lines[1].split())  # pobranie szerokości i wysokości obrazu z nagłówka
        max_val = int(lines[2])  # pobranie maksymalnej wartości koloru

        pixels = [int(val) for val in ' '.join(lines[3:]).split()]  # odczyt wartości RGB pikseli z pozostałych linii

        return np.array(pixels).reshape((height, width, 3)).astype(np.uint8)  # konwersja danych na tablicę NumPy

# Funkcja do odczytu obrazu z formatu P6 - binarny
def read_ppm_p6(filename):
    with open(filename, 'rb') as f:  # otwarcie pliku w trybie odczytu binarnego
        header = []  # lista na przechowywanie nagłówka pliku

        while len(header) < 3:  # czytanie pierwszych trzech linii nagłówka
            line = f.readline().strip()  # odczyt jednej linii i usunięcie białych znaków
            if not line.startswith(b'#'):  # pomijanie komentarzy zaczynających się od "#"
                header.append(line.decode())  # dekodowanie linii jako tekst i dodanie do listy nagłówka

        assert header[0] == "P6"  # sprawdzenie, czy format to "P6"

        width, height = map(int, header[1].split())  # pobranie szerokości i wysokości obrazu z nagłówka
        max_val = int(header[2])  # pobranie maksymalnej wartości koloru

        pixel_data = np.frombuffer(f.read(), dtype=np.uint8)  # odczyt danych pikseli jako bajty

        return pixel_data.reshape((height, width, 3)).astype(np.uint8)  # konwersja danych na tablicę NumPy

# Główna część programu
if __name__ == "__main__":
    width, height = 5, 5 # ustawienie wymiarów szkicu (szerokość i wysokość)
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
        photo_rgb = cv2.imread("obrazek.ppm")[:, :, ::-1]  # odczyt zdjęcia RGB z pliku (zakładamy format PPM)

        save_ppm_p3("photo_p3.ppm", photo_rgb)  # zapis zdjęcia w formacie tekstowym P3
        save_ppm_p6("photo_p6.ppm", photo_rgb)  # zapis zdjęcia w formacie binarnym P6

        photo_p3_read = read_ppm_p3("photo_p3.ppm")  # odczyt pliku tekstowego P3
        photo_p6_read = read_ppm_p6("photo_p6.ppm")  # odczyt pliku binarnego P6

        photo_p3_size = len(open("photo_p3.ppm", "rb").read())
        photo_p6_size = len(open("photo_p6.ppm", "rb").read())

        print(f"\nRozmiar pliku zdjęcia P3: {photo_p3_size} bajtów")
        print(f"Rozmiar pliku zdjęcia P6: {photo_p6_size} bajtów")

    except FileNotFoundError:
        print("\nPlik 'obrazek.ppm' nie istnieje")

# Wyświetlenie szkicu za pomocą Matplotlib
plt.imshow(sketch_rgb)
plt.title("Szkic RGB")
plt.axis("off")
plt.show()