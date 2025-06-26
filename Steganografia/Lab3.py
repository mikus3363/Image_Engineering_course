import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math
import zlib

def resource_path(relative_path):
    try:
        # Jeśli uruchamiane z pliku .exe
        base_path = sys._MEIPASS
    except AttributeError:
        # Jeśli uruchamiane jako skrypt .py
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def zad1_1():
    # Funkcja do konwersji RGB -> YCbCr
    def rgb_to_ycbcr(image):
        return cv.cvtColor(image, cv.COLOR_RGB2YCrCb)

    # Funkcja do konwersji YCbCr -> RGB
    def ycbcr_to_rgb(image):
        return cv.cvtColor(image, cv.COLOR_YCrCb2RGB)

    # Funkcja do próbkowania (Downsampling)
    def downsample(channel, factor):
        if factor == 1:
            return channel
        elif factor == 2:
            return channel[::2, ::2]
        elif factor == 4:
            return channel[::4, ::4]
        else:
            raise ValueError("Invalid sampling factor")

    # Funkcja do upsamplingu (odwrotne próbkowanie)
    def upsample(channel, original_shape):
        return cv.resize(channel, (original_shape[1], original_shape[0]), interpolation=cv.INTER_NEAREST)

    # Funkcja do podziału na bloki 8x8
    def split_into_blocks(channel):
        h, w = channel.shape
        blocks = []
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                blocks.append(channel[i:i + 8, j:j + 8])
        return blocks

    # Funkcja do składania bloków z powrotem w obraz
    def reconstruct_image(blocks, shape):
        reconstructed_image = np.zeros(shape)

        h_blocks = shape[0] // 8
        w_blocks = shape[1] // 8

        idx = 0

        for i in range(h_blocks):
            for j in range(w_blocks):
                reconstructed_image[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = blocks[idx]
                idx += 1
        return reconstructed_image

    # Funkcja do wyznaczania macierzy kwantyzacji w zależności od QF
    def get_quantization_matrix(QF):
        base_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])
        if QF < 50:
            scale = max(1.0 * (5000 / QF), 1)
        else:
            scale = 200 - QF * 2
        quantization_matrix = np.floor((base_matrix * scale + 50) / 100).astype(np.int32)
        quantization_matrix[quantization_matrix == 0] = 1
        return quantization_matrix

    # Funkcja do obliczania DCT dla bloku
    def apply_dct(block):
        return cv.dct(block.astype(np.float32))

    # Funkcja do obliczania odwrotnej DCT dla bloku
    def apply_idct(block):
        return cv.idct(block)

    # Funkcja do kwantyzacji współczynników DCT
    def quantize(block_dct, quantization_matrix):
        return np.round(block_dct / quantization_matrix).astype(np.int32)

    # Funkcja do dekwantyzacji współczynników DCT
    def dequantize(block_quantized, quantization_matrix):
        return (block_quantized * quantization_matrix).astype(np.float32)

    # Funkcja kompresji danych za pomocą zlib
    def compress_data(data):
        return zlib.compress(data.tobytes())

    # Główna funkcja implementacji algorytmu JPEG z etapami DCT i kwantyzacji
    def jpeg_algorithm_with_qf(image_path, sampling_factor=1, QF=50):
        # Krok 0: Wczytanie obrazu tęczy
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Konwersja na RGB

        # Krok 1: Konwersja RGB -> YCbCr
        ycbcr_image = rgb_to_ycbcr(image)
        Y_channel = ycbcr_image[:, :, 0]
        Cb_channel = ycbcr_image[:, :, 1]
        Cr_channel = ycbcr_image[:, :, 2]

        # Krok 2: Próbkowanie kanałów chromatycznych Cb i Cr
        Cb_downsampled = downsample(Cb_channel, sampling_factor)
        Cr_downsampled = downsample(Cr_channel, sampling_factor)

        # Krok 3: Podział na bloki (8x8)
        Y_blocks = split_into_blocks(Y_channel)
        Cb_blocks = split_into_blocks(Cb_downsampled)
        Cr_blocks = split_into_blocks(Cr_downsampled)

        # Krok 4: Wyznaczenie macierzy kwantyzacji na podstawie QF
        quantization_matrix = get_quantization_matrix(QF)

        # Krok 5: Obliczanie DCT i kwantyzacja dla każdego bloku
        Y_quantized_blocks = [quantize(apply_dct(block), quantization_matrix) for block in Y_blocks]
        Cb_quantized_blocks = [quantize(apply_dct(block), quantization_matrix) for block in Cb_blocks]
        Cr_quantized_blocks = [quantize(apply_dct(block), quantization_matrix) for block in Cr_blocks]

        # Krok 6: Dekwantyzacja i odwrotna DCT dla każdego bloku (odwrotne operacje)
        Y_reconstructed_blocks = [apply_idct(dequantize(block, quantization_matrix)) for block in Y_quantized_blocks]
        Cb_reconstructed_blocks = [apply_idct(dequantize(block, quantization_matrix)) for block in Cb_quantized_blocks]
        Cr_reconstructed_blocks = [apply_idct(dequantize(block, quantization_matrix)) for block in Cr_quantized_blocks]

        # Składanie bloków z powrotem w obrazy
        Y_reconstructed = reconstruct_image(Y_reconstructed_blocks, Y_channel.shape)
        Cb_reconstructed_upsampled = upsample(reconstruct_image(Cb_reconstructed_blocks, Cb_downsampled.shape),
                                              Y_channel.shape)
        Cr_reconstructed_upsampled = upsample(reconstruct_image(Cr_reconstructed_blocks, Cr_downsampled.shape),
                                              Y_channel.shape)

        # Połączenie kanałów w obraz wynikowy (YCbCr -> RGB)
        reconstructed_ycbcr_image = np.stack((Y_reconstructed, Cb_reconstructed_upsampled, Cr_reconstructed_upsampled),
                                             axis=-1)

        # Skalowanie wartości do zakresu 0–255 i konwersja do RGB
        reconstructed_ycbcr_image = np.clip(reconstructed_ycbcr_image, 0, 255)
        reconstructed_rgb_image = ycbcr_to_rgb(reconstructed_ycbcr_image.astype(np.uint8))

        # Krok końcowy: Kompresja danych i pomiar rozmiaru pliku w bajtach
        compressed_Y = compress_data(np.array(Y_quantized_blocks))
        compressed_Cb = compress_data(np.array(Cb_quantized_blocks))
        compressed_Cr = compress_data(np.array(Cr_quantized_blocks))

        total_size = len(compressed_Y) + len(compressed_Cb) + len(compressed_Cr)

        print(f"Rozmiar skompresowanego obrazu dla QF={QF}, próbkowania={sampling_factor}: {total_size} bajtów")

        return reconstructed_rgb_image

    # Parametry testowe i wywołanie funkcji
    image_path = resource_path("rainbow.png")  # Ścieżka do obrazu tęczy

    # Testowanie dla różnych wartości QF i próbkowania
    reconstructed_image_qf50 = jpeg_algorithm_with_qf(image_path=image_path, sampling_factor=1, QF=50)
    reconstructed_image_qf25 = jpeg_algorithm_with_qf(image_path=image_path, sampling_factor=1, QF=25)
    reconstructed_image_qf75 = jpeg_algorithm_with_qf(image_path=image_path, sampling_factor=1, QF=75)

    # Wyświetlenie wyników dla różnych wartości QF
    plt.figure(figsize=(15, 5))

    plt.subplot(3, 1, 1)
    plt.imshow(reconstructed_image_qf50)
    plt.title("QF=50")
    plt.axis("off")

    plt.subplot(3, 1, 2)
    plt.imshow(reconstructed_image_qf25)
    plt.title("QF=25")
    plt.axis("off")

    plt.subplot(3, 1, 3)
    plt.imshow(reconstructed_image_qf75)
    plt.title("QF=75")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def zad1_2():
    def encode_as_binary_array(msg):
        msg = msg.encode("utf-8")  # koduje wiadomość jako bajty
        msg = msg.hex()  # konwertuje bajty do postaci szesnastkowej
        msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]  # dzieli ciąg hex na pary znaków
        msg = ["{:08b}".format(int(el, base=16)) for el in
               msg]  # konwertuje każdy bajt z hex do 8-bitowego ciągu binarnego

        return "".join(msg)  # łączy binarne ciągi w jeden długi ciąg i go zwraca

    def decode_from_binary_array(array):
        array = [array[i:i + 8] for i in range(0, len(array), 8)]  # dzieli ciąg na 8-bitowe segmenty

        if len(array[-1]) != 8:  # jeśli ostatni segment ma mniej niż 8 bitów
            array[-1] = array[-1] + "0" * (8 - len(array[-1]))  # uzupełnia brakujące bity zerami

        array = ["{:02x}".format(int(el, 2)) for el in array]  # konwertuje każdy ciąg 8-bitowy na dwuznakowy hex
        array = "".join(array)  # łączy ciągi hex w jeden
        result = binascii.unhexlify(array)  # konwertuje hex na bajty

        return result.decode("utf-8", errors="replace")  # dekoduje bajty na tekst zamieniając błędy

    def load_image(path, pad=False):
        image = cv.imread(path)  # wczytuje obraz
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # konwertuje kolor z BGR na RGB

        if pad:  # jeśli pad jest True
            y_pad = 8 - (image.shape[0] % 8)  # oblicza ile linii dodać, by wysokość była podzielna przez 8
            x_pad = 8 - (image.shape[1] % 8)  # oblicza ile kolumn dodać, by szerokość była podzielna przez 8
            image = np.pad(image, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')  # dodaje padding zerami

        return image  # zwraca obraz

    def save_image(path, image):
        plt.imsave(path, image)  # zapisuje obraz

    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)  # zwraca n ograniczone do zakresu min max

    def hide_message(image, message, nbits=1):
        nbits = clamp(nbits, 1, 8)  # ogranicza nbits do przedziału 1-8
        shape = image.shape  # zapisuje oryginalny kształt
        image = np.copy(image).flatten()  # tworzy kopię i spłaszcza ją do 1D

        if len(message) > len(image) * nbits:  # sprawdza czy wiadomość zmieści się w obrazie
            raise ValueError("Message is too long :(")  # jeśli nie, ValueError

        chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]  # dzieli wiadomość binarną na segmenty

        for i, chunk in enumerate(chunks):  # iteruje po segmentach
            byte = "{:08b}".format(image[i])  # konwertuje piksel do postaci binarnej
            new_byte = byte[:-nbits] + chunk  # zamienia ostatnie bity na bity wiadomości
            image[i] = int(new_byte, 2)  # zamienia z powrotem na całkowitą liczbę

        return image.reshape(shape)  # przekształca obraz do oryginalnego kształtu

    def reveal_message(image, nbits=1, length=0):
        nbits = clamp(nbits, 1, 8)  # ogranicza nbits do zakresu 1-8
        shape = image.shape  # zapisuje kształt obrazu
        image = np.copy(image).flatten()  # tworzy kopię i spłaszcza ją do 1D
        length_in_pixels = math.ceil(length / nbits)  # ile pikseli potrzeba by odczytać wiadomość

        if len(image) < length_in_pixels or length_in_pixels <= 0:  # jeśli długość wiadomości przekracza rozmiar obrazu
            length_in_pixels = len(image)  # ustawia długość na maksymalną

        message = ""  # ciąg na wiadomość

        for i in range(length_in_pixels):
            byte = "{:08b}".format(image[i])  # konwertuje piksel na binarny ciąg
            message += byte[-nbits:]  # dodaje ostatnie nbits bitów do wiadomości

        mod = length % -nbits  # sprawdza czy długość wiadomości wymaga obcięcia

        if mod != 0:  # jeśli tak
            message = message[:mod]  # obcina wiadomość

        return message

    original_image = load_image(resource_path("../lab4/image.png"))  # wczytuje obrazek

    message = "Tekst ukryty w obrazku"  # ustawia wiadomość do ukrycia
    binary_message = encode_as_binary_array(message)  # konwertuje wiadomość do postaci binarnej

    nbits = 1
    image_with_message = hide_message(original_image, binary_message, nbits)  # ukrywa wiadomość w obrazie

    save_image(resource_path("image_with_text.png"), image_with_message)  # zapisuje obraz w PNG
    save_image(resource_path("image_with_text.jpg"), image_with_message)  # zapisuje obraz w JPG

    image_with_message_png = load_image(resource_path("image_with_text.png"))  # ładuje obraz PNG
    image_with_message_jpg = load_image(resource_path("image_with_text.jpg"))  # ładuje obraz JPG

    secret_message_png = decode_from_binary_array(
        reveal_message(image_with_message_png, nbits=nbits, length=len(binary_message)))  # odczytuje wiadomość z PNG
    secret_message_jpg = decode_from_binary_array(
        reveal_message(image_with_message_jpg, nbits=nbits, length=len(binary_message)))  # odczytuje wiadomość z JPG

    print(secret_message_png)  # wiadomość odczytaną z PNG
    print(secret_message_jpg)  # wiadomość odczytaną z JPG

    f, ar = plt.subplots(2, 2)
    ar[0, 0].imshow(original_image)
    ar[0, 0].set_title("Original image")
    ar[0, 1].imshow(image_with_message)
    ar[0, 1].set_title("Image with message")
    ar[1, 0].imshow(image_with_message_png)
    ar[1, 0].set_title("PNG image")
    ar[1, 1].imshow(image_with_message_jpg)
    ar[1, 1].set_title("JPG image")
    ar[0, 0].axis('off')
    ar[0, 1].axis('off')
    ar[1, 0].axis('off')
    ar[1, 1].axis('off')

    plt.show()

def zad1_3():
    def encode_as_binary_array(msg):
        msg = msg.encode("utf-8")  # koduje wiadomość jako bajty
        msg = msg.hex()  # konwertuje bajty do postaci szesnastkowej
        msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]  # dzieli ciąg hex na pary znaków
        msg = ["{:08b}".format(int(el, base=16)) for el in
               msg]  # konwertuje każdy bajt z hex do 8-bitowego ciągu binarnego

        return "".join(msg)  # łączy binarne ciągi w jeden długi ciąg i go zwraca

    def decode_from_binary_array(array):
        array = [array[i:i + 8] for i in range(0, len(array), 8)]  # dzieli ciąg na 8-bitowe segmenty

        if len(array[-1]) != 8:  # jeśli ostatni segment ma mniej niż 8 bitów
            array[-1] = array[-1] + "0" * (8 - len(array[-1]))  # uzupełnia brakujące bity zerami

        array = ["{:02x}".format(int(el, 2)) for el in array]  # konwertuje każdy ciąg 8-bitowy na dwuznakowy hex
        array = "".join(array)  # łączy ciągi hex w jeden
        result = binascii.unhexlify(array)  # konwertuje hex na bajty

        return result.decode("utf-8", errors="replace")  # dekoduje bajty na tekst zamieniając błędy

    def load_image(path, pad=False):
        image = cv.imread(path)  # wczytuje obraz
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # konwertuje kolor z BGR na RGB

        if pad:  # jeśli pad jest True
            y_pad = 8 - (image.shape[0] % 8)  # oblicza ile linii dodać, by wysokość była podzielna przez 8
            x_pad = 8 - (image.shape[1] % 8)  # oblicza ile kolumn dodać, by szerokość była podzielna przez 8
            image = np.pad(image, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')  # dodaje padding zerami

        return image  # zwraca obraz

    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)  # zwraca n ograniczone do zakresu min max

    def hide_message(image, message, nbits=1):
        nbits = clamp(nbits, 1, 8)  # ogranicza nbits do przedziału 1-8
        shape = image.shape  # zapisuje oryginalny kształt
        image = np.copy(image).flatten()  # tworzy kopię i spłaszcza ją do 1D

        if len(message) > len(image) * nbits:  # sprawdza czy wiadomość zmieści się w obrazie
            raise ValueError("Message is too long :(")  # jeśli nie, ValueError

        chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]  # dzieli wiadomość binarną na segmenty

        for i, chunk in enumerate(chunks):  # iteruje po segmentach
            byte = "{:08b}".format(image[i])  # konwertuje piksel do postaci binarnej
            new_byte = byte[:-nbits] + chunk  # zamienia ostatnie bity na bity wiadomości
            image[i] = int(new_byte, 2)  # zamienia z powrotem na całkowitą liczbę

        return image.reshape(shape)  # przekształca obraz do oryginalnego kształtu

    def mse(imageA, imageB):
        err = np.mean(
            (imageA.astype("float") - imageB.astype("float")) ** 2)  # oblicza średnią różnicę kwadratów pikseli
        return err

    original_image = load_image(resource_path("../lab4/image.png"))  # wczytuje obraz

    max_bits = int(original_image.size * 0.8)  # 80% pojemności obrazu
    long_message = "To jest przykładowy tekst testowy." * 100

    binary_message = ""

    while len(binary_message) < max_bits:  # while dopóki długość zakodowanej wiadomości jest mniejsza niż maksymalna
        long_message += " " + "To jest przykładowy tekst testowy." * 100  # dodawanie tekstu
        binary_message = encode_as_binary_array(long_message)  # koduje wiadomość do postaci binarnej

    binary_message = binary_message[:max_bits]  # obcina wiadomość

    mse_values = []
    images = []

    for nbits in range(1, 9):
        try:
            img_with_msg = hide_message(original_image, binary_message, nbits=nbits)  # ukrycie wiadomości w obrazie
            err = mse(original_image, img_with_msg)  # obliczanie MSE
            mse_values.append(err)  # zapisywanie MSE
            images.append((nbits, img_with_msg))
        except ValueError:
            mse_values.append(None)  # None jako MSE
            images.append((nbits, None))  # None jako obraz

    plt.figure(figsize=(16, 8))  # tworzenie wykresu na obrazki

    for idx, (nbits, img) in enumerate(images):  # iteracja po parach (nbits, obraz)
        plt.subplot(2, 4, idx + 1)  # tworzy podwykres w układzie 2x4

        if img is not None:  # jeśli obraz istnieje
            plt.imshow(img)  # wyświetla obraz
            plt.title(f"nbits = {nbits}")  # ustawia tytuł z liczbą bitów
        else:
            plt.text(0.5, 0.5, "Za duża wiadomość", ha='center', va='center')  # komunikat w razie obrazu None

        plt.axis("off")

    plt.suptitle("Obrazy z ukrytą wiadomością dla różnych nbits")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))  # tworzy nowy wykres

    values = [v or 0 for v in mse_values]
    plt.plot(range(1, 9), values, marker='o')  # rysuje wykres MSE, zastępując None zerami
    plt.title("MSE między oryginałem a obrazem z tesktem")
    plt.xlabel("nbits")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.show()


def zad1_4():
    def encode_as_binary_array(msg):
        msg = msg.encode("utf-8")  # koduje wiadomość jako bajty
        msg = msg.hex()  # konwertuje bajty do postaci szesnastkowej
        msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]  # dzieli ciąg hex na pary znaków
        msg = ["{:08b}".format(int(el, base=16)) for el in
               msg]  # konwertuje każdy bajt z hex do 8-bitowego ciągu binarnego

        return "".join(msg)  # łączy binarne ciągi w jeden długi ciąg i go zwraca

    def decode_from_binary_array(array):
        array = [array[i:i + 8] for i in range(0, len(array), 8)]  # dzieli ciąg na 8-bitowe segmenty

        if len(array[-1]) != 8:  # jeśli ostatni segment ma mniej niż 8 bitów
            array[-1] = array[-1] + "0" * (8 - len(array[-1]))  # uzupełnia brakujące bity zerami

        array = ["{:02x}".format(int(el, 2)) for el in array]  # konwertuje każdy ciąg 8-bitowy na dwuznakowy hex
        array = "".join(array)  # łączy ciągi hex w jeden
        result = binascii.unhexlify(array)  # konwertuje hex na bajty

        return result.decode("utf-8", errors="replace")  # dekoduje bajty na tekst zamieniając błędy

    def load_image(path, pad=False):
        image = cv.imread(path)  # wczytuje obraz
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # konwertuje kolor z BGR na RGB

        if pad:  # jeśli pad jest True
            y_pad = 8 - (image.shape[0] % 8)  # oblicza ile linii dodać, by wysokość była podzielna przez 8
            x_pad = 8 - (image.shape[1] % 8)  # oblicza ile kolumn dodać, by szerokość była podzielna przez 8
            image = np.pad(image, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')  # dodaje padding zerami

        return image  # zwraca obraz

    def save_image(path, image):
        plt.imsave(path, image)  # zapisuje obraz

    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)  # zwraca n ograniczone do zakresu min max

    def hide_message(image, message, nbits, spos):
        nbits = clamp(nbits, 1, 8)  # ogranicza liczbę bitów do przedziału 1-8
        shape = image.shape  # zapisuje oryginalny kształt obrazu
        flat_image = np.copy(image).flatten()  # spłaszcza obraz do 1D

        if spos < 0 or spos >= len(flat_image):
            raise ValueError("Invalid starting position (spos).")  # jeśli nie, rzuca wyjątek

        available_space = (len(flat_image) - spos) * nbits  # oblicza maksymalną liczbę bitów dostępnych do zapisu

        if len(message) > available_space:  # jeśli wiadomość się nie zmieści
            raise ValueError("Message is too long for the selected starting position and nbits.")  # rzuca wyjątek

        chunks = [message[i:i + nbits] for i in
                  range(0, len(message), nbits)]  # dzieli wiadomość binarną na kawałki po nbits

        for i, chunk in enumerate(chunks):
            idx = spos + i  # oblicza indeks w obrazie
            byte = "{:08b}".format(flat_image[idx])  # konwertuje bajt na 8-bitowy ciąg binarny
            new_byte = byte[:-nbits] + chunk  # zastępuje nbits najmłodszych bitów wiadomością
            flat_image[idx] = int(new_byte, 2)  # konwertuje z powrotem do liczby i zapisuje w obrazie

        return flat_image.reshape(shape)

    def reveal_message(image, nbits, length, spos):
        nbits = clamp(nbits, 1, 8)  # ogranicza liczbę bitów do zakresu 1–8
        shape = image.shape  # zapisuje kształt obrazu
        flat_image = np.copy(image).flatten()  # spłaszcza obraz do 1D

        if spos < 0 or spos >= len(flat_image):  # sprawdza czy spos jest poprawną pozycją startową
            raise ValueError("Invalid starting position (spos).")  # rzuca wyjątek

        length_in_pixels = math.ceil(length / nbits)  # oblicza ile pikseli potrzeba do odczytania wiadomości

        if spos + length_in_pixels > len(flat_image):  # sprawdza czy nie wychodzimy poza obraz
            length_in_pixels = len(flat_image) - spos  # dostosowuje długość, żeby nie przekroczyć granicy obrazu

        message = ""
        for i in range(length_in_pixels):  # iteruje po pikselach potrzebnych do odczytu
            idx = spos + i  # oblicza indeks pikselu
            byte = "{:08b}".format(flat_image[idx])  # zamienia wartość pikselu na binarny ciąg
            message += byte[-nbits:]  # dodaje nbits najmłodszych bitów do wiadomości

        mod = length % -nbits  # oblicza ile nadmiarowych bitów może być na końcu

        if mod != 0:  # jeśli są nadmiarowe
            message = message[:mod]  # ucina je

        return message

    original_image = load_image(resource_path("../lab4/image.png"))

    message = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    binary_message = encode_as_binary_array(message)

    nbits = 8
    spos = 100000  # pozycja startowa w spłaszczonym obrazie
    image_with_message = hide_message(original_image, binary_message, nbits=nbits, spos=spos)

    save_image(resource_path("image_with_hidden_message.png"), image_with_message)

    loaded_image = load_image(resource_path("image_with_hidden_message.png"))

    recovered_bits = reveal_message(loaded_image, nbits=nbits, length=len(binary_message), spos=spos)
    recovered_message = decode_from_binary_array(recovered_bits)

    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_image)
    ax[0].set_title("Oryginalny obraz")
    ax[0].axis('off')

    ax[1].imshow(image_with_message)
    ax[1].set_title("Z ukrytą wiadomością (od pozycji 1000)")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()


def zad1_5():
    def hide_message(image, message, nbits=1):
        nbits = clamp(nbits, 1, 8)  # ogranicza nbits do przedziału 1-8
        shape = image.shape  # zapisuje oryginalny kształt
        image = np.copy(image).flatten()  # tworzy kopię i spłaszcza ją do 1D

        if len(message) > len(image) * nbits:  # sprawdza czy wiadomość zmieści się w obrazie
            raise ValueError("Message is too long :(")  # jeśli nie, ValueError

        chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]  # dzieli wiadomość binarną na segmenty

        for i, chunk in enumerate(chunks):  # iteruje po segmentach
            byte = "{:08b}".format(image[i])  # konwertuje piksel do postaci binarnej
            new_byte = byte[:-nbits] + chunk  # zamienia ostatnie bity na bity wiadomości
            image[i] = int(new_byte, 2)  # zamienia z powrotem na całkowitą liczbę

        return image.reshape(shape)  # przekształca obraz do oryginalnego kształtu

    def encode_file_to_binary(file_path):
        with open(file_path, "rb") as file:
            data = file.read()
        hex_data = data.hex()
        bin_data = ["{:08b}".format(int(hex_data[i:i + 2], 16)) for i in range(0, len(hex_data), 2)]
        return "".join(bin_data)

    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)

    def load_image(path, pad=False):
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if pad:
            y_pad = 8 - (image.shape[0] % 8)
            x_pad = 8 - (image.shape[1] % 8)
            image = np.pad(image, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')
        return image

    def save_image(path, image):
        plt.imsave(path, image)

    def hide_image(image, secret_image_path, nbits=1):
        secret_binary = encode_file_to_binary(secret_image_path)
        hidden_image = hide_message(image, secret_binary, nbits)
        return hidden_image, len(secret_binary)

    def reveal_message(image, nbits=1, length=0):
        nbits = clamp(nbits, 1, 8)
        image = np.copy(image).flatten()
        length_in_pixels = math.ceil(length / nbits)
        if len(image) < length_in_pixels or length_in_pixels <= 0:
            length_in_pixels = len(image)
        message = ""
        for i in range(length_in_pixels):
            byte = "{:08b}".format(image[i])
            message += byte[-nbits:]
        mod = length % -nbits
        if mod != 0:
            message = message[:mod]
        return message

    def reveal_image(image, length, nbits=1, output_path="recovered_image.png"):
        bits = reveal_message(image, nbits, length)

        if len(bits) % 8 != 0:
            bits += '0' * (8 - len(bits) % 8)

        byte_data = bytearray()
        for i in range(0, len(bits), 8):
            byte_data.append(int(bits[i:i + 8], 2))

        with open(output_path, "wb") as f:
            f.write(byte_data)

        print("Obraz odzyskany:", output_path)
        return output_path

    original_image = load_image(resource_path("../lab4/image.png"))

    image_with_secret, length_of_secret = hide_image(original_image, resource_path("image2.png"), nbits=1)

    save_image(resource_path("image_with_hidden.png"), image_with_secret)

    loaded_image_with_secret = load_image(resource_path("image_with_hidden.png"))

    reveal_image(loaded_image_with_secret, length=length_of_secret, nbits=1, output_path="recovered_image.png")

    from PIL import Image

    f, ar = plt.subplots(1, 3, figsize=(12, 4))
    ar[0].imshow(original_image)
    ar[0].set_title("Oryginalny obraz")
    ar[1].imshow(image_with_secret)
    ar[1].set_title("Z ukrytym obrazkiem")
    ar[2].imshow(Image.open("recovered_image.png"))
    ar[2].set_title("Odzyskany obrazek")

    for ax in ar:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def zad1_6():
    def hide_message(image, message, nbits=1):
        nbits = clamp(nbits, 1, 8)
        shape = image.shape
        image = np.copy(image).flatten()
        if len(message) > len(image) * nbits:
            raise ValueError("Message is too long :(")
        chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]
        for i, chunk in enumerate(chunks):
            byte = "{:08b}".format(image[i])
            new_byte = byte[:-nbits] + chunk
            image[i] = int(new_byte, 2)
        return image.reshape(shape)

    def encode_file_to_binary(file_path):
        with open(file_path, "rb") as file:
            data = file.read()
        hex_data = data.hex()
        bin_data = ["{:08b}".format(int(hex_data[i:i + 2], 16)) for i in range(0, len(hex_data), 2)]
        return "".join(bin_data)

    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)

    def load_image(path, pad=False):
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if pad:
            y_pad = 8 - (image.shape[0] % 8)
            x_pad = 8 - (image.shape[1] % 8)
            image = np.pad(image, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')
        return image

    def save_image(path, image):
        plt.imsave(path, image)

    def hide_image(image, secret_image_path, nbits=1):
        secret_binary = encode_file_to_binary(secret_image_path)
        hidden_image = hide_message(image, secret_binary, nbits)
        return hidden_image, len(secret_binary)

    def reveal_message(image, nbits=1, length=0):
        nbits = clamp(nbits, 1, 8)
        image = np.copy(image).flatten()
        length_in_pixels = math.ceil(length / nbits)
        if len(image) < length_in_pixels or length_in_pixels <= 0:
            length_in_pixels = len(image)
        message = ""
        for i in range(length_in_pixels):
            byte = "{:08b}".format(image[i])
            message += byte[-nbits:]
        mod = length % -nbits
        if mod != 0:
            message = message[:mod]
        return message

    def reveal_png_image(image, nbits=1, output_path="recovered_image_v2.png"):
        bits = reveal_message(image, nbits)

        # stopka pliku PNG (IEND + CRC) – 8 bajtów = 64 bity
        png_footer = "0100100101000101010011100100010010101110010000100110000010000010"  # 49 45 4E 44 AE 42 60 82

        # szukamy pierwszego wystąpienia stopki
        idx = bits.find(png_footer)
        if idx == -1:
            print("Nie znaleziono stopki pliku PNG.")
            return None

        end = idx + len(png_footer)

        # uzupełnienie do pełnych bajtów
        if end % 8 != 0:
            end += 8 - (end % 8)

        bits = bits[:end]

        # zamiana bitów na bajty
        byte_data = bytearray()
        for i in range(0, len(bits), 8):
            byte_data.append(int(bits[i:i + 8], 2))

        with open(output_path, "wb") as f:
            f.write(byte_data)

        print(f"Odzyskano ukryty obraz PNG: {output_path}")
        return output_path

    original_image = load_image(resource_path("../lab4/image.png"))

    image_with_hidden, _ = hide_image(original_image, resource_path("kostki.png"), nbits=1)

    save_image(resource_path("image_with_hidden_v2.png"), image_with_hidden)

    loaded_image = load_image(resource_path("image_with_hidden_v2.png"))

    reveal_png_image(loaded_image, nbits=1, output_path="recovered_image_v2.png")

    from PIL import Image

    f, ar = plt.subplots(1, 3, figsize=(12, 4))
    ar[0].imshow(original_image)
    ar[0].set_title("Oryginalny obraz")
    ar[1].imshow(image_with_hidden)
    ar[1].set_title("Z ukrytym obrazkiem")
    ar[2].imshow(Image.open("recovered_image_v2.png"))
    ar[2].set_title("Odzyskany obrazek")

    for ax in ar:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def menu():
    print("\nAutor: Mikołaj Lipiński")
    while True:
        print("\nMenu:")
        print("1. Zadanie 3.0 - algorytm JPEG")
        print("2. Zadanie 3.1 - ukrywanie wiadomosci w obrazku")
        print("3. Zadanie 3.2 - tekst w obrazku i MSE")
        print("4. Zadanie 3.3 - ukrywanie wiadomosci od danej pozycji")
        print("5. Zadanie 3.4 - odzyskiwanie obrazka z obrazka")
        print("6. Zadanie 3.5 - zmodyfikowany sposób odzyskiwania obrazka ze zdjęcia")
        print("7. Wyjście z programu")

        choice = input("Wybierz opcję (1-7): ")

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
            print("Wybrano Zadanie 6")
            zad1_6()
        elif choice == "7":
            print("Wyjście z menu. Żegnaj!")
            break
        else:
            print("Nieprawidłowy wybór. Proszę spróbuj ponownie.")

menu()