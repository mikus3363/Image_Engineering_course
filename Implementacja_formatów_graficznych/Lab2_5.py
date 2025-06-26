import cv2
import numpy as np
import matplotlib.pyplot as plt
import zlib


# Funkcja do konwersji RGB -> YCbCr
def rgb_to_ycbcr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)


# Funkcja do konwersji YCbCr -> RGB
def ycbcr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)


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
    return cv2.resize(channel, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)


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
    return cv2.dct(block.astype(np.float32))


# Funkcja do obliczania odwrotnej DCT dla bloku
def apply_idct(block):
    return cv2.idct(block)


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
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konwersja na RGB

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
image_path = "rainbow.png"  # Ścieżka do obrazu tęczy

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




