from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math
import lorem

def encode_as_binary_array(msg):
    msg = msg.encode("utf-8")  # koduje wiadomość jako bajty
    msg = msg.hex()  # konwertuje bajty do postaci szesnastkowej
    msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]  # dzieli ciąg hex na pary znaków
    msg = [ "{:08b}".format(int(el, base=16)) for el in msg]  # konwertuje każdy bajt z hex do 8-bitowego ciągu binarnego

    return "".join(msg)  # łączy binarne ciągi w jeden długi ciąg i go zwraca

def decode_from_binary_array(array):
    array = [array[i:i+8] for i in range(0, len(array), 8)]  # dzieli ciąg na 8-bitowe segmenty

    if len(array[-1]) != 8:  # jeśli ostatni segment ma mniej niż 8 bitów
        array[-1] = array[-1] + "0" * (8 - len(array[-1]))  # uzupełnia brakujące bity zerami

    array = [ "{:02x}".format(int(el, 2)) for el in array]  # konwertuje każdy ciąg 8-bitowy na dwuznakowy hex
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
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2) # oblicza średnią różnicę kwadratów pikseli
    return err

original_image = load_image("../lab4/image.png")  # wczytuje obraz

max_bits = int(original_image.size * 0.8)  # 80% pojemności obrazu
long_message = lorem.text()

binary_message = ""

while len(binary_message) < max_bits: # while dopóki długość zakodowanej wiadomości jest mniejsza niż maksymalna
    long_message += " " + lorem.text() # dodawanie tekstu
    binary_message = encode_as_binary_array(long_message) # koduje wiadomość do postaci binarnej

binary_message = binary_message[:max_bits] # obcina wiadomość

mse_values = []
images = []

for nbits in range(1, 9):
    try:
        img_with_msg = hide_message(original_image, binary_message, nbits=nbits) # ukrycie wiadomości w obrazie
        err = mse(original_image, img_with_msg) # obliczanie MSE
        mse_values.append(err) # zapisywanie MSE
        images.append((nbits, img_with_msg))
    except ValueError:
        mse_values.append(None) # None jako MSE
        images.append((nbits, None)) # None jako obraz

plt.figure(figsize=(16, 8)) # tworzenie wykresu na obrazki

for idx, (nbits, img) in enumerate(images): # iteracja po parach (nbits, obraz)
    plt.subplot(2, 4, idx + 1)  # tworzy podwykres w układzie 2x4

    if img is not None:  # jeśli obraz istnieje
        plt.imshow(img)  # wyświetla obraz
        plt.title(f"nbits = {nbits}")  # ustawia tytuł z liczbą bitów
    else:
        plt.text(0.5, 0.5, "Za duża wiadomość", ha='center', va='center') # komunikat w razie obrazu None

    plt.axis("off")

plt.suptitle("Obrazy z ukrytą wiadomością dla różnych nbits")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5)) # tworzy nowy wykres

values = [v or 0 for v in mse_values]
plt.plot(range(1, 9), values, marker='o') # rysuje wykres MSE, zastępując None zerami
plt.title("MSE między oryginałem a obrazem z tesktem")
plt.xlabel("nbits")
plt.ylabel("MSE")
plt.grid(True)
plt.show()
