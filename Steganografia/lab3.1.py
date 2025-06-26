from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math

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
    length_in_pixels = math.ceil(length/nbits)  # ile pikseli potrzeba by odczytać wiadomość

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

original_image = load_image("../lab4/image.png")  # wczytuje obrazek

message = "Tekst ukryty w obrazku"  # ustawia wiadomość do ukrycia
binary_message = encode_as_binary_array(message)  # konwertuje wiadomość do postaci binarnej

nbits = 1
image_with_message = hide_message(original_image, binary_message, nbits)  # ukrywa wiadomość w obrazie

save_image("image_with_text.png", image_with_message)  # zapisuje obraz w PNG
save_image("image_with_text.jpg", image_with_message)  # zapisuje obraz w JPG

image_with_message_png = load_image("image_with_text.png")  # ładuje obraz PNG
image_with_message_jpg = load_image("image_with_text.jpg")  # ładuje obraz JPG

secret_message_png = decode_from_binary_array(reveal_message(image_with_message_png, nbits=nbits, length=len(binary_message)))  # odczytuje wiadomość z PNG
secret_message_jpg = decode_from_binary_array(reveal_message(image_with_message_jpg, nbits=nbits,length=len(binary_message)))  # odczytuje wiadomość z JPG

print(secret_message_png)  # wiadomość odczytaną z PNG
print(secret_message_jpg)  # wiadomość odczytaną z JPG

f, ar = plt.subplots(2,2)
ar[0,0].imshow(original_image)
ar[0,0].set_title("Original image")
ar[0,1].imshow(image_with_message)
ar[0,1].set_title("Image with message")
ar[1,0].imshow(image_with_message_png)
ar[1,0].set_title("PNG image")
ar[1,1].imshow(image_with_message_jpg)
ar[1,1].set_title("JPG image")
ar[0,0].axis('off')
ar[0,1].axis('off')
ar[1,0].axis('off')
ar[1,1].axis('off')

plt.show()
