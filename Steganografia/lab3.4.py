from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math


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
    bin_data = [ "{:08b}".format(int(hex_data[i:i+2], 16)) for i in range(0, len(hex_data), 2) ]
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
        byte_data.append(int(bits[i:i+8], 2))

    with open(output_path, "wb") as f:
        f.write(byte_data)

    print("Obraz odzyskany:", output_path)
    return output_path


original_image = load_image("../lab4/image.png")

image_with_secret, length_of_secret = hide_image(original_image, "image2.png", nbits=1)

save_image("image_with_hidden.png", image_with_secret)

loaded_image_with_secret = load_image("image_with_hidden.png")

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
