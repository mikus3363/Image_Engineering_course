from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math

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
        byte_data.append(int(bits[i:i+8], 2))

    with open(output_path, "wb") as f:
        f.write(byte_data)

    print(f"Odzyskano ukryty obraz PNG: {output_path}")
    return output_path


original_image = load_image("../lab4/image.png")

image_with_hidden, _ = hide_image(original_image, "kostki.png", nbits=1)

save_image("image_with_hidden_v2.png", image_with_hidden)

loaded_image = load_image("image_with_hidden_v2.png")

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
