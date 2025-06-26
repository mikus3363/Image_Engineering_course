import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def find_closest_palette_color(value, k=2):
    return round((k - 1) * value / 255) * 255 / (k - 1)   # zaokrąglamy wartość do najbliższego poziomu palety


def color_reduction(image, k_values):

    result = image.copy().astype(np.float64)

    for channel in range(3): # redukcja kolorów dla każdego kanału RGB
        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                result[y, x, channel] = find_closest_palette_color(result[y, x, channel], k_values[channel])

    np.clip(result, 0, 255, out=result) # przycinamy wartości

    return result.astype(np.uint8)


def floyd_steinberg_dithering(image, k_values):

    result = image.copy().astype(np.float64)

    height, width, channels = result.shape

    for channel in range(channels): # dla każdego kanału RGB
        k = k_values[channel]

        for y in range(height): # dla każdego piksela w obrazie
            for x in range(width):
                old_value = result[y, x, channel] # stara wartość piksela

                new_value = find_closest_palette_color(old_value, k) # najbliższy kolor z palety dla danej składowej

                result[y, x, channel] = new_value # nowa wartość piksela

                error = old_value - new_value # oblicz błąd kwantyzacji

                # rozprowadź błąd do sąsiednich pikseli
                if x < width - 1:
                    result[y, x + 1, channel] += error * 7 / 16

                if y < height - 1:
                    if x > 0:
                        result[y + 1, x - 1, channel] += error * 3 / 16

                    result[y + 1, x, channel] += error * 5 / 16

                    if x < width - 1:
                        result[y + 1, x + 1, channel] += error * 1 / 16

    np.clip(result, 0, 255, out=result) # przycinamy wartości

    return result.astype(np.uint8)


def create_color_histogram(image, title="Color Histogram"):

    colors = ("red", "green", "blue") # kolory dla każdego kanału

    fig, ax = plt.subplots(figsize=(10, 6)) # wykres histogramu
    ax.set_xlim([0, 256])

    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(image[:, :, channel_id], bins=256, range=(0, 256)) # obliczamy histogram dla danego kanału
        ax.plot(bin_edges[0:-1], histogram, color=color) # rysujemy linię histogramu w odpowiednim kolorze

    ax.set_title(title)
    ax.set_xlabel("Wartość koloru")
    ax.set_ylabel("Liczba pikseli")

    return fig


def main():
    image_path = 'obrazek.jpg'  # scieżka do obrazka
    image = np.array(Image.open(image_path))

    k_values = (2, 2, 2)

    reduced_image = color_reduction(image, k_values) # redukcja palety kolorów bez dithering

    dithered_image = floyd_steinberg_dithering(image, k_values) # z dithering


    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Oryginalny obraz')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(reduced_image)
    plt.title('Redukcja kolorów (bez dithering)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(dithered_image)
    plt.title('Floyd-Steinberg Dithering')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


    create_color_histogram(image, "Histogram oryginalnego obrazu")
    plt.show()
    create_color_histogram(reduced_image, "Histogram obrazu po redukcji kolorów")
    plt.show()
    create_color_histogram(dithered_image, "Histogram obrazu po dithering")
    plt.show()


if __name__ == "__main__":
    main()
