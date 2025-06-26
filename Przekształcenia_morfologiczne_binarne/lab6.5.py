import cv2
import numpy as np
from matplotlib import pyplot as plt

image_files = {
    '1': 'test1.png',
    '2': 'test2.png'
}

kernels = {
    '1': {
        'name': 'krzyżyk 3x3',
        'kernel': cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    },
    '2': {
        'name': 'pełny 3x3',
        'kernel': np.ones((3, 3), np.uint8)
    },
    '3': {
        'name': 'pionowy 7x1',
        'kernel': np.ones((7, 1), np.uint8)
    },
    '4': {
        'name': 'duży krzyżyk 7x7',
        'kernel': np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.uint8)
    }
}


def przetworz_obraz(image_path):
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(original, 127, 255, cv2.THRESH_BINARY)
    return binary


def pokaz_wyniki(binary, kernel_name, gradient, laplacian):
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(f"Gradient i Laplasjan - {kernel_name}", fontsize=14)

    axs[0].imshow(binary, cmap='gray')
    axs[0].set_title("Oryginał")
    axs[0].axis('off')

    axs[1].imshow(gradient, cmap='gray')
    axs[1].set_title("Gradient")
    axs[1].axis('off')

    axs[2].imshow(laplacian, cmap='gray')
    axs[2].set_title("Laplasjan")
    axs[2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


def menu_obrazow():
    while True:
        print("\n=== WYBOR OBRAZU ===")
        print("1. test1.png")
        print("2. test2.png")
        print("3. Powrot")
        print("4. Wyjscie")

        wybor = input("Wybierz opcję (1-4): ")

        if wybor in ['1', '2']:
            binary = przetworz_obraz(image_files[wybor])
            menu_kerneli(binary)
        elif wybor == '3':
            return
        elif wybor == '4':
            print("Zamykanie programu...")
            exit()
        else:
            print("Nieprawidłowy wybór!")


def menu_kerneli(binary):
    while True:
        print("\n=== WYBOR KERNELA ===")
        print("1. Krzyzyk 3x3")
        print("2. Pelny 3x3")
        print("3. Pionowy 7x1")
        print("4. Duzy krzyzyk 7x7")
        print("5. Powrot")

        wybor = input("Wybierz opcję (1-5): ")

        if wybor in ['1', '2', '3', '4']:
            kernel = kernels[wybor]
            dilation = cv2.dilate(binary, kernel['kernel'])
            erosion = cv2.erode(binary, kernel['kernel'])

            gradient = cv2.subtract(dilation, erosion)
            laplacian = cv2.subtract(
                cv2.subtract(dilation, binary),
                cv2.subtract(binary, erosion)
            )

            pokaz_wyniki(binary, kernel['name'], gradient, laplacian)
        elif wybor == '5':
            return
        else:
            print("Nieprawidłowy wybór!")


if __name__ == "__main__":
    menu_obrazow()
