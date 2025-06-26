import cv2
import numpy as np
from matplotlib import pyplot as plt

original = cv2.imread('test3.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(original, 127, 255, cv2.THRESH_BINARY)

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

def pokaz_wyniki(kernel_name, closed):
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle(f"Domknięcie (closing) – {kernel_name}", fontsize=14)

    axs[0].imshow(binary, cmap='gray')
    axs[0].set_title("Oryginał")
    axs[0].axis('off')

    axs[1].imshow(closed, cmap='gray')
    axs[1].set_title(f"Domknięcie")
    axs[1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    plt.show()

def podmenu():
    while True:
        print("\n=== MENU WYBORU KERNELI ===")
        print("1. Krzyzyk 3x3")
        print("2. Pelny 3x3")
        print("3. Pionowy 7x1")
        print("4. Duzy krzyzyk 7x7")
        print("5. Wyjscie")

        wybor = input("Wybierz opcję (1-5): ")

        if wybor in ['1', '2', '3', '4']:
            kernel_data = kernels[wybor]
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_data['kernel'])
            pokaz_wyniki(kernel_data['name'], closed)
        elif wybor == '5':
            print("Powrot")
            break
        else:
            print("Nieprawidłowy wybór, spróbuj ponownie.")

if __name__ == "__main__":
    podmenu()
