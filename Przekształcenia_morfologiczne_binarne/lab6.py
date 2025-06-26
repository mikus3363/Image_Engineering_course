import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def lab6_zad1():
    original = cv2.imread(resource_path('test2.png'), cv2.IMREAD_GRAYSCALE)
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
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0]
            ], dtype=np.uint8)
        }
    }

    def pokaz_wyniki(kernel_name, eroded, contour):
        fig, axs = plt.subplots(1, 3, figsize=(12, 5))
        fig.suptitle(f"Erozja i kontur – {kernel_name}", fontsize=16)

        axs[0].imshow(binary, cmap='gray')
        axs[0].set_title("Oryginal")
        axs[0].axis('off')

        axs[1].imshow(eroded, cmap='gray')
        axs[1].set_title(f"Erozja: {kernel_name}")
        axs[1].axis('off')

        axs[2].imshow(contour, cmap='gray')
        axs[2].set_title("Kontur (roznica)")
        axs[2].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.82)
        plt.show()

    def podmenu():
        while True:
            print("\n=== MENU WYBORU KERNELI ===")
            print("1. Krzyzyk 3x3")
            print("2. Pelny 3x3")
            print("3. Pionowy 7x1")
            print("4. Duzy krzyzyk 7x7")
            print("5. Powrot")

            wybor = input("Wybierz opcje (1-5): ")

            if wybor in ['1', '2', '3', '4']:
                kernel_data = kernels[wybor]
                eroded = cv2.erode(binary, kernel_data['kernel'], iterations=1)
                contour = cv2.subtract(binary, eroded)
                pokaz_wyniki(kernel_data['name'], eroded, contour)
            elif wybor == '5':
                print("Powrot")
                break
            else:
                print("Nieprawidłowy wybor, sprobuj ponownie.")
    podmenu()

def lab6_zad2():
    original = cv2.imread(resource_path('test1.png'), cv2.IMREAD_GRAYSCALE)
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
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0]
            ], dtype=np.uint8)
        }
    }

    def pokaz_wyniki(kernel_name, eroded, dilated):
        fig, axs = plt.subplots(1, 3, figsize=(12, 5))
        fig.suptitle(f"Erozja i dylacja – {kernel_name}", fontsize=14)

        axs[0].imshow(binary, cmap='gray')
        axs[0].set_title("Oryginal")
        axs[0].axis('off')

        axs[1].imshow(eroded, cmap='gray')
        axs[1].set_title(f"Erozja")
        axs[1].axis('off')

        axs[2].imshow(dilated, cmap='gray')
        axs[2].set_title("Dylacja")
        axs[2].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

    def podmenu():
        while True:
            print("\n=== MENU WYBORU KERNELI ===")
            print("1. Krzyzyk 3x3")
            print("2. Pelny 3x3")
            print("3. Pionowy 7x1")
            print("4. Duzy krzyzyk 7x7")
            print("5. Powrot")

            wybor = input("Wybierz opcję (1-5): ")

            if wybor in ['1', '2', '3', '4']:
                kernel_data = kernels[wybor]
                eroded = cv2.erode(binary, kernel_data['kernel'], iterations=1)
                dilated = cv2.dilate(binary, kernel_data['kernel'], iterations=1)
                pokaz_wyniki(kernel_data['name'], eroded, dilated)
            elif wybor == '5':
                print("Powrot")
                break
            else:
                print("Nieprawidlowy wybor, sprobuj ponownie.")
    podmenu()

def lab6_zad3():
    original = cv2.imread(resource_path('test3.png'), cv2.IMREAD_GRAYSCALE)
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
        axs[0].set_title("Oryginal")
        axs[0].axis('off')

        axs[1].imshow(closed, cmap='gray')
        axs[1].set_title(f"Domkniecie")
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
            print("5. Powrot")

            wybor = input("Wybierz opcję (1-5): ")

            if wybor in ['1', '2', '3', '4']:
                kernel_data = kernels[wybor]
                closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_data['kernel'])
                pokaz_wyniki(kernel_data['name'], closed)
            elif wybor == '5':
                print("Powrot")
                break
            else:
                print("Nieprawidlowy wybor, sprobuj ponownie.")
    podmenu()

def lab6_zad4():
    original = cv2.imread(resource_path('test3.png'), cv2.IMREAD_GRAYSCALE)
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

    def pokaz_wyniki(kernel_name, opened):
        fig, axs = plt.subplots(1, 2, figsize=(8, 5))
        fig.suptitle(f"Otwarcie (opening) – {kernel_name}", fontsize=14)

        axs[0].imshow(binary, cmap='gray')
        axs[0].set_title("Oryginal")
        axs[0].axis('off')

        axs[1].imshow(opened, cmap='gray')
        axs[1].set_title("Otwarcie")
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
            print("5. Powrot")

            wybor = input("Wybierz opcję (1-5): ")

            if wybor in ['1', '2', '3', '4']:
                kernel_data = kernels[wybor]
                opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_data['kernel'])
                pokaz_wyniki(kernel_data['name'], opened)
            elif wybor == '5':
                print("Powrot")
                break
            else:
                print("Nieprawidlowy wybor, sprobuj ponownie.")
    podmenu()

def lab6_zad5():
    image_files = {
        '1': resource_path('test1.png'),
        '2': resource_path('test2.png')
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
        axs[0].set_title("Oryginal")
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

            wybor = input("Wybierz opcję (1-3): ")

            if wybor in ['1', '2']:
                binary = przetworz_obraz(image_files[wybor])
                menu_kerneli(binary)
            elif wybor == '3':
                print("Powrot")
                break
            else:
                print("Nieprawidlowy wybor!")

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
                print("Powrot")
                break
            else:
                print("Nieprawidlowy wybor!")
    menu_obrazow()

def menu():
    while True:
        print("\n=== LABORATORIUM 6 ===")
        print("1. Zadanie 1 - Procedura erozja")
        print("2. Zadanie 2 - Procedura dylacji")
        print("3. Zadanie 3 - Procedura domknieta")
        print("4. Zadanie 4 - Procedura otwarcia")
        print("5. Zadanie 5 - Procedura wyznaczajaca gradient oraz laplasjan")
        print("6. Wyjście")

        choice = input("Wybierz opcję (1-6): ")

        if choice == "1":
            lab6_zad1()
        elif choice == "2":
            lab6_zad2()
        elif choice == "3":
            lab6_zad3()
        elif choice == "4":
            lab6_zad4()
        elif choice == "5":
            lab6_zad5()
        elif choice == "6":
            print("Wyjście z programu.")
            break
        else:
            print("Nieprawidłowa opcja. Spróbuj ponownie.")

if __name__ == "__main__":
    menu()