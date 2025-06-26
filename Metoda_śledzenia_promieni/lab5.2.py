import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [12, 8]  # rozmiar obrazka 12x8

EPSILON = 0.0001  # do błędów numerycznych przy przecięciach

def reflect(vector, normal_vector):  # wektor odbicia
    n_dot_l = np.dot(vector, normal_vector)  # obliczenie iloczynu skalarnego wektora i normalnej
    return vector - normal_vector * (2 * n_dot_l)  # zwrócenie odbitego wektora

def normalize(vector):  # funkcja normalizująca wektor
    return vector / np.sqrt((vector**2).sum())  # dzielenie przez długość wektora

class Ray:  # klasa promień
    def __init__(self, starting_point, direction):
        self.starting_point = starting_point  # punkt początkowy
        self.direction = direction  # kierunek promienia

class Light:  # klasa źródło światła
    def __init__(self, position):
        self.position = position  # pozycja światła
        self.ambient = np.array([0, 0, 0])  # składowa otoczenia
        self.diffuse = np.array([0, 1, 1])  # składowa rozproszenia
        self.specular = np.array([1, 1, 0])  #  składowa odbicia

class SceneObject:  # klasa bazowa dla obiektów
    def __init__(self, ambient=np.array([0, 0, 0]), diffuse=np.array([0.6, 0.7, 0.8]), specular=np.array([0.8, 0.8, 0.8]), shining=25):
        self.ambient = ambient  # kolor ambient
        self.diffuse = diffuse  # kolor diffuse
        self.specular = specular  # kolor specular
        self.shining = shining  # współczynnik połysku

    def get_color(self, cross_point, obs_vector, scene):  # oblicza kolor w punkcie przecięcia
        color = self.ambient * scene.ambient  # kolor z oświetlenia ambient
        light = scene.light  # źródło światła
        normal = self.get_normal(cross_point)  # wektor normalny w punkcie kolizji
        light_vector = normalize(light.position - cross_point)  # wektor do źródła światła
        n_dot_l = np.dot(light_vector, normal)
        reflection_vector = normalize(reflect(-1 * light_vector, normal))  # wektor odbicia
        v_dot_r = np.dot(reflection_vector, -obs_vector)

        if v_dot_r < 0:  # odbicie jest w przeciwną stronę to 0
            v_dot_r = 0

        if n_dot_l > 0:  # jeśli światło pada na widoczną stronę
            color += (
                (self.diffuse * light.diffuse * n_dot_l) +  # składowa rozproszona
                (self.specular * light.specular * v_dot_r**self.shining) +  # składowa zwierciadlana
                (self.ambient * light.ambient)  # ambient ze światła
            )

        return color

class Sphere(SceneObject):  # klasa kuli
    def __init__(self, position, radius, ambient=np.array([0, 0, 0]), diffuse=np.array([0.6, 0.7, 0.8]), specular=np.array([0.8, 0.8, 0.8]), shining=25):
        super(Sphere, self).__init__(ambient=ambient, diffuse=diffuse, specular=specular, shining=shining)  # wywołanie konstruktora
        self.position = position  # pozycja środka kuli
        self.radius = radius  # promień kuli

    def get_normal(self, cross_point):  # wektor normalny na powierzchni kuli
        return normalize(cross_point - self.position)  # wektor od środka do punktu

    def trace(self, ray):  # metoda przecięcia promienia z kulą
        distance = ray.starting_point - self.position  # wektor od środka kuli do początku promienia
        a = np.dot(ray.direction, ray.direction)  # współczynnik a
        b = 2 * np.dot(ray.direction, distance)  # współczynnik b
        c = np.dot(distance, distance) - self.radius**2  # współczynnik c
        d = b**2 - 4*a*c  # delta

        if d < 0:  # jeśli ujemna to brak
            return (None, None)

        sqrt_d = d**(0.5)
        denominator = 1 / (2 * a)

        if d > 0:  # dwa rozwiązania
            r1 = (-b - sqrt_d) * denominator  # pierwsze przecięcie
            r2 = (-b + sqrt_d) * denominator  # drugie przecięcie
            if r1 < EPSILON:  # jeśli pierwsze za blisko
                if r2 < EPSILON:  # jeśli drugie też za blisko to brak
                    return (None, None)
                r1 = r2  # użyj drugiego przecięcia
        else:  # jedno przecięcie
            r1 = -b * denominator
            if r1 < EPSILON:  # jeśli zbyt blisko to brak
                return (None, None)

        cross_point = ray.starting_point + r1 * ray.direction  # obliczenie punktu przecięcia
        return cross_point, r1  # zwrócenie punktu i odległości

class Camera:  # klasa kamery
    def __init__(self, position=np.array([0, 0, -3]), look_at=np.array([0, 0, 0])):
        self.z_near = 1  # dystans do płaszczyzny obrazu
        self.pixel_height = 500  # wysokość obrazu
        self.pixel_width = 700  # szerokość obrazu
        self.povy = 45  # pionowy kąt widzenia

        look = normalize(look_at - position)

        self.up = normalize(np.cross(np.cross(look, np.array([0, 1, 0])), look))
        self.position = position  # zapisanie pozycji
        self.look_at = look_at  # zapisanie punktu patrzenia
        self.direction = normalize(look_at - position)  # kierunek patrzenia

        aspect = self.pixel_width / self.pixel_height  # proporcje obrazu
        povy = self.povy * np.pi / 180  # konwersja kąta na radiany

        self.world_height = 2 * np.tan(povy/2) * self.z_near
        self.world_width = aspect * self.world_height

        center = self.position + self.direction * self.z_near  # środek płaszczyzny
        width_vector = normalize(np.cross(self.up, self.direction))  # wektor szerokości

        self.translation_vector_x = width_vector * -(self.world_width / self.pixel_width)  # przeskok w poziomie
        self.translation_vector_y = self.up * -(self.world_height / self.pixel_height)  # przeskok w pionie

        self.starting_point = center + width_vector * (self.world_width / 2) + (self.up * self.world_height / 2)  # górny lewy punkt startowy

    def get_world_pixel(self, x, y):  # metoda pozycji piksela
        return self.starting_point + self.translation_vector_x * x + self.translation_vector_y * y

class Scene:  # klasa sceny
    def __init__(self, objects, light, camera):
        self.objects = objects  # zapisanie listy obiektów
        self.light = light  # zapisanie światła
        self.camera = camera  # zapisanie kamery
        self.ambient = np.array([0.1, 0.1, 0.1])  # kolor ambient sceny
        self.background = np.array([0, 0, 0])  # kolor tła

class RayTracer:  # klasa renderowanie
    def __init__(self, scene, max_depth=3):  # maksymalna głębokość odbić *
        self.scene = scene
        self.max_depth = max_depth  # głębokość śledzenia odbić *

    def generate_image(self):  # generowanie obrazu
        camera = self.scene.camera
        image = np.zeros((camera.pixel_height, camera.pixel_width, 3))
        for y in range(image.shape[0]):  # pętla po wierszach
            for x in range(image.shape[1]):  # pętla po kolumnach
                world_pixel = camera.get_world_pixel(x, y)
                direction = normalize(world_pixel - camera.position)  # kierunek promienia
                image[y][x] = self._get_pixel_color(Ray(world_pixel, direction))  # kolor z funkcji śledzenia
        return image

    def _get_pixel_color(self, ray, depth=0):  # nowy parametr depth dla promieni wtórnych *
        obj, distance, cross_point = self._get_closest_object(ray)
        if not obj:
            return self.scene.background

        base_color = obj.get_color(cross_point, ray.direction, self.scene)  # kolor własny obiektu

        if depth >= self.max_depth:  # jeśli przekroczono maksymalną głębokość zakończ *
            return base_color

        normal = obj.get_normal(cross_point)  # wektor normalny
        reflection_dir = reflect(ray.direction, normal)  # kierunek odbicia *
        reflection_ray = Ray(cross_point + normal * EPSILON, normalize(reflection_dir))  # nowy promień z offsetem *

        reflection_color = self._get_pixel_color(reflection_ray, depth + 1)  # rekurencyjne śledzenie odbicia *

        final_color = 0.7 * base_color + 0.3 * reflection_color  # mieszanie koloru obiektu z odbiciem *
        return final_color  # zwrócenie koloru końcowego

    def _get_closest_object(self, ray):  # funkcja najbliższego przecięcia
        closest = None  # brak trafienia
        min_distance = np.inf  # minimalna odległość początkowa
        min_cross_point = None  # punkt przecięcia
        for obj in self.scene.objects:
            cross_point, distance = obj.trace(ray)  # jeżeli przecięcie
            if cross_point is not None and distance < min_distance:
                min_distance = distance
                closest = obj
                min_cross_point = cross_point
        return (closest, min_distance, min_cross_point)

scene = Scene(
    objects=[
        Sphere(position=np.array([0, 0, 0]), radius=1.5, diffuse=np.array([1, 0.2, 0.2])),  # pierwsza kula
        Sphere(position=np.array([-2, -1, 0]), radius=0.5, diffuse=np.array([0.2, 0.9, 0.3]))  # druga kula
    ],
    light=Light(position=np.array([3, 2, 5])),  # światło sceny
    camera=Camera(position=np.array([0, 0, 5]))  # kamera
)

rt = RayTracer(scene)  # raytracer z daną sceną
image = np.clip(rt.generate_image(), 0, 1)  # obraz i przycięcie wartości RGB do [0,1]
plt.imshow(image)
plt.show()
