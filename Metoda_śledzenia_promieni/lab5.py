import numpy as np
from matplotlib import pyplot as plt

def lab5_zad1():
    plt.rcParams['figure.figsize'] = [12, 8]  # rozmiar obrazka 12x8

    EPSILON = 0.0001  # do błędów numerycznych przy przecięciach

    def reflect(vector, normal_vector):  # wektor odbicia
        n_dot_l = np.dot(vector, normal_vector)  # obliczenie iloczynu skalarnego wektora i normalnej
        return vector - normal_vector * (2 * n_dot_l)  # zwrócenie odbitego wektora

    def normalize(vector):  # funkcja normalizująca wektor
        return vector / np.sqrt((vector ** 2).sum())  # dzielenie przez długość wektora

    class Ray:  # klasa promień
        def __init__(self, starting_point, direction):
            self.starting_point = starting_point  # punkt początkowy
            self.direction = direction  # kierunek promienia

    class Light:  # klasa źródło światła
        def __init__(self, position):
            self.position = position  # pozycja światła
            self.ambient = np.array([0, 0, 0])  # składowa otoczenia
            self.diffuse = np.array([0, 1, 1])  # składowa rozproszenia
            self.specular = np.array([1, 1, 0])  # składowa odbicia

    class SceneObject:  # klasa bazowa dla obiektów
        def __init__(self, ambient=np.array([0, 0, 0]), diffuse=np.array([0.6, 0.7, 0.8]),
                     specular=np.array([0.8, 0.8, 0.8]), shining=25):
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
                        (self.specular * light.specular * v_dot_r ** self.shining) +  # składowa zwierciadlana
                        (self.ambient * light.ambient)  # ambient ze światła
                )

            return color

    class Sphere(SceneObject):  # klasa kuli
        def __init__(self, position, radius, ambient=np.array([0, 0, 0]), diffuse=np.array([0.6, 0.7, 0.8]),
                     specular=np.array([0.8, 0.8, 0.8]), shining=25):
            super(Sphere, self).__init__(ambient=ambient, diffuse=diffuse, specular=specular,
                                         shining=shining)  # wywołanie konstruktora
            self.position = position  # pozycja środka kuli
            self.radius = radius  # promień kuli

        def get_normal(self, cross_point):  # wektor normalny na powierzchni kuli
            return normalize(cross_point - self.position)  # wektor od środka do punktu

        def trace(self, ray):  # metoda przecięcia promienia z kulą
            distance = ray.starting_point - self.position  # wektor od środka kuli do początku promienia
            a = np.dot(ray.direction, ray.direction)  # współczynnik a
            b = 2 * np.dot(ray.direction, distance)  # współczynnik b
            c = np.dot(distance, distance) - self.radius ** 2  # współczynnik c
            d = b ** 2 - 4 * a * c  # delta

            if d < 0:  # jeśli ujemna to brak
                return (None, None)

            sqrt_d = d ** (0.5)
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

            self.world_height = 2 * np.tan(povy / 2) * self.z_near
            self.world_width = aspect * self.world_height

            center = self.position + self.direction * self.z_near  # środek płaszczyzny
            width_vector = normalize(np.cross(self.up, self.direction))  # wektor szerokości

            self.translation_vector_x = width_vector * -(self.world_width / self.pixel_width)  # przeskok w poziomie
            self.translation_vector_y = self.up * -(self.world_height / self.pixel_height)  # przeskok w pionie

            self.starting_point = center + width_vector * (self.world_width / 2) + (
                        self.up * self.world_height / 2)  # górny lewy punkt startowy

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
        def __init__(self, scene):
            self.scene = scene

        def generate_image(self):  # generowanie obrazu
            camera = self.scene.camera
            image = np.zeros((camera.pixel_height, camera.pixel_width, 3))
            for y in range(image.shape[0]):  # pętla po w
                for x in range(image.shape[1]):  # pętla po h
                    world_pixel = camera.get_world_pixel(x, y)
                    direction = normalize(world_pixel - camera.position)  # kierunek promienia
                    image[y][x] = self._get_pixel_color(Ray(world_pixel, direction))  # kolor z funkcji śledzenia
            return image

        def _get_pixel_color(self, ray):
            obj, distance, cross_point = self._get_closest_object(ray)
            if not obj:  # jeśli nic nie trafiono
                return self.scene.background
            return obj.get_color(cross_point, ray.direction, self.scene)

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


def lab5_zad2():
    plt.rcParams['figure.figsize'] = [12, 8]  # rozmiar obrazka 12x8

    EPSILON = 0.0001  # do błędów numerycznych przy przecięciach

    def reflect(vector, normal_vector):  # wektor odbicia
        n_dot_l = np.dot(vector, normal_vector)  # obliczenie iloczynu skalarnego wektora i normalnej
        return vector - normal_vector * (2 * n_dot_l)  # zwrócenie odbitego wektora

    def normalize(vector):  # funkcja normalizująca wektor
        return vector / np.sqrt((vector ** 2).sum())  # dzielenie przez długość wektora

    class Ray:  # klasa promień
        def __init__(self, starting_point, direction):
            self.starting_point = starting_point  # punkt początkowy
            self.direction = direction  # kierunek promienia

    class Light:  # klasa źródło światła
        def __init__(self, position):
            self.position = position  # pozycja światła
            self.ambient = np.array([0, 0, 0])  # składowa otoczenia
            self.diffuse = np.array([0, 1, 1])  # składowa rozproszenia
            self.specular = np.array([1, 1, 0])  # składowa odbicia

    class SceneObject:  # klasa bazowa dla obiektów
        def __init__(self, ambient=np.array([0, 0, 0]), diffuse=np.array([0.6, 0.7, 0.8]),
                     specular=np.array([0.8, 0.8, 0.8]), shining=25):
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
                        (self.specular * light.specular * v_dot_r ** self.shining) +  # składowa zwierciadlana
                        (self.ambient * light.ambient)  # ambient ze światła
                )

            return color

    class Sphere(SceneObject):  # klasa kuli
        def __init__(self, position, radius, ambient=np.array([0, 0, 0]), diffuse=np.array([0.6, 0.7, 0.8]),
                     specular=np.array([0.8, 0.8, 0.8]), shining=25):
            super(Sphere, self).__init__(ambient=ambient, diffuse=diffuse, specular=specular,
                                         shining=shining)  # wywołanie konstruktora
            self.position = position  # pozycja środka kuli
            self.radius = radius  # promień kuli

        def get_normal(self, cross_point):  # wektor normalny na powierzchni kuli
            return normalize(cross_point - self.position)  # wektor od środka do punktu

        def trace(self, ray):  # metoda przecięcia promienia z kulą
            distance = ray.starting_point - self.position  # wektor od środka kuli do początku promienia
            a = np.dot(ray.direction, ray.direction)  # współczynnik a
            b = 2 * np.dot(ray.direction, distance)  # współczynnik b
            c = np.dot(distance, distance) - self.radius ** 2  # współczynnik c
            d = b ** 2 - 4 * a * c  # delta

            if d < 0:  # jeśli ujemna to brak
                return (None, None)

            sqrt_d = d ** (0.5)
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

            self.world_height = 2 * np.tan(povy / 2) * self.z_near
            self.world_width = aspect * self.world_height

            center = self.position + self.direction * self.z_near  # środek płaszczyzny
            width_vector = normalize(np.cross(self.up, self.direction))  # wektor szerokości

            self.translation_vector_x = width_vector * -(self.world_width / self.pixel_width)  # przeskok w poziomie
            self.translation_vector_y = self.up * -(self.world_height / self.pixel_height)  # przeskok w pionie

            self.starting_point = center + width_vector * (self.world_width / 2) + (
                        self.up * self.world_height / 2)  # górny lewy punkt startowy

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


def lab5_zad3():
    plt.rcParams['figure.figsize'] = [12, 8]  # rozmiar obrazka 12x8

    EPSILON = 0.0001  # do błędów numerycznych przy przecięciach

    def reflect(vector, normal_vector):  # funkcja zwracająca wektor odbity
        n_dot_l = np.dot(vector, normal_vector)  # obliczenie iloczynu skalarnego
        return vector - normal_vector * (2 * n_dot_l)  # wzór na odbicie

    def normalize(vector):  # funkcja normalizująca wektor
        return vector / np.sqrt((vector ** 2).sum())  # dzielenie przez długość wektora

    class Ray:  # klasa promienia
        def __init__(self, starting_point, direction):
            self.starting_point = starting_point  # punkt początkowy promienia
            self.direction = direction  # kierunek promienia

    class Light:  # klasa źródła światła
        def __init__(self, position):
            self.position = position  # pozycja światła
            self.ambient = np.array([0, 0, 0])  # składowa ambient
            self.diffuse = np.array([0, 1, 1])  # składowa diffuse
            self.specular = np.array([1, 1, 0])  # składowa specular

    class SceneObject:  # klasa bazowa dla obiektów
        def __init__(self, ambient=np.array([0, 0, 0]), diffuse=np.array([0.6, 0.7, 0.8]),
                     specular=np.array([0.8, 0.8, 0.8]), shining=25):
            self.ambient = ambient  # kolor ambient
            self.diffuse = diffuse  # kolor diffuse
            self.specular = specular  # kolor specular
            self.shining = shining  # współczynnik połysku

        def get_color(self, cross_point, obs_vector, scene):  # oblicza kolor w punkcie przecięcia
            color = self.ambient * scene.ambient  # kolor z oświetlenia ambient
            light = scene.light  # źródło światła
            normal = self.get_normal(cross_point)  # wektor normalny w punkcie kolizji
            light_vector = normalize(light.position - cross_point)  # wektor do źródła światła

            if scene.raytracer.is_in_shadow(cross_point, light.position):  # sprawdzenie czy punkt jest w cieniu *
                return color  # jeśli tak, zwracamy ambient *

            n_dot_l = np.dot(light_vector, normal)
            reflection_vector = normalize(reflect(-1 * light_vector, normal))  # wektor odbicia
            v_dot_r = np.dot(reflection_vector, -obs_vector)

            if v_dot_r < 0:  # odbicie jest w przeciwną stronę to 0
                v_dot_r = 0

            if n_dot_l > 0:  # jeśli światło pada na widoczną stronę
                color += (
                        (self.diffuse * light.diffuse * n_dot_l) +  # składowa rozproszona
                        (self.specular * light.specular * v_dot_r ** self.shining) +  # składowa zwierciadlana
                        (self.ambient * light.ambient)  # ambient ze światła
                )

            return color

    class Sphere(SceneObject):  # klasa kuli
        def __init__(self, position, radius, ambient=np.array([0, 0, 0]), diffuse=np.array([0.6, 0.7, 0.8]),
                     specular=np.array([0.8, 0.8, 0.8]), shining=25):
            super(Sphere, self).__init__(ambient=ambient, diffuse=diffuse, specular=specular,
                                         shining=shining)  # wywołanie konstruktora
            self.position = position  # pozycja środka kuli
            self.radius = radius  # promień kuli

        def get_normal(self, cross_point):  # wektor normalny na powierzchni kuli
            return normalize(cross_point - self.position)  # wektor od środka do punktu

        def trace(self, ray):  # metoda przecięcia promienia z kulą
            distance = ray.starting_point - self.position  # wektor od środka kuli do początku promienia
            a = np.dot(ray.direction, ray.direction)  # współczynnik a
            b = 2 * np.dot(ray.direction, distance)  # współczynnik b
            c = np.dot(distance, distance) - self.radius ** 2  # współczynnik c
            d = b ** 2 - 4 * a * c  # delta

            if d < 0:  # jeśli ujemna to brak
                return (None, None)

            sqrt_d = d ** (0.5)
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

            look = normalize(look_at - position)  # wektor patrzenia
            self.up = normalize(np.cross(np.cross(look, np.array([0, 1, 0])), look))  # wektor "góry" kamery
            self.position = position  # zapisanie pozycji
            self.look_at = look_at  # zapisanie punktu patrzenia
            self.direction = normalize(look_at - position)  # kierunek patrzenia

            aspect = self.pixel_width / self.pixel_height  # proporcje obrazu
            povy = self.povy * np.pi / 180  # konwersja kąta na radiany
            self.world_height = 2 * np.tan(povy / 2) * self.z_near  # wysokość w świecie
            self.world_width = aspect * self.world_height  # szerokość w świecie

            center = self.position + self.direction * self.z_near  # środek płaszczyzny obrazu
            width_vector = normalize(np.cross(self.up, self.direction))  # wektor szerokości obrazu

            self.translation_vector_x = width_vector * -(self.world_width / self.pixel_width)  # przeskok w poziomie
            self.translation_vector_y = self.up * -(self.world_height / self.pixel_height)  # przeskok w pionie

            self.starting_point = center + width_vector * (self.world_width / 2) + (
                        self.up * self.world_height / 2)  # górny lewy punkt startowy

        def get_world_pixel(self, x, y):  # zwraca pozycję piksela w przestrzeni świata
            return self.starting_point + self.translation_vector_x * x + self.translation_vector_y * y

    class Scene:  # klasa sceny
        def __init__(self, objects, light, camera):
            self.objects = objects  # zapisanie listy obiektów
            self.light = light  # zapisanie światła
            self.camera = camera  # zapisanie kamery
            self.ambient = np.array([0.1, 0.1, 0.1])  # kolor ambient sceny
            self.background = np.array([0, 0, 0])  # kolor tła (czarny)

    class RayTracer:  # klasa raytracera
        def __init__(self, scene, max_depth=3):
            self.scene = scene
            self.scene.raytracer = self  # dodanie raytracera do sceny *
            self.max_depth = max_depth  # maksymalna głębokość odbić

        def generate_image(self):  # generuje obraz
            camera = self.scene.camera
            image = np.zeros((camera.pixel_height, camera.pixel_width, 3))
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    world_pixel = camera.get_world_pixel(x, y)
                    direction = normalize(world_pixel - camera.position)
                    image[y][x] = self._get_pixel_color(Ray(world_pixel, direction))
            return image

        def _get_pixel_color(self, ray, depth=0):  # oblicza kolor piksela
            obj, distance, cross_point = self._get_closest_object(ray)
            if not obj:  # jeśli brak przecięcia
                return self.scene.background

            base_color = obj.get_color(cross_point, ray.direction, self.scene)  # kolor z materiału

            if depth >= self.max_depth:  # ograniczenie głębokości
                return base_color

            normal = obj.get_normal(cross_point)
            reflection_dir = reflect(ray.direction, normal)
            reflection_ray = Ray(cross_point + normal * EPSILON, normalize(reflection_dir))

            reflection_color = self._get_pixel_color(reflection_ray, depth + 1)

            final_color = 0.7 * base_color + 0.3 * reflection_color  # mieszanie kolorów
            return final_color

        def _get_closest_object(self, ray):  # znajdź najbliższy obiekt
            closest = None
            min_distance = np.inf
            min_cross_point = None
            for obj in self.scene.objects:
                cross_point, distance = obj.trace(ray)
                if cross_point is not None and distance < min_distance:
                    min_distance = distance
                    closest = obj
                    min_cross_point = cross_point
            return (closest, min_distance, min_cross_point)

        def is_in_shadow(self, point, light_position):  # czy punkt jest w cieniu *
            direction_to_light = normalize(light_position - point)  # kierunek do światła *
            shadow_ray = Ray(point + direction_to_light * EPSILON,
                             direction_to_light)  # promień cienia z przesunięciem *
            obj, distance, _ = self._get_closest_object(shadow_ray)  # najbliższy obiekt na drodze *
            if obj is None:
                return False
            light_distance = np.linalg.norm(light_position - point)
            return distance < light_distance

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


def lab5_zad4():
    plt.rcParams['figure.figsize'] = [12, 8]

    EPSILON = 0.0001  # do błędów numerycznych przy przecięciach

    def reflect(vector, normal_vector):  # funkcja zwracająca wektor odbity
        n_dot_l = np.dot(vector, normal_vector)  # obliczenie iloczynu skalarnego
        return vector - normal_vector * (2 * n_dot_l)  # zwrócenie odbitego wektora

    def normalize(vector):  # funkcja normalizująca wektor
        return vector / np.sqrt((vector ** 2).sum())  # dzielenie przez długość wektora

    def refract(incoming, normal, ior):  # funkcja załamująca promień
        cos = np.clip(np.dot(-incoming, normal), -1, 1)  # cosinus kąta padania *
        etai = 1  # współczynnik załamania *
        etat = ior  # współczynnik załamania obiektu *
        n = normal
        if cos < 0:  # jeżeli promień wchodzi do obiektu *
            cos = -cos
        else:  # promień wychodzi z obiektu *
            etai, etat = etat, etai
            n = -normal
        eta = etai / etat  # stosunek współczynników *
        k = 1 - eta ** 2 * (1 - cos ** 2)  # sprawdzenie warunku całkowitego wewnętrznego odbicia *
        if k < 0:
            return None  # brak załamania *
        return normalize(eta * incoming + (eta * cos - np.sqrt(k)) * n)  # obliczenie kierunku promienia załamanego *

    class Ray:  # klasa promienia
        def __init__(self, starting_point, direction):
            self.starting_point = starting_point  # punkt początkowy promienia
            self.direction = direction  # kierunek promienia

    class Light:  # klasa światła
        def __init__(self, position):
            self.position = position  # pozycja źródła światła
            self.ambient = np.array([0, 0, 0])  # składowa ambient
            self.diffuse = np.array([0, 1, 1])  # składowa diffuse
            self.specular = np.array([1, 1, 0])  # składowa specular

    class SceneObject:  # klasa bazowa dla obiektów sceny
        def __init__(self, ambient=np.array([0, 0, 0]),
                     diffuse=np.array([0.6, 0.7, 0.8]),
                     specular=np.array([0.8, 0.8, 0.8]),
                     shining=25,
                     transparency=0.0,  # poziom przezroczystości obiektu *
                     refraction_index=1.0):  # współczynnik załamania *
            self.ambient = ambient  # kolor ambient
            self.diffuse = diffuse  # kolor diffuse
            self.specular = specular  # kolor specular
            self.shining = shining  # współczynnik połysku
            self.transparency = transparency  # zapisanie przezroczystości *
            self.refraction_index = refraction_index  # zapisanie współczynnika załamania *

        def get_color(self, cross_point, obs_vector, scene):  # oblicza kolor w punkcie przecięcia
            color = self.ambient * scene.ambient  # kolor z oświetlenia ambient
            light = scene.light  # źródło światła
            normal = self.get_normal(cross_point)  # wektor normalny w punkcie kolizji
            light_vector = normalize(light.position - cross_point)  # wektor do źródła światła

            if scene.raytracer.is_in_shadow(cross_point, light.position):  # sprawdzenie czy punkt jest w cieniu
                return color  # tylko ambient

            n_dot_l = np.dot(light_vector, normal)  # iloczyn skalarny normalnej i wektora do światła
            reflection_vector = normalize(reflect(-1 * light_vector, normal))  # wektor odbicia
            v_dot_r = np.dot(reflection_vector, -obs_vector)  # iloczyn skalarny obserwatora i odbicia

            if v_dot_r < 0:
                v_dot_r = 0  # jeśli odbicie nie widoczne, ustaw na 0

            if n_dot_l > 0:  # jeśli kąt padania sensowny
                color += (
                        (self.diffuse * light.diffuse * n_dot_l) +  # składowa diffuse
                        (self.specular * light.specular * v_dot_r ** self.shining) +  # składowa specular
                        (self.ambient * light.ambient)  # ambient ze światła
                )

            return color

    class Sphere(SceneObject):  # klasa reprezentująca kulę
        def __init__(self, position, radius,
                     ambient=np.array([0, 0, 0]),
                     diffuse=np.array([0.6, 0.7, 0.8]),
                     specular=np.array([0.8, 0.8, 0.8]),
                     shining=25,
                     transparency=0.0,
                     refraction_index=1.0):
            super(Sphere, self).__init__(ambient, diffuse, specular, shining, transparency, refraction_index)
            self.position = position  # środek kuli
            self.radius = radius  # promień kuli

        def get_normal(self, cross_point):  # zwraca normalną do powierzchni
            return normalize(cross_point - self.position)  # normalna to wektor od środka do punktu

        def trace(self, ray):  # sprawdzenie przecięcia promienia z kulą
            distance = ray.starting_point - self.position
            a = np.dot(ray.direction, ray.direction)
            b = 2 * np.dot(ray.direction, distance)
            c = np.dot(distance, distance) - self.radius ** 2
            d = b ** 2 - 4 * a * c

            if d < 0:
                return (None, None)  # brak przecięcia

            sqrt_d = d ** (0.5)
            denominator = 1 / (2 * a)

            if d > 0:
                r1 = (-b - sqrt_d) * denominator
                r2 = (-b + sqrt_d) * denominator
                if r1 < EPSILON:
                    if r2 < EPSILON:
                        return (None, None)
                    r1 = r2
            else:
                r1 = -b * denominator
                if r1 < EPSILON:
                    return (None, None)

            cross_point = ray.starting_point + r1 * ray.direction
            return cross_point, r1  # punkt przecięcia i odległość

    class Camera:  # klasa kamery
        def __init__(self, position=np.array([0, 0, -3]), look_at=np.array([0, 0, 0])):
            self.z_near = 1  # odległość do ekranu
            self.pixel_height = 500  # przywrócona wysokość obrazu *
            self.pixel_width = 700  # przywrócona szerokość obrazu *
            self.povy = 45  # kąt widzenia pionowy

            look = normalize(look_at - position)
            self.up = normalize(np.cross(np.cross(look, np.array([0, 1, 0])), look))
            self.position = position
            self.look_at = look_at
            self.direction = normalize(look_at - position)

            aspect = self.pixel_width / self.pixel_height
            povy = self.povy * np.pi / 180
            self.world_height = 2 * np.tan(povy / 2) * self.z_near
            self.world_width = aspect * self.world_height

            center = self.position + self.direction * self.z_near
            width_vector = normalize(np.cross(self.up, self.direction))
            self.translation_vector_x = width_vector * -(self.world_width / self.pixel_width)
            self.translation_vector_y = self.up * -(self.world_height / self.pixel_height)
            self.starting_point = center + width_vector * (self.world_width / 2) + (self.up * self.world_height / 2)

        def get_world_pixel(self, x, y):  # zwraca współrzędne piksela w przestrzeni 3D
            return self.starting_point + self.translation_vector_x * x + self.translation_vector_y * y

    class Scene:  # klasa sceny
        def __init__(self, objects, light, camera):
            self.objects = objects  # lista obiektów
            self.light = light  # światło
            self.camera = camera  # kamera
            self.ambient = np.array([0.1, 0.1, 0.1])  # oświetlenie ogólne
            self.background = np.array([0, 0, 0])  # kolor tła

    class RayTracer:  # klasa śledząca promienie
        def __init__(self, scene, max_depth=2):  # zmniejszona głębokość rekursji *
            self.scene = scene
            self.scene.raytracer = self
            self.max_depth = max_depth

        def generate_image(self):  # generowanie obrazu
            camera = self.scene.camera
            image = np.zeros((camera.pixel_height, camera.pixel_width, 3))
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    world_pixel = camera.get_world_pixel(x, y)
                    direction = normalize(world_pixel - camera.position)
                    image[y][x] = self._get_pixel_color(Ray(world_pixel, direction))
            return image

        def _get_pixel_color(self, ray, depth=0):  # kolor pojedynczego piksela
            obj, distance, cross_point = self._get_closest_object(ray)
            if not obj:
                return self.scene.background

            base_color = obj.get_color(cross_point, ray.direction, self.scene)

            if depth >= self.max_depth:
                return base_color

            normal = obj.get_normal(cross_point)
            reflection_dir = reflect(ray.direction, normal)
            reflection_ray = Ray(cross_point + normal * EPSILON, normalize(reflection_dir))
            reflection_color = self._get_pixel_color(reflection_ray, depth + 1)

            if obj.transparency > 0:  # jeśli obiekt jest przezroczysty *
                refraction_dir = refract(ray.direction, normal, obj.refraction_index)  # oblicz kierunek załamania *
                if refraction_dir is not None:
                    refraction_ray = Ray(cross_point - normal * EPSILON, refraction_dir)  # nowy promień załamany *
                    refraction_color = self._get_pixel_color(refraction_ray,
                                                             depth + 1)  # kolor z promienia załamanego *
                    return (1 - obj.transparency) * (
                                0.7 * base_color + 0.3 * reflection_color) + obj.transparency * refraction_color  # mieszanie kolorów *

            return 0.7 * base_color + 0.3 * reflection_color

        def _get_closest_object(self, ray):  # najbliższy obiekt przecinany przez promień
            closest = None
            min_distance = np.inf
            min_cross_point = None
            for obj in self.scene.objects:
                cross_point, distance = obj.trace(ray)
                if cross_point is not None and distance < min_distance:
                    min_distance = distance
                    closest = obj
                    min_cross_point = cross_point
            return (closest, min_distance, min_cross_point)

        def is_in_shadow(self, point, light_position):  # sprawdzanie czy punkt jest w cieniu
            direction_to_light = normalize(light_position - point)
            shadow_ray = Ray(point + direction_to_light * EPSILON, direction_to_light)
            obj, distance, _ = self._get_closest_object(shadow_ray)
            if obj is None:
                return False
            light_distance = np.linalg.norm(light_position - point)
            return distance < light_distance

    scene = Scene(
        objects=[
            Sphere(position=np.array([-1.5, -1, 0]), radius=0.5, diffuse=np.array([0.2, 1.0, 0.3]), transparency=0.5,
                   refraction_index=1.3),  # zielona półprzezroczysta kula *
            Sphere(position=np.array([0, 0, 0]), radius=1.0, diffuse=np.array([0.1, 0.3, 0.8]), transparency=0.6,
                   refraction_index=1.5)  # przezroczysta niebieska kula *
        ],
        light=Light(position=np.array([3, 2, 5])),  # pozycja światła
        camera=Camera(position=np.array([0, 0, 5]))  # pozycja kamery
    )

    rt = RayTracer(scene)  # utworzenie raytracera
    image = np.clip(rt.generate_image(), 0, 1)  # generowanie obrazu i ograniczenie wartosci RGB
    plt.imshow(image)
    plt.show()


def lab5_zad5():
    plt.rcParams['figure.figsize'] = [12, 8]

    EPSILON = 0.0001  # do błędów numerycznych przy przecięciach

    def reflect(vector, normal_vector):  # funkcja zwracająca wektor odbity
        n_dot_l = np.dot(vector, normal_vector)  # obliczenie iloczynu skalarnego
        return vector - normal_vector * (2 * n_dot_l)  # zwrócenie odbitego wektora

    def normalize(vector):  # funkcja normalizująca wektor
        return vector / np.sqrt((vector ** 2).sum())  # dzielenie przez długość wektora

    def refract(incoming, normal, ior):  # funkcja załamująca promień wg prawa Snelliusa *
        cosi = np.clip(np.dot(-incoming, normal), -1, 1)  # cosinus kąta padania
        etai = 1  # współczynnik załamania powietrza
        etat = ior  # współczynnik załamania obiektu
        n = normal
        if cosi < 0:  # promień wchodzi do obiektu
            cosi = -cosi
        else:  # promień wychodzi z obiektu
            etai, etat = etat, etai
            n = -normal
        eta = etai / etat  # stosunek współczynników
        k = 1 - eta ** 2 * (1 - cosi ** 2)  # sprawdzenie warunku całkowitego wewnętrznego odbicia
        if k < 0:
            return None  # brak załamania *
        return normalize(eta * incoming + (eta * cosi - np.sqrt(k)) * n)  # obliczenie kierunku promienia załamanego

    class Ray:  # klasa promienia
        def __init__(self, starting_point, direction):
            self.starting_point = starting_point  # punkt początkowy promienia
            self.direction = direction  # kierunek promienia

    class Light:  # klasa światła
        def __init__(self, position):
            self.position = position  # pozycja źródła światła
            self.ambient = np.array([0, 0, 0])  # składowa ambient
            self.diffuse = np.array([0, 1, 1])  # składowa diffuse
            self.specular = np.array([1, 1, 0])  # składowa specular

    class SceneObject:  # klasa bazowa dla obiektów sceny
        def __init__(self, ambient=np.array([0, 0, 0]),
                     diffuse=np.array([0.6, 0.7, 0.8]),
                     specular=np.array([0.8, 0.8, 0.8]),
                     shining=25,
                     transparency=0.0,  # poziom przezroczystości obiektu
                     refraction_index=1.0):  # współczynnik załamania
            self.ambient = ambient  # kolor ambient
            self.diffuse = diffuse  # kolor diffuse
            self.specular = specular  # kolor specular
            self.shining = shining  # współczynnik połysku
            self.transparency = transparency  # zapisanie przezroczystości
            self.refraction_index = refraction_index  # zapisanie współczynnika załamania

        def get_color(self, cross_point, obs_vector, scene):  # oblicza kolor w punkcie przecięcia
            color = self.ambient * scene.ambient  # kolor z oświetlenia ambient
            light = scene.light  # źródło światła
            normal = self.get_normal(cross_point)  # wektor normalny w punkcie kolizji
            light_vector = normalize(light.position - cross_point)  # wektor do źródła światła

            if scene.raytracer.is_in_shadow(cross_point, light.position):  # sprawdzenie czy punkt jest w cieniu
                return color  # tylko ambient

            n_dot_l = np.dot(light_vector, normal)  # iloczyn skalarny normalnej i wektora do światła
            reflection_vector = normalize(reflect(-1 * light_vector, normal))  # wektor odbicia
            v_dot_r = np.dot(reflection_vector, -obs_vector)  # iloczyn skalarny obserwatora i odbicia

            if v_dot_r < 0:
                v_dot_r = 0  # jeśli odbicie nie widoczne, ustaw na 0

            if n_dot_l > 0:  # jeśli kąt padania sensowny
                color += (
                        (self.diffuse * light.diffuse * n_dot_l) +  # składowa diffuse
                        (self.specular * light.specular * v_dot_r ** self.shining) +  # składowa specular
                        (self.ambient * light.ambient)  # ambient ze światła
                )

            return color

    class Sphere(SceneObject):  # klasa reprezentująca kulę
        def __init__(self, position, radius,
                     ambient=np.array([0, 0, 0]),
                     diffuse=np.array([0.6, 0.7, 0.8]),
                     specular=np.array([0.8, 0.8, 0.8]),
                     shining=25,
                     transparency=0.0,
                     refraction_index=1.0):
            super(Sphere, self).__init__(ambient, diffuse, specular, shining, transparency, refraction_index)
            self.position = position  # środek kuli
            self.radius = radius  # promień kuli

        def get_normal(self, cross_point):  # zwraca normalną do powierzchni
            return normalize(cross_point - self.position)  # normalna to wektor od środka do punktu

        def trace(self, ray):  # sprawdzenie przecięcia promienia z kulą
            distance = ray.starting_point - self.position
            a = np.dot(ray.direction, ray.direction)
            b = 2 * np.dot(ray.direction, distance)
            c = np.dot(distance, distance) - self.radius ** 2
            d = b ** 2 - 4 * a * c

            if d < 0:
                return (None, None)  # brak przecięcia

            sqrt_d = d ** (0.5)
            denominator = 1 / (2 * a)

            if d > 0:
                r1 = (-b - sqrt_d) * denominator
                r2 = (-b + sqrt_d) * denominator
                if r1 < EPSILON:
                    if r2 < EPSILON:
                        return (None, None)
                    r1 = r2
            else:
                r1 = -b * denominator
                if r1 < EPSILON:
                    return (None, None)

            cross_point = ray.starting_point + r1 * ray.direction
            return cross_point, r1  # punkt przecięcia i odległość

    class Triangle(SceneObject):  # klasa reprezentująca trójkąt *
        def __init__(self, v0, v1, v2, **kwargs):
            super().__init__(**kwargs)
            self.v0 = v0  # pierwszy wierzchołek *
            self.v1 = v1  # drugi wierzchołek *
            self.v2 = v2  # trzeci wierzchołek *
            self.normal = normalize(np.cross(v1 - v0, v2 - v0))  # normalna trójkąta *

        def get_normal(self, cross_point):
            return self.normal  # zwracamy stałą normalną dla trójkąta *

        def trace(self, ray):  # sprawdzanie przecięcia promienia z trójkątem *
            edge1 = self.v1 - self.v0
            edge2 = self.v2 - self.v0
            h = np.cross(ray.direction, edge2)
            a = np.dot(edge1, h)
            if -EPSILON < a < EPSILON:
                return (None, None)  # promień równoległy do trójkąta *
            f = 1.0 / a
            s = ray.starting_point - self.v0
            u = f * np.dot(s, h)
            if u < 0.0 or u > 1.0:
                return (None, None)
            q = np.cross(s, edge1)
            v = f * np.dot(ray.direction, q)
            if v < 0.0 or u + v > 1.0:
                return (None, None)
            t = f * np.dot(edge2, q)
            if t > EPSILON:
                cross_point = ray.starting_point + ray.direction * t
                return cross_point, t  # punkt przecięcia i odległość *
            else:
                return (None, None)

    class Camera:  # klasa kamery
        def __init__(self, position=np.array([0, 0, -3]), look_at=np.array([0, 0, 0])):
            self.z_near = 1  # odległość do ekranu
            self.pixel_height = 500  # przywrócona wysokość obrazu
            self.pixel_width = 700  # przywrócona szerokość obrazu
            self.povy = 45  # kąt widzenia pionowy

            look = normalize(look_at - position)
            self.up = normalize(np.cross(np.cross(look, np.array([0, 1, 0])), look))
            self.position = position
            self.look_at = look_at
            self.direction = normalize(look_at - position)

            aspect = self.pixel_width / self.pixel_height
            povy = self.povy * np.pi / 180
            self.world_height = 2 * np.tan(povy / 2) * self.z_near
            self.world_width = aspect * self.world_height

            center = self.position + self.direction * self.z_near
            width_vector = normalize(np.cross(self.up, self.direction))
            self.translation_vector_x = width_vector * -(self.world_width / self.pixel_width)
            self.translation_vector_y = self.up * -(self.world_height / self.pixel_height)
            self.starting_point = center + width_vector * (self.world_width / 2) + (self.up * self.world_height / 2)

        def get_world_pixel(self, x, y):  # zwraca współrzędne piksela w przestrzeni 3D
            return self.starting_point + self.translation_vector_x * x + self.translation_vector_y * y

    class Scene:  # klasa sceny
        def __init__(self, objects, light, camera):
            self.objects = objects  # lista obiektów
            self.light = light  # światło
            self.camera = camera  # kamera
            self.ambient = np.array([0.1, 0.1, 0.1])  # oświetlenie ogólne
            self.background = np.array([0, 0, 0])  # kolor tła

    class RayTracer:  # klasa śledząca promienie
        def __init__(self, scene, max_depth=2):  # zmniejszona głębokość rekursji
            self.scene = scene
            self.scene.raytracer = self
            self.max_depth = max_depth

        def generate_image(self):  # generowanie obrazu
            camera = self.scene.camera
            image = np.zeros((camera.pixel_height, camera.pixel_width, 3))
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    world_pixel = camera.get_world_pixel(x, y)
                    direction = normalize(world_pixel - camera.position)
                    image[y][x] = self._get_pixel_color(Ray(world_pixel, direction))
            return image

        def _get_pixel_color(self, ray, depth=0):  # kolor pojedynczego piksela
            obj, distance, cross_point = self._get_closest_object(ray)
            if not obj:
                return self.scene.background

            base_color = obj.get_color(cross_point, ray.direction, self.scene)

            if depth >= self.max_depth:
                return base_color

            normal = obj.get_normal(cross_point)
            reflection_dir = reflect(ray.direction, normal)
            reflection_ray = Ray(cross_point + normal * EPSILON, normalize(reflection_dir))
            reflection_color = self._get_pixel_color(reflection_ray, depth + 1)

            if obj.transparency > 0:  # jeśli obiekt jest przezroczysty
                refraction_dir = refract(ray.direction, normal, obj.refraction_index)  # oblicz kierunek załamania
                if refraction_dir is not None:
                    refraction_ray = Ray(cross_point - normal * EPSILON, refraction_dir)  # nowy promień załamany
                    refraction_color = self._get_pixel_color(refraction_ray, depth + 1)  # kolor z promienia załamanego
                    return (1 - obj.transparency) * (
                                0.7 * base_color + 0.3 * reflection_color) + obj.transparency * refraction_color  # mieszanie kolorów

            return 0.7 * base_color + 0.3 * reflection_color

        def _get_closest_object(self, ray):  # najbliższy obiekt przecinany przez promień
            closest = None
            min_distance = np.inf
            min_cross_point = None
            for obj in self.scene.objects:
                cross_point, distance = obj.trace(ray)
                if cross_point is not None and distance < min_distance:
                    min_distance = distance
                    closest = obj
                    min_cross_point = cross_point
            return (closest, min_distance, min_cross_point)

        def is_in_shadow(self, point, light_position):  # sprawdzanie czy punkt jest w cieniu
            direction_to_light = normalize(light_position - point)
            shadow_ray = Ray(point + direction_to_light * EPSILON, direction_to_light)
            obj, distance, _ = self._get_closest_object(shadow_ray)
            if obj is None:
                return False
            light_distance = np.linalg.norm(light_position - point)
            return distance < light_distance

    scene = Scene(
        objects=[
            Sphere(position=np.array([-1.5, -1, 0]), radius=0.5, diffuse=np.array([0.2, 1.0, 0.3]), transparency=0.5,
                   refraction_index=1.3),  # półprzezroczysta kula
            Sphere(position=np.array([0, 0, 0]), radius=1.0, diffuse=np.array([0.1, 0.3, 0.8]), transparency=0.6,
                   refraction_index=1.5),  # przezroczysta kula
            Triangle(np.array([1.5, -0.5, -1]), np.array([2.5, -0.5, -1]), np.array([2.0, 0.5, -1]),
                     diffuse=np.array([1.0, 0.7, 0.2]))  # dodatkowy trójkąt *
        ],
        light=Light(position=np.array([3, 2, 5])),  # pozycja światła
        camera=Camera(position=np.array([0, 0, 5]))  # pozycja kamery
    )

    rt = RayTracer(scene)  # utworzenie raytracera
    image = np.clip(rt.generate_image(), 0, 1)  # generowanie obrazu i ograniczenie wartosci RGB
    plt.imshow(image)  # wyświetlenie obrazu
    plt.show()


def menu():
    while True:
        print("\n=== LABORATORIUM 5 ===")
        print("1. Zadanie 1 - Przykladowa scena ze sferami")
        print("2. Zadanie 2 - Promienie wtorne")
        print("3. Zadanie 3 - Obsluga cieni")
        print("4. Zadanie 4 - Obsluga obiektow (pol)przezroczystych")
        print("5. Zadanie 5 - Wprowadzenie kolejnego elementu innego niz sfera")
        print("6. Wyjście")

        choice = input("Wybierz opcję (1-6): ")

        if choice == "1":
            lab5_zad1()
        elif choice == "2":
            lab5_zad2()
        elif choice == "3":
            lab5_zad3()
        elif choice == "4":
            lab5_zad4()
        elif choice == "5":
            lab5_zad5()
        elif choice == "6":
            print("Wyjście z programu.")
            break
        else:
            print("Nieprawidłowa opcja. Spróbuj ponownie.")

if __name__ == "__main__":
    menu()