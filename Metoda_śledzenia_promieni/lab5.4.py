import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [12, 8]

EPSILON = 0.0001  # do błędów numerycznych przy przecięciach

def reflect(vector, normal_vector):  # funkcja zwracająca wektor odbity
    n_dot_l = np.dot(vector, normal_vector)  # obliczenie iloczynu skalarnego
    return vector - normal_vector * (2 * n_dot_l)  # zwrócenie odbitego wektora

def normalize(vector):  # funkcja normalizująca wektor
    return vector / np.sqrt((vector**2).sum())  # dzielenie przez długość wektora

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
    k = 1 - eta**2 * (1 - cos**2)  # sprawdzenie warunku całkowitego wewnętrznego odbicia *
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
                (self.specular * light.specular * v_dot_r**self.shining) +  # składowa specular
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
        c = np.dot(distance, distance) - self.radius**2
        d = b**2 - 4*a*c

        if d < 0:
            return (None, None)  # brak przecięcia

        sqrt_d = d**(0.5)
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
        self.world_height = 2 * np.tan(povy/2) * self.z_near
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
                refraction_color = self._get_pixel_color(refraction_ray, depth + 1)  # kolor z promienia załamanego *
                return (1 - obj.transparency) * (0.7 * base_color + 0.3 * reflection_color) + obj.transparency * refraction_color  # mieszanie kolorów *

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
        Sphere(position=np.array([-1.5, -1, 0]), radius=0.5, diffuse=np.array([0.2, 1.0, 0.3]), transparency=0.5, refraction_index=1.3),  # zielona półprzezroczysta kula *
        Sphere(position=np.array([0, 0, 0]), radius=1.0, diffuse=np.array([0.1, 0.3, 0.8]), transparency=0.6, refraction_index=1.5)  # przezroczysta niebieska kula *
    ],
    light=Light(position=np.array([3, 2, 5])),  # pozycja światła
    camera=Camera(position=np.array([0, 0, 5]))  # pozycja kamery
)

rt = RayTracer(scene)  # utworzenie raytracera
image = np.clip(rt.generate_image(), 0, 1)  # generowanie obrazu i ograniczenie wartosci RGB
plt.imshow(image)  # wyświetlenie obrazu
plt.show()
