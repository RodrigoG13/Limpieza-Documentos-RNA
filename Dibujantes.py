from abc import ABC, abstractmethod
import random
import math
from PIL import Image, ImageDraw, ImageFont
import os
import string

class Dibujante(ABC):

    @abstractmethod
    def rayar(self, ruta):
        pass


class DibujanteGeometrico(Dibujante):
    def __init__(self, color='red', num_formas=5, tam_min=5, tam_max=15):
        self.color = color
        self.num_formas = num_formas
        self.tam_min = tam_min
        self.tam_max = tam_max

    def rayar(self, ruta):
        with Image.open(ruta) as img:
            draw = ImageDraw.Draw(img)
            width, height = img.size
            for _ in range(self.num_formas):
                shape_type = random.choice(['circle', 'triangle'])
                if shape_type == 'circle':
                    self._draw_circle(draw, width, height)
                elif shape_type == 'rectangle':
                    self._draw_rectangle(draw, width, height)
                elif shape_type == 'triangle':
                    self._draw_triangle(draw, width, height)
            return img

    def _draw_circle(self, draw, width, height):
        x, y = random.randint(0, width), random.randint(0, height)
        radius = random.randint(self.tam_min, min(self.tam_max, min(width, height) // 4))
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=self.color)

    def _draw_rectangle(self, draw, width, height):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(x1 + self.tam_min, min(x1 + self.tam_max, width)), random.randint(y1 + self.tam_min, min(y1 + self.tam_max, height))
        draw.rectangle((x1, y1, x2, y2), fill=self.color)

    def _draw_triangle(self, draw, width, height):
        cx, cy = random.randint(0, width), random.randint(0, height)
        max_side = random.randint(self.tam_min, self.tam_max)
        angle = random.uniform(0, 2 * math.pi)

        points = []
        for i in range(3):
            theta = angle + i * 2 * math.pi / 3
            x = cx + max_side * math.cos(theta)
            y = cy + max_side * math.sin(theta)
            x = min(max(0, x), width)
            y = min(max(0, y), height)
            points.append((x, y))

        draw.polygon(points, fill=self.color)

class DibujanteEscritor(Dibujante):
    def __init__(self, folder_path='fonts', rotmin=1, rotmax=360):
        self.folder_path = folder_path
        self.font_size = random.randrange(12, 40)
        self.text_rotation = random.randrange(rotmin, rotmax)

    def _get_random_font(self):
        fonts = [f for f in os.listdir(self.folder_path) if f.endswith('.ttf')]
        return os.path.join(self.folder_path, random.choice(fonts))

    def rayar(self, ruta):
        text = self._generar_cadena_aleatoria(random.randrange(10, 60))
        image = Image.open(ruta)
        font_path = self._get_random_font()
        print(font_path, self.text_rotation)
        font = ImageFont.truetype(font_path, self.font_size)
        width, height = image.size

        temp_image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp_image)
        
        temp_draw.text((0, 0), text, font=font, fill=(255, 0, 0))
        rotated_text = temp_image.rotate(self.text_rotation, expand=1)
        rotated_bbox = rotated_text.getbbox()
        cropped_rotated_text = rotated_text.crop(rotated_bbox)
        x, y = random.randint(0, max(1, width-30)), random.randint(0, max(1, height-30))
        image.paste(cropped_rotated_text, (x, y), cropped_rotated_text)

        return image
    
    def _generar_cadena_aleatoria(self,longitud):
        caracteres = string.ascii_letters + string.digits  
        cadena_aleatoria = ''.join(random.choice(caracteres) for _ in range(longitud))
        return cadena_aleatoria

class DibujanteEscritorDeLinea(Dibujante):
    def __init__(self, folder_path='fonts', rotmin=0, rotmax=1):
        self.folder_path = folder_path
        self.font_size = random.randrange(12, 30)
        self.text_rotation = random.randrange(rotmin, rotmax)

    def _get_random_font(self):
        fonts = [f for f in os.listdir(self.folder_path) if f.endswith('.ttf')]
        return os.path.join(self.folder_path, random.choice(fonts))

    def rayar(self, ruta):
        text = self._generar_cadena_aleatoria(random.randrange(10, 60))
        image = Image.open(ruta)
        font_path = self._get_random_font()
        print(font_path, self.text_rotation)
        font = ImageFont.truetype(font_path, self.font_size)
        width, height = image.size

        temp_image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp_image)
        
        temp_draw.text((0, 0), text, font=font, fill=(255, 0, 0))
        rotated_text = temp_image.rotate(self.text_rotation, expand=1)
        rotated_bbox = rotated_text.getbbox()
        cropped_rotated_text = rotated_text.crop(rotated_bbox)
        x, y = random.randint(0, max(1, width-30)), random.randint(0, max(1, height-30))
        image.paste(cropped_rotated_text, (x, y), cropped_rotated_text)

        return image
    
    def _generar_cadena_aleatoria(self,longitud):
        caracteres = string.ascii_letters + string.digits  
        cadena_aleatoria = ''.join(random.choice(caracteres) for _ in range(longitud))
        return cadena_aleatoria

class DibujanteGenesis(Dibujante):
    def __init__(self):
        self.longitud_linea = random.randrange(500, 3000)
        self.rotacion_max = random.randrange(5, 20)
        self.grosor_linea = random.randrange(1, 3)
        
    def rayar(self, ruta):
        self.imagen = Image.open(ruta)
        self.ancho, self.alto = self.imagen.size
        draw = ImageDraw.Draw(self.imagen)
        
        x, y = random.randint(10, self.ancho), random.randint(10, self.alto)
        angulo = random.uniform(0, 360)

        for _ in range(self.longitud_linea):
            x_nuevo = x + math.cos(math.radians(angulo))
            y_nuevo = y + math.sin(math.radians(angulo))

            x_nuevo = min(max(0, x_nuevo), self.ancho - 1)
            y_nuevo = min(max(0, y_nuevo), self.alto - 1)

            draw.line((x, y, x_nuevo, y_nuevo), fill='red', width=self.grosor_linea)

            x, y = x_nuevo, y_nuevo
            angulo += random.uniform(-self.rotacion_max, self.rotacion_max)
        return self.imagen

class DibujanteSazon(Dibujante):
    def __init__(self, num_puntos=40, tam_min=2, tam_max=4):
        self.num_puntos = num_puntos
        self.tam_min = tam_min
        self.tam_max = tam_max

    def rayar(self, ruta):
        with Image.open(ruta) as img:
            draw = ImageDraw.Draw(img)
            width, height = img.size

            for _ in range(self.num_puntos):
                x = random.randint(0, width)
                y = random.randint(0, height)
                r = random.randint(self.tam_min, self.tam_max)  
                draw.ellipse((x-r, y-r, x+r, y+r), fill='red', outline='red')

            return img


if __name__ == '__main__':
    
    #PRUEBAS
    ruta_imagen = "capturas\9.png"
    josue = DibujanteGenesis()
    urias = DibujanteEscritor()
    pushy = DibujanteGeometrico()
    fst = DibujanteSazon()
    
    lista = [josue,urias,fst,pushy]
    dibujante = random.choice(lista)
    #dibujante = urias
    

    dibujante.rayar(ruta_imagen).show()
