"""
Синтетическая генерация данных для обучения детекторов меток
и дообучения NLI моделей
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import random
from typing import List, Dict, Tuple, Optional
import json
import os
from PIL import Image, ImageDraw, ImageFont
import cv2

class GeometricGenerator:
    """Генератор синтетических геометрических чертежей"""
    
    def __init__(self, image_size: Tuple[int, int] = (800, 600)):
        self.image_size = image_size
        self.font_size = 20
        
    def generate_triangle(self, vertices: List[Tuple[float, float]], 
                         labels: List[str] = None) -> Tuple[Image.Image, Dict]:
        """Генерация треугольника с метками"""
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlim(0, self.image_size[0])
        ax.set_ylim(0, self.image_size[1])
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Рисуем треугольник
        triangle = plt.Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        
        # Добавляем точки
        for i, (x, y) in enumerate(vertices):
            ax.plot(x, y, 'ko', markersize=8)
            label = labels[i] if labels and i < len(labels) else chr(65 + i)
            ax.text(x + 10, y + 10, label, fontsize=self.font_size, fontweight='bold')
        
        # Добавляем углы
        self._add_angle_marks(ax, vertices)
        
        # Добавляем равенства сторон
        self._add_equality_marks(ax, vertices)
        
        # Добавляем перпендикуляры
        self._add_perpendicular_marks(ax, vertices)
        
        # Сохраняем в PIL Image
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = Image.fromarray(image_array)
        
        plt.close(fig)
        
        # Метаданные
        metadata = {
            "type": "triangle",
            "vertices": vertices,
            "labels": labels or [chr(65 + i) for i in range(len(vertices))],
            "features": self._extract_features(vertices)
        }
        
        return image, metadata
    
    def generate_rectangle(self, corners: List[Tuple[float, float]], 
                          labels: List[str] = None) -> Tuple[Image.Image, Dict]:
        """Генерация прямоугольника с метками"""
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlim(0, self.image_size[0])
        ax.set_ylim(0, self.image_size[1])
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Рисуем прямоугольник
        rect = plt.Rectangle(corners[0], 
                           corners[1][0] - corners[0][0], 
                           corners[1][1] - corners[0][1],
                           fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Добавляем точки
        for i, (x, y) in enumerate(corners):
            ax.plot(x, y, 'ko', markersize=8)
            label = labels[i] if labels and i < len(labels) else chr(65 + i)
            ax.text(x + 10, y + 10, label, fontsize=self.font_size, fontweight='bold')
        
        # Добавляем прямые углы
        self._add_right_angle_marks(ax, corners)
        
        # Добавляем параллели
        self._add_parallel_marks(ax, corners)
        
        # Сохраняем в PIL Image
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = Image.fromarray(image_array)
        
        plt.close(fig)
        
        metadata = {
            "type": "rectangle",
            "corners": corners,
            "labels": labels or [chr(65 + i) for i in range(len(corners))],
            "features": self._extract_rectangle_features(corners)
        }
        
        return image, metadata
    
    def generate_circle(self, center: Tuple[float, float], radius: float,
                       labels: List[str] = None) -> Tuple[Image.Image, Dict]:
        """Генерация окружности с метками"""
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlim(0, self.image_size[0])
        ax.set_ylim(0, self.image_size[1])
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Рисуем окружность
        circle = Circle(center, radius, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Добавляем центр
        ax.plot(center[0], center[1], 'ko', markersize=8)
        ax.text(center[0] + 10, center[1] + 10, 'O', fontsize=self.font_size, fontweight='bold')
        
        # Добавляем радиус
        radius_point = (center[0] + radius, center[1])
        ax.plot(radius_point[0], radius_point[1], 'ko', markersize=6)
        ax.text(radius_point[0] + 10, radius_point[1] + 10, 'A', fontsize=self.font_size, fontweight='bold')
        
        # Рисуем радиус
        ax.plot([center[0], radius_point[0]], [center[1], radius_point[1]], 
                'k--', linewidth=1, alpha=0.7)
        
        # Добавляем диаметр
        diameter_end = (center[0] - radius, center[1])
        ax.plot(diameter_end[0], diameter_end[1], 'ko', markersize=6)
        ax.text(diameter_end[0] - 20, diameter_end[1] + 10, 'B', fontsize=self.font_size, fontweight='bold')
        ax.plot([radius_point[0], diameter_end[0]], [radius_point[1], diameter_end[1]], 
                'k-', linewidth=2)
        
        # Сохраняем в PIL Image
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = Image.fromarray(image_array)
        
        plt.close(fig)
        
        metadata = {
            "type": "circle",
            "center": center,
            "radius": radius,
            "labels": labels or ['O', 'A', 'B'],
            "features": self._extract_circle_features(center, radius)
        }
        
        return image, metadata
    
    def _add_angle_marks(self, ax, vertices: List[Tuple[float, float]]):
        """Добавление меток углов"""
        for i in range(len(vertices)):
            # Простые метки углов
            x, y = vertices[i]
            ax.text(x - 15, y - 15, f'α{i}', fontsize=14, color='red')
    
    def _add_equality_marks(self, ax, vertices: List[Tuple[float, float]]):
        """Добавление меток равенства сторон"""
        # Добавляем штрихи для равных сторон
        for i in range(len(vertices)):
            start = vertices[i]
            end = vertices[(i + 1) % len(vertices)]
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            # Добавляем штрихи
            ax.plot([mid_x - 5, mid_x + 5], [mid_y - 5, mid_y + 5], 'k-', linewidth=2)
    
    def _add_perpendicular_marks(self, ax, vertices: List[Tuple[float, float]]):
        """Добавление меток перпендикуляров"""
        # Добавляем квадратики для прямых углов
        for i in range(len(vertices)):
            if i == 0:  # Только для первого угла
                x, y = vertices[i]
                square = Rectangle((x - 8, y - 8), 16, 16, 
                                  fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(square)
    
    def _add_right_angle_marks(self, ax, corners: List[Tuple[float, float]]):
        """Добавление меток прямых углов для прямоугольника"""
        for i in range(len(corners)):
            x, y = corners[i]
            square = Rectangle((x - 8, y - 8), 16, 16, 
                              fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(square)
    
    def _add_parallel_marks(self, ax, corners: List[Tuple[float, float]]):
        """Добавление меток параллелей"""
        # Добавляем стрелки для параллельных сторон
        for i in range(0, len(corners), 2):
            if i + 1 < len(corners):
                start = corners[i]
                end = corners[i + 1]
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                
                # Стрелки для параллелей
                ax.annotate('', xy=(end[0], end[1]), xytext=(start[0], start[1]),
                           arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    def _extract_features(self, vertices: List[Tuple[float, float]]) -> Dict:
        """Извлечение геометрических признаков"""
        features = {
            "has_perpendicular": True,  # Есть прямой угол
            "has_parallel": False,      # Нет параллелей
            "has_equal_sides": True,    # Есть равные стороны
            "angle_count": len(vertices),
            "is_triangle": len(vertices) == 3
        }
        return features
    
    def _extract_rectangle_features(self, corners: List[Tuple[float, float]]) -> Dict:
        """Извлечение признаков прямоугольника"""
        features = {
            "has_perpendicular": True,  # Все углы прямые
            "has_parallel": True,      # Противоположные стороны параллельны
            "has_equal_sides": True,   # Противоположные стороны равны
            "angle_count": 4,
            "is_rectangle": True
        }
        return features
    
    def _extract_circle_features(self, center: Tuple[float, float], radius: float) -> Dict:
        """Извлечение признаков окружности"""
        features = {
            "has_perpendicular": False,
            "has_parallel": False,
            "has_equal_sides": True,    # Все радиусы равны
            "angle_count": 0,
            "is_circle": True,
            "radius": radius
        }
        return features

class PhysicsGenerator:
    """Генератор синтетических физических экспериментов"""
    
    def __init__(self, image_size: Tuple[int, int] = (800, 600)):
        self.image_size = image_size
        self.font_size = 16
    
    def generate_liquid_experiment(self, num_containers: int = 2) -> Tuple[Image.Image, Dict]:
        """Генерация эксперимента с жидкостями"""
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlim(0, self.image_size[0])
        ax.set_ylim(0, self.image_size[1])
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Параметры контейнеров
        container_width = 80
        container_height = 200
        container_spacing = 150
        
        containers = []
        liquid_levels = []
        
        for i in range(num_containers):
            x = 100 + i * container_spacing
            y = 100
            
            # Рисуем контейнер
            container = Rectangle((x, y), container_width, container_height,
                               fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(container)
            
            # Уровень жидкости (случайный)
            liquid_height = random.uniform(50, container_height - 20)
            liquid_level = y + liquid_height
            
            # Рисуем жидкость
            liquid = Rectangle((x + 2, y + 2), container_width - 4, liquid_height - 4,
                              facecolor='lightblue', alpha=0.7, edgecolor='blue', linewidth=1)
            ax.add_patch(liquid)
            
            # Горизонтальная линия уровня
            ax.axhline(y=liquid_level, xmin=x/self.image_size[0], 
                      xmax=(x + container_width)/self.image_size[0], 
                      color='blue', linewidth=2)
            
            containers.append((x, y, container_width, container_height))
            liquid_levels.append(liquid_level)
            
            # Подписи
            ax.text(x + container_width/2, y - 30, f'Сосуд {i+1}', 
                   fontsize=self.font_size, ha='center')
            ax.text(x + container_width/2, liquid_level + 10, f'{liquid_height:.0f} мм', 
                   fontsize=12, ha='center', color='blue')
        
        # Соединяем сосуды трубкой (если сообщающиеся)
        if num_containers > 1:
            tube_y = max(liquid_levels) + 50
            ax.plot([containers[0][0] + containers[0][2], 
                    containers[1][0]], [tube_y, tube_y], 
                   'k-', linewidth=3)
        
        # Сохраняем в PIL Image
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = Image.fromarray(image_array)
        
        plt.close(fig)
        
        metadata = {
            "type": "liquid_experiment",
            "num_containers": num_containers,
            "liquid_levels": liquid_levels,
            "containers": containers,
            "features": self._extract_liquid_features(liquid_levels)
        }
        
        return image, metadata
    
    def generate_sphere_experiment(self, num_spheres: int = 3) -> Tuple[Image.Image, Dict]:
        """Генерация эксперимента с шариками в жидкости"""
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlim(0, self.image_size[0])
        ax.set_ylim(0, self.image_size[1])
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Контейнер
        container_width = 400
        container_height = 300
        container_x = 200
        container_y = 100
        
        container = Rectangle((container_x, container_y), container_width, container_height,
                            fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(container)
        
        # Жидкость
        liquid_height = container_height - 50
        liquid = Rectangle((container_x + 2, container_y + 2), 
                         container_width - 4, liquid_height - 4,
                         facecolor='lightblue', alpha=0.7, edgecolor='blue', linewidth=1)
        ax.add_patch(liquid)
        
        # Шарики
        spheres = []
        for i in range(num_spheres):
            radius = random.uniform(15, 25)
            x = container_x + 50 + i * 100
            y = container_y + liquid_height - 30 - i * 20
            
            # Рисуем шарик
            sphere = Circle((x, y), radius, facecolor='red', alpha=0.8, 
                           edgecolor='darkred', linewidth=2)
            ax.add_patch(sphere)
            
            # Номер шарика
            ax.text(x, y, str(i+1), fontsize=14, ha='center', va='center', 
                   color='white', fontweight='bold')
            
            spheres.append((x, y, radius))
        
        # Уровень жидкости
        liquid_level = container_y + liquid_height
        ax.axhline(y=liquid_level, xmin=container_x/self.image_size[0], 
                  xmax=(container_x + container_width)/self.image_size[0], 
                  color='blue', linewidth=2)
        
        # Сохраняем в PIL Image
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = Image.fromarray(image_array)
        
        plt.close(fig)
        
        metadata = {
            "type": "sphere_experiment",
            "num_spheres": num_spheres,
            "spheres": spheres,
            "liquid_level": liquid_level,
            "features": self._extract_sphere_features(spheres, liquid_level)
        }
        
        return image, metadata
    
    def _extract_liquid_features(self, liquid_levels: List[float]) -> Dict:
        """Извлечение признаков жидкости"""
        features = {
            "has_liquid_levels": True,
            "num_levels": len(liquid_levels),
            "level_differences": [abs(liquid_levels[i] - liquid_levels[i+1]) 
                                for i in range(len(liquid_levels)-1)],
            "is_communicating": len(liquid_levels) > 1
        }
        return features
    
    def _extract_sphere_features(self, spheres: List[Tuple[float, float, float]], 
                               liquid_level: float) -> Dict:
        """Извлечение признаков шариков"""
        features = {
            "num_spheres": len(spheres),
            "spheres_above_liquid": sum(1 for _, y, _ in spheres if y > liquid_level),
            "spheres_below_liquid": sum(1 for _, y, _ in spheres if y < liquid_level),
            "has_spheres": len(spheres) > 0
        }
        return features

class SyntheticDatasetGenerator:
    """Генератор синтетических датасетов для обучения"""
    
    def __init__(self, output_dir: str = "synthetic_data"):
        self.output_dir = output_dir
        self.geometric_generator = GeometricGenerator()
        self.physics_generator = PhysicsGenerator()
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/annotations", exist_ok=True)
    
    def generate_geometric_dataset(self, num_samples: int = 100) -> str:
        """Генерация геометрического датасета"""
        
        dataset = []
        
        for i in range(num_samples):
            # Случайный выбор типа фигуры
            figure_type = random.choice(['triangle', 'rectangle', 'circle'])
            
            if figure_type == 'triangle':
                # Случайный треугольник
                vertices = [
                    (random.uniform(100, 300), random.uniform(100, 300)),
                    (random.uniform(400, 600), random.uniform(100, 300)),
                    (random.uniform(250, 450), random.uniform(400, 500))
                ]
                image, metadata = self.geometric_generator.generate_triangle(vertices)
                
            elif figure_type == 'rectangle':
                # Случайный прямоугольник
                x1, y1 = random.uniform(100, 200), random.uniform(100, 200)
                x2, y2 = x1 + random.uniform(200, 400), y1 + random.uniform(150, 300)
                corners = [(x1, y1), (x2, y2)]
                image, metadata = self.geometric_generator.generate_rectangle(corners)
                
            else:  # circle
                # Случайная окружность
                center = (random.uniform(200, 600), random.uniform(200, 400))
                radius = random.uniform(50, 150)
                image, metadata = self.geometric_generator.generate_circle(center, radius)
            
            # Сохранение изображения
            image_path = f"{self.output_dir}/images/geometric_{i:04d}.png"
            image.save(image_path)
            
            # Создание аннотаций
            annotation = {
                "image_path": image_path,
                "metadata": metadata,
                "statements": self._generate_statements(metadata),
                "true_statements": self._generate_true_statements(metadata),
                "false_statements": self._generate_false_statements(metadata)
            }
            
            dataset.append(annotation)
        
        # Сохранение датасета
        dataset_path = f"{self.output_dir}/annotations/geometric_dataset.json"
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        return dataset_path
    
    def generate_physics_dataset(self, num_samples: int = 100) -> str:
        """Генерация физического датасета"""
        
        dataset = []
        
        for i in range(num_samples):
            # Случайный выбор типа эксперимента
            experiment_type = random.choice(['liquid', 'spheres'])
            
            if experiment_type == 'liquid':
                num_containers = random.randint(2, 4)
                image, metadata = self.physics_generator.generate_liquid_experiment(num_containers)
            else:
                num_spheres = random.randint(2, 5)
                image, metadata = self.physics_generator.generate_sphere_experiment(num_spheres)
            
            # Сохранение изображения
            image_path = f"{self.output_dir}/images/physics_{i:04d}.png"
            image.save(image_path)
            
            # Создание аннотаций
            annotation = {
                "image_path": image_path,
                "metadata": metadata,
                "statements": self._generate_physics_statements(metadata),
                "true_statements": self._generate_physics_true_statements(metadata),
                "false_statements": self._generate_physics_false_statements(metadata)
            }
            
            dataset.append(annotation)
        
        # Сохранение датасета
        dataset_path = f"{self.output_dir}/annotations/physics_dataset.json"
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        return dataset_path
    
    def _generate_statements(self, metadata: Dict) -> List[str]:
        """Генерация утверждений для геометрических фигур"""
        statements = []
        
        if metadata["type"] == "triangle":
            statements.extend([
                "Треугольник имеет три угла",
                "Все стороны треугольника равны",
                "Треугольник имеет прямой угол"
            ])
        elif metadata["type"] == "rectangle":
            statements.extend([
                "Прямоугольник имеет четыре угла",
                "Противоположные стороны прямоугольника параллельны",
                "Все углы прямоугольника прямые"
            ])
        elif metadata["type"] == "circle":
            statements.extend([
                "Окружность имеет центр",
                "Все радиусы окружности равны",
                "Окружность имеет диаметр"
            ])
        
        return statements
    
    def _generate_true_statements(self, metadata: Dict) -> List[str]:
        """Генерация истинных утверждений"""
        true_statements = []
        
        if metadata["type"] == "triangle":
            true_statements.append("Треугольник имеет три угла")
        elif metadata["type"] == "rectangle":
            true_statements.extend([
                "Противоположные стороны прямоугольника параллельны",
                "Все углы прямоугольника прямые"
            ])
        elif metadata["type"] == "circle":
            true_statements.extend([
                "Окружность имеет центр",
                "Все радиусы окружности равны"
            ])
        
        return true_statements
    
    def _generate_false_statements(self, metadata: Dict) -> List[str]:
        """Генерация ложных утверждений"""
        false_statements = []
        
        if metadata["type"] == "triangle":
            false_statements.extend([
                "Треугольник имеет четыре угла",
                "Все стороны треугольника равны"
            ])
        elif metadata["type"] == "rectangle":
            false_statements.append("Смежные стороны прямоугольника перпендикулярны")
        elif metadata["type"] == "circle":
            false_statements.append("Окружность имеет углы")
        
        return false_statements
    
    def _generate_physics_statements(self, metadata: Dict) -> List[str]:
        """Генерация утверждений для физических экспериментов"""
        statements = []
        
        if metadata["type"] == "liquid_experiment":
            statements.extend([
                "Уровни жидкости в сосудах одинаковые",
                "Жидкость находится в сообщающихся сосудах",
                "Уровень жидкости зависит от давления"
            ])
        elif metadata["type"] == "sphere_experiment":
            statements.extend([
                "Шарики находятся в жидкости",
                "Шарики имеют разную плотность",
                "Шарики всплывают на поверхность"
            ])
        
        return statements
    
    def _generate_physics_true_statements(self, metadata: Dict) -> List[str]:
        """Генерация истинных физических утверждений"""
        true_statements = []
        
        if metadata["type"] == "liquid_experiment":
            true_statements.append("Жидкость находится в сосудах")
        elif metadata["type"] == "sphere_experiment":
            true_statements.append("Шарики находятся в жидкости")
        
        return true_statements
    
    def _generate_physics_false_statements(self, metadata: Dict) -> List[str]:
        """Генерация ложных физических утверждений"""
        false_statements = []
        
        if metadata["type"] == "liquid_experiment":
            false_statements.append("Уровни жидкости в сосудах разные")
        elif metadata["type"] == "sphere_experiment":
            false_statements.append("Шарики находятся в воздухе")
        
        return false_statements

def main():
    """Генерация синтетических датасетов"""
    generator = SyntheticDatasetGenerator()
    
    print("Генерация геометрического датасета...")
    geometric_path = generator.generate_geometric_dataset(50)
    print(f"Геометрический датасет сохранен: {geometric_path}")
    
    print("Генерация физического датасета...")
    physics_path = generator.generate_physics_dataset(50)
    print(f"Физический датасет сохранен: {physics_path}")
    
    print("Генерация завершена!")

if __name__ == "__main__":
    main()
