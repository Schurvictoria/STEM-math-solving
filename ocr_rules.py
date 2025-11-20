"""
OCR и геометрические правила для анализа изображений
Включает детекцию перпендикуляров, параллелей, равенств и физических явлений
"""

import cv2
import numpy as np
from PIL import Image
import re
from typing import List, Dict, Tuple, Optional, Set
import logging
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class GeometricFeature:
    """Геометрическая особенность"""
    feature_type: str  # "perpendicular", "parallel", "equal", "angle", "length"
    confidence: float
    details: Dict

@dataclass
class SceneGraph:
    """Граф сцены с точками, линиями и их свойствами"""
    points: Dict[str, Tuple[float, float]]  # A: (x, y)
    lines: List[Dict]  # {"start": "A", "end": "B", "type": "line"}
    features: List[GeometricFeature]

class OCRProcessor:
    """Обработка OCR и извлечение текста"""
    
    def __init__(self):
        self.ocr_engine = None
        self._init_ocr()
    
    def _init_ocr(self):
        """Инициализация OCR движка"""
        try:
            import paddleocr
            self.ocr_engine = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')
            logger.info("PaddleOCR инициализирован")
        except ImportError:
            logger.warning("PaddleOCR не найден, используем fallback")
            self.ocr_engine = None
    
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """Извлечение текста из изображения"""
        if self.ocr_engine is None:
            return []
        
        try:
            results = self.ocr_engine.ocr(image, cls=True)
            text_boxes = []
            
            for line in results[0] if results[0] else []:
                bbox, (text, confidence) = line
                text_boxes.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox
                })
            
            return text_boxes
        except Exception as e:
            logger.error(f"Ошибка OCR: {e}")
            return []
    
    def extract_geometric_labels(self, text_boxes: List[Dict]) -> Dict:
        """Извлечение геометрических меток из текста"""
        labels = {
            "points": {},  # A, B, C, D...
            "angles": {},   # α, β, γ...
            "lengths": {},  # 4 см, 5.2...
            "relations": []  # AB ⟂ CD, AB ∥ CD, AB = CD
        }
        
        for box in text_boxes:
            text = box["text"].strip()
            confidence = box["confidence"]
            
            # Точки (A, B, C, D, E...)
            point_match = re.search(r'\b([A-Z])\b', text)
            if point_match and confidence > 0.7:
                point = point_match.group(1)
                labels["points"][point] = {
                    "text": text,
                    "confidence": confidence,
                    "bbox": box["bbox"]
                }
            
            # Углы (α, β, γ, 30°, 45°...)
            angle_patterns = [
                r'([αβγδεζηθικλμνξοπρστυφχψω])',  # Греческие буквы
                r'(\d+°)',  # Углы в градусах
                r'(\d+\.?\d*°)'
            ]
            
            for pattern in angle_patterns:
                angle_match = re.search(pattern, text)
                if angle_match and confidence > 0.6:
                    angle = angle_match.group(1)
                    labels["angles"][angle] = {
                        "text": text,
                        "confidence": confidence,
                        "bbox": box["bbox"]
                    }
            
            # Длины (4 см, 5.2, 10 мм...)
            length_patterns = [
                r'(\d+\.?\d*)\s*(см|mm|мм|m|м)',
                r'(\d+\.?\d*)\s*(cm|mm|m)',
                r'(\d+\.?\d*)\b(?!°)'  # Числа без градусов
            ]
            
            for pattern in length_patterns:
                length_match = re.search(pattern, text)
                if length_match and confidence > 0.6:
                    length = length_match.group(1)
                    labels["lengths"][length] = {
                        "text": text,
                        "confidence": confidence,
                        "bbox": box["bbox"]
                    }
            
            # Отношения (⟂, ∥, =)
            if '⟂' in text or '⊥' in text:
                labels["relations"].append({
                    "type": "perpendicular",
                    "text": text,
                    "confidence": confidence,
                    "bbox": box["bbox"]
                })
            elif '∥' in text or '||' in text:
                labels["relations"].append({
                    "type": "parallel", 
                    "text": text,
                    "confidence": confidence,
                    "bbox": box["bbox"]
                })
            elif '=' in text:
                labels["relations"].append({
                    "type": "equal",
                    "text": text,
                    "confidence": confidence,
                    "bbox": box["bbox"]
                })
        
        return labels

class GeometricAnalyzer:
    """Анализ геометрических особенностей"""
    
    def __init__(self):
        self.template_cache = {}
    
    def detect_lines(self, image: np.ndarray) -> List[Dict]:
        """Детекция линий с помощью Hough Transform"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return []
        
        detected_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            detected_lines.append({
                "start": (x1, y1),
                "end": (x2, y2),
                "angle": angle,
                "length": length
            })
        
        return detected_lines
    
    def detect_perpendiculars(self, lines: List[Dict], threshold: float = 15.0) -> List[GeometricFeature]:
        """Детекция перпендикулярных линий"""
        perpendiculars = []
        
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i+1:], i+1):
                angle_diff = abs(abs(line1["angle"] - line2["angle"]) - 90)
                
                if angle_diff < threshold:
                    # Проверяем пересечение
                    if self._lines_intersect(line1, line2):
                        confidence = 1.0 - (angle_diff / threshold)
                        perpendiculars.append(GeometricFeature(
                            feature_type="perpendicular",
                            confidence=confidence,
                            details={
                                "line1": i,
                                "line2": j,
                                "angle_diff": angle_diff
                            }
                        ))
        
        return perpendiculars
    
    def detect_parallels(self, lines: List[Dict], threshold: float = 10.0) -> List[GeometricFeature]:
        """Детекция параллельных линий"""
        parallels = []
        
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i+1:], i+1):
                angle_diff = abs(line1["angle"] - line2["angle"])
                angle_diff = min(angle_diff, 180 - angle_diff)  # Учитываем периодичность
                
                if angle_diff < threshold:
                    confidence = 1.0 - (angle_diff / threshold)
                    parallels.append(GeometricFeature(
                        feature_type="parallel",
                        confidence=confidence,
                        details={
                            "line1": i,
                            "line2": j,
                            "angle_diff": angle_diff
                        }
                    ))
        
        return parallels
    
    def detect_right_angles(self, image: np.ndarray) -> List[GeometricFeature]:
        """Детекция прямых углов по шаблону"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        right_angles = []
        
        # Создаем шаблон прямого угла
        template_size = 20
        template = np.zeros((template_size, template_size), dtype=np.uint8)
        cv2.line(template, (5, 5), (5, 15), 255, 2)
        cv2.line(template, (5, 15), (15, 15), 255, 2)
        
        # Поиск совпадений
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= 0.7)
        
        for pt in zip(*locations[::-1]):
            confidence = result[pt[1], pt[0]]
            right_angles.append(GeometricFeature(
                feature_type="right_angle",
                confidence=confidence,
                details={
                    "position": pt,
                    "template_size": template_size
                }
            ))
        
        return right_angles
    
    def detect_equal_marks(self, image: np.ndarray) -> List[GeometricFeature]:
        """Детекция меток равенства (штрихи)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equal_marks = []
        
        # Поиск коротких горизонтальных линий (метки равенства)
        kernel = np.ones((1, 5), np.uint8)
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Фильтруем по соотношению сторон (горизонтальные линии)
            if aspect_ratio > 3 and w > 10 and h < 5:
                confidence = min(1.0, aspect_ratio / 10)
                equal_marks.append(GeometricFeature(
                    feature_type="equal_mark",
                    confidence=confidence,
                    details={
                        "position": (x, y),
                        "size": (w, h),
                        "aspect_ratio": aspect_ratio
                    }
                ))
        
        return equal_marks
    
    def _lines_intersect(self, line1: Dict, line2: Dict) -> bool:
        """Проверка пересечения двух линий"""
        x1, y1 = line1["start"]
        x2, y2 = line1["end"]
        x3, y3 = line2["start"]
        x4, y4 = line2["end"]
        
        # Простая проверка пересечения отрезков
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw((x1, y1), (x3, y3), (x4, y4)) != ccw((x2, y2), (x3, y3), (x4, y4)) and \
               ccw((x1, y1), (x2, y2), (x3, y3)) != ccw((x1, y1), (x2, y2), (x4, y4))

class PhysicsAnalyzer:
    """Анализ физических явлений (жидкости, уровни, силы)"""
    
    def detect_liquid_levels(self, image: np.ndarray) -> List[GeometricFeature]:
        """Детекция уровней жидкости"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        levels = []
        
        # Поиск горизонтальных линий (уровни жидкости)
        kernel = np.ones((1, 10), np.uint8)
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if aspect_ratio > 5 and w > 20:  # Горизонтальные линии
                confidence = min(1.0, aspect_ratio / 20)
                levels.append(GeometricFeature(
                    feature_type="liquid_level",
                    confidence=confidence,
                    details={
                        "position": (x, y),
                        "width": w,
                        "height": h,
                        "aspect_ratio": aspect_ratio
                    }
                ))
        
        return levels
    
    def detect_objects_in_fluid(self, image: np.ndarray) -> List[GeometricFeature]:
        """Детекция объектов в жидкости (шарики, тела)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        objects = []
        
        # Поиск окружностей (шарики)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=5, maxRadius=50)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                confidence = 0.8  # Базовая уверенность для детектированных окружностей
                objects.append(GeometricFeature(
                    feature_type="sphere",
                    confidence=confidence,
                    details={
                        "center": (x, y),
                        "radius": r
                    }
                ))
        
        return objects

class RuleEngine:
    """Движок правил для проверки утверждений"""
    
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.geometric_analyzer = GeometricAnalyzer()
        self.physics_analyzer = PhysicsAnalyzer()
    
    def analyze_image(self, image: np.ndarray) -> SceneGraph:
        """Полный анализ изображения"""
        # OCR
        text_boxes = self.ocr_processor.extract_text(image)
        labels = self.ocr_processor.extract_geometric_labels(text_boxes)
        
        # Геометрический анализ
        lines = self.geometric_analyzer.detect_lines(image)
        perpendiculars = self.geometric_analyzer.detect_perpendiculars(lines)
        parallels = self.geometric_analyzer.detect_parallels(lines)
        right_angles = self.geometric_analyzer.detect_right_angles(image)
        equal_marks = self.geometric_analyzer.detect_equal_marks(image)
        
        # Физический анализ
        liquid_levels = self.physics_analyzer.detect_liquid_levels(image)
        objects = self.physics_analyzer.detect_objects_in_fluid(image)
        
        # Создание графа сцены
        scene_graph = SceneGraph(
            points=labels["points"],
            lines=[{"start": "unknown", "end": "unknown", "type": "line"} for _ in lines],
            features=perpendiculars + parallels + right_angles + equal_marks + liquid_levels + objects
        )
        
        return scene_graph
    
    def check_statement(self, statement: str, scene_graph: SceneGraph) -> Dict:
        """Проверка конкретного утверждения"""
        statement_lower = statement.lower()
        
        result = {
            "verdict": "U",  # U = Not enough info
            "confidence": 0.0,
            "evidence": []
        }
        
        # Проверка перпендикуляров
        if "perpendicular" in statement_lower or "⟂" in statement or "⊥" in statement:
            perpendiculars = [f for f in scene_graph.features if f.feature_type == "perpendicular"]
            if perpendiculars:
                max_confidence = max(f.confidence for f in perpendiculars)
                if max_confidence > 0.7:
                    result["verdict"] = "T"
                    result["confidence"] = max_confidence
                    result["evidence"].append("Detected perpendicular lines")
                else:
                    result["verdict"] = "F"
                    result["confidence"] = 1.0 - max_confidence
                    result["evidence"].append("No clear perpendicular lines")
        
        # Проверка параллелей
        elif "parallel" in statement_lower or "∥" in statement or "||" in statement:
            parallels = [f for f in scene_graph.features if f.feature_type == "parallel"]
            if parallels:
                max_confidence = max(f.confidence for f in parallels)
                if max_confidence > 0.7:
                    result["verdict"] = "T"
                    result["confidence"] = max_confidence
                    result["evidence"].append("Detected parallel lines")
                else:
                    result["verdict"] = "F"
                    result["confidence"] = 1.0 - max_confidence
                    result["evidence"].append("No clear parallel lines")
        
        # Проверка равенств
        elif "equal" in statement_lower or "=" in statement:
            equal_marks = [f for f in scene_graph.features if f.feature_type == "equal_mark"]
            if equal_marks:
                max_confidence = max(f.confidence for f in equal_marks)
                if max_confidence > 0.6:
                    result["verdict"] = "T"
                    result["confidence"] = max_confidence
                    result["evidence"].append("Detected equality marks")
                else:
                    result["verdict"] = "F"
                    result["confidence"] = 1.0 - max_confidence
                    result["evidence"].append("No equality marks found")
        
        # Проверка углов
        elif "angle" in statement_lower or "°" in statement:
            # Ищем числовые значения углов в утверждении
            angle_match = re.search(r'(\d+\.?\d*)°?', statement)
            if angle_match:
                target_angle = float(angle_match.group(1))
                # Простая проверка на наличие прямых углов
                right_angles = [f for f in scene_graph.features if f.feature_type == "right_angle"]
                if right_angles and abs(target_angle - 90) < 10:
                    result["verdict"] = "T"
                    result["confidence"] = max(f.confidence for f in right_angles)
                    result["evidence"].append("Detected right angle")
                else:
                    result["verdict"] = "U"
                    result["confidence"] = 0.5
                    result["evidence"].append("Cannot verify specific angle")
        
        # Проверка уровней жидкости
        elif "level" in statement_lower or "liquid" in statement_lower:
            liquid_levels = [f for f in scene_graph.features if f.feature_type == "liquid_level"]
            if liquid_levels:
                max_confidence = max(f.confidence for f in liquid_levels)
                if max_confidence > 0.6:
                    result["verdict"] = "T"
                    result["confidence"] = max_confidence
                    result["evidence"].append("Detected liquid levels")
                else:
                    result["verdict"] = "F"
                    result["confidence"] = 1.0 - max_confidence
                    result["evidence"].append("No clear liquid levels")
        
        return result
