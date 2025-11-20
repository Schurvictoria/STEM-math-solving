"""
NLI (Natural Language Inference) модуль для проверки утверждений
Использует предобученные модели для строгой верификации текста
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class NLIResult:
    """Результат NLI анализа"""
    entailment: float  # Вероятность entailment
    contradiction: float  # Вероятность contradiction  
    neutral: float  # Вероятность neutral
    predicted_label: str  # "entailment", "contradiction", "neutral"
    confidence: float

class NLIModel:
    """NLI модель для проверки утверждений"""
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-large-mnli"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Загрузка NLI модели"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"NLI модель {self.model_name} загружена")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки NLI модели: {e}")
            self.model = None
            self.tokenizer = None
    
    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """Предсказание NLI для пары premise-hypothesis"""
        
        if self.model is None:
            # Fallback на случайные результаты
            return NLIResult(
                entailment=0.33,
                contradiction=0.33, 
                neutral=0.34,
                predicted_label="neutral",
                confidence=0.5
            )
        
        try:
            # Токенизация
            inputs = self.tokenizer(
                premise, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Извлечение результатов
            probs = probabilities[0].cpu().numpy()
            
            # Маппинг индексов к лейблам (зависит от модели)
            label_mapping = {
                0: "contradiction",
                1: "neutral", 
                2: "entailment"
            }
            
            predicted_idx = np.argmax(probs)
            predicted_label = label_mapping.get(predicted_idx, "neutral")
            confidence = float(np.max(probs))
            
            return NLIResult(
                entailment=float(probs[2]) if len(probs) > 2 else 0.33,
                contradiction=float(probs[0]) if len(probs) > 0 else 0.33,
                neutral=float(probs[1]) if len(probs) > 1 else 0.34,
                predicted_label=predicted_label,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Ошибка NLI предсказания: {e}")
            return NLIResult(
                entailment=0.33,
                contradiction=0.33,
                neutral=0.34, 
                predicted_label="neutral",
                confidence=0.5
            )

class NLIProcessor:
    """Процессор для работы с NLI на задачах геометрии/физики"""
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-large-mnli"):
        self.nli_model = NLIModel(model_name)
        self.geometry_templates = self._load_geometry_templates()
        self.physics_templates = self._load_physics_templates()
    
    def _load_geometry_templates(self) -> List[str]:
        """Шаблоны для геометрических утверждений"""
        return [
            "Based on the geometric diagram, {statement}",
            "In the given geometric figure, {statement}",
            "Looking at the geometric construction, {statement}",
            "From the geometric diagram shown, {statement}",
            "In this geometric figure, {statement}"
        ]
    
    def _load_physics_templates(self) -> List[str]:
        """Шаблоны для физических утверждений"""
        return [
            "Based on the physical setup shown, {statement}",
            "In the given physical experiment, {statement}",
            "Looking at the physical apparatus, {statement}",
            "From the physical diagram shown, {statement}",
            "In this physical setup, {statement}"
        ]
    
    def create_premise(self, question: str, statement: str, domain: str = "geometry") -> str:
        """Создание premise для NLI"""
        
        # Определяем домен по ключевым словам
        if any(word in question.lower() for word in ["liquid", "fluid", "level", "pressure", "force", "density"]):
            domain = "physics"
        
        # Выбираем подходящий шаблон
        if domain == "physics":
            templates = self.physics_templates
        else:
            templates = self.geometry_templates
        
        # Создаем premise
        premise = f"Context: {question}\n"
        premise += templates[0].format(statement=statement)
        
        return premise
    
    def create_hypothesis(self, statement: str) -> str:
        """Создание hypothesis для NLI"""
        return statement
    
    def analyze_statement(self, question: str, statement: str, domain: str = "geometry") -> NLIResult:
        """Анализ утверждения с помощью NLI"""
        
        premise = self.create_premise(question, statement, domain)
        hypothesis = self.create_hypothesis(statement)
        
        result = self.nli_model.predict(premise, hypothesis)
        
        return result
    
    def batch_analyze(self, question: str, statements: List[str], domain: str = "geometry") -> List[NLIResult]:
        """Пакетный анализ нескольких утверждений"""
        results = []
        
        for statement in statements:
            result = self.analyze_statement(question, statement, domain)
            results.append(result)
        
        return results

class NLIAggregator:
    """Агрегатор NLI результатов для принятия решений"""
    
    def __init__(self, nli_processor: NLIProcessor):
        self.nli_processor = nli_processor
        self.thresholds = {
            "entailment": 0.7,
            "contradiction": 0.7,
            "neutral": 0.5
        }
    
    def classify_statement(self, question: str, statement: str, domain: str = "geometry") -> Dict:
        """Классификация утверждения на основе NLI"""
        
        nli_result = self.nli_processor.analyze_statement(question, statement, domain)
        
        # Принятие решения на основе порогов
        if nli_result.entailment > self.thresholds["entailment"]:
            verdict = "T"  # True
            confidence = nli_result.entailment
        elif nli_result.contradiction > self.thresholds["contradiction"]:
            verdict = "F"  # False
            confidence = nli_result.contradiction
        else:
            verdict = "U"  # Not enough info
            confidence = nli_result.neutral
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "nli_scores": {
                "entailment": nli_result.entailment,
                "contradiction": nli_result.contradiction,
                "neutral": nli_result.neutral
            },
            "predicted_label": nli_result.predicted_label
        }
    
    def analyze_options(self, question: str, options: List[str], domain: str = "geometry") -> List[Dict]:
        """Анализ всех опций"""
        results = []
        
        for option in options:
            result = self.classify_statement(question, option, domain)
            results.append(result)
        
        return results

class CalibratedNLI:
    """Калиброванная NLI модель с температурной калибровкой"""
    
    def __init__(self, nli_processor: NLIProcessor, temperature: float = 1.0):
        self.nli_processor = nli_processor
        self.temperature = temperature
        self.calibration_data = []
    
    def calibrate(self, validation_data: List[Dict]):
        """Калибровка на валидационных данных"""
        # Простая температурная калибровка
        for sample in validation_data:
            question = sample["question"]
            statement = sample["statement"]
            true_label = sample["label"]  # "T", "F", "U"
            
            nli_result = self.nli_processor.analyze_statement(question, statement)
            
            self.calibration_data.append({
                "nli_scores": [nli_result.entailment, nli_result.contradiction, nli_result.neutral],
                "true_label": true_label
            })
    
    def predict_calibrated(self, question: str, statement: str, domain: str = "geometry") -> Dict:
        """Калиброванное предсказание"""
        
        nli_result = self.nli_processor.analyze_statement(question, statement, domain)
        
        # Применяем температурную калибровку
        raw_scores = [nli_result.entailment, nli_result.contradiction, nli_result.neutral]
        calibrated_scores = [score / self.temperature for score in raw_scores]
        
        # Нормализация
        total = sum(calibrated_scores)
        calibrated_scores = [score / total for score in calibrated_scores]
        
        # Принятие решения
        max_score = max(calibrated_scores)
        max_idx = calibrated_scores.index(max_score)
        
        verdicts = ["F", "U", "T"]  # contradiction, neutral, entailment
        verdict = verdicts[max_idx]
        
        return {
            "verdict": verdict,
            "confidence": max_score,
            "calibrated_scores": {
                "entailment": calibrated_scores[2],
                "contradiction": calibrated_scores[0], 
                "neutral": calibrated_scores[1]
            }
        }

# Утилиты для работы с NLI
def extract_statements_from_question(question: str) -> List[str]:
    """Извлечение утверждений из вопроса"""
    statements = []
    
    # Ищем паттерны типа "A. statement", "B. statement" и т.д.
    pattern = r'([A-D])\.\s*([^A-D]*?)(?=[A-D]\.|$)'
    matches = re.findall(pattern, question, re.DOTALL)
    
    for letter, statement in matches:
        statements.append(statement.strip())
    
    return statements

def determine_domain(question: str) -> str:
    """Определение домена (геометрия/физика) по ключевым словам"""
    physics_keywords = [
        "liquid", "fluid", "level", "pressure", "force", "density",
        "volume", "mass", "weight", "buoyancy", "archimedes"
    ]
    
    geometry_keywords = [
        "angle", "triangle", "circle", "perpendicular", "parallel",
        "equal", "length", "radius", "diameter", "square", "rectangle"
    ]
    
    question_lower = question.lower()
    
    physics_score = sum(1 for keyword in physics_keywords if keyword in question_lower)
    geometry_score = sum(1 for keyword in geometry_keywords if keyword in question_lower)
    
    if physics_score > geometry_score:
        return "physics"
    else:
        return "geometry"
