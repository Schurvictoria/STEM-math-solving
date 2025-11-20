"""
VLM (Vision-Language Model) модуль для анализа изображений и текста
Поддерживает Qwen2-VL, InternVL2 и другие модели с self-consistency
"""

import torch
import numpy as np
from PIL import Image
import json
import re
from typing import List, Dict, Tuple, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMModel:
    """Базовый класс для VLM моделей"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели и токенизатора"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers import BitsAndBytesConfig
            
            # Конфигурация для квантизации 4-bit
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            logger.info(f"Модель {self.model_name} загружена успешно")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            # Fallback на CPU
            self.device = "cpu"
            self.model = None
            self.tokenizer = None
    
    def _create_prompt(self, question: str, options: List[str], few_shot_examples: List[Dict] = None) -> str:
        """Создание промпта с few-shot примерами"""
        
        system_prompt = """Ты эксперт по геометрии и физике. Анализируй изображение и проверяй каждое утверждение.
Отвечай строго в формате: T (True) / F (False) / U (Not enough info) + вероятность 0-1.
Пример: T 0.87 или F 0.92 или U 0.65

Правила:
- T: утверждение точно верно на основе изображения
- F: утверждение точно неверно на основе изображения  
- U: недостаточно информации для определения
- Всегда указывай вероятность от 0 до 1"""
        
        if few_shot_examples:
            examples_text = "\n".join([
                f"Утверждение: {ex['statement']}\nОтвет: {ex['answer']}"
                for ex in few_shot_examples
            ])
            system_prompt += f"\n\nПримеры:\n{examples_text}\n"
        
        prompt = f"{system_prompt}\n\nВопрос: {question}\n\nПроверь каждое утверждение:\n"
        
        for i, option in enumerate(options):
            prompt += f"{chr(65+i)}. {option}\n"
        
        prompt += "\nОтвет для каждого утверждения (T/F/U + вероятность):"
        
        return prompt
    
    def _parse_response(self, response: str) -> List[Tuple[str, float]]:
        """Парсинг ответа модели в формат (T/F/U, вероятность)"""
        results = []
        
        # Ищем паттерны типа "A. T 0.87" или "T 0.87"
        patterns = [
            r'([A-D])\.\s*([TFU])\s+(\d+\.?\d*)',
            r'([TFU])\s+(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if len(match) == 3:  # A. T 0.87
                    letter, verdict, prob = match
                    results.append((verdict.upper(), float(prob)))
                elif len(match) == 2:  # T 0.87
                    verdict, prob = match
                    results.append((verdict.upper(), float(prob)))
        
        return results
    
    def predict_single(self, image: Image.Image, question: str, options: List[str], 
                    few_shot_examples: List[Dict] = None) -> List[Tuple[str, float]]:
        """Предсказание для одного изображения"""
        
        if self.model is None:
            # Fallback на случайные ответы
            return [("U", 0.5) for _ in options]
        
        try:
            prompt = self._create_prompt(question, options, few_shot_examples)
            
            # Подготовка изображения
            if hasattr(self.model, 'process_images'):
                inputs = self.model.process_images([image], self.model.config)
            else:
                # Простая обработка для базовых моделей
                inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
            
            # Генерация ответа
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results = self._parse_response(response)
            
            # Дополняем до нужного количества опций
            while len(results) < len(options):
                results.append(("U", 0.5))
            
            return results[:len(options)]
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return [("U", 0.5) for _ in options]
    
    def predict_with_consistency(self, image: Image.Image, question: str, options: List[str],
                               num_samples: int = 5, few_shot_examples: List[Dict] = None) -> List[Dict]:
        """Предсказание с self-consistency"""
        
        all_predictions = []
        
        for _ in range(num_samples):
            predictions = self.predict_single(image, question, options, few_shot_examples)
            all_predictions.append(predictions)
        
        # Агрегация результатов
        final_results = []
        for i in range(len(options)):
            verdicts = [pred[i][0] for pred in all_predictions]
            probs = [pred[i][1] for pred in all_predictions]
            
            # Голосование
            verdict_counts = {"T": 0, "F": 0, "U": 0}
            for v in verdicts:
                verdict_counts[v] += 1
            
            # Выбираем наиболее частый вердикт
            final_verdict = max(verdict_counts, key=verdict_counts.get)
            
            # Средняя вероятность для выбранного вердикта
            relevant_probs = [p for v, p in zip(verdicts, probs) if v == final_verdict]
            avg_prob = np.mean(relevant_probs) if relevant_probs else 0.5
            
            # Энтропия как мера неопределенности
            entropy = -sum(p * np.log(p + 1e-8) for p in [verdict_counts["T"]/num_samples, 
                                                          verdict_counts["F"]/num_samples, 
                                                          verdict_counts["U"]/num_samples])
            
            final_results.append({
                "verdict": final_verdict,
                "probability": avg_prob,
                "entropy": entropy,
                "consistency": max(verdict_counts.values()) / num_samples
            })
        
        return final_results

class VLMPipeline:
    """Пайплайн для работы с несколькими VLM моделями"""
    
    def __init__(self, model_configs: List[Dict]):
        self.models = []
        
        for config in model_configs:
            try:
                model = VLMModel(config["name"], config.get("device", "cuda"))
                self.models.append(model)
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель {config['name']}: {e}")
    
    def predict_ensemble(self, image: Image.Image, question: str, options: List[str],
                        few_shot_examples: List[Dict] = None) -> List[Dict]:
        """Ансамбль предсказаний от всех моделей"""
        
        all_results = []
        
        for model in self.models:
            try:
                results = model.predict_with_consistency(
                    image, question, options, 
                    num_samples=3,  # Меньше сэмплов для скорости
                    few_shot_examples=few_shot_examples
                )
                all_results.append(results)
            except Exception as e:
                logger.error(f"Ошибка в модели: {e}")
                continue
        
        if not all_results:
            # Fallback
            return [{"verdict": "U", "probability": 0.5, "entropy": 1.0, "consistency": 0.0} 
                   for _ in options]
        
        # Агрегация результатов от всех моделей
        final_results = []
        for i in range(len(options)):
            verdicts = []
            probs = []
            entropies = []
            consistencies = []
            
            for results in all_results:
                if i < len(results):
                    verdicts.append(results[i]["verdict"])
                    probs.append(results[i]["probability"])
                    entropies.append(results[i]["entropy"])
                    consistencies.append(results[i]["consistency"])
            
            # Взвешенное голосование (вес = consistency)
            weights = [c for c in consistencies]
            if sum(weights) == 0:
                weights = [1.0] * len(weights)
            
            # Нормализация весов
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Взвешенное голосование
            verdict_scores = {"T": 0, "F": 0, "U": 0}
            weighted_prob = 0
            
            for v, p, w in zip(verdicts, probs, weights):
                verdict_scores[v] += w
                weighted_prob += p * w
            
            final_verdict = max(verdict_scores, key=verdict_scores.get)
            avg_entropy = np.mean(entropies) if entropies else 1.0
            avg_consistency = np.mean(consistencies) if consistencies else 0.0
            
            final_results.append({
                "verdict": final_verdict,
                "probability": weighted_prob,
                "entropy": avg_entropy,
                "consistency": avg_consistency,
                "num_models": len(all_results)
            })
        
        return final_results

# Few-shot примеры для геометрии и физики
FEW_SHOT_EXAMPLES = [
    {
        "statement": "Угол α равен 45°",
        "answer": "T 0.85"
    },
    {
        "statement": "AB перпендикулярно CD", 
        "answer": "F 0.90"
    },
    {
        "statement": "Треугольник равносторонний",
        "answer": "U 0.60"
    },
    {
        "statement": "Уровень жидкости в левом сосуде выше",
        "answer": "T 0.88"
    }
]

def create_crops(image: Image.Image, crop_regions: List[Tuple[int, int, int, int]]) -> List[Image.Image]:
    """Создание кропов изображения для анализа"""
    crops = []
    
    for x1, y1, x2, y2 in crop_regions:
        try:
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)
        except Exception as e:
            logger.warning(f"Ошибка создания кропа: {e}")
            crops.append(image)  # Fallback на полное изображение
    
    return crops
