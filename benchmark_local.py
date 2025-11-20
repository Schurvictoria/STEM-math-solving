"""
Скрипт для локального тестирования и бенчмарка решения
"""

import json
import pickle
import time
import os
import logging
from typing import List, Dict
import numpy as np
from PIL import Image
import io

from solution import SolutionPipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Запуск бенчмарков и тестирования"""
    
    def __init__(self):
        self.pipeline = SolutionPipeline()
        self.results = []
    
    def load_test_data(self, data_path: str) -> List[Dict]:
        """Загрузка тестовых данных"""
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def run_single_test(self, question: str, image: Image.Image, expected_answer: str = None) -> Dict:
        """Запуск одного теста"""
        start_time = time.time()
        
        try:
            predicted_answer = self.pipeline.analyze_single_problem(question, image)
            processing_time = time.time() - start_time
            
            result = {
                "question": question,
                "predicted_answer": predicted_answer,
                "expected_answer": expected_answer,
                "processing_time": processing_time,
                "correct": predicted_answer == expected_answer if expected_answer else None
            }
            
            logger.info(f"Тест завершен: {predicted_answer} (время: {processing_time:.2f}с)")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка в тесте: {e}")
            return {
                "question": question,
                "predicted_answer": "ERROR",
                "expected_answer": expected_answer,
                "processing_time": time.time() - start_time,
                "correct": False,
                "error": str(e)
            }
    
    def run_batch_test(self, test_data: List[Dict]) -> List[Dict]:
        """Запуск батча тестов"""
        results = []
        
        for i, sample in enumerate(test_data):
            logger.info(f"Запуск теста {i+1}/{len(test_data)}")
            
            try:
                question = sample["question"]
                image = Image.open(io.BytesIO(sample["image"]))
                expected_answer = sample.get("answer", None)
                
                result = self.run_single_test(question, image, expected_answer)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Ошибка в тесте {i+1}: {e}")
                results.append({
                    "question": sample.get("question", "Unknown"),
                    "predicted_answer": "ERROR",
                    "expected_answer": sample.get("answer", None),
                    "processing_time": 0,
                    "correct": False,
                    "error": str(e)
                })
        
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Расчет метрик"""
        total_tests = len(results)
        correct_tests = sum(1 for r in results if r.get("correct", False))
        error_tests = sum(1 for r in results if "error" in r)
        
        accuracy = correct_tests / total_tests if total_tests > 0 else 0
        error_rate = error_tests / total_tests if total_tests > 0 else 0
        
        processing_times = [r["processing_time"] for r in results if "processing_time" in r]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        max_processing_time = max(processing_times) if processing_times else 0
        min_processing_time = min(processing_times) if processing_times else 0
        
        # Анализ по типам ответов
        answer_distribution = {}
        for r in results:
            answer = r.get("predicted_answer", "ERROR")
            answer_distribution[answer] = answer_distribution.get(answer, 0) + 1
        
        metrics = {
            "total_tests": total_tests,
            "correct_tests": correct_tests,
            "error_tests": error_tests,
            "accuracy": accuracy,
            "error_rate": error_rate,
            "avg_processing_time": avg_processing_time,
            "max_processing_time": max_processing_time,
            "min_processing_time": min_processing_time,
            "answer_distribution": answer_distribution
        }
        
        return metrics
    
    def save_results(self, results: List[Dict], metrics: Dict, output_path: str):
        """Сохранение результатов"""
        output_data = {
            "results": results,
            "metrics": metrics,
            "timestamp": time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Результаты сохранены в {output_path}")
    
    def print_summary(self, metrics: Dict):
        """Вывод сводки результатов"""
        print("\n" + "="*50)
        print("СВОДКА РЕЗУЛЬТАТОВ БЕНЧМАРКА")
        print("="*50)
        print(f"Всего тестов: {metrics['total_tests']}")
        print(f"Правильных ответов: {metrics['correct_tests']}")
        print(f"Ошибок: {metrics['error_tests']}")
        print(f"Точность: {metrics['accuracy']:.2%}")
        print(f"Частота ошибок: {metrics['error_rate']:.2%}")
        print(f"Среднее время обработки: {metrics['avg_processing_time']:.2f}с")
        print(f"Максимальное время: {metrics['max_processing_time']:.2f}с")
        print(f"Минимальное время: {metrics['min_processing_time']:.2f}с")
        
        print("\nРаспределение ответов:")
        for answer, count in metrics['answer_distribution'].items():
            percentage = count / metrics['total_tests'] * 100
            print(f"  {answer}: {count} ({percentage:.1f}%)")
        
        print("="*50)

def create_sample_test_data() -> List[Dict]:
    """Создание примеров тестовых данных"""
    sample_data = [
        {
            "rid": 1,
            "question": "A. Угол α равен 45°\nB. Угол α равен 90°\nC. Угол α равен 30°\nD. Угол α равен 60°",
            "image": b"fake_image_data",  # Заглушка
            "answer": "B"  # Ожидаемый ответ
        },
        {
            "rid": 2,
            "question": "A. AB перпендикулярно CD\nB. AB параллельно CD\nC. AB равно CD\nD. AB пересекает CD",
            "image": b"fake_image_data",
            "answer": "A"
        }
    ]
    return sample_data

def run_component_tests():
    """Тестирование отдельных компонентов"""
    print("Тестирование компонентов...")
    
    # Тест VLM
    try:
        from vlm import VLMModel
        vlm = VLMModel()
        print("✓ VLM модуль загружен")
    except Exception as e:
        print(f"✗ Ошибка VLM: {e}")
    
    # Тест OCR
    try:
        from ocr_rules import RuleEngine
        ocr = RuleEngine()
        print("✓ OCR модуль загружен")
    except Exception as e:
        print(f"✗ Ошибка OCR: {e}")
    
    # Тест NLI
    try:
        from nli import NLIProcessor
        nli = NLIProcessor()
        print("✓ NLI модуль загружен")
    except Exception as e:
        print(f"✗ Ошибка NLI: {e}")
    
    # Тест мета-модели
    try:
        from meta import MetaModel
        meta = MetaModel()
        print("✓ Мета-модель загружена")
    except Exception as e:
        print(f"✗ Ошибка мета-модели: {e}")

def main():
    """Основная функция бенчмарка"""
    print("Запуск бенчмарка решения...")
    
    # Тестирование компонентов
    run_component_tests()
    
    # Создание бенчмарка
    benchmark = BenchmarkRunner()
    
    # Загрузка тестовых данных
    test_data_path = "example-dataset.pickle"
    if os.path.exists(test_data_path):
        print(f"Загрузка тестовых данных из {test_data_path}")
        test_data = benchmark.load_test_data(test_data_path)
    else:
        print("Создание примеров тестовых данных")
        test_data = create_sample_test_data()
    
    print(f"Загружено {len(test_data)} тестовых примеров")
    
    # Запуск тестов
    print("Запуск тестов...")
    results = benchmark.run_batch_test(test_data)
    
    # Расчет метрик
    metrics = benchmark.calculate_metrics(results)
    
    # Вывод результатов
    benchmark.print_summary(metrics)
    
    # Сохранение результатов
    output_path = "benchmark_results.json"
    benchmark.save_results(results, metrics, output_path)
    
    print(f"\nБенчмарк завершен. Результаты сохранены в {output_path}")

if __name__ == "__main__":
    main()
