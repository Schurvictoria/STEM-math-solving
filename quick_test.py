"""
Быстрый тест решения без тяжелых моделей
Проверяет работоспособность пайплайна с fallback режимом
"""

import json
import pickle
import io
import os
from PIL import Image
import random

def create_fallback_solution():
    """Создание fallback решения для быстрого тестирования"""
    
    # Проверяем наличие входного файла
    if not os.path.exists('input.pickle'):
        if os.path.exists('example-dataset.pickle'):
            print("Копируем example-dataset.pickle в input.pickle...")
            import shutil
            shutil.copy('example-dataset.pickle', 'input.pickle')
        else:
            print("ОШИБКА: Не найден файл input.pickle или example-dataset.pickle")
            return
    
    # Загрузка данных
    with open('input.pickle', "rb") as input_file:
        model_input = pickle.load(input_file)
    
    print(f"Загружено {len(model_input)} примеров")
    
    # Fallback решения
    answers = ["A", "B", "C", "D", "AB", "BC", "CD", "ABC", "BCD", "ABCD"]
    model_output = []
    
    for i, row in enumerate(model_input):
        print(f"Обрабатываем пример {i+1}/{len(model_input)}")
        rid = row["rid"]
        question = row["question"]
        
        # Простая эвристика на основе текста вопроса
        question_lower = question.lower()
        
        if "perpendicular" in question_lower or "⟂" in question or "⊥" in question:
            # Для перпендикуляров чаще выбираем A или B
            prediction = random.choice(["A", "B", "AB"])
        elif "parallel" in question_lower or "∥" in question or "||" in question:
            # Для параллелей чаще выбираем C или D
            prediction = random.choice(["C", "D", "CD"])
        elif "equal" in question_lower or "=" in question:
            # Для равенств чаще выбираем комбинации
            prediction = random.choice(["AB", "CD", "ABC"])
        elif "angle" in question_lower or "degree" in question_lower:
            # Для углов чаще выбираем одиночные ответы
            prediction = random.choice(["A", "B", "C", "D"])
        else:
            # Общий случай - случайный выбор
            prediction = random.choice(answers)
        
        model_output.append({"rid": rid, "answer": prediction})
        print(f"Пример {i+1} обработан: {prediction}")
    
    # Сохранение результатов
    with open('output.json', 'w') as output_file:
        json.dump(model_output, output_file, ensure_ascii=False)
    
    print(f"Обработано {len(model_output)} примеров")
    print("Результаты сохранены в output.json")

def test_components():
    """Тестирование компонентов без загрузки тяжелых моделей"""
    print("Тестирование компонентов...")
    
    # Тест импортов
    try:
        from vlm import VLMModel, FEW_SHOT_EXAMPLES
        print("✓ VLM модуль импортирован")
    except Exception as e:
        print(f"✗ Ошибка VLM: {e}")
    
    try:
        from ocr_rules import RuleEngine, GeometricAnalyzer
        print("✓ OCR модуль импортирован")
    except Exception as e:
        print(f"✗ Ошибка OCR: {e}")
    
    try:
        from nli import NLIProcessor, NLIAggregator
        print("✓ NLI модуль импортирован")
    except Exception as e:
        print(f"✗ Ошибка NLI: {e}")
    
    try:
        from meta import MetaModel, ExpertResult
        print("✓ Мета-модель импортирована")
    except Exception as e:
        print(f"✗ Ошибка мета-модели: {e}")
    
    try:
        from synthetic_data import GeometricGenerator, PhysicsGenerator
        print("✓ Генератор синтетических данных импортирован")
    except Exception as e:
        print(f"✗ Ошибка генератора: {e}")

def main():
    """Основная функция быстрого теста"""
    print("="*50)
    print("БЫСТРЫЙ ТЕСТ РЕШЕНИЯ")
    print("="*50)
    
    # Тестирование компонентов
    test_components()
    
    print("\n" + "="*50)
    print("ЗАПУСК FALLBACK РЕШЕНИЯ")
    print("="*50)
    
    # Запуск fallback решения
    create_fallback_solution()
    
    print("\n" + "="*50)
    print("ТЕСТ ЗАВЕРШЕН")
    print("="*50)

if __name__ == "__main__":
    main()
