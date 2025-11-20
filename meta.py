"""
Мета-модель для объединения результатов от всех экспертов
Использует LightGBM/XGBoost для финального принятия решений
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import pickle
import json

logger = logging.getLogger(__name__)

@dataclass
class ExpertResult:
    """Результат от одного эксперта"""
    expert_name: str
    verdict: str  # "T", "F", "U"
    confidence: float
    features: Dict[str, float]  # Дополнительные признаки

@dataclass
class MetaFeatures:
    """Мета-признаки для финальной модели"""
    # VLM признаки
    vlm_verdict: str
    vlm_confidence: float
    vlm_entropy: float
    vlm_consistency: float
    vlm_num_models: int
    
    # OCR/правила признаки
    ocr_verdict: str
    ocr_confidence: float
    ocr_evidence_count: int
    
    # NLI признаки
    nli_verdict: str
    nli_confidence: float
    nli_entailment: float
    nli_contradiction: float
    nli_neutral: float
    
    # Комбинированные признаки
    agreement_score: float
    conflict_score: float
    uncertainty_score: float

class FeatureExtractor:
    """Извлечение признаков из результатов экспертов"""
    
    def __init__(self):
        self.verdict_mapping = {"T": 1, "F": 0, "U": 0.5}
    
    def extract_features(self, expert_results: List[ExpertResult]) -> MetaFeatures:
        """Извлечение мета-признаков"""
        
        # Группировка по типам экспертов
        vlm_results = [r for r in expert_results if "vlm" in r.expert_name.lower()]
        ocr_results = [r for r in expert_results if "ocr" in r.expert_name.lower()]
        nli_results = [r for r in expert_results if "nli" in r.expert_name.lower()]
        
        # VLM признаки
        vlm_verdict, vlm_confidence, vlm_entropy, vlm_consistency, vlm_num_models = \
            self._extract_vlm_features(vlm_results)
        
        # OCR признаки
        ocr_verdict, ocr_confidence, ocr_evidence_count = \
            self._extract_ocr_features(ocr_results)
        
        # NLI признаки
        nli_verdict, nli_confidence, nli_entailment, nli_contradiction, nli_neutral = \
            self._extract_nli_features(nli_results)
        
        # Комбинированные признаки
        agreement_score, conflict_score, uncertainty_score = \
            self._extract_agreement_features(expert_results)
        
        return MetaFeatures(
            vlm_verdict=vlm_verdict,
            vlm_confidence=vlm_confidence,
            vlm_entropy=vlm_entropy,
            vlm_consistency=vlm_consistency,
            vlm_num_models=vlm_num_models,
            ocr_verdict=ocr_verdict,
            ocr_confidence=ocr_confidence,
            ocr_evidence_count=ocr_evidence_count,
            nli_verdict=nli_verdict,
            nli_confidence=nli_confidence,
            nli_entailment=nli_entailment,
            nli_contradiction=nli_contradiction,
            nli_neutral=nli_neutral,
            agreement_score=agreement_score,
            conflict_score=conflict_score,
            uncertainty_score=uncertainty_score
        )
    
    def _extract_vlm_features(self, vlm_results: List[ExpertResult]) -> Tuple[str, float, float, float, int]:
        """Извлечение VLM признаков"""
        if not vlm_results:
            return "U", 0.5, 1.0, 0.0, 0
        
        # Агрегация VLM результатов
        verdicts = [r.verdict for r in vlm_results]
        confidences = [r.confidence for r in vlm_results]
        
        # Голосование
        verdict_counts = {"T": 0, "F": 0, "U": 0}
        for v in verdicts:
            verdict_counts[v] += 1
        
        final_verdict = max(verdict_counts, key=verdict_counts.get)
        avg_confidence = np.mean(confidences)
        
        # Энтропия
        total = len(verdicts)
        entropy = -sum((count/total) * np.log(count/total + 1e-8) 
                      for count in verdict_counts.values())
        
        # Консистентность
        consistency = max(verdict_counts.values()) / total
        
        return final_verdict, avg_confidence, entropy, consistency, len(vlm_results)
    
    def _extract_ocr_features(self, ocr_results: List[ExpertResult]) -> Tuple[str, float, int]:
        """Извлечение OCR признаков"""
        if not ocr_results:
            return "U", 0.5, 0
        
        result = ocr_results[0]  # Берем первый результат
        evidence_count = len(result.features.get("evidence", []))
        
        return result.verdict, result.confidence, evidence_count
    
    def _extract_nli_features(self, nli_results: List[ExpertResult]) -> Tuple[str, float, float, float, float]:
        """Извлечение NLI признаков"""
        if not nli_results:
            return "U", 0.5, 0.33, 0.33, 0.34
        
        result = nli_results[0]  # Берем первый результат
        features = result.features
        
        entailment = features.get("entailment", 0.33)
        contradiction = features.get("contradiction", 0.33)
        neutral = features.get("neutral", 0.34)
        
        return result.verdict, result.confidence, entailment, contradiction, neutral
    
    def _extract_agreement_features(self, expert_results: List[ExpertResult]) -> Tuple[float, float, float]:
        """Извлечение признаков согласованности"""
        if len(expert_results) < 2:
            return 1.0, 0.0, 0.0
        
        verdicts = [r.verdict for r in expert_results]
        confidences = [r.confidence for r in expert_results]
        
        # Согласованность (доля экспертов с одинаковым вердиктом)
        verdict_counts = {"T": 0, "F": 0, "U": 0}
        for v in verdicts:
            verdict_counts[v] += 1
        
        agreement_score = max(verdict_counts.values()) / len(verdicts)
        
        # Конфликт (противоположные вердикты)
        has_true = verdict_counts["T"] > 0
        has_false = verdict_counts["F"] > 0
        conflict_score = 1.0 if (has_true and has_false) else 0.0
        
        # Неопределенность (доля "U")
        uncertainty_score = verdict_counts["U"] / len(verdicts)
        
        return agreement_score, conflict_score, uncertainty_score

class MetaModel:
    """Мета-модель для финального принятия решений"""
    
    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
    
    def _create_model(self):
        """Создание модели"""
        if self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbose=-1
                )
            except ImportError:
                logger.warning("LightGBM не найден, используем простую модель")
                self.model = None
        elif self.model_type == "xgboost":
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            except ImportError:
                logger.warning("XGBoost не найден, используем простую модель")
                self.model = None
        else:
            self.model = None
    
    def _features_to_vector(self, features: MetaFeatures) -> np.ndarray:
        """Преобразование признаков в вектор"""
        vector = [
            self.feature_extractor.verdict_mapping.get(features.vlm_verdict, 0.5),
            features.vlm_confidence,
            features.vlm_entropy,
            features.vlm_consistency,
            features.vlm_num_models,
            self.feature_extractor.verdict_mapping.get(features.ocr_verdict, 0.5),
            features.ocr_confidence,
            features.ocr_evidence_count,
            self.feature_extractor.verdict_mapping.get(features.nli_verdict, 0.5),
            features.nli_confidence,
            features.nli_entailment,
            features.nli_contradiction,
            features.nli_neutral,
            features.agreement_score,
            features.conflict_score,
            features.uncertainty_score
        ]
        
        return np.array(vector)
    
    def train(self, training_data: List[Dict]):
        """Обучение мета-модели"""
        self._create_model()
        
        if self.model is None:
            logger.warning("Модель не создана, используем простое правило")
            self.is_trained = True
            return
        
        # Подготовка данных
        X = []
        y = []
        
        for sample in training_data:
            expert_results = sample["expert_results"]
            true_label = sample["true_label"]  # "T", "F", "U"
            
            features = self.feature_extractor.extract_features(expert_results)
            feature_vector = self._features_to_vector(features)
            
            X.append(feature_vector)
            y.append(self.feature_extractor.verdict_mapping[true_label])
        
        X = np.array(X)
        y = np.array(y)
        
        # Обучение
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logger.info("Мета-модель обучена")
        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            self.model = None
            self.is_trained = True
    
    def predict(self, expert_results: List[ExpertResult]) -> Dict:
        """Предсказание на основе результатов экспертов"""
        
        features = self.feature_extractor.extract_features(expert_results)
        
        if self.model is None or not self.is_trained:
            # Простое правило
            return self._simple_rule(features)
        
        try:
            feature_vector = self._features_to_vector(features)
            feature_vector = feature_vector.reshape(1, -1)
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(feature_vector)[0]
                confidence = np.max(probabilities)
                predicted_class = np.argmax(probabilities)
            else:
                predicted_class = self.model.predict(feature_vector)[0]
                confidence = 0.8  # Базовая уверенность
            
            # Маппинг класса на вердикт
            class_to_verdict = {0: "F", 1: "T", 2: "U"}
            verdict = class_to_verdict.get(predicted_class, "U")
            
            return {
                "verdict": verdict,
                "confidence": confidence,
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return self._simple_rule(features)
    
    def _simple_rule(self, features: MetaFeatures) -> Dict:
        """Простое правило для принятия решений"""
        
        # Приоритет: OCR > VLM > NLI
        if features.ocr_confidence > 0.7 and features.ocr_verdict != "U":
            verdict = features.ocr_verdict
            confidence = features.ocr_confidence
        elif features.vlm_confidence > 0.6 and features.vlm_verdict != "U":
            verdict = features.vlm_verdict
            confidence = features.vlm_confidence
        elif features.nli_confidence > 0.6 and features.nli_verdict != "U":
            verdict = features.nli_verdict
            confidence = features.nli_confidence
        else:
            # Голосование
            verdicts = [features.vlm_verdict, features.ocr_verdict, features.nli_verdict]
            verdict_counts = {"T": 0, "F": 0, "U": 0}
            for v in verdicts:
                verdict_counts[v] += 1
            
            verdict = max(verdict_counts, key=verdict_counts.get)
            confidence = max(verdict_counts.values()) / len(verdicts)
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "features": features
        }
    
    def save_model(self, path: str):
        """Сохранение модели"""
        if self.model is not None:
            try:
                if self.model_type == "lightgbm":
                    self.model.booster_.save_model(f"{path}.txt")
                else:
                    pickle.dump(self.model, open(f"{path}.pkl", "wb"))
                logger.info(f"Модель сохранена в {path}")
            except Exception as e:
                logger.error(f"Ошибка сохранения модели: {e}")
    
    def load_model(self, path: str):
        """Загрузка модели"""
        try:
            if self.model_type == "lightgbm":
                import lightgbm as lgb
                self.model = lgb.Booster(model_file=f"{path}.txt")
            else:
                self.model = pickle.load(open(f"{path}.pkl", "rb"))
            self.is_trained = True
            logger.info(f"Модель загружена из {path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.model = None

class ThresholdOptimizer:
    """Оптимизация порогов для финального принятия решений"""
    
    def __init__(self):
        self.thresholds = {
            "geometry": {"T": 0.6, "F": 0.6, "U": 0.4},
            "physics": {"T": 0.7, "F": 0.7, "U": 0.3}
        }
    
    def optimize_thresholds(self, validation_data: List[Dict]):
        """Оптимизация порогов на валидационных данных"""
        # Простая оптимизация - поиск лучших порогов
        best_thresholds = self.thresholds.copy()
        best_accuracy = 0.0
        
        for domain in ["geometry", "physics"]:
            domain_data = [d for d in validation_data if d.get("domain") == domain]
            if not domain_data:
                continue
            
            for threshold in np.arange(0.3, 0.9, 0.1):
                accuracy = self._evaluate_threshold(domain_data, threshold)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_thresholds[domain] = {
                        "T": threshold,
                        "F": threshold, 
                        "U": 1.0 - threshold
                    }
        
        self.thresholds = best_thresholds
        logger.info(f"Оптимизированные пороги: {self.thresholds}")
    
    def _evaluate_threshold(self, data: List[Dict], threshold: float) -> float:
        """Оценка точности для данного порога"""
        correct = 0
        total = len(data)
        
        for sample in data:
            predicted_verdict = sample["predicted_verdict"]
            true_verdict = sample["true_verdict"]
            confidence = sample["confidence"]
            
            # Применяем порог
            if confidence >= threshold:
                final_verdict = predicted_verdict
            else:
                final_verdict = "U"
            
            if final_verdict == true_verdict:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def apply_threshold(self, verdict: str, confidence: float, domain: str = "geometry") -> str:
        """Применение порога к результату"""
        thresholds = self.thresholds.get(domain, self.thresholds["geometry"])
        
        if confidence >= thresholds.get(verdict, 0.5):
            return verdict
        else:
            return "U"
