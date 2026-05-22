"""
OCR処理モジュール - PaddleOCRを使用したテキスト認識
Модуль OCR обработки - распознавание текста с использованием PaddleOCR
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Tuple, Dict
import re


class OCRProcessor:
    """PaddleOCRを使用したテキスト認識クラス"""
    
    def __init__(self, languages=['uk', 'en'], use_gpu=False):
        """
        PaddleOCRの初期化
        
        Args:
            languages: 認識する言語リスト (デフォルト: ウクライナ語、英語)
            use_gpu: GPUを使用するか (デフォルト: False)
        """
        print("🚀 Завантажую PaddleOCR модель...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=languages,
            use_gpu=use_gpu,
            enable_mkldnn=True
        )
        self.confidence_threshold = 0.5
        print("✅ PaddleOCR завантажена успішно!")
    
    def recognize_text(self, image: np.ndarray) -> List[Dict]:
        """
        Розпізнавання тексту на зображенні
        
        Args:
            image: cv2 зображення (RGB)
            
        Returns:
            Список слів з координатами та впевненістю
        """
        if image is None or image.size == 0:
            return []
        
        # Конвертуємо BGR в RGB якщо треба
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        try:
            results = self.ocr.ocr(rgb_image, cls=True)
            
            extracted_text = []
            if results and results[0]:
                for line in results[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    if confidence >= self.confidence_threshold:
                        # Отримуємо координати (середня точка)
                        points = np.array(line[0], dtype=np.float32)
                        x = int(np.mean(points[:, 0]))
                        y = int(np.mean(points[:, 1]))
                        
                        extracted_text.append({
                            'text': text.strip(),
                            'confidence': confidence,
                            'x': x,
                            'y': y,
                            'box': points.tolist()
                        })
            
            return extracted_text
            
        except Exception as e:
            print(f"❌ Помилка OCR: {str(e)}")
            return []
    
    def recognize_cell(self, cell_image: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Розпізнавання тексту в одній комірці таблиці
        
        Args:
            cell_image: зображення комірки
            
        Returns:
            Кортеж (текст, впевненість, метаінформація)
        """
        if cell_image is None or cell_image.size == 0:
            return '', 0.0, {'status': 'empty', 'reason': 'Пуста комірка'}
        
        # Попередня обробка для цифр
        processed = self._preprocess_for_digits(cell_image)
        
        try:
            # RGB конвертація
            if len(processed.shape) == 3 and processed.shape[2] == 3:
                rgb_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = processed
            
            results = self.ocr.ocr(rgb_image, cls=True)
            
            if not results or not results[0]:
                return '', 0.0, {'status': 'no_text', 'reason': 'Текст не знайдений'}
            
            # Беремо найліпший результат
            best_result = max(results[0], key=lambda x: x[1][1])
            text = best_result[1][0].strip()
            confidence = best_result[1][1]
            
            # Валідація для цифр
            validation = self._validate_digits(text, confidence)
            
            return text, confidence, validation
            
        except Exception as e:
            return '', 0.0, {'status': 'error', 'reason': f'Помилка: {str(e)}'}
    
    def _preprocess_for_digits(self, image: np.ndarray) -> np.ndarray:
        """
        Спеціальна предобробка для рукописних цифр
        """
        if image is None or image.size == 0:
            return image
        
        # Конвертуємо в сірий
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Посилення контрасту CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Морфологічні операції для з'єднання розривів
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=1)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Дилатація для посилення цифр
        enhanced = cv2.dilate(enhanced, kernel, iterations=1)
        
        # Конвертуємо назад в BGR для PaddleOCR
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def _validate_digits(self, text: str, confidence: float) -> Dict:
        """
        Валідація розпізнаного тексту (перевірка для цифр)
        """
        problems = []
        status = 'ok'
        
        # Очищаємо від неприпустимих символів
        cleaned = re.sub(r'[^0-9,.\-\+\(\)% ]', '', text)
        
        if not cleaned or not any(c.isdigit() for c in cleaned):
            status = 'warning'
            problems.append('Не знайдено цифр')
        
        if confidence < 0.7:
            status = 'warning'
            problems.append(f'Низька впевненість: {confidence:.2%}')
        
        if len(cleaned) > 20:
            status = 'warning'
            problems.append('Надто довгий текст')
        
        return {
            'status': status,
            'confidence': confidence,
            'cleaned_text': cleaned.strip(),
            'problems': problems
        }
    
    def batch_recognize_cells(self, cells: List[np.ndarray]) -> List[Tuple[str, float, Dict]]:
        """
        Розпізнавання кількох комірок за раз
        """
        results = []
        for i, cell in enumerate(cells):
            text, conf, validation = self.recognize_cell(cell)
            results.append((text, conf, validation))
            
            if (i + 1) % 10 == 0:
                print(f"  ✓ Розпізнано {i + 1}/{len(cells)} комірок...")
        
        return results
