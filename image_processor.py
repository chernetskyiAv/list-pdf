"""
Обробка зображень - передобробка для високої якості розпізнавання
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImageProcessor:
    """Клас для обробки зображень перед розпізнаванням"""
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Завантаження зображення з файлу
        
        Args:
            image_path: шлях до файлу
            
        Returns:
            cv2 зображення або None якщо помилка
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Не вдалося завантажити: {image_path}")
                return None
            
            self.original_image = img.copy()
            print(f"✅ Зображення завантажено: {img.shape}")
            return img
            
        except Exception as e:
            print(f"❌ Помилка завантаження: {str(e)}")
            return None
    
    def preprocess_for_table_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Передобробка для детекції таблиці
        
        Args:
            image: вхідне зображення
            
        Returns:
            обробљене зображення для детекції
        """
        # Конвертуємо в сірий
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Посилення контрасту CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Гаусівське розмиття для видалення шумів
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Морфологічні операції для з'єднання ліній таблиці
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return closed
    
    def preprocess_for_ocr(self, image: np.ndarray, scale: float = 2.0) -> np.ndarray:
        """
        Передобробка для OCR розпізнавання
        
        Args:
            image: вхідне зображення
            scale: коефіцієнт масштабування для кращої якості
            
        Returns:
            обробљене зображення для OCR
        """
        # Масштабування для кращої якості
        if scale > 1:
            h, w = image.shape[:2]
            image = cv2.resize(image, (int(w * scale), int(h * scale)), 
                             interpolation=cv2.INTER_CUBIC)
        
        # Конвертуємо в сірий
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # CLAHE для посилення контрасту
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Біль гаусівське розмиття для видалення шумів
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Морфологічні операції спеціально для цифр
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Адаптивна бінаризація для рукопису
        binary = cv2.adaptiveThreshold(
            morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Конвертуємо назад в BGR для PaddleOCR
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def detect_skew(self, image: np.ndarray) -> float:
        """
        Детекція нахилу документа
        
        Args:
            image: сіре зображення
            
        Returns:
            кут нахилу в градусах
        """
        # Детекція ліній Hough
        edges = cv2.Canny(image, 100, 200)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)
            if angle > 90:
                angle = angle - 180
            angles.append(angle)
        
        # Середній кут
        median_angle = np.median(angles)
        return median_angle
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Повертання зображення на заданий кут
        
        Args:
            image: вхідне зображення
            angle: кут повертання в градусах
            
        Returns:
            повернене зображення
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Матриця повертання
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Виконуємо повертання
        rotated = cv2.warpAffine(
            image, M, (w, h),
            borderMode=cv2.BORDER_REFLECT,
            flags=cv2.INTER_CUBIC
        )
        
        return rotated
    
    def extract_roi(self, image: np.ndarray, x: int, y: int, 
                   w: int, h: int) -> np.ndarray:
        """
        Виділення регіону інтересу (ROI)
        
        Args:
            image: вхідне зображення
            x, y, w, h: координати та розміри ROI
            
        Returns:
            виділена область
        """
        # Перевірка меж
        y = max(0, y)
        x = max(0, x)
        h = min(h, image.shape[0] - y)
        w = min(w, image.shape[1] - x)
        
        if h <= 0 or w <= 0:
            return np.array([])
        
        return image[y:y+h, x:x+w]
    
    def resize_image(self, image: np.ndarray, max_width: int = 1200,
                    max_height: int = 1600) -> np.ndarray:
        """
        Зміна розміру зображення з збереженням пропорцій
        
        Args:
            image: вхідне зображення
            max_width, max_height: максимальні розміри
            
        Returns:
            змінене зображення
        """
        h, w = image.shape[:2]
        
        # Обчислюємо коефіцієнт масштабування
        scale = min(1.0, max_width / w, max_height / h)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
    
    def get_image_info(self, image: np.ndarray) -> dict:
        """
        Отримання інформації про зображення
        
        Args:
            image: зображення
            
        Returns:
            словник з інформацією
        """
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        return {
            'height': h,
            'width': w,
            'channels': channels,
            'size_mb': (h * w * channels) / (1024 * 1024),
            'aspect_ratio': w / h if h > 0 else 0
        }
