"""
Детекція таблиці та структури колонок
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.cluster import DBSCAN


class TableDetector:
    """Клас для детекції таблиці та її структури"""
    
    def __init__(self):
        self.table_roi = None
        self.rows = []
        self.columns = []
        self.cells = []
    
    def detect_table(self, image: np.ndarray, 
                    min_table_area: int = 5000) -> Optional[Tuple]:
        """
        Детекція таблиці на зображенні
        
        Args:
            image: обробљене зображення
            min_table_area: мінімальна площа таблиці в пікселях
            
        Returns:
            (x, y, w, h) координати таблиці або None
        """
        # Бінаризація
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Інверсія для пошуку об'єктів
        binary = cv2.bitwise_not(binary)
        
        # Знаходження контурів
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("❌ Таблиця не знайдена")
            return None
        
        # Фільтруємо контури за площею
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_table_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Фільтруємо за aspect ratio (не надто витягнута)
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0:
                    valid_contours.append((contour, area, (x, y, w, h)))
        
        if not valid_contours:
            print("❌ Таблиця не знайдена (недостатньо великих об'єктів)")
            return None
        
        # Беремо найбільший контур (таблиця)
        largest = max(valid_contours, key=lambda x: x[1])
        x, y, w, h = largest[2]
        
        self.table_roi = (x, y, w, h)
        print(f"✅ Таблиця знайдена: x={x}, y={y}, w={w}, h={h}")
        
        return self.table_roi
    
    def detect_lines(self, image: np.ndarray) -> Tuple[List, List]:
        """
        Детекція горизонтальних та вертикальних ліній таблиці
        
        Args:
            image: сіре зображення таблиці
            
        Returns:
            (горизонтальні_лінії, вертикальні_лінії)
        """
        # Бінаризація
        _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
        
        # Гоор лінії (горизонтальне ядро)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
        
        # Вертикальні лінії (вертикальне ядро)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
        
        # Детекція контурів ліній
        h_contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_contours, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h_lines_coords = []
        for contour in h_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10:  # Мінімальна ширина
                h_lines_coords.append(y + h // 2)
        
        v_lines_coords = []
        for contour in v_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 10:  # Мінімальна висота
                v_lines_coords.append(x + w // 2)
        
        # Сортуємо
        h_lines_coords = sorted(set(h_lines_coords))
        v_lines_coords = sorted(set(v_lines_coords))
        
        print(f"🔍 Знайдено {len(h_lines_coords)} горизонтальних ліній")
        print(f"🔍 Знайдено {len(v_lines_coords)} вертикальних ліній")
        
        return h_lines_coords, v_lines_coords
    
    def detect_columns_by_projection(self, image: np.ndarray) -> List[int]:
        """
        Детекція колонок через вертикальну проекцію
        
        Args:
            image: сіре зображення таблиці
            
        Returns:
            список x-координат границь колонок
        """
        # Бінаризація
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Вертикальна проекція (сумуємо пікселі по вертикалі)
        vertical_projection = np.sum(binary, axis=0)
        
        # Нормалізуємо
        if vertical_projection.max() > 0:
            vertical_projection = vertical_projection / vertical_projection.max()
        
        # Знаходимо долини (місця з мало пікселей = разделители колонок)
        threshold = 0.1
        valleys = np.where(vertical_projection < threshold)[0]
        
        # Групуємо близькі точки
        column_boundaries = []
        if len(valleys) > 0:
            current_valley = valleys[0]
            for valley in valleys[1:]:
                if valley - current_valley > 5:  # Поріг групування
                    column_boundaries.append(int(np.mean([current_valley, valley])))
                    current_valley = valley
        
        # Додаємо крайні границі
        column_boundaries = [0] + sorted(set(column_boundaries)) + [image.shape[1]]
        
        print(f"📊 Детектовано {len(column_boundaries) - 1} колонок")
        
        return column_boundaries
    
    def extract_cells(self, image: np.ndarray, h_lines: List[int], 
                     v_lines: List[int]) -> List[Tuple[np.ndarray, Dict]]:
        """
        Виділення комірок таблиці
        
        Args:
            image: вхідне зображення (кольорове)
            h_lines: горизонтальні лінії (y-координати)
            v_lines: вертикальні лінії (x-координати)
            
        Returns:
            список (зображення_комірки, метаінформація)
        """
        cells = []
        h, w = image.shape[:2]
        
        # Додаємо крайні границі якщо їх немає
        if h_lines[0] != 0:
            h_lines = [0] + h_lines
        if h_lines[-1] != h:
            h_lines = h_lines + [h]
        
        if v_lines[0] != 0:
            v_lines = [0] + v_lines
        if v_lines[-1] != w:
            v_lines = v_lines + [w]
        
        # Витягуємо комірки
        for row_idx in range(len(h_lines) - 1):
            for col_idx in range(len(v_lines) - 1):
                y1, y2 = h_lines[row_idx], h_lines[row_idx + 1]
                x1, x2 = v_lines[col_idx], v_lines[col_idx + 1]
                
                # Пропускаємо дуже малі комірки
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue
                
                cell_img = image[y1:y2, x1:x2]
                
                cell_info = {
                    'row': row_idx,
                    'col': col_idx,
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1
                }
                
                cells.append((cell_img, cell_info))
        
        self.cells = cells
        print(f"📦 Витягнено {len(cells)} комірок")
        
        return cells
    
    def visualize_table(self, image: np.ndarray, h_lines: List[int],
                       v_lines: List[int], output_path: str = None) -> np.ndarray:
        """
        Візуалізація детектованої таблиці та ліній
        
        Args:
            image: вхідне зображення
            h_lines: горизонтальні лінії
            v_lines: вертикальні лінії
            output_path: шлях для збереження (опціонально)
            
        Returns:
            зображення з намальованими лініями
        """
        viz = image.copy()
        if len(viz.shape) == 2:
            viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
        
        # Малюємо горизонтальні лінії (червоні)
        for y in h_lines:
            cv2.line(viz, (0, y), (viz.shape[1], y), (0, 0, 255), 2)
        
        # Малюємо вертикальні лінії (зелені)
        for x in v_lines:
            cv2.line(viz, (x, 0), (x, viz.shape[0]), (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, viz)
            print(f"💾 Візуалізація збережена: {output_path}")
        
        return viz
    
    def get_cell_by_position(self, row: int, col: int) -> Optional[np.ndarray]:
        """
        Отримання комірки за позицією
        
        Args:
            row: номер рядка
            col: номер колонки
            
        Returns:
            зображення комірки або None
        """
        for cell_img, info in self.cells:
            if info['row'] == row and info['col'] == col:
                return cell_img
        
        return None
