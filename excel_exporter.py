"""
Експорт розпізнаних даних в Excel з двома аркушами
"""

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from typing import Dict, List
from datetime import datetime


class ExcelExporter:
    """Клас для експорту даних в Excel"""
    
    def __init__(self):
        self.workbook = None
        self.ws_all = None
        self.ws_problems = None
    
    def export_table(self, extracted_data: Dict, output_path: str) -> bool:
        """
        Експорт таблиці в Excel з двома аркушами
        
        Args:
            extracted_data: словник з розпізнаними даними
            output_path: шлях для збереження
            
        Returns:
            успішність експорту
        """
        try:
            print(f"📊 Подготовка экспорта...")
            
            # Створюємо робочу книгу
            self.workbook = Workbook()
            self.workbook.remove(self.workbook.active)
            
            # Створюємо два аркуші
            self.ws_all = self.workbook.create_sheet("All Data", 0)
            self.ws_problems = self.workbook.create_sheet("Problem Cells", 1)
            
            # Визначаємо розміри таблиці
            max_row = max(extracted_data.keys()) + 1 if extracted_data else 0
            max_col = max(
                max(cols.keys()) + 1 
                for cols in extracted_data.values()
            ) if extracted_data else 0
            
            print(f"📐 Размер таблицы: {max_row} рядків × {max_col} колонок")
            
            # Заповнюємо аркуш "All Data"
            self._fill_all_data_sheet(extracted_data, max_row, max_col)
            
            # Заповнюємо аркуш "Problem Cells"
            self._fill_problem_cells_sheet(extracted_data)
            
            # Форматуємо
            self._format_sheets()
            
            # Зберігаємо
            print(f"💾 Сохранение файла...")
            self.workbook.save(output_path)
            
            print(f"✅ Успешный экспорт в: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка экспорта: {str(e)}")
            return False
    
    def _fill_all_data_sheet(self, extracted_data: Dict, max_row: int, max_col: int):
        """Заповнення аркуша з усіма даними"""
        print("  📝 Заполнение листа 'All Data'...")
        
        # Заголовок
        self.ws_all['A1'] = "Table Recognition Results"
        self.ws_all['A2'] = f"Дата обработки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Таблиця з даними
        for row_idx, cols in extracted_data.items():
            for col_idx, data in cols.items():
                cell = self.ws_all.cell(
                    row=row_idx + 4,
                    column=col_idx + 1,
                    value=data['text'] or ""
                )
                
                # Колірування проблемних комірок
                if data['validation']['status'] != 'ok':
                    cell.fill = PatternFill(
                        start_color="FFFF99",
                        end_color="FFFF99",
                        fill_type="solid"
                    )
        
        # Автоширина колонок
        for col_idx in range(1, max_col + 1):
            self.ws_all.column_dimensions[chr(64 + col_idx)].width = 12
    
    def _fill_problem_cells_sheet(self, extracted_data: Dict):
        """Заповнення аркуша з проблемними комірками"""
        print("  ⚠️  Заполнение листа 'Problem Cells'...")
        
        # Заголовки
        headers = ["Row", "Column", "Text", "Confidence", "Problems", "Reason"]
        for col_idx, header in enumerate(headers, 1):
            cell = self.ws_problems.cell(row=1, column=col_idx, value=header)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center")
        
        # Заповнюємо проблемні комірки
        row_num = 2
        problem_count = 0
        
        for row_idx, cols in extracted_data.items():
            for col_idx, data in cols.items():
                if data['validation']['status'] != 'ok':
                    problems = data['validation'].get('problems', [])
                    reason = "; ".join(problems) if problems else "Unknown"
                    
                    self.ws_problems.cell(row=row_num, column=1, value=row_idx)
                    self.ws_problems.cell(row=row_num, column=2, value=col_idx)
                    self.ws_problems.cell(row=row_num, column=3, value=data['text'])
                    self.ws_problems.cell(row=row_num, column=4, value=f"{data['confidence']:.1%}")
                    self.ws_problems.cell(row=row_num, column=5, value=data['validation']['status'])
                    self.ws_problems.cell(row=row_num, column=6, value=reason)
                    
                    row_num += 1
                    problem_count += 1
        
        if problem_count == 0:
            self.ws_problems.cell(row=2, column=1, value="No problems found!")
            self.ws_problems.cell(row=2, column=1).font = Font(italic=True, color="008000")
        
        # Автоширина колонок
        for col_idx in range(1, len(headers) + 1):
            self.ws_problems.column_dimensions[chr(64 + col_idx)].width = 15
    
    def _format_sheets(self):
        """Форматування листів"""
        # Бордери
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # All Data
        for row in self.ws_all.iter_rows(min_row=4):
            for cell in row:
                if cell.value is not None:
                    cell.border = border
                    cell.alignment = Alignment(horizontal="center", vertical="center")
        
        # Problem Cells
        for row in self.ws_problems.iter_rows():
            for cell in row:
                cell.border = border
                cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        
        # Заморозка рядків
        self.ws_all.freeze_panes = "A4"
        self.ws_problems.freeze_panes = "A2"
    
    def export_with_statistics(self, extracted_data: Dict, output_path: str) -> bool:
        """
        Експорт з додатковою статистикою
        """
        try:
            self.export_table(extracted_data, output_path)
            
            # Додаємо лист зі статистикою
            ws_stats = self.workbook.create_sheet("Statistics", 2)
            
            total_cells = sum(len(cols) for cols in extracted_data.values())
            problem_cells = sum(
                1 for row, cols in extracted_data.items()
                for col, data in cols.items()
                if data['validation']['status'] != 'ok'
            )
            
            accuracy = (total_cells - problem_cells) / total_cells * 100 if total_cells > 0 else 0
            
            # Заповнюємо статистику
            stats = [
                ("Metric", "Value"),
                ("Total Cells", total_cells),
                ("Recognized", total_cells - problem_cells),
                ("Problems", problem_cells),
                ("Accuracy", f"{accuracy:.1f}%"),
                ("Export Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            ]
            
            for row_idx, (metric, value) in enumerate(stats, 1):
                ws_stats.cell(row=row_idx, column=1, value=metric).font = Font(bold=True)
                ws_stats.cell(row=row_idx, column=2, value=value)
            
            self.workbook.save(output_path)
            return True
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return False
