"""
GUI на Tkinter для розпізнавання таблиць
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import os
from pathlib import Path

from main import TableRecognitionPipeline


class TableRecognitionGUI:
    """Клас GUI для розпізнавання таблиць"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("📊 Table Recognition System - PaddleOCR")
        self.root.geometry("1200x800")
        
        # Інітіалізація конвеєра
        self.pipeline = TableRecognitionPipeline()
        self.current_image_path = None
        self.is_processing = False
        
        # Створюємо GUI
        self._create_widgets()
        
        print("✅ GUI ініціалізовано")
    
    def _create_widgets(self):
        """Створення елементів GUI"""
        
        # Головна контейнер
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 1. Верхня панель з кнопками
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            button_frame, text="📁 Open Image",
            command=self._open_image
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, text="🔍 Detect & Extract",
            command=self._process_image
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, text="💾 Export to Excel",
            command=self._export_excel
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, text="🗑️ Clear",
            command=self._clear_all
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, text="❌ Exit",
            command=self.root.quit
        ).pack(side=tk.LEFT, padx=5)
        
        # 2. Основна область з перегляду та інформацією
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 2.1 Ліва сторона - зображення
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        ttk.Label(left_frame, text="📷 Preview:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.image_canvas = tk.Canvas(
            left_frame, bg="gray30", width=600, height=600
        )
        self.image_canvas.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # 2.2 Права сторона - інформація та результати
        right_frame = ttk.Frame(content_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # Інформація про файл
        ttk.Label(right_frame, text="📄 File Info:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.info_text = tk.Text(right_frame, height=6, width=40, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X, pady=(5, 10))
        self.info_text.config(state=tk.DISABLED)
        
        # Статистика
        ttk.Label(right_frame, text="📊 Statistics:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.stats_text = tk.Text(right_frame, height=8, width=40, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        self.stats_text.config(state=tk.DISABLED)
        
        # 3. Нижня панель - логування
        log_frame = ttk.LabelFrame(main_frame, text="📝 Log", height=100)
        log_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 0))
        
        # Скролбар для логу
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(
            log_frame, height=6, wrap=tk.WORD,
            yscrollcommand=scrollbar.set
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)
        
        # Перенаправляємо stdout до логу
        self._setup_logging()
    
    def _setup_logging(self):
        """Налаштування логування"""
        class TextHandler:
            def __init__(self, text_widget):
                self.text = text_widget
            
            def write(self, message):
                if message:
                    self.text.config(state=tk.NORMAL)
                    self.text.insert(tk.END, message)
                    self.text.see(tk.END)
                    self.text.config(state=tk.DISABLED)
                    self.text.update()
            
            def flush(self):
                pass
        
        # Зберігаємо оригінальний stdout
        import sys
        self.original_stdout = sys.stdout
        sys.stdout = TextHandler(self.log_text)
    
    def _open_image(self):
        """Відкриття файлу зображення"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.current_image_path = file_path
        print(f"✅ File selected: {file_path}")
        
        # Показуємо переглядження
        self._display_image(file_path)
        
        # Оновлюємо інформацію про файл
        self._update_file_info(file_path)
    
    def _display_image(self, image_path: str, max_width: int = 580, max_height: int = 580):
        """Показ зображення в canvas"""
        try:
            # Завантажуємо зображення
            img = Image.open(image_path)
            
            # Масштабуємо
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Конвертуємо в PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Показуємо на canvas
            self.image_canvas.create_image(
                0, 0, anchor=tk.NW, image=photo
            )
            self.image_canvas.image = photo  # Залишаємо посилання
            
        except Exception as e:
            print(f"❌ Error displaying image: {e}")
    
    def _update_file_info(self, file_path: str):
        """Оновлення інформації про файл"""
        try:
            img = cv2.imread(file_path)
            h, w = img.shape[:2]
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            info = f"""File: {os.path.basename(file_path)}
Size: {size_mb:.2f} MB
Resolution: {w}×{h}
Aspect Ratio: {w/h:.2f}"""
            
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info)
            self.info_text.config(state=tk.DISABLED)
            
        except Exception as e:
            print(f"❌ Error reading file info: {e}")
    
    def _process_image(self):
        """Обробка зображення (в окремому потоці)"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        
        if self.is_processing:
            messagebox.showinfo("Info", "Processing is already in progress...")
            return
        
        # Запускаємо в окремому потоці щоб GUI не замерзав
        thread = threading.Thread(target=self._process_in_thread)
        thread.start()
    
    def _process_in_thread(self):
        """Обробка в окремому потоці"""
        try:
            self.is_processing = True
            print("\n" + "="*50)
            print("🔄 Processing started...")
            print("="*50 + "\n")
            
            # Обробка
            success = self.pipeline.process_image(self.current_image_path)
            
            if success:
                self._update_statistics()
                messagebox.showinfo("Success", "Image processed successfully!")
                
                # Показуємо детектовану таблицю
                if self.pipeline.table_image is not None:
                    table_path = "/tmp/table_preview.png"
                    cv2.imwrite(table_path, self.pipeline.table_image)
                    self._display_image(table_path)
            else:
                messagebox.showerror("Error", "Failed to process image!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            messagebox.showerror("Error", f"Processing error:\n{str(e)}")
        
        finally:
            self.is_processing = False
            print("\n" + "="*50)
            print("✅ Processing finished")
            print("="*50)
    
    def _update_statistics(self):
        """Оновлення статистики"""
        if not self.pipeline.extracted_data:
            return
        
        total_cells = sum(len(cols) for cols in self.pipeline.extracted_data.values())
        problem_cells = 0
        
        for row, cols in self.pipeline.extracted_data.items():
            for col, data in cols.items():
                if data['validation']['status'] != 'ok':
                    problem_cells += 1
        
        accuracy = (total_cells - problem_cells) / total_cells * 100 if total_cells > 0 else 0
        
        stats = f"""Total Cells: {total_cells}
Recognized: {total_cells - problem_cells}
Problems: {problem_cells}
Accuracy: {accuracy:.1f}%

Status: Ready for export"""
        
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats)
        self.stats_text.config(state=tk.DISABLED)
    
    def _export_excel(self):
        """Експорт в Excel"""
        if not self.pipeline.extracted_data:
            messagebox.showwarning("Warning", "No data to export! Process an image first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        print(f"\n💾 Exporting to: {file_path}")
        
        if self.pipeline.export_to_excel(file_path):
            messagebox.showinfo("Success", f"Exported successfully to:\n{file_path}")
            print("✅ Export completed!")
        else:
            messagebox.showerror("Error", "Export failed!")
    
    def _clear_all(self):
        """Очистка всіх даних"""
        self.current_image_path = None
        self.pipeline = TableRecognitionPipeline()
        
        self.image_canvas.delete("all")
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.config(state=tk.DISABLED)
        
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.config(state=tk.DISABLED)
        
        print("\n🗑️ Cleared all data")


def main():
    root = tk.Tk()
    gui = TableRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
