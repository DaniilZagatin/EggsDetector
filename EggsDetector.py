import os
import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, maximum_filter
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class EggCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Egg Counter (Contours & Maxima)")
        self.root.geometry("1600x900")

        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.eggs_mask = None
        self.local_maxima_img = None
        self.white_count = 0
        self.red_count = 0

        self.white_lower_r = StringVar(value="205")
        self.white_lower_g = StringVar(value="215")
        self.white_lower_b = StringVar(value="212")

        self.red_lower_l = StringVar(value="50")
        self.red_lower_a = StringVar(value="125")
        self.red_lower_b = StringVar(value="120")

        self.red_upper_l = StringVar(value="210")
        self.red_upper_a = StringVar(value="160")
        self.red_upper_b = StringVar(value="150")

        self.open_iter = IntVar(value=14)
        self.close_iter = IntVar(value=0)
        self.min_area = IntVar(value=10000)

        self.kernel_blur = StringVar(value="31")

        self.clahe_clip = StringVar(value="3.0")
        self.clahe_tilesize = IntVar(value=8)
        self.l_dilate_iter = IntVar(value=1)

        self.sigma = DoubleVar(value=2.0)
        self.neighborhood_size = IntVar(value=200)

        self.create_widgets()

    def create_widgets(self):

        control_frame = Frame(self.root)
        control_frame.pack(side=TOP, fill=X, padx=10, pady=10)

        btn_frame = Frame(control_frame)
        btn_frame.pack(side=LEFT, padx=20)

        self.load_btn = Button(btn_frame, text="Загрузить изображение",
                               command=self.load_image, width=20)
        self.load_btn.pack(pady=5)

        self.preprocess_btn = Button(btn_frame, text="Предобработка",
                                     command=self.preprocess_image, state=DISABLED, width=20)
        self.preprocess_btn.pack(pady=5)

        self.count_btn = Button(btn_frame, text="Подсчитать яйца",
                                command=self.count_eggs, state=DISABLED, width=20)
        self.count_btn.pack(pady=5)


        self.filename_entry = Entry(btn_frame, width=20)
        self.filename_entry.insert(0, "result")
        self.filename_entry.pack(pady=5)

        self.save_btn = Button(btn_frame, text="Скачать результат",
                               command=self.save_results, state=DISABLED, width=20)
        self.save_btn.pack(pady=5)

        # --- Пороговые значения для белых (RGB) ---
        white_thresh_frame = Frame(control_frame)
        white_thresh_frame.pack(side=LEFT, padx=10)

        Label(white_thresh_frame, text="Порог белых (min R,G,B)",
              font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=3)

        Label(white_thresh_frame, text="R:").grid(row=1, column=0, sticky=E)
        Entry(white_thresh_frame, textvariable=self.white_lower_r, width=5).grid(row=1, column=1)

        Label(white_thresh_frame, text="G:").grid(row=2, column=0, sticky=E)
        Entry(white_thresh_frame, textvariable=self.white_lower_g, width=5).grid(row=2, column=1)

        Label(white_thresh_frame, text="B:").grid(row=3, column=0, sticky=E)
        Entry(white_thresh_frame, textvariable=self.white_lower_b, width=5).grid(row=3, column=1)

        # --- Пороговые значения для красных (LAB) ---
        red_thresh_frame = Frame(control_frame)
        red_thresh_frame.pack(side=LEFT, padx=10)

        Label(red_thresh_frame, text="Порог красных (min..max) [LAB]",
              font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=4)

        # Min L, A, B
        Label(red_thresh_frame, text="Min L:").grid(row=1, column=0, sticky=E)
        Entry(red_thresh_frame, textvariable=self.red_lower_l, width=5).grid(row=1, column=1)

        Label(red_thresh_frame, text="Min A:").grid(row=2, column=0, sticky=E)
        Entry(red_thresh_frame, textvariable=self.red_lower_a, width=5).grid(row=2, column=1)

        Label(red_thresh_frame, text="Min B:").grid(row=3, column=0, sticky=E)
        Entry(red_thresh_frame, textvariable=self.red_lower_b, width=5).grid(row=3, column=1)

        # Max L, A, B
        Label(red_thresh_frame, text="Max L:").grid(row=1, column=2, sticky=E)
        Entry(red_thresh_frame, textvariable=self.red_upper_l, width=5).grid(row=1, column=3)

        Label(red_thresh_frame, text="Max A:").grid(row=2, column=2, sticky=E)
        Entry(red_thresh_frame, textvariable=self.red_upper_a, width=5).grid(row=2, column=3)

        Label(red_thresh_frame, text="Max B:").grid(row=3, column=2, sticky=E)
        Entry(red_thresh_frame, textvariable=self.red_upper_b, width=5).grid(row=3, column=3)

        # --- Морфология ---
        morph_frame = Frame(control_frame)
        morph_frame.pack(side=LEFT, padx=20)

        Label(morph_frame, text="Морфология", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2)

        Label(morph_frame, text="Открытие:").grid(row=1, column=0, sticky=E)
        Scale(morph_frame, from_=0, to=15, orient=HORIZONTAL, variable=self.open_iter).grid(row=1, column=1)

        Label(morph_frame, text="Закрытие:").grid(row=2, column=0, sticky=E)
        Scale(morph_frame, from_=0, to=15, orient=HORIZONTAL, variable=self.close_iter).grid(row=2, column=1)

        Label(morph_frame, text="Мин площадь:").grid(row=3, column=0, sticky=E)
        Scale(morph_frame, from_=0, to=20000, orient=HORIZONTAL, variable=self.min_area).grid(row=3, column=1)

        blur_frame = Frame(control_frame)
        blur_frame.pack(side=LEFT, padx=30)

        Label(blur_frame, text="Размытие (kernel)", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2)
        Entry(blur_frame, textvariable=self.kernel_blur, width=5).grid(row=1, column=0, padx=5, pady=5)

        preproc_params_frame = Frame(control_frame)
        preproc_params_frame.pack(side=LEFT, padx=30)

        Label(preproc_params_frame, text="Параметры предобработки", font=('Arial', 10, 'bold')
              ).grid(row=0, column=0, columnspan=2)

        Label(preproc_params_frame, text="ClipLimit:").grid(row=1, column=0, sticky=E)
        Entry(preproc_params_frame, textvariable=self.clahe_clip, width=5).grid(row=1, column=1)

        Label(preproc_params_frame, text="TileGrid:").grid(row=2, column=0, sticky=E)
        Entry(preproc_params_frame, textvariable=self.clahe_tilesize, width=5).grid(row=2, column=1)

        Label(preproc_params_frame, text="Dilate L:").grid(row=3, column=0, sticky=E)
        Scale(preproc_params_frame, from_=0, to=5, orient=HORIZONTAL, variable=self.l_dilate_iter).grid(row=3, column=1)

        maxima_frame = Frame(control_frame)
        maxima_frame.pack(side=LEFT, padx=20)

        Label(maxima_frame, text="Максимумы & Изолинии", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2)

        Label(maxima_frame, text="sigma:").grid(row=1, column=0, sticky=E)
        Scale(maxima_frame, from_=0.5, to=5.0, resolution=0.5,
              orient=HORIZONTAL, variable=self.sigma, command=self.update_local_maxima).grid(row=1, column=1)

        Label(maxima_frame, text="size:").grid(row=2, column=0, sticky=E)
        Scale(maxima_frame, from_=20, to=500, resolution=20,
              orient=HORIZONTAL, variable=self.neighborhood_size, command=self.update_local_maxima).grid(row=2, column=1)

        images_frame = Frame(self.root)
        images_frame.pack(side=TOP, fill=BOTH, expand=True)

        # 1) Оригинал
        self.original_label = Label(images_frame, text="Оригинал")
        self.original_label.grid(row=0, column=0, padx=10, pady=10)

        self.original_canvas = Canvas(images_frame, width=400, height=350, bg='white')
        self.original_canvas.grid(row=1, column=0, padx=10, pady=10)

        # 2) Обработанное
        self.processed_label = Label(images_frame, text="Обработанное изображение")
        self.processed_label.grid(row=0, column=1, padx=10, pady=10)

        self.processed_canvas = Canvas(images_frame, width=400, height=350, bg='white')
        self.processed_canvas.grid(row=1, column=1, padx=10, pady=10)

        # 3) Карта лок. максимумов + изолинии
        self.maxima_label = Label(images_frame, text="Максимумы + изолинии")
        self.maxima_label.grid(row=0, column=2, padx=10, pady=10)

        self.maxima_canvas = Canvas(images_frame, width=400, height=350, bg='white')
        self.maxima_canvas.grid(row=1, column=2, padx=10, pady=10)

        self.result_label = Label(images_frame, text="", font=('Arial', 14))
        self.result_label.grid(row=2, column=0, columnspan=3, pady=10)


    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображение.")
                return
            self.image_path = file_path
            self.original_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            self.processed_image = self.original_image.copy()
            self.eggs_mask = None
            self.local_maxima_img = None
            self.white_count = 0
            self.red_count = 0

            self.display_image(self.original_image, self.original_canvas)

            self.preprocess_btn.config(state=NORMAL)
            self.count_btn.config(state=NORMAL)
            self.result_label.config(text="")

            # Сформировать/обновить карту яркости
            self.update_local_maxima()

    def preprocess_image(self):
        if self.original_image is None:
            return

        clip_val = float(self.clahe_clip.get())
        tile_val = int(self.clahe_tilesize.get())
        ld_iter = self.l_dilate_iter.get()
        blur_size = int(self.kernel_blur.get())

        # Переводим в LAB
        lab = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Расширяем L
        kernel = np.ones((3, 3), np.uint8)
        l_dilated = cv2.dilate(l, kernel, iterations=ld_iter)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_val, tileGridSize=(tile_val, tile_val))
        l_clahe = clahe.apply(l_dilated)

        # MedianBlur
        blur = cv2.medianBlur(l_clahe, blur_size)

        # Доп. dilate
        l_final = cv2.dilate(blur, kernel, iterations=3)
        lab_processed = cv2.merge((l_final, a, b))

        result = cv2.cvtColor(lab_processed, cv2.COLOR_LAB2RGB)
        self.processed_image = result
        self.display_image(self.processed_image, self.processed_canvas)

    def count_eggs(self):
        if self.processed_image is None:
            return


        if not self.validate_thresholds():
            messagebox.showerror("Ошибка", "Некорректные значения (0..255) в порогах.")
            return

        w_lower = np.array([
            int(self.white_lower_r.get()),
            int(self.white_lower_g.get()),
            int(self.white_lower_b.get())
        ])
        w_upper = np.array([255, 255, 255])

        mask_white = cv2.inRange(self.processed_image, w_lower, w_upper)

        kernel = np.ones((5, 5), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel, iterations=self.open_iter.get())
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=self.close_iter.get())

        cont_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_w = [c for c in cont_white if cv2.contourArea(c) > self.min_area.get()]
        self.white_count = len(filtered_w)

        lab_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2LAB)

        r_lower = np.array([
            int(self.red_lower_l.get()),
            int(self.red_lower_a.get()),
            int(self.red_lower_b.get())
        ])
        r_upper = np.array([
            int(self.red_upper_l.get()),
            int(self.red_upper_a.get()),
            int(self.red_upper_b.get())
        ])

        mask_red_lab = cv2.inRange(lab_image, r_lower, r_upper)

        mask_red_lab = cv2.morphologyEx(mask_red_lab, cv2.MORPH_OPEN, kernel, iterations=self.open_iter.get())
        mask_red_lab = cv2.morphologyEx(mask_red_lab, cv2.MORPH_CLOSE, kernel, iterations=self.close_iter.get())

        cont_red, _ = cv2.findContours(mask_red_lab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_r = [c for c in cont_red if cv2.contourArea(c) > self.min_area.get()]
        self.red_count = len(filtered_r)


        h, w_ = mask_white.shape
        color_mask = np.zeros((h, w_, 3), dtype=np.uint8)
        color_mask[mask_white != 0] = (255, 255, 255)
        color_mask[mask_red_lab != 0] = (255, 0, 0)

        self.eggs_mask = color_mask
        self.display_image(self.eggs_mask, self.processed_canvas)

        self.result_label.config(
            text=f"Белые яйца: {self.white_count} | Красные яйца: {self.red_count}"
        )
        self.save_btn.config(state=NORMAL)

    def update_local_maxima(self, event=None):
        if self.original_image is None:
            return

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

        sigma_val = float(self.sigma.get())
        smoothed = gaussian_filter(gray, sigma=sigma_val)

        size_val = int(self.neighborhood_size.get())
        maxfilt = maximum_filter(smoothed, size_val)
        local_max = (smoothed == maxfilt)

        fig, ax = plt.subplots(figsize=(4, 3.5), dpi=100)
        fig.subplots_adjust(0, 0, 1, 1)

        minv, maxv = smoothed.min(), smoothed.max()
        ax.imshow(smoothed, cmap='inferno', vmin=minv, vmax=maxv)
        ax.axis('off')

        levels = np.linspace(minv, maxv, 10)
        ax.contour(smoothed, levels=levels, colors='white', linewidths=1)

        yy, xx = np.where(local_max)
        ax.scatter(xx, yy, c='red', s=20, marker='o')

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        plt.close(fig)

        self.local_maxima_img = buf
        self.display_image(buf, self.maxima_canvas)

    def display_image(self, image: np.ndarray, canvas: Canvas):
        if image is None:
            return

        fixed_w, fixed_h = 400, 350
        resized_image = cv2.resize(image, (fixed_w, fixed_h), interpolation=cv2.INTER_AREA)

        img_pil = Image.fromarray(resized_image)
        imgtk = ImageTk.PhotoImage(image=img_pil)

        canvas.delete("all")
        canvas.config(width=fixed_w, height=fixed_h)
        canvas.create_image(0, 0, anchor=NW, image=imgtk)
        canvas.image = imgtk

    def save_results(self):
        if self.eggs_mask is None or self.processed_image is None or self.local_maxima_img is None:
            messagebox.showinfo(
                "Сохранение",
                "Сначала загрузите изображение, выполните предобработку, "
                "подсчитайте яйца и сформируйте карту яркости."
            )
            return

        base_filename = self.filename_entry.get().strip()
        if not base_filename:
            messagebox.showerror("Ошибка", "Пожалуйста, введите имя для файлов перед сохранением.")
            return

        dir_path = filedialog.askdirectory(title="Выберите папку для сохранения:")
        if not dir_path:
            return

        mask_path = f"{dir_path}/{base_filename}_mask.png"
        preprocessed_path = f"{dir_path}/{base_filename}_original.png"
        maxima_path = f"{dir_path}/{base_filename}_heatmap.png"

        # Маска
        mask_bgr = cv2.cvtColor(self.eggs_mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_path, mask_bgr)

        # Предобработанное изображение
        prep_bgr = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(preprocessed_path, prep_bgr)

        # Карта яркости
        maxima_bgr = cv2.cvtColor(self.local_maxima_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(maxima_path, maxima_bgr)

        messagebox.showinfo(
            "Сохранение",
            f"Файлы сохранены:\n"
            f"1) {mask_path}\n"
            f"2) {preprocessed_path}\n"
            f"3) {maxima_path}"
        )

    def validate_thresholds(self) -> bool:
        # Проверяем белые (RGB)
        if not self.check_val(self.white_lower_r.get()): return False
        if not self.check_val(self.white_lower_g.get()): return False
        if not self.check_val(self.white_lower_b.get()): return False

        # Проверяем красные (LAB)
        if not self.check_val(self.red_lower_l.get()): return False
        if not self.check_val(self.red_lower_a.get()): return False
        if not self.check_val(self.red_lower_b.get()): return False
        if not self.check_val(self.red_upper_l.get()): return False
        if not self.check_val(self.red_upper_a.get()): return False
        if not self.check_val(self.red_upper_b.get()): return False

        return True

    def check_val(self, val_str: str) -> bool:
        try:
            iv = int(val_str)
            return 0 <= iv <= 255
        except ValueError:
            return False


if __name__ == "__main__":
    root = Tk()
    app = EggCounterApp(root)
    root.mainloop()
