# -*- coding: utf-8 -*-
"""
PDI – Color (Actividad práctica)
Interfaz en Tkinter que implementa los 8 pasos del workflow propuesto:

1) Normalizar RGB del píxel (0..255 -> 0..1)
2) Convertir RGB -> YIQ (usando la "segunda matriz")
3) Y' := a * Y  (coeficiente de luminancia) a
4) I' := b * I ; Q' := b * Q  (coeficiente de saturación) b
5) Chequear Y' <= 1
6) Chequear rangos de crominancia: -0.5957 < I' < 0.5957 ; -0.5226 < Q' < 0.5226
7) Convertir Y'I'Q' -> R'G'B' (RGB normalizado procesado)
8) Convertir R'G'B' a bytes (0..255) y graficar el píxel/imagen

"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import os

# ============================
# Matrices/coeficientes YIQ
# ============================
# RGB (normalizado) -> YIQ
# "Segunda matriz" típica (NTSC):
#   Y = 0.299 R + 0.587 G + 0.114 B
#   I = 0.595716 R - 0.274453 G - 0.321263 B
#   Q = 0.211456 R - 0.522591 G + 0.311135 B

RGB2YIQ = np.array([
    [0.299000,   0.587000,   0.114000],
    [0.595716,  -0.274453,  -0.321263],
    [0.211456,  -0.522591,   0.311135],
], dtype=np.float64)

# YIQ -> RGB (normalizado)
#   R = Y + 0.9563 I + 0.6210 Q
#   G = Y - 0.2721 I - 0.6474 Q
#   B = Y - 1.1070 I + 1.7046 Q

YIQ2RGB = np.array([
    [1.0,  0.9563,  0.6210],
    [1.0, -0.2721, -0.6474],
    [1.0, -1.1070,  1.7046],
], dtype=np.float64)

# Rangos válidos para crominancia I y Q (paso 6)
I_MIN, I_MAX = -0.5957, 0.5957
Q_MIN, Q_MAX = -0.5226, 0.5226


class PDIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDI – YIQ (Luminancia/Saturación) – Tkinter")
        self.geometry("1150x700")
        self.minsize(1000, 640)

        # Estado
        self.original_image = None   # PIL.Image
        self.processed_image = None  # PIL.Image
        self.photo_orig = None       # ImageTk.PhotoImage
        self.photo_proc = None       # ImageTk.PhotoImage

        # Parámetros a (luminancia) y b (saturación)
        self.a_var = tk.DoubleVar(value=1.0)
        self.b_var = tk.DoubleVar(value=1.0)
        self.autoupdate_var = tk.BooleanVar(value=True)

        self._build_ui()

    # ----------------------
    # Construcción de la UI
    # ----------------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        btn_load = ttk.Button(top, text="📂 Cargar imagen", command=self.load_image)
        btn_load.pack(side=tk.LEFT)

        ttk.Label(top, text="   a (Luminancia)").pack(side=tk.LEFT, padx=(20, 4))
        a_scale = ttk.Scale(top, from_=0.0, to=2.0, variable=self.a_var, command=self._maybe_auto)
        a_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.a_entry = ttk.Entry(top, width=6, textvariable=self.a_var)
        self.a_entry.pack(side=tk.LEFT, padx=4)
        self.a_entry.bind("<Return>", lambda e: self._maybe_auto(None))

        ttk.Label(top, text="   b (Saturación)").pack(side=tk.LEFT, padx=(20, 4))
        b_scale = ttk.Scale(top, from_=0.0, to=2.0, variable=self.b_var, command=self._maybe_auto)
        b_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.b_entry = ttk.Entry(top, width=6, textvariable=self.b_var)
        self.b_entry.pack(side=tk.LEFT, padx=4)
        self.b_entry.bind("<Return>", lambda e: self._maybe_auto(None))

        ttk.Checkbutton(top, text="Actualizar automáticamente", variable=self.autoupdate_var).pack(side=tk.LEFT, padx=10)

        btn_apply = ttk.Button(top, text="▶ Procesar (8 pasos)", command=self.apply_processing)
        btn_apply.pack(side=tk.LEFT, padx=10)

        btn_save = ttk.Button(top, text="💾 Guardar procesada", command=self.save_processed)
        btn_save.pack(side=tk.LEFT)

        # Paneles de imágenes
        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Original
        self.panel_orig = ttk.LabelFrame(mid, text="Imagen original (RGB)")
        self.panel_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.label_orig = ttk.Label(self.panel_orig, anchor="center")
        self.label_orig.pack(fill=tk.BOTH, expand=True)

        # Procesada
        self.panel_proc = ttk.LabelFrame(mid, text="Imagen procesada (8 pasos)")
        self.panel_proc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.label_proc = ttk.Label(self.panel_proc, anchor="center")
        self.label_proc.pack(fill=tk.BOTH, expand=True)

        # Barra de estado
        self.status = tk.StringVar(value="Cargá una imagen para empezar…")
        statusbar = ttk.Label(self, textvariable=self.status, anchor="w")
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _maybe_auto(self, _):
        if self.autoupdate_var.get():
            self.apply_processing()

    # ----------------------
    # Cargar y guardar
    # ----------------------
    def load_image(self):
        path = filedialog.askopenfilename(title="Seleccionar imagen",
                                          filetypes=[
                                              ("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                                              ("Todos los archivos", "*.*")
                                          ])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")
            return
        self.original_image = img
        self._show_original()
        self.apply_processing()
        self.status.set(f"Imagen cargada: {os.path.basename(path)} – {img.size[0]}x{img.size[1]} px")

    def save_processed(self):
        if self.processed_image is None:
            messagebox.showinfo("Guardar", "Primero procesá una imagen.")
            return
        path = filedialog.asksaveasfilename(title="Guardar imagen procesada",
                                            defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")])
        if not path:
            return
        try:
            self.processed_image.save(path)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la imagen:\n{e}")
            return
        self.status.set(f"Imagen guardada en: {path}")

    # ----------------------
    # Mostrar imágenes
    # ----------------------
    def _fit_to_label(self, pil_img, label_widget, max_scale=1.0):
        # Ajuste manteniendo aspecto dentro del label
        lw = label_widget.winfo_width() or 1
        lh = label_widget.winfo_height() or 1
        iw, ih = pil_img.size
        scale = min(lw / iw, lh / ih) * max_scale
        scale = max(1e-6, min(scale, 1.0))  # Evitar cero
        new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        return pil_img.resize(new_size, Image.LANCZOS)

    def _show_original(self):
        if self.original_image is None:
            return
        to_show = self._fit_to_label(self.original_image, self.label_orig)
        self.photo_orig = ImageTk.PhotoImage(to_show)
        self.label_orig.configure(image=self.photo_orig)

    def _show_processed(self):
        if self.processed_image is None:
            return
        to_show = self._fit_to_label(self.processed_image, self.label_proc)
        self.photo_proc = ImageTk.PhotoImage(to_show)
        self.label_proc.configure(image=self.photo_proc)

    # ----------------------
    # Núcleo de procesamiento (8 pasos)
    # ----------------------
    def apply_processing(self):
        if self.original_image is None:
            return

        a = float(self.a_var.get())
        b = float(self.b_var.get())

        # Convertir PIL -> NumPy array (uint8)
        img = np.array(self.original_image, dtype=np.uint8)  # [H, W, 3], 0..255

        # (1) Normalizar RGB: 0..255 -> 0..1 (float64)
        rgb = img.astype(np.float64) / 255.0  # [H, W, 3]

        # Reacomodar a 2D para multiplicación matricial: N x 3
        h, w, _ = rgb.shape
        rgb_flat = rgb.reshape((-1, 3))  # N x 3

        # (2) RGB -> YIQ (usando matriz RGB2YIQ)
        yiq_flat = rgb_flat @ RGB2YIQ.T  # N x 3
        Y = yiq_flat[:, 0]
        I = yiq_flat[:, 1]
        Q = yiq_flat[:, 2]

        # (3) Y' := a * Y  (luminancia)
        Yp = a * Y

        # (5) Chequear que Y' <= 1 (y >= 0 por robustez)
        Yp = np.clip(Yp, 0.0, 1.0)

        # (4) I' := b * I ; Q' := b * Q  (saturación)
        Ip = b * I
        Qp = b * Q

        # (6) Chequear rangos válidos de I', Q'
        Ip = np.clip(Ip, I_MIN, I_MAX)
        Qp = np.clip(Qp, Q_MIN, Q_MAX)

        # Reconstruir plano Y'I'Q'
        yiq_p_flat = np.stack([Yp, Ip, Qp], axis=1)  # N x 3

        # (7) Y'I'Q' -> R'G'B' (normalizado)
        rgb_p_flat = yiq_p_flat @ YIQ2RGB.T  # N x 3

        # Importante: puede haber pequeñas desviaciones -> limitar 0..1
        rgb_p_flat = np.clip(rgb_p_flat, 0.0, 1.0)

        # (8) Convertir R'G'B' a bytes (0..255) para graficar/guardar
        rgb_p = (rgb_p_flat.reshape((h, w, 3)) * 255.0 + 0.5).astype(np.uint8)

        # Array -> PIL
        self.processed_image = Image.fromarray(rgb_p, mode="RGB")
        self._show_processed()

        self.status.set(f"Procesado con a={a:.3f} (luminancia), b={b:.3f} (saturación)")


if __name__ == "__main__":
    app = PDIApp()
    # Ensayar tamaños iniciales de labels para un mejor fit en primer render
    app.update_idletasks()
    app.mainloop()

