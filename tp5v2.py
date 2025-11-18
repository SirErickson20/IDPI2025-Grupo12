# -*- coding: utf-8 -*-
"""
PDI – Procesamiento morfológico en niveles de gris (FI–UNJu)

Aplicativo que:
- Carga una imagen (RGB o gris).
- La convierte a niveles de gris.
- Aplica operaciones morfológicas con elemento estructurante 3x3:
    * Erosión  (mínimo de la vecindad)
    * Dilatación (máximo)
    * Apertura  = Erosión seguida de Dilatación
    * Cierre    = Dilatación seguida de Erosión
    * Borde morfológico (gradiente) = Dilatación - Erosión
    * Mediana (filtro no lineal típico)
- Incluye un botón para copiar la imagen procesada como nueva imagen original,
  para poder aplicar filtros en secuencia.

Requisitos:
    pip install pillow numpy
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import os


# =========================
# Helpers morfológicos 3x3
# =========================

def pad_replicate(arr: np.ndarray, r: int = 1) -> np.ndarray:
    """Padding por replicación de bordes (edge) para vecindad r (3x3 => r=1)."""
    return np.pad(arr, pad_width=r, mode="edge")


def erosion_3x3(arr: np.ndarray) -> np.ndarray:
    """Erosión en niveles de gris: mínimo en vecindad 3x3."""
    r = 1
    padded = pad_replicate(arr, r)
    h, w = arr.shape
    out = np.empty_like(arr)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i+3, j:j+3]
            out[i, j] = np.min(patch)

    return out


def dilatacion_3x3(arr: np.ndarray) -> np.ndarray:
    """Dilatación en niveles de gris: máximo en vecindad 3x3."""
    r = 1
    padded = pad_replicate(arr, r)
    h, w = arr.shape
    out = np.empty_like(arr)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i+3, j:j+3]
            out[i, j] = np.max(patch)

    return out


def mediana_3x3(arr: np.ndarray) -> np.ndarray:
    """Filtro de mediana en vecindad 3x3."""
    r = 1
    padded = pad_replicate(arr, r)
    h, w = arr.shape
    out = np.empty_like(arr)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i+3, j:j+3]
            out[i, j] = np.median(patch)

    return out


# =========================
# App Tkinter
# =========================

OPS = [
    "Erosión 3x3",
    "Dilatación 3x3",
    "Apertura (Erosión + Dilatación)",
    "Cierre (Dilatación + Erosión)",
    "Borde morfológico (Gradiente)",
    "Mediana 3x3",
]


class MorfologiaGrisApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PDI – Morfología en niveles de gris (3x3)")
        self.geometry("1100x700")
        self.minsize(950, 600)

        # Imágenes:
        self.img_in = None       # PIL.Image (L) - imagen actual/original
        self.arr_in = None       # numpy float64 [0,1]
        self.img_out = None      # PIL.Image (L) - resultado actual

        self.tk_in = None        # PhotoImage para mostrar
        self.tk_out = None

        self.op_var = tk.StringVar(value=OPS[0])
        self.autoupdate = tk.BooleanVar(value=True)

        self._build_ui()

    # -------------- UI --------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Button(top, text="📂 Cargar imagen", command=self.load_image).pack(side=tk.LEFT)

        ttk.Label(top, text="   Operación:").pack(side=tk.LEFT, padx=(16, 4))
        cb = ttk.Combobox(top, textvariable=self.op_var, values=OPS,
                          state="readonly", width=34)
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda e: self._maybe_auto())

        ttk.Checkbutton(top, text="Actualizar automáticamente",
                        variable=self.autoupdate).pack(side=tk.LEFT, padx=12)

        ttk.Button(top, text="▶ Aplicar", command=self.apply_operation).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="⮐ Copiar salida a entrada",
                   command=self.copy_output_to_input).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="💾 Guardar salida", command=self.save_output).pack(side=tk.LEFT, padx=(8, 0))

        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        lf1 = ttk.LabelFrame(mid, text="Imagen original / actual (gris)")
        lf1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.lbl_in = ttk.Label(lf1, anchor="center")
        self.lbl_in.pack(fill=tk.BOTH, expand=True)

        lf2 = ttk.LabelFrame(mid, text="Imagen procesada")
        lf2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.lbl_out = ttk.Label(lf2, anchor="center")
        self.lbl_out.pack(fill=tk.BOTH, expand=True)

        self.status = tk.StringVar(
            value="Cargá una imagen. Se convertirá a niveles de gris y se aplicarán operaciones morfológicas 3x3."
        )
        ttk.Label(self, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    # -------------- Helpers de UI --------------
    def _fit(self, pil_img, label):
        lw = label.winfo_width() or 1
        lh = label.winfo_height() or 1
        iw, ih = pil_img.size
        s = min(lw / iw, lh / ih)
        s = max(1e-6, min(s, 1.0))
        return pil_img.resize((max(1, int(iw * s)), max(1, int(ih * s))), Image.LANCZOS)

    def _refresh_view(self):
        if self.img_in is not None:
            x = self._fit(self.img_in, self.lbl_in)
            self.tk_in = ImageTk.PhotoImage(x)
            self.lbl_in.configure(image=self.tk_in)

        if self.img_out is not None:
            y = self._fit(self.img_out, self.lbl_out)
            self.tk_out = ImageTk.PhotoImage(y)
            self.lbl_out.configure(image=self.tk_out)

    def _maybe_auto(self):
        if self.autoupdate.get():
            self.apply_operation()

    # -------------- IO --------------
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not path:
            return
        try:
            img = Image.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")
            return

        # Convertir a escala de grises (luminancia)
        self.img_in = img.convert("L")
        arr_u8 = np.array(self.img_in, dtype=np.uint8)
        self.arr_in = arr_u8.astype(np.float64) / 255.0  # [0,1]

        self.img_out = None
        self._refresh_view()
        self.status.set(f"Imagen cargada: {os.path.basename(path)} – {self.img_in.size[0]}x{self.img_in.size[1]} px")
        self._maybe_auto()

    def save_output(self):
        if self.img_out is None:
            messagebox.showinfo("Guardar", "No hay imagen procesada todavía.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar salida",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")]
        )
        if not path:
            return
        try:
            self.img_out.save(path)
            self.status.set(f"Salida guardada en: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar:\n{e}")

    def copy_output_to_input(self):
        """Copia la imagen procesada como nueva imagen original para encadenar filtros."""
        if self.img_out is None:
            messagebox.showinfo("Copiar", "No hay imagen procesada para copiar.")
            return
        # Nueva entrada = salida actual:
        self.img_in = self.img_out.copy()
        arr_u8 = np.array(self.img_in, dtype=np.uint8)
        self.arr_in = arr_u8.astype(np.float64) / 255.0
        self.status.set("Salida copiada como nueva imagen de entrada.")
        self._refresh_view()

    # -------------- Núcleo: operaciones morfológicas --------------
    def apply_operation(self):
        if self.arr_in is None:
            return

        op = self.op_var.get()
        img = self.arr_in  # float [0,1]

        if op == "Erosión 3x3":
            out = erosion_3x3(img)
        elif op == "Dilatación 3x3":
            out = dilatacion_3x3(img)
        elif op == "Apertura (Erosión + Dilatación)":
            er = erosion_3x3(img)
            out = dilatacion_3x3(er)
        elif op == "Cierre (Dilatación + Erosión)":
            di = dilatacion_3x3(img)
            out = erosion_3x3(di)
        elif op == "Borde morfológico (Gradiente)":
            # Gradiente morfológico en gris: dilatación - erosión
            er = erosion_3x3(img)
            di = dilatacion_3x3(img)
            grad = di - er
            # Normalizamos a [0,1] por robustez:
            grad = grad - grad.min()
            if grad.max() > 0:
                grad = grad / grad.max()
            out = grad
        elif op == "Mediana 3x3":
            out = mediana_3x3(img)
        else:
            out = img.copy()

        out_u8 = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        self.img_out = Image.fromarray(out_u8, mode="L")
        self._refresh_view()
        self.status.set(f"Operación aplicada: {op}")


if __name__ == "__main__":
    app = MorfologiaGrisApp()
    app.update_idletasks()
    app.mainloop()
