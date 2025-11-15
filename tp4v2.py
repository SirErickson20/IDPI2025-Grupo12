# -*- coding: utf-8 -*-
"""
PDI – Procesamiento por convolución (FI-UNJu)

App Tkinter que:
- Carga una imagen (RGB o gris).
- La convierte a nivel de gris (luminancia).
- Aplica filtrado por convolución con padding de REPETICIÓN DE BORDES.
- Filtros implementados:

1) Pasabajos:
   - Plano 3x3, 5x5, 7x7
   - Bartlett 3x3, 5x5, 7x7
   - Gaussiano 5x5, 7x7 (aprox. Pascal)

2) Detectores de bordes:
   - Laplaciano v4, Laplaciano v8
   - Sobel 8 orientaciones (N, NE, E, SE, S, SW, W, NW)

3) Pasabanda y pasaaltos (frecuencia de corte ~0.2 y ~0.4)
   - Pasaaltos fc=0.2 ≈ identidad - LP gaussiano 7x7
   - Pasaaltos fc=0.4 ≈ identidad - LP gaussiano 5x5
   - Pasabanda fc=0.2 ≈ LP5 - LP7
   - Pasabanda fc=0.4 ≈ LP3 - LP5 (usando pasabajos plano 3x3 y gaussiano 5x5)

Padding: replicación de bordes (edge), como pide el enunciado.
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np

# ====================
# Kernels auxiliares
# ====================

def mean_kernel(size: int) -> np.ndarray:
    """Kernel pasabajos plano (todos unos normalizados)."""
    k = np.ones((size, size), dtype=np.float64)
    return k / k.size

def bartlett_kernel(size: int) -> np.ndarray:
    """Kernel Bartlett (triangular 1D -> outer product, normalizado)."""
    assert size in (3, 5, 7)
    n = size // 2
    seq = np.array(list(range(1, n + 2)) + list(range(n, 0, -1)), dtype=np.float64)
    k = np.outer(seq, seq)
    return k / np.sum(k)

def gauss_kernel_pascal(size: int) -> np.ndarray:
    """Kernel Gaussiano aproximado usando filas del triángulo de Pascal."""
    assert size in (5, 7)
    if size == 5:
        row = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    else:
        # fila de 7 elementos del triángulo de Pascal
        row = np.array([1, 6, 15, 20, 15, 6, 1], dtype=np.float64)
    k = np.outer(row, row)
    return k / np.sum(k)

# Laplacianos
LAPLACE_V4 = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]], dtype=np.float64)

LAPLACE_V8 = np.array([[1,  1, 1],
                       [1, -8, 1],
                       [1,  1, 1]], dtype=np.float64)

# Sobel 8 orientaciones (3x3)
SOBEL_KERNELS = {
    "Sobel Norte": np.array([[ 1,  2,  1],
                             [ 0,  0,  0],
                             [-1, -2, -1]], dtype=np.float64),

    "Sobel Sur": np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float64),

    "Sobel Este": np.array([[-1,  0,  1],
                            [-2,  0,  2],
                            [-1,  0,  1]], dtype=np.float64),

    "Sobel Oeste": np.array([[ 1,  0, -1],
                             [ 2,  0, -2],
                             [ 1,  0, -1]], dtype=np.float64),

    "Sobel Noreste": np.array([[ 0,  1,  2],
                               [-1,  0,  1],
                               [-2, -1,  0]], dtype=np.float64),

    "Sobel Noroeste": np.array([[ 2,  1,  0],
                               [ 1,  0, -1],
                               [ 0, -1, -2]], dtype=np.float64),

    "Sobel Sudeste": np.array([[-2, -1,  0],
                               [-1,  0,  1],
                               [ 0,  1,  2]], dtype=np.float64),

    "Sobel Sudoeste": np.array([[ 0, -1, -2],
                                [ 1,  0, -1],
                                [ 2,  1,  0]], dtype=np.float64),
}

# ====================
# Convolución con padding por replicación
# ====================

def convolve_replicate(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolución 2D con padding de replicación de bordes.
    img: imagen en float64 normalizada [0,1], 2D.
    kernel: matriz cuadrada impar.
    """
    k = kernel.shape[0]
    assert kernel.shape[0] == kernel.shape[1], "Kernel debe ser cuadrado"
    r = k // 2

    # padding por réplica de filas/columnas
    padded = np.pad(img, pad_width=r, mode="edge")

    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float64)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + k, j:j + k]
            out[i, j] = np.sum(region * kernel)

    # para la mayoría de los filtros queremos forzar a [0,1]
    out = np.clip(out, 0.0, 1.0)
    return out

# ====================
# App Tkinter
# ====================

FILTERS = [
    # Pasabajos plano
    "Pasabajos plano 3x3",
    "Pasabajos plano 5x5",
    "Pasabajos plano 7x7",
    # Bartlett
    "Pasabajos Bartlett 3x3",
    "Pasabajos Bartlett 5x5",
    "Pasabajos Bartlett 7x7",
    # Gauss
    "Pasabajos Gaussiano 5x5",
    "Pasabajos Gaussiano 7x7",
    # Laplaciano
    "Laplaciano v4",
    "Laplaciano v8",
    # Sobel 8 orientaciones
    "Sobel Norte",
    "Sobel Noreste",
    "Sobel Este",
    "Sobel Sudeste",
    "Sobel Sur",
    "Sobel Sudoeste",
    "Sobel Oeste",
    "Sobel Noroeste",
    # Pasaaltos / Pasabanda
    "Pasaaltos fc=0.2",
    "Pasaaltos fc=0.4",
    "Pasabanda fc=0.2",
    "Pasabanda fc=0.4",
]


class ConvolucionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDI – Procesamiento por convolución")
        self.geometry("1200x720")
        self.minsize(1000, 600)

        self.img_gray = None      # PIL (L) original
        self.img_gray_arr = None  # numpy float [0,1]
        self.img_out = None       # PIL (L) resultado

        self.tk_orig = None
        self.tk_out = None

        self.filter_var = tk.StringVar(value=FILTERS[0])
        self.autoupdate = tk.BooleanVar(value=True)

        # precalcular algunos kernels reutilizados
        self.kernel_gauss5 = gauss_kernel_pascal(5)
        self.kernel_gauss7 = gauss_kernel_pascal(7)
        self.kernel_mean3 = mean_kernel(3)

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Button(top, text="📂 Cargar imagen", command=self.load_image).pack(side=tk.LEFT)
        ttk.Label(top, text="   Filtro:").pack(side=tk.LEFT, padx=(16, 4))
        cb = ttk.Combobox(top, textvariable=self.filter_var, values=FILTERS,
                          state="readonly", width=32)
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda e: self._maybe_auto())

        ttk.Checkbutton(top, text="Actualizar automáticamente",
                        variable=self.autoupdate).pack(side=tk.LEFT, padx=12)
        ttk.Button(top, text="▶ Aplicar", command=self.apply_filter).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="💾 Guardar resultado", command=self.save_output).pack(side=tk.LEFT, padx=(8, 0))

        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        lf1 = ttk.LabelFrame(mid, text="Imagen gris / luminancia (entrada)")
        lf1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.lbl_orig = ttk.Label(lf1, anchor="center")
        self.lbl_orig.pack(fill=tk.BOTH, expand=True)

        lf2 = ttk.LabelFrame(mid, text="Resultado filtrado (salida)")
        lf2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.lbl_out = ttk.Label(lf2, anchor="center")
        self.lbl_out.pack(fill=tk.BOTH, expand=True)

        self.status = tk.StringVar(
            value="Cargá una imagen. Se convierte a gris/luminancia y se filtra por convolución."
        )
        ttk.Label(self, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    # ------------- utilidades de UI -------------

    def _fit(self, pil_img, label):
        lw = label.winfo_width() or 1
        lh = label.winfo_height() or 1
        iw, ih = pil_img.size
        s = min(lw / iw, lh / ih)
        s = max(1e-6, min(s, 1.0))
        return pil_img.resize((max(1, int(iw * s)), max(1, int(ih * s))), Image.LANCZOS)

    def _refresh_images(self):
        if self.img_gray is not None:
            x = self._fit(self.img_gray, self.lbl_orig)
            self.tk_orig = ImageTk.PhotoImage(x)
            self.lbl_orig.configure(image=self.tk_orig)

        if self.img_out is not None:
            y = self._fit(self.img_out, self.lbl_out)
            self.tk_out = ImageTk.PhotoImage(y)
            self.lbl_out.configure(image=self.tk_out)

    def _maybe_auto(self):
        if self.autoupdate.get():
            self.apply_filter()

    # ------------- IO -------------

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

        # Convertir a nivel de gris (luminancia)
        self.img_gray = img.convert("L")  # PIL interno hace combinación de canales
        arr_u8 = np.array(self.img_gray, dtype=np.uint8)
        self.img_gray_arr = arr_u8.astype(np.float64) / 255.0

        self.img_out = None
        self._refresh_images()
        self.status.set(f"Imagen cargada: {path} – tamaño {self.img_gray.size[0]}x{self.img_gray.size[1]}")
        self._maybe_auto()

    def save_output(self):
        if self.img_out is None:
            messagebox.showinfo("Guardar", "No hay resultado todavía.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar resultado",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")]
        )
        if not path:
            return
        try:
            self.img_out.save(path)
            self.status.set(f"Resultado guardado en: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar:\n{e}")

    # ------------- núcleo: selección de filtro -------------

    def apply_filter(self):
        if self.img_gray_arr is None:
            return

        f = self.filter_var.get()
        img = self.img_gray_arr

        # 1) Pasabajos
        if f == "Pasabajos plano 3x3":
            kernel = mean_kernel(3)
            out = convolve_replicate(img, kernel)
        elif f == "Pasabajos plano 5x5":
            kernel = mean_kernel(5)
            out = convolve_replicate(img, kernel)
        elif f == "Pasabajos plano 7x7":
            kernel = mean_kernel(7)
            out = convolve_replicate(img, kernel)
        elif f == "Pasabajos Bartlett 3x3":
            kernel = bartlett_kernel(3)
            out = convolve_replicate(img, kernel)
        elif f == "Pasabajos Bartlett 5x5":
            kernel = bartlett_kernel(5)
            out = convolve_replicate(img, kernel)
        elif f == "Pasabajos Bartlett 7x7":
            kernel = bartlett_kernel(7)
            out = convolve_replicate(img, kernel)
        elif f == "Pasabajos Gaussiano 5x5":
            out = convolve_replicate(img, self.kernel_gauss5)
        elif f == "Pasabajos Gaussiano 7x7":
            out = convolve_replicate(img, self.kernel_gauss7)

        # 2) Laplaciano
        elif f == "Laplaciano v4":
            # puede producir valores negativos o >1, por eso luego se recorta
            out = convolve_replicate(img, LAPLACE_V4)
        elif f == "Laplaciano v8":
            out = convolve_replicate(img, LAPLACE_V8)

        # 3) Sobel 8 direcciones
        elif f in SOBEL_KERNELS:
            k = SOBEL_KERNELS[f]
            # Sobel puede dar valores negativos; tomamos valor absoluto y reescalamos
            resp = convolve_replicate(img, k)
            out = np.abs(resp)
            out = out / (out.max() + 1e-8)
        # 4) Pasaaltos / Pasabanda
        elif f == "Pasaaltos fc=0.2":
            # fc más baja: usamos blur más fuerte (gauss7)
            low = convolve_replicate(img, self.kernel_gauss7)
            hp = img - low
            hp = (hp - hp.min()) / (hp.max() - hp.min() + 1e-8)
            out = hp
        elif f == "Pasaaltos fc=0.4":
            # fc más alta: blur más suave (gauss5)
            low = convolve_replicate(img, self.kernel_gauss5)
            hp = img - low
            hp = (hp - hp.min()) / (hp.max() - hp.min() + 1e-8)
            out = hp
        elif f == "Pasabanda fc=0.2":
            # banda media: diferencia entre LP5 y LP7
            low5 = convolve_replicate(img, self.kernel_gauss5)
            low7 = convolve_replicate(img, self.kernel_gauss7)
            bp = low5 - low7
            bp = (bp - bp.min()) / (bp.max() - bp.min() + 1e-8)
            out = bp
        elif f == "Pasabanda fc=0.4":
            # otra banda: diferencia entre pasabajos 3x3 y gauss5
            low3 = convolve_replicate(img, self.kernel_mean3)
            low5 = convolve_replicate(img, self.kernel_gauss5)
            bp = low3 - low5
            bp = (bp - bp.min()) / (bp.max() - bp.min() + 1e-8)
            out = bp
        else:
            out = img.copy()

        # convertir a imagen PIL en escala de grises
        out_u8 = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        self.img_out = Image.fromarray(out_u8, mode="L")
        self._refresh_images()
        self.status.set(f"Filtro aplicado: {f}")


if __name__ == "__main__":
    app = ConvolucionApp()
    app.update_idletasks()
    app.mainloop()
