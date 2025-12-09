# -*- coding: utf-8 -*-
"""
Mega-panel PDI – FI-UNJu

Integra en una sola interfaz Tkinter:

- Espacio cromático RGB<->YIQ, canal Y
- Aritmética entre dos imágenes (cuasi-sumas, cuasi-restas, if-lighter/darker)
- Operaciones de luminancia (raíz, cuadrática, lineal a trozos)
- Convolución (pasabajos, Gauss, Bartlett, Laplaciano, Sobel)
- Morfología en niveles de gris (erosión, dilatación, apertura, cierre, borde, mediana)
- Binarización (global, 50/50, 2 modas, Otsu)
- Segmentación (Marching Squares, Color Fill)

Requisitos:
    pip install pillow numpy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os

# ============================================================
# Utilidades de conversión y helpers comunes
# ============================================================

MAT_RGB2YIQ = np.array([
    [0.299, 0.587, 0.114],
    [0.596, -0.275, -0.321],
    [0.212, -0.523, 0.311]
], dtype=np.float64)

MAT_YIQ2RGB = np.linalg.inv(MAT_RGB2YIQ)


def pil_to_rgb_f32(img: Image.Image) -> np.ndarray:
    """PIL -> float32 RGB [0,1]"""
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def rgb_f32_to_pil(arr: np.ndarray) -> Image.Image:
    arr_u8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_u8, mode="RGB")


def rgb_to_yiq(arr_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = arr_rgb.shape
    flat = arr_rgb.reshape(-1, 3)
    yiq = flat @ MAT_RGB2YIQ.T
    return yiq.reshape(h, w, 3)


def yiq_to_rgb(arr_yiq: np.ndarray) -> np.ndarray:
    h, w, _ = arr_yiq.shape
    flat = arr_yiq.reshape(-1, 3)
    rgb = flat @ MAT_YIQ2RGB.T
    return np.clip(rgb.reshape(h, w, 3), 0.0, 1.0)


def get_Y_from_rgb(arr_rgb: np.ndarray) -> np.ndarray:
    """Devuelve canal Y (luminancia) a partir de RGB."""
    yiq = rgb_to_yiq(arr_rgb)
    return np.clip(yiq[..., 0], 0.0, 1.0)


def Y_to_rgb_like(arr_Y: np.ndarray, base_rgb: np.ndarray) -> np.ndarray:
    """Reemplaza canal Y en una imagen base, dejando I,Q como en base."""
    yiq = rgb_to_yiq(base_rgb)
    yiq[..., 0] = np.clip(arr_Y, 0.0, 1.0)
    return yiq_to_rgb(yiq)


def pad_edge(arr: np.ndarray, r: int) -> np.ndarray:
    return np.pad(arr, pad_width=r, mode="edge")


# ============================================================
# Aritmética / binarización helpers
# ============================================================

def bin_global(arr_f: np.ndarray, thr_f: float) -> np.ndarray:
    return (arr_f >= thr_f).astype(np.float64)


def bin_mediana(arr_f: np.ndarray) -> np.ndarray:
    thr = np.median(arr_f)
    return bin_global(arr_f, thr)


def otsu_threshold(arr_f: np.ndarray) -> float:
    arr_u8 = (np.clip(arr_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    hist, _ = np.histogram(arr_u8, bins=256, range=(0, 255))
    total = arr_u8.size
    sum_total = np.dot(hist, np.arange(256))
    sumB = 0.0
    wB = 0.0
    varMax = 0.0
    thr = 0
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > varMax:
            varMax = var_between
            thr = t
    return thr / 255.0


def bin_otsu(arr_f: np.ndarray) -> np.ndarray:
    return bin_global(arr_f, otsu_threshold(arr_f))


def bin_dos_modas(arr_f: np.ndarray) -> np.ndarray:
    arr_u8 = (np.clip(arr_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    hist, _ = np.histogram(arr_u8, bins=256, range=(0, 255))
    peaks = np.argsort(hist)[::-1]
    if len(peaks) < 2:
        return bin_mediana(arr_f)
    m1 = peaks[0]
    m2 = peaks[1]
    for idx in peaks[1:]:
        if abs(idx - m1) >= 10:
            m2 = idx
            break
    A = arr_u8.astype(np.int16)
    d1 = np.abs(A - int(m1))
    d2 = np.abs(A - int(m2))
    mask = d1 <= d2
    return np.where(mask, 1.0, 0.0)


# ============================================================
# Convolución
# ============================================================

def convolve(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.shape[0]
    r = k // 2
    padded = pad_edge(arr, r)
    h, w = arr.shape
    out = np.zeros_like(arr, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            patch = padded[i:i + k, j:j + k]
            out[i, j] = np.sum(patch * kernel)
    return out


def gauss_kernel(size: int) -> np.ndarray:
    # usando coeficientes de Pascal (1,4,6,4,1) etc.
    if size == 5:
        row = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    elif size == 7:
        row = np.array([1, 6, 15, 20, 15, 6, 1], dtype=np.float64)
    else:
        raise ValueError("Gauss solo 5x5 o 7x7")
    kernel = np.outer(row, row)
    kernel = kernel / kernel.sum()
    return kernel


def plano_kernel(size: int) -> np.ndarray:
    k = np.ones((size, size), dtype=np.float64)
    return k / k.size


def bartlett_kernel(size: int) -> np.ndarray:
    # triangular 1,2,3,2,1 etc.
    if size == 3:
        v = np.array([1, 2, 1], dtype=np.float64)
    elif size == 5:
        v = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    elif size == 7:
        v = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float64)
    else:
        raise ValueError("Bartlett 3,5,7")
    k = np.outer(v, v)
    k = k / k.sum()
    return k


LAPLACE_V4 = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float64)

LAPLACE_V8 = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]], dtype=np.float64)


def sobel_kernel(direction: str) -> np.ndarray:
    # básico N-E-S-O y diagonales
    if direction == "N":
        return np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]], dtype=np.float64)
    if direction == "S":
        return -sobel_kernel("N")
    if direction == "E":
        return np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float64)
    if direction == "O":
        return -sobel_kernel("E")
    if direction == "NE":
        return np.array([[0, -1, -2],
                         [1, 0, -1],
                         [2, 1, 0]], dtype=np.float64)
    if direction == "SE":
        return -sobel_kernel("NO")
    if direction == "NO":
        return np.array([[-2, -1, 0],
                         [-1, 0, 1],
                         [0, 1, 2]], dtype=np.float64)
    if direction == "SO":
        return -sobel_kernel("NE")
    raise ValueError("Dirección Sobel inválida")


# ============================================================
# Morfología 3x3
# ============================================================

def erosion_3x3(arr: np.ndarray) -> np.ndarray:
    padded = pad_edge(arr, 1)
    h, w = arr.shape
    out = np.empty_like(arr)
    for i in range(h):
        for j in range(w):
            out[i, j] = np.min(padded[i:i+3, j:j+3])
    return out


def dilatacion_3x3(arr: np.ndarray) -> np.ndarray:
    padded = pad_edge(arr, 1)
    h, w = arr.shape
    out = np.empty_like(arr)
    for i in range(h):
        for j in range(w):
            out[i, j] = np.max(padded[i:i+3, j:j+3])
    return out


def mediana_3x3(arr: np.ndarray) -> np.ndarray:
    padded = pad_edge(arr, 1)
    h, w = arr.shape
    out = np.empty_like(arr)
    for i in range(h):
        for j in range(w):
            out[i, j] = np.median(padded[i:i+3, j:j+3])
    return out


def gradiente_morfologico(arr: np.ndarray) -> np.ndarray:
    er = erosion_3x3(arr)
    di = dilatacion_3x3(arr)
    g = di - er
    g = g - g.min()
    if g.max() > 0:
        g = g / g.max()
    return g


# ============================================================
# Marching Squares y Color Fill
# ============================================================

def marching_squares(arr_f: np.ndarray, thr: float = None) -> Image.Image:
    """Devuelve imagen L con contorno blanco en fondo negro."""
    h, w = arr_f.shape
    if thr is None:
        thr = otsu_threshold(arr_f)
    mask = arr_f >= thr
    out = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(out)
    for y in range(h - 1):
        for x in range(w - 1):
            tl = mask[y, x]
            tr = mask[y, x+1]
            br = mask[y+1, x+1]
            bl = mask[y+1, x]
            pts = []
            if tl != tr:
                pts.append((x + 0.5, y))
            if tr != br:
                pts.append((x + 1, y + 0.5))
            if br != bl:
                pts.append((x + 0.5, y + 1))
            if bl != tl:
                pts.append((x, y + 0.5))
            if len(pts) == 2:
                draw.line((pts[0][0], pts[0][1], pts[1][0], pts[1][1]), fill=255)
            elif len(pts) == 4:
                draw.line((pts[0][0], pts[0][1], pts[1][0], pts[1][1]), fill=255)
                draw.line((pts[2][0], pts[2][1], pts[3][0], pts[3][1]), fill=255)
    return out


def flood_fill(arr_u8: np.ndarray, x0: int, y0: int, tol: int) -> np.ndarray:
    h, w = arr_u8.shape
    seed = int(arr_u8[y0, x0])
    visited = np.zeros_like(arr_u8, dtype=bool)
    mask = np.zeros_like(arr_u8, dtype=bool)
    stack = [(x0, y0)]
    while stack:
        x, y = stack.pop()
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if visited[y, x]:
            continue
        visited[y, x] = True
        if abs(int(arr_u8[y, x]) - seed) <= tol:
            mask[y, x] = True
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
    return mask


# ============================================================
# Interfaz Tkinter
# ============================================================

class PDIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mega-panel PDI – FI-UNJu")
        self.geometry("1300x800")
        self.minsize(1100, 700)

        # Estado
        self.img_base = None      # PIL RGB
        self.arr_base = None      # float RGB [0,1]
        self.img_sec = None       # para aritmética
        self.arr_sec = None
        self.img_out = None

        self.tk_base = None
        self.tk_out = None

        self.disp_size_out = (1, 1)

        # Variables de UI
        self.ymin_var = tk.IntVar(value=50)
        self.ymax_var = tk.IntVar(value=180)
        self.thr_var = tk.IntVar(value=128)
        self.tol_fill = tk.IntVar(value=15)
        self.var_mode_fill = tk.BooleanVar(value=False)  # si está activa la varita

        self._build_ui()

    # --------------- Construcción UI ----------------

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Button(top, text="📂 Cargar imagen base",
                   command=self.load_base).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(top, text="📂 Cargar imagen secundaria (aritmética)",
                   command=self.load_sec).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(top, text="⮐ Copiar salida a base",
                   command=self.copy_output_to_base).pack(side=tk.LEFT, padx=(10, 5))
        ttk.Button(top, text="💾 Guardar salida",
                   command=self.save_output).pack(side=tk.LEFT, padx=(0, 5))

        self.status = tk.StringVar(value="Cargá una imagen para comenzar.")
        ttk.Label(self, textvariable=self.status,
                  anchor="w").pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=3)

        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Notebook con pestañas de operaciones
        right = ttk.Frame(mid)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.nb = ttk.Notebook(right)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self._build_tab_esp()
        self._build_tab_arit()
        self._build_tab_lum()
        self._build_tab_conv()
        self._build_tab_morf()
        self._build_tab_bin()

        # Visores
        left = ttk.Frame(mid)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        lf1 = ttk.LabelFrame(left, text="Imagen base / actual")
        lf1.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 4))
        self.lbl_base = ttk.Label(lf1, anchor="center")
        self.lbl_base.pack(fill=tk.BOTH, expand=True)

        lf2 = ttk.LabelFrame(left, text="Resultado")
        lf2.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(4, 0))
        self.lbl_out = ttk.Label(lf2, anchor="center")
        self.lbl_out.pack(fill=tk.BOTH, expand=True)

        # Para varita mágica
        self.lbl_out.bind("<Button-1>", self.on_click_out)

    # ---------- Tabs ----------

    def _build_tab_esp(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Esp. cromático")

        ttk.Button(tab, text="RGB → YIQ (solo efecto interno)",
                   command=self.do_rgb2yiq).pack(fill=tk.X, pady=3, padx=5)
        ttk.Button(tab, text="YIQ → RGB (recomponer)",
                   command=self.do_yiq2rgb).pack(fill=tk.X, pady=3, padx=5)
        ttk.Button(tab, text="Promediar 3 canales (RGB → gris)",
                   command=self.do_promedio_canales).pack(fill=tk.X, pady=3, padx=5)
        ttk.Button(tab, text="Mostrar solo canal Y (luminancia)",
                   command=self.do_solo_Y).pack(fill=tk.X, pady=3, padx=5)

    def _build_tab_arit(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Aritmética")

        ttk.Label(tab, text="Usa imagen base + imagen secundaria (mismo tamaño)."
                 ).pack(padx=5, pady=(5, 3))

        ttk.Button(tab, text="Cuasi-suma promediada",
                   command=self.do_sum_prom).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tab, text="Cuasi-suma clampeada",
                   command=self.do_sum_clamp).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tab, text="Cuasi-resta promediada",
                   command=self.do_rest_prom).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tab, text="Cuasi-resta clampeada",
                   command=self.do_rest_clamp).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tab, text="If-lighter (pixel más claro)",
                   command=self.do_if_lighter).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tab, text="If-darker (pixel más oscuro)",
                   command=self.do_if_darker).pack(fill=tk.X, padx=5, pady=2)

    def _build_tab_lum(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Luminancia")

        ttk.Button(tab, text="Raíz (aclara sombras)",
                   command=self.do_lum_sqrt).pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(tab, text="Cuadrado (oscurece)",
                   command=self.do_lum_square).pack(fill=tk.X, padx=5, pady=3)

        box = ttk.Frame(tab)
        box.pack(fill=tk.X, padx=5, pady=(10, 2))
        ttk.Label(box, text="Ymin").pack(side=tk.LEFT)
        ttk.Entry(box, width=6, textvariable=self.ymin_var).pack(side=tk.LEFT, padx=3)
        ttk.Label(box, text="Ymax").pack(side=tk.LEFT)
        ttk.Entry(box, width=6, textvariable=self.ymax_var).pack(side=tk.LEFT, padx=3)
        ttk.Button(tab, text="Función lineal a trozos",
                   command=self.do_lum_piecewise).pack(fill=tk.X, padx=5, pady=3)

    def _build_tab_conv(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Convolución")

        # Filtros pasabajos
        sec1 = ttk.LabelFrame(tab, text="Pasabajos")
        sec1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        for size in (3, 5, 7):
            ttk.Button(sec1, text=f"Plano {size}x{size}",
                       command=lambda s=size: self.do_conv_plano(s)
                       ).pack(fill=tk.X, padx=3, pady=1)
        for size in (3, 5, 7):
            ttk.Button(sec1, text=f"Bartlett {size}x{size}",
                       command=lambda s=size: self.do_conv_bartlett(s)
                       ).pack(fill=tk.X, padx=3, pady=1)
        for size in (5, 7):
            ttk.Button(sec1, text=f"Gaussiano {size}x{size}",
                       command=lambda s=size: self.do_conv_gauss(s)
                       ).pack(fill=tk.X, padx=3, pady=1)

        # Derivativos
        sec2 = ttk.LabelFrame(tab, text="Derivativos / Bordes")
        sec2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Button(sec2, text="Laplaciano v4",
                   command=lambda: self.do_conv_lap("v4")
                   ).pack(fill=tk.X, padx=3, pady=1)
        ttk.Button(sec2, text="Laplaciano v8",
                   command=lambda: self.do_conv_lap("v8")
                   ).pack(fill=tk.X, padx=3, pady=1)

        for d, txt in [
            ("N", "Sobel Norte"), ("S", "Sobel Sur"),
            ("E", "Sobel Este"), ("O", "Sobel Oeste"),
            ("NE", "Sobel Noreste"), ("SE", "Sobel Sudeste"),
            ("NO", "Sobel Noroeste"), ("SO", "Sobel Sudoeste")
        ]:
            ttk.Button(sec2, text=txt,
                       command=lambda dd=d: self.do_conv_sobel(dd)
                       ).pack(fill=tk.X, padx=3, pady=1)

    def _build_tab_morf(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Morfología")

        ttk.Button(tab, text="Erosión 3x3",
                   command=self.do_erosion).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tab, text="Dilatación 3x3",
                   command=self.do_dilatacion).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tab, text="Apertura (Ero→Dila)",
                   command=self.do_apertura).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tab, text="Cierre (Dila→Ero)",
                   command=self.do_cierre).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tab, text="Borde morfológico (gradiente)",
                   command=self.do_grad_morf).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tab, text="Mediana 3x3",
                   command=self.do_mediana).pack(fill=tk.X, padx=5, pady=2)

    def _build_tab_bin(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Binarización / Segm.")

        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(top, text="Umbral (0–255):").pack(side=tk.LEFT)
        scale = ttk.Scale(top, from_=0, to=255,
                          variable=self.thr_var,
                          orient="horizontal")
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(top, textvariable=self.thr_var, width=4
                  ).pack(side=tk.LEFT)

        ttk.Button(tab, text="Binarización global",
                   command=self.do_bin_global).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tab, text="Invertir colores",
                   command=self.do_invert).pack(fill=tk.X, padx=5, pady=2)

        ttk.Separator(tab, orient="horizontal").pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(tab, text="Binarización 50/50 (mediana)",
                   command=self.do_bin_mediana).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tab, text="Binarización dos modas",
                   command=self.do_bin_dos_modas).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tab, text="Binarización Otsu",
                   command=self.do_bin_otsu).pack(fill=tk.X, padx=5, pady=2)

        ttk.Separator(tab, orient="horizontal").pack(fill=tk.X, padx=5, pady=5)

        ms_box = ttk.LabelFrame(tab, text="Segmentación avanzada")
        ms_box.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(ms_box, text="Marching Squares (bordes)",
                   command=self.do_marching).pack(fill=tk.X, padx=5, pady=2)

        fill_box = ttk.Frame(ms_box)
        fill_box.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(fill_box, text="Tolerancia varita:").pack(side=tk.LEFT)
        ttk.Scale(fill_box, from_=0, to=100,
                  variable=self.tol_fill,
                  orient="horizontal").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(fill_box, textvariable=self.tol_fill, width=4
                  ).pack(side=tk.LEFT)

        ttk.Button(ms_box, text="Color Fill (clic en la imagen de salida)",
                   command=self.enable_varita).pack(fill=tk.X, padx=5, pady=2)

    # ------------- IO -------------

    def load_base(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen base",
            filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not path:
            return
        try:
            img = Image.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir:\n{e}")
            return
        self.img_base = img.convert("RGB")
        self.arr_base = pil_to_rgb_f32(self.img_base)
        self.img_out = None
        self._refresh_view()
        self.status.set(f"Imagen base: {os.path.basename(path)} {self.img_base.size[0]}x{self.img_base.size[1]}")

    def load_sec(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen secundaria (mismo tamaño)",
            filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not path:
            return
        if self.img_base is None:
            messagebox.showinfo("Aviso", "Primero cargá la imagen base.")
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir:\n{e}")
            return
        if img.size != self.img_base.size:
            messagebox.showerror("Error", "La imagen secundaria debe tener el mismo tamaño que la base.")
            return
        self.img_sec = img
        self.arr_sec = pil_to_rgb_f32(self.img_sec)
        self.status.set(f"Imagen secundaria cargada: {os.path.basename(path)}")

    def save_output(self):
        if self.img_out is None:
            messagebox.showinfo("Guardar", "No hay imagen de salida.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar salida",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp"),("TIF", "*.tif;*.tiff")]
        )
        if not path:
            return
        try:
            self.img_out.save(path)
            self.status.set(f"Salida guardada en {path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar:\n{e}")

    def copy_output_to_base(self):
        if self.img_out is None:
            messagebox.showinfo("Copiar", "No hay salida para copiar.")
            return
        self.img_base = self.img_out.convert("RGB")
        self.arr_base = pil_to_rgb_f32(self.img_base)
        self.status.set("Salida copiada como nueva base.")
        self._refresh_view()

    # ------------- Helpers UI -------------

    def _fit(self, img: Image.Image, label: ttk.Label):
        lw = label.winfo_width() or 1
        lh = label.winfo_height() or 1
        iw, ih = img.size
        s = min(lw / iw, lh / ih)
        s = max(s, 1e-6)
        new_size = (max(1, int(iw * s)), max(1, int(ih * s)))
        return img.resize(new_size, Image.LANCZOS)

    def _refresh_view(self):
        if self.img_base is not None:
            disp = self._fit(self.img_base, self.lbl_base)
            self.tk_base = ImageTk.PhotoImage(disp)
            self.lbl_base.configure(image=self.tk_base)
        if self.img_out is not None:
            disp = self._fit(self.img_out, self.lbl_out)
            self.disp_size_out = disp.size
            self.tk_out = ImageTk.PhotoImage(disp)
            self.lbl_out.configure(image=self.tk_out)

    def _require_base(self):
        if self.arr_base is None:
            messagebox.showinfo("Aviso", "Cargá primero una imagen base.")
            return False
        return True

    # ------------- Espacio cromático -------------

    def do_rgb2yiq(self):
        if not self._require_base():
            return
        # solo dejamos nota, las demás operaciones usan YIQ de forma implícita
        self.status.set("Transformación RGB→YIQ se aplica implícitamente para las operaciones en Y.")
        # no cambiamos imagen visiblemente

    def do_yiq2rgb(self):
        if not self._require_base():
            return
        self.status.set("La imagen ya está en RGB; YIQ→RGB se usa internamente al recomponer.")

    def do_promedio_canales(self):
        if not self._require_base():
            return
        arrY = get_Y_from_rgb(self.arr_base)
        rgb = np.repeat(arrY[..., None], 3, axis=2)
        self.img_out = rgb_f32_to_pil(rgb)
        self._refresh_view()
        self.status.set("Promedio de canales (gris).")

    def do_solo_Y(self):
        if not self._require_base():
            return
        arrY = get_Y_from_rgb(self.arr_base)
        rgb = np.repeat(arrY[..., None], 3, axis=2)
        self.img_out = rgb_f32_to_pil(rgb)
        self._refresh_view()
        self.status.set("Mostrando solo canal Y (luminancia).")

    # ------------- Aritmética -------------

    def _check_arit(self):
        if self.arr_base is None or self.arr_sec is None:
            messagebox.showinfo("Aviso", "Necesitás imagen base y secundaria (mismo tamaño).")
            return False
        return True

    def _Y_both(self):
        Ya = get_Y_from_rgb(self.arr_base)
        Yb = get_Y_from_rgb(self.arr_sec)
        return Ya, Yb

    def do_sum_prom(self):
        if not self._check_arit():
            return
        Ya, Yb = self._Y_both()
        C = (Ya + Yb) / 2.0
        rgb = Y_to_rgb_like(C, self.arr_base)
        self.img_out = rgb_f32_to_pil(rgb)
        self._refresh_view()
        self.status.set("Cuasi-suma promediada sobre Y.")

    def do_sum_clamp(self):
        if not self._check_arit():
            return
        Ya, Yb = self._Y_both()
        C = np.clip(Ya + Yb, 0.0, 1.0)
        rgb = Y_to_rgb_like(C, self.arr_base)
        self.img_out = rgb_f32_to_pil(rgb)
        self._refresh_view()
        self.status.set("Cuasi-suma clampeada sobre Y.")

    def do_rest_prom(self):
        if not self._check_arit():
            return
        Ya, Yb = self._Y_both()
        C = (Ya - Yb) / 2.0 + 0.5
        C = np.clip(C, 0.0, 1.0)
        rgb = Y_to_rgb_like(C, self.arr_base)
        self.img_out = rgb_f32_to_pil(rgb)
        self._refresh_view()
        self.status.set("Cuasi-resta promediada (centrada en gris).")

    def do_rest_clamp(self):
        if not self._check_arit():
            return
        Ya, Yb = self._Y_both()
        C = np.clip(Ya - Yb, 0.0, 1.0)
        rgb = Y_to_rgb_like(C, self.arr_base)
        self.img_out = rgb_f32_to_pil(rgb)
        self._refresh_view()
        self.status.set("Cuasi-resta clampeada (máx(Ya−Yb,0)).")

    def do_if_lighter(self):
        if not self._check_arit():
            return
        Ya, Yb = self._Y_both()
        mask = Ya >= Yb
        C = np.where(mask, Ya, Yb)
        rgb = Y_to_rgb_like(C, self.arr_base)
        self.img_out = rgb_f32_to_pil(rgb)
        self._refresh_view()
        self.status.set("If-lighter sobre Y.")

    def do_if_darker(self):
        if not self._check_arit():
            return
        Ya, Yb = self._Y_both()
        mask = Ya <= Yb
        C = np.where(mask, Ya, Yb)
        rgb = Y_to_rgb_like(C, self.arr_base)
        self.img_out = rgb_f32_to_pil(rgb)
        self._refresh_view()
        self.status.set("If-darker sobre Y.")

    # ------------- Luminancia -------------

    def _Y_from_base(self):
        return get_Y_from_rgb(self.arr_base)

    def _Y_to_out(self, Y):
        rgb = Y_to_rgb_like(np.clip(Y, 0.0, 1.0), self.arr_base)
        self.img_out = rgb_f32_to_pil(rgb)
        self._refresh_view()

    def do_lum_sqrt(self):
        if not self._require_base():
            return
        Y = self._Y_from_base()
        self._Y_to_out(np.sqrt(Y))
        self.status.set("Filtro de luminancia: raíz (aclara sombras).")

    def do_lum_square(self):
        if not self._require_base():
            return
        Y = self._Y_from_base()
        self._Y_to_out(Y ** 2)
        self.status.set("Filtro de luminancia: cuadrático (oscurece).")

    def do_lum_piecewise(self):
        if not self._require_base():
            return
        Y = self._Y_from_base()
        ymin = self.ymin_var.get() / 255.0
        ymax = self.ymax_var.get() / 255.0
        a = 1.0 / max(ymax - ymin, 1e-6)
        b = -a * ymin
        Y2 = a * Y + b
        Y2 = np.clip(Y2, 0.0, 1.0)
        self._Y_to_out(Y2)
        self.status.set(f"Función lineal a trozos en rango [{ymin:.2f},{ymax:.2f}].")

    # ------------- Convolución -------------

    def _conv_on_Y(self, kernel: np.ndarray, take_abs=False, renorm=True):
        if not self._require_base():
            return
        Y = self._Y_from_base()
        R = convolve(Y, kernel)
        if take_abs:
            R = np.abs(R)
        if renorm:
            R = R - R.min()
            if R.max() > 0:
                R = R / R.max()
        self._Y_to_out(R)

    def do_conv_plano(self, size):
        self._conv_on_Y(plano_kernel(size))
        self.status.set(f"Filtro pasabajos plano {size}x{size} sobre Y.")

    def do_conv_bartlett(self, size):
        self._conv_on_Y(bartlett_kernel(size))
        self.status.set(f"Filtro Bartlett {size}x{size} sobre Y.")

    def do_conv_gauss(self, size):
        self._conv_on_Y(gauss_kernel(size))
        self.status.set(f"Filtro Gaussiano {size}x{size} sobre Y.")

    def do_conv_lap(self, kind):
        k = LAPLACE_V4 if kind == "v4" else LAPLACE_V8
        self._conv_on_Y(k, take_abs=True, renorm=True)
        self.status.set(f"Laplaciano {kind} sobre Y.")

    def do_conv_sobel(self, direction):
        k = sobel_kernel(direction)
        self._conv_on_Y(k, take_abs=True, renorm=True)
        self.status.set(f"Sobel {direction} sobre Y.")

    # ------------- Morfología -------------

    def _morf_on_Y(self, func, label):
        if not self._require_base():
            return
        Y = self._Y_from_base()
        R = func(Y)
        self._Y_to_out(R)
        self.status.set(label)

    def do_erosion(self):
        self._morf_on_Y(erosion_3x3, "Erosión 3x3 sobre Y.")

    def do_dilatacion(self):
        self._morf_on_Y(dilatacion_3x3, "Dilatación 3x3 sobre Y.")

    def do_apertura(self):
        if not self._require_base():
            return
        Y = self._Y_from_base()
        R = dilatacion_3x3(erosion_3x3(Y))
        self._Y_to_out(R)
        self.status.set("Apertura (erosión→dilatación) sobre Y.")

    def do_cierre(self):
        if not self._require_base():
            return
        Y = self._Y_from_base()
        R = erosion_3x3(dilatacion_3x3(Y))
        self._Y_to_out(R)
        self.status.set("Cierre (dilatación→erosión) sobre Y.")

    def do_grad_morf(self):
        self._morf_on_Y(gradiente_morfologico, "Borde morfológico (gradiente).")

    def do_mediana(self):
        self._morf_on_Y(mediana_3x3, "Filtro de mediana 3x3 sobre Y.")

    # ------------- Binarización / segmentación -------------

    def _Y_or_gray_from_current(self):
        if self.img_out is not None:
            arr = np.asarray(self.img_out.convert("L"), dtype=np.float32) / 255.0
        else:
            if not self._require_base():
                return None
            arr = self._Y_from_base()
        return arr

    def _set_from_bin(self, bin_arr: np.ndarray, keep_gray=False):
        if keep_gray:
            arr_u8 = (bin_arr * 255.0 + 0.5).astype(np.uint8)
        else:
            arr_u8 = (bin_arr > 0.5).astype(np.uint8) * 255
        self.img_out = Image.fromarray(arr_u8, mode="L")
        self._refresh_view()

    def do_bin_global(self):
        arr = self._Y_or_gray_from_current()
        if arr is None:
            return
        thr = self.thr_var.get() / 255.0
        bin_arr = bin_global(arr, thr)
        self._set_from_bin(bin_arr, keep_gray=False)
        self.status.set(f"Binarización global con umbral {self.thr_var.get()}.")

    def do_invert(self):
        if self.img_out is None:
            if not self._require_base():
                return
            self.img_out = self.img_base.copy().convert("L")
        arr = np.asarray(self.img_out, dtype=np.uint8)
        self.img_out = Image.fromarray(255 - arr, mode="L")
        self._refresh_view()
        self.status.set("Imagen de salida invertida.")

    def do_bin_mediana(self):
        arr = self._Y_or_gray_from_current()
        if arr is None:
            return
        self._set_from_bin(bin_mediana(arr), keep_gray=False)
        self.status.set("Binarización 50/50 (mediana).")

    def do_bin_dos_modas(self):
        arr = self._Y_or_gray_from_current()
        if arr is None:
            return
        self._set_from_bin(bin_dos_modas(arr), keep_gray=False)
        self.status.set("Binarización por dos modas (distancia).")

    def do_bin_otsu(self):
        arr = self._Y_or_gray_from_current()
        if arr is None:
            return
        self._set_from_bin(bin_otsu(arr), keep_gray=False)
        self.status.set("Binarización de Otsu.")

    def do_marching(self):
        arr = self._Y_or_gray_from_current()
        if arr is None:
            return
        ms_img = marching_squares(arr, thr=None)
        self.img_out = ms_img
        self._refresh_view()
        self.status.set("Marching Squares aplicado (contorno).")

    def enable_varita(self):
        if self._Y_or_gray_from_current() is None:
            return
        self.var_mode_fill.set(True)
        self.status.set("Modo Color Fill activado: hacé clic en la imagen de salida.")

    def on_click_out(self, event):
        if not self.var_mode_fill.get():
            return
        if self.img_out is None:
            return
        # mapa de click sobre imagen real
        w0, h0 = self.img_out.size
        disp_w, disp_h = self.disp_size_out
        x = int(event.x * w0 / max(disp_w, 1))
        y = int(event.y * h0 / max(disp_h, 1))
        x = max(0, min(w0 - 1, x))
        y = max(0, min(h0 - 1, y))

        arr_u8 = np.asarray(self.img_out.convert("L"), dtype=np.uint8)
        tol = int(self.tol_fill.get())
        mask = flood_fill(arr_u8, x, y, tol)
        arr_new = arr_u8.copy()
        arr_new[mask] = 255
        self.img_out = Image.fromarray(arr_new, mode="L")
        self._refresh_view()
        self.status.set(f"Color Fill aplicado desde ({x},{y}) con tolerancia {tol}.")
        self.var_mode_fill.set(False)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    app = PDIApp()
    app.mainloop()
