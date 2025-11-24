# -*- coding: utf-8 -*-
"""
PDI – Segmentación de imágenes (FI-UNJu)
Actividad práctica:

1) Dadas imágenes a niveles de gris, binarizarlas y comparar resultados:
   - Binarización 50/50 (mediana)
   - Binarización por dos modas clara/oscura (distancia mínima)
   - Binarización de Otsu

2) En imágenes como la (a) encontrar los bordes:
   - Laplaciano
   - Borde morfológico
   - Marching Squares (contorno vectorial)

3) En imágenes como la (b), implementar color fill con "varita mágica".

Requisitos:
    pip install pillow numpy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np


# ===============================
# Utilidades generales
# ===============================

def to_gray(img: Image.Image) -> Image.Image:
    """Convierte a escala de grises (luminancia)."""
    return img.convert("L")


def img_to_float(arr_u8: np.ndarray) -> np.ndarray:
    return arr_u8.astype(np.float64) / 255.0


def float_to_u8(arr_f: np.ndarray) -> np.ndarray:
    return (np.clip(arr_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ===============================
# Binarizaciones
# ===============================

def binarizar_mediana(arr_f: np.ndarray) -> np.ndarray:
    """Binarización 50/50: umbral = mediana de luminancia."""
    thr = np.median(arr_f)
    return (arr_f >= thr).astype(np.float64)


def otsu_threshold(arr_f: np.ndarray) -> float:
    """Calcula el umbral de Otsu en [0,1]."""
    arr_u8 = float_to_u8(arr_f)
    hist, _ = np.histogram(arr_u8, bins=256, range=(0, 255))
    total = arr_u8.size
    sum_total = np.dot(hist, np.arange(256))

    sumB = 0.0
    wB = 0.0
    varMax = 0.0
    threshold = 0

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
            threshold = t

    return threshold / 255.0


def binarizar_otsu(arr_f: np.ndarray) -> np.ndarray:
    thr = otsu_threshold(arr_f)
    return (arr_f >= thr).astype(np.float64)


def binarizar_dos_modas(arr_f: np.ndarray) -> np.ndarray:
    """
    Binarización por dos modas:
    - Encuentra dos picos principales del histograma
    - Asigna cada píxel a la moda más cercana (clara u oscura)
    """
    arr_u8 = float_to_u8(arr_f)
    hist, _ = np.histogram(arr_u8, bins=256, range=(0, 255))
    peaks = np.argsort(hist)[::-1]  # índices de histogramas ordenados por frecuencia desc.

    if len(peaks) < 2:
        # caso raro, usamos simplemente mediana
        return binarizar_mediana(arr_f)

    m1 = peaks[0]
    m2 = peaks[1]

    # Intentar que estén algo separados
    for idx in peaks[1:]:
        if abs(idx - m1) >= 10:
            m2 = idx
            break

    # Distancia a cada moda
    A = arr_u8.astype(np.int16)
    d1 = np.abs(A - int(m1))
    d2 = np.abs(A - int(m2))

    mask = d1 <= d2
    return np.where(mask, 1.0, 0.0)


# ===============================
# Filtros de bordes (convolución, morfología)
# ===============================

def pad_replicate(arr: np.ndarray, r: int) -> np.ndarray:
    return np.pad(arr, pad_width=r, mode="edge")


def convolve_replicate(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolución 2D con padding por replicación."""
    k = kernel.shape[0]
    assert k == kernel.shape[1]
    r = k // 2
    padded = pad_replicate(arr, r)
    h, w = arr.shape
    out = np.zeros_like(arr, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + k, j:j + k]
            out[i, j] = np.sum(region * kernel)
    return out


LAPLACE_V4 = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float64)


def laplaciano_v4(arr: np.ndarray) -> np.ndarray:
    resp = convolve_replicate(arr, LAPLACE_V4)
    # Tomamos valor absoluto y normalizamos a [0,1]
    resp = np.abs(resp)
    resp = resp / (resp.max() + 1e-8)
    return resp


def erosion_3x3(arr: np.ndarray) -> np.ndarray:
    padded = pad_replicate(arr, 1)
    h, w = arr.shape
    out = np.empty_like(arr)
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+3, j:j+3]
            out[i, j] = np.min(patch)
    return out


def dilatacion_3x3(arr: np.ndarray) -> np.ndarray:
    padded = pad_replicate(arr, 1)
    h, w = arr.shape
    out = np.empty_like(arr)
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+3, j:j+3]
            out[i, j] = np.max(patch)
    return out


def borde_morfologico(arr: np.ndarray) -> np.ndarray:
    er = erosion_3x3(arr)
    di = dilatacion_3x3(arr)
    grad = di - er
    grad = grad - grad.min()
    if grad.max() > 0:
        grad = grad / grad.max()
    return grad


# ===============================
# Marching Squares (contorno vectorial)
# ===============================

def marching_squares_contour(arr_f: np.ndarray, thr: float = None) -> Image.Image:
    """
    Marching Squares completo sobre una imagen en gris.
    - threshold -> si None, usa Otsu
    - devuelve una imagen en L con fondo negro y contorno blanco
    """
    h, w = arr_f.shape
    if thr is None:
        thr = otsu_threshold(arr_f)
    mask = arr_f >= thr

    out_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(out_img)

    for y in range(h - 1):
        for x in range(w - 1):
            tl = mask[y, x]
            tr = mask[y, x+1]
            br = mask[y+1, x+1]
            bl = mask[y+1, x]

            points = []
            # top edge
            if tl != tr:
                points.append((x + 0.5, y))
            # right edge
            if tr != br:
                points.append((x + 1.0, y + 0.5))
            # bottom edge
            if br != bl:
                points.append((x + 0.5, y + 1.0))
            # left edge
            if bl != tl:
                points.append((x, y + 0.5))

            if len(points) == 2:
                draw.line((points[0][0], points[0][1],
                           points[1][0], points[1][1]), fill=255)
            elif len(points) == 4:
                # caso ambiguo: hacemos dos segmentos
                draw.line((points[0][0], points[0][1],
                           points[1][0], points[1][1]), fill=255)
                draw.line((points[2][0], points[2][1],
                           points[3][0], points[3][1]), fill=255)

    return out_img


# ===============================
# Color fill (varita mágica)
# ===============================

def flood_fill_region(arr_u8: np.ndarray, x0: int, y0: int, tol: int) -> np.ndarray:
    """
    Flood fill 4-conexo sobre una imagen de 0..255.
    Devuelve una máscara booleana con los píxeles pertenecientes a la región.
    """
    h, w = arr_u8.shape
    seed_val = int(arr_u8[y0, x0])
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
        if abs(int(arr_u8[y, x]) - seed_val) <= tol:
            mask[y, x] = True
            stack.append((x+1, y))
            stack.append((x-1, y))
            stack.append((x, y+1))
            stack.append((x, y-1))

    return mask


# ===============================
# App Tkinter
# ===============================

OPS = [
    "Binarización 50/50 (mediana)",
    "Binarización dos modas (distancia)",
    "Binarización Otsu",
    "Bordes: Laplaciano v4",
    "Bordes: Borde morfológico",
    "Bordes: Marching Squares",
    "Color Fill (Varita mágica)",
]


class SegmentacionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PDI – Segmentación de imágenes")
        self.geometry("1200x750")
        self.minsize(1000, 650)

        # Imágenes de trabajo
        self.img_in = None      # PIL L
        self.arr_in = None      # float [0,1]
        self.img_out = None     # PIL L

        self.tk_in = None
        self.tk_out = None

        # Para mapear clicks en la varita
        self.disp_size_in = (1, 1)
        self.disp_size_out = (1, 1)

        self.op_var = tk.StringVar(value=OPS[0])
        self.autoupdate = tk.BooleanVar(value=True)
        self.tolerancia = tk.IntVar(value=15)  # para color fill

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Button(top, text="📂 Cargar imagen",
                   command=self.load_image).pack(side=tk.LEFT)

        ttk.Label(top, text="   Operación:").pack(side=tk.LEFT, padx=(16, 4))
        cb = ttk.Combobox(top, textvariable=self.op_var,
                          values=OPS, state="readonly", width=35)
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda e: self._maybe_auto())

        ttk.Checkbutton(top, text="Actualizar automáticamente",
                        variable=self.autoupdate).pack(side=tk.LEFT, padx=12)

        ttk.Button(top, text="▶ Aplicar",
                   command=self.apply_operation).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="⮐ Copiar salida a entrada",
                   command=self.copy_output_to_input).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="💾 Guardar salida",
                   command=self.save_output).pack(side=tk.LEFT, padx=(6, 0))

        # Parámetro tolerancia para color fill
        tol_frame = ttk.LabelFrame(self, text="Parámetro – Varita mágica")
        tol_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 8))

        ttk.Label(tol_frame, text="Tolerancia: ").pack(side=tk.LEFT, padx=(8, 4))
        tol_scale = ttk.Scale(tol_frame, from_=0, to=100,
                              variable=self.tolerancia,
                              command=lambda _=None: None)
        tol_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Label(tol_frame, textvariable=self.tolerancia, width=4
                  ).pack(side=tk.LEFT, padx=(0, 8))

        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        lf1 = ttk.LabelFrame(mid, text="Imagen original / actual (gris)")
        lf1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.lbl_in = ttk.Label(lf1, anchor="nw")
        self.lbl_in.pack(fill=tk.BOTH, expand=True)

        lf2 = ttk.LabelFrame(mid, text="Resultado")
        lf2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.lbl_out = ttk.Label(lf2, anchor="nw")
        self.lbl_out.pack(fill=tk.BOTH, expand=True)

        # Click para varita mágica en la imagen de salida
        self.lbl_out.bind("<Button-1>", self.on_click_out)

        self.status = tk.StringVar(
            value="Cargá una imagen en gris. Elegí binarización, bordes o color fill."
        )
        ttk.Label(self, textvariable=self.status,
                  anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    # ---------------- Helpers UI ----------------
    def _fit(self, pil_img, label):
        lw = label.winfo_width() or 1
        lh = label.winfo_height() or 1
        iw, ih = pil_img.size
        s = min(lw / iw, lh / ih)
        s = max(1e-6, min(s, 1.0))
        return pil_img.resize((max(1, int(iw * s)),
                               max(1, int(ih * s))), Image.LANCZOS)

    def _refresh_view(self):
        # entrada
        if self.img_in is not None:
            disp = self._fit(self.img_in, self.lbl_in)
            self.disp_size_in = disp.size
            self.tk_in = ImageTk.PhotoImage(disp)
            self.lbl_in.configure(image=self.tk_in)

        # salida
        if self.img_out is not None:
            disp = self._fit(self.img_out, self.lbl_out)
            self.disp_size_out = disp.size
            self.tk_out = ImageTk.PhotoImage(disp)
            self.lbl_out.configure(image=self.tk_out)

    def _maybe_auto(self):
        if self.autoupdate.get():
            self.apply_operation()

    # ---------------- IO ----------------
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

        self.img_in = to_gray(img)
        arr_u8 = np.array(self.img_in, dtype=np.uint8)
        self.arr_in = img_to_float(arr_u8)
        self.img_out = None
        self._refresh_view()
        self.status.set(f"Imagen cargada: {path}")
        self._maybe_auto()

    def save_output(self):
        if self.img_out is None:
            messagebox.showinfo("Guardar", "No hay salida todavía.")
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
        """Copia la salida como nueva imagen de entrada (para encadenar)."""
        if self.img_out is None:
            messagebox.showinfo("Copiar", "No hay salida para copiar.")
            return
        self.img_in = self.img_out.copy()
        arr_u8 = np.array(self.img_in, dtype=np.uint8)
        self.arr_in = img_to_float(arr_u8)
        self.status.set("Salida copiada como nueva imagen de entrada.")
        self._refresh_view()
        self._maybe_auto()

    # ---------------- Núcleo de operaciones ----------------
    def apply_operation(self):
        if self.arr_in is None:
            return

        op = self.op_var.get()
        arr = self.arr_in

        if op == "Binarización 50/50 (mediana)":
            bin_arr = binarizar_mediana(arr)
            out_u8 = float_to_u8(bin_arr)
            self.img_out = Image.fromarray(out_u8, mode="L")

        elif op == "Binarización dos modas (distancia)":
            bin_arr = binarizar_dos_modas(arr)
            out_u8 = float_to_u8(bin_arr)
            self.img_out = Image.fromarray(out_u8, mode="L")

        elif op == "Binarización Otsu":
            bin_arr = binarizar_otsu(arr)
            out_u8 = float_to_u8(bin_arr)
            self.img_out = Image.fromarray(out_u8, mode="L")

        elif op == "Bordes: Laplaciano v4":
            edge = laplaciano_v4(arr)
            out_u8 = float_to_u8(edge)
            self.img_out = Image.fromarray(out_u8, mode="L")

        elif op == "Bordes: Borde morfológico":
            edge = borde_morfologico(arr)
            out_u8 = float_to_u8(edge)
            self.img_out = Image.fromarray(out_u8, mode="L")

        elif op == "Bordes: Marching Squares":
            # usa la imagen de entrada actual, umbral Otsu
            ms_img = marching_squares_contour(arr, thr=None)
            self.img_out = ms_img

        elif op == "Color Fill (Varita mágica)":
            # sólo actualizamos status; la acción ocurre en on_click_out
            self.status.set(
                "Modo Color Fill: hacé clic en la imagen de salida para aplicar la varita."
            )
            # si aún no hay salida, usamos la entrada como base visual
            if self.img_out is None and self.img_in is not None:
                self.img_out = self.img_in.copy()
        else:
            self.img_out = None

        self._refresh_view()
        if op != "Color Fill (Varita mágica)":
            self.status.set(f"Operación aplicada: {op}")

    # ---------------- Varita mágica (click) ----------------
    def on_click_out(self, event):
        if self.op_var.get() != "Color Fill (Varita mágica)":
            return
        if self.img_out is None:
            return

        base = self.img_out
        w0, h0 = base.size
        disp_w, disp_h = self.disp_size_out

        # Mapear coordenadas de click a coordenadas de imagen
        x = int(event.x * w0 / max(disp_w, 1))
        y = int(event.y * h0 / max(disp_h, 1))
        x = max(0, min(w0 - 1, x))
        y = max(0, min(h0 - 1, y))

        arr_u8 = np.array(base, dtype=np.uint8)
        tol = int(self.tolerancia.get())

        mask = flood_fill_region(arr_u8, x, y, tol)

        # Pintamos la región de blanco (255) y dejamos el resto igual
        arr_new = arr_u8.copy()
        arr_new[mask] = 255

        self.img_out = Image.fromarray(arr_new, mode="L")
        self._refresh_view()
        self.status.set(
            f"Color Fill aplicado desde ({x},{y}) con tolerancia {tol}."
        )


if __name__ == "__main__":
    app = SegmentacionApp()
    app.update_idletasks()
    app.mainloop()
