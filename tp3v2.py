# -*- coding: utf-8 -*-
"""
PDI – Operaciones de Luminancia (YIQ) + Histogramas
- Filtros: raíz, cuadrática, lineal a trozos.
- Flujo: RGB -> YIQ, Y' = f(Y), Y'IQ -> RGB.
- I y Q se mantienen.
- Muestra imagen original y procesada, y sus histogramas de luminancia.

Requisitos:
    pip install pillow numpy matplotlib
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import os

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# YIQ transforms (NTSC)
# =========================
RGB2YIQ = np.array([
    [0.299000,   0.587000,   0.114000],
    [0.595716,  -0.274453,  -0.321263],
    [0.211456,  -0.522591,   0.311135],
], dtype=np.float64)

YIQ2RGB = np.array([
    [1.0,  0.9563,  0.6210],
    [1.0, -0.2721, -0.6474],
    [1.0, -1.1070,  1.7046],
], dtype=np.float64)

I_MIN, I_MAX = -0.5957, 0.5957
Q_MIN, Q_MAX = -0.5226, 0.5226

OPS = ["Raíz (sqrt) – iluminar",
       "Cuadrática (Y*Y) – oscurecer",
       "Lineal a trozos (Ymin/Ymax)"]


class LuminanciaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDI – Operaciones de Luminancia (YIQ) + Histogramas")
        self.geometry("1300x780")
        self.minsize(1100, 700)

        self.img = None          # PIL original
        self.proc = None         # PIL procesada
        self.tk_orig = None
        self.tk_proc = None

        self.op_var = tk.StringVar(value=OPS[0])
        self.autoupdate = tk.BooleanVar(value=True)
        self.ymin = tk.DoubleVar(value=0.2)
        self.ymax = tk.DoubleVar(value=0.8)

        # figuras para histogramas
        self.fig_orig = None
        self.ax_orig = None
        self.canvas_orig_hist = None

        self.fig_proc = None
        self.ax_proc = None
        self.canvas_proc_hist = None

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Button(top, text="📂 Cargar imagen", command=self.load_image).pack(side=tk.LEFT)
        ttk.Label(top, text="   Filtro:").pack(side=tk.LEFT, padx=(16, 6))
        cb = ttk.Combobox(top, textvariable=self.op_var, values=OPS,
                          state="readonly", width=32)
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda e: self._maybe_auto())

        ttk.Checkbutton(top, text="Actualizar automáticamente",
                        variable=self.autoupdate).pack(side=tk.LEFT, padx=14)
        ttk.Button(top, text="▶ Procesar", command=self.apply).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="💾 Guardar", command=self.save).pack(side=tk.LEFT, padx=(8, 0))

        # Controles de lineal a trozos
        seg = ttk.LabelFrame(self, text="Parámetros – Lineal a trozos")
        seg.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 8))

        ttk.Label(seg, text="Ymin").pack(side=tk.LEFT, padx=(8, 4))
        smin = ttk.Scale(seg, from_=0.0, to=0.9, variable=self.ymin,
                         command=lambda _=None: self._maybe_auto())
        smin.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        e1 = ttk.Entry(seg, width=6, textvariable=self.ymin)
        e1.pack(side=tk.LEFT, padx=(0, 12))
        e1.bind("<Return>", lambda e: self._maybe_auto())

        ttk.Label(seg, text="Ymax").pack(side=tk.LEFT, padx=(8, 4))
        smax = ttk.Scale(seg, from_=0.1, to=1.0, variable=self.ymax,
                         command=lambda _=None: self._maybe_auto())
        smax.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        e2 = ttk.Entry(seg, width=6, textvariable=self.ymax)
        e2.pack(side=tk.LEFT, padx=(0, 8))
        e2.bind("<Return>", lambda e: self._maybe_auto())

        ttk.Button(seg, text="Auto (percentiles 5–95)",
                   command=self.auto_percentiles).pack(side=tk.LEFT, padx=6)

        # === Zona de imágenes + histogramas ===
        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        # --- Columna izquierda: original ---
        colL = ttk.LabelFrame(mid, text="Original")
        colL.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        img_frame_L = ttk.Frame(colL)
        img_frame_L.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.lbl_orig = ttk.Label(img_frame_L, anchor="center")
        self.lbl_orig.pack(fill=tk.BOTH, expand=True)

        hist_frame_L = ttk.Frame(colL, height=200)
        hist_frame_L.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

        self.fig_orig = Figure(figsize=(4, 2), dpi=100)
        self.ax_orig = self.fig_orig.add_subplot(111)
        self.ax_orig.set_title("Histograma Y (original)")
        self.ax_orig.set_xlim(0, 1)
        self.canvas_orig_hist = FigureCanvasTkAgg(self.fig_orig, master=hist_frame_L)
        self.canvas_orig_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Columna derecha: procesada ---
        colR = ttk.LabelFrame(mid, text="Procesada")
        colR.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        img_frame_R = ttk.Frame(colR)
        img_frame_R.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.lbl_proc = ttk.Label(img_frame_R, anchor="center")
        self.lbl_proc.pack(fill=tk.BOTH, expand=True)

        hist_frame_R = ttk.Frame(colR, height=200)
        hist_frame_R.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

        self.fig_proc = Figure(figsize=(4, 2), dpi=100)
        self.ax_proc = self.fig_proc.add_subplot(111)
        self.ax_proc.set_title("Histograma Y (procesada)")
        self.ax_proc.set_xlim(0, 1)
        self.canvas_proc_hist = FigureCanvasTkAgg(self.fig_proc, master=hist_frame_R)
        self.canvas_proc_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status
        self.status = tk.StringVar(
            value="Cargá una imagen y compará luminancia + histogramas."
        )
        ttk.Label(self, textvariable=self.status, anchor="w").pack(
            side=tk.BOTTOM, fill=tk.X
        )

    # -------------- Helpers de UI --------------
    def _fit(self, pil_img, label):
        lw = label.winfo_width() or 1
        lh = label.winfo_height() or 1
        iw, ih = pil_img.size
        s = min(lw / iw, lh / ih)
        s = max(1e-6, min(s, 1.0))
        return pil_img.resize((max(1, int(iw * s)), max(1, int(ih * s))), Image.LANCZOS)

    def _update_hist_orig(self):
        """Histograma de Y de la imagen original (10 bins, %)."""
        self.ax_orig.clear()
        self.ax_orig.set_title("Histograma Y (original)")
        self.ax_orig.set_xlabel("Luminancia")
        self.ax_orig.set_ylabel("Frec. relativa de aparición (%)")
        self.ax_orig.set_xlim(0, 1)
        self.ax_orig.set_ylim(0, 100)
        self.ax_orig.grid(True, axis="y", linestyle="-", linewidth=0.5)

        if self.img is None:
            self.canvas_orig_hist.draw()
            return

        arr = np.array(self.img, dtype=np.uint8)
        Y, _, _ = self._rgb_to_yiq_uint8(arr)

        # 10 bins: [0,0.1), [0.1,0.2), ... [0.9,1.0]
        bins = np.linspace(0.0, 1.0, 11)
        counts, edges = np.histogram(Y.ravel(), bins=bins)
        total = counts.sum() if counts.sum() > 0 else 1
        perc = counts * 100.0 / total

        centers = (edges[:-1] + edges[1:]) / 2.0
        width = 0.1  # ancho de cada bin
        self.ax_orig.bar(centers, perc, width=width, align="center")

        self.canvas_orig_hist.draw()

    def _update_hist_proc(self):
        """Histograma de Y de la imagen procesada (10 bins, %)."""
        self.ax_proc.clear()
        self.ax_proc.set_title("Histograma Y (procesada)")
        self.ax_proc.set_xlabel("Luminancia")
        self.ax_proc.set_ylabel("Frec. relativa de aparición (%)")
        self.ax_proc.set_xlim(0, 1)
        self.ax_proc.set_ylim(0, 100)
        self.ax_proc.grid(True, axis="y", linestyle="-", linewidth=0.5)

        if self.proc is None:
            self.canvas_proc_hist.draw()
            return

        arr = np.array(self.proc, dtype=np.uint8)
        Y, _, _ = self._rgb_to_yiq_uint8(arr)

        bins = np.linspace(0.0, 1.0, 11)
        counts, edges = np.histogram(Y.ravel(), bins=bins)
        total = counts.sum() if counts.sum() > 0 else 1
        perc = counts * 100.0 / total

        centers = (edges[:-1] + edges[1:]) / 2.0
        width = 0.1
        self.ax_proc.bar(centers, perc, width=width, align="center")

        self.canvas_proc_hist.draw()


    def _show(self):
        # imágenes
        if self.img is not None:
            x = self._fit(self.img, self.lbl_orig)
            self.tk_orig = ImageTk.PhotoImage(x)
            self.lbl_orig.configure(image=self.tk_orig)
        if self.proc is not None:
            y = self._fit(self.proc, self.lbl_proc)
            self.tk_proc = ImageTk.PhotoImage(y)
            self.lbl_proc.configure(image=self.tk_proc)

        # histogramas
        self._update_hist_orig()
        self._update_hist_proc()

    def _maybe_auto(self):
        if self.autoupdate.get():
            self.apply()

    # -------------- IO --------------
    def load_image(self):
        p = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")],
        )
        if not p:
            return
        try:
            self.img = Image.open(p).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")
            return
        self.status.set(
            f"Imagen: {os.path.basename(p)} – {self.img.size[0]}x{self.img.size[1]} px"
        )
        self.apply(force=True)

    def save(self):
        if self.proc is None:
            messagebox.showinfo("Guardar", "No hay imagen procesada.")
            return
        p = filedialog.asksaveasfilename(
            title="Guardar procesada",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")],
        )
        if not p:
            return
        try:
            self.proc.save(p)
            self.status.set(f"Guardado en: {p}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar:\n{e}")

    # ----------- Núcleo (YIQ) -------------
    @staticmethod
    def _rgb_to_yiq_uint8(arr_u8):
        rgb = arr_u8.astype(np.float64) / 255.0
        flat = rgb.reshape(-1, 3)
        yiq = flat @ RGB2YIQ.T
        H, W = rgb.shape[:2]
        return yiq[:, 0].reshape(H, W), yiq[:, 1].reshape(H, W), yiq[:, 2].reshape(H, W)

    @staticmethod
    def _yiq_to_rgb_u8(Y, I, Q):
        Y = np.clip(Y, 0.0, 1.0)
        I = np.clip(I, I_MIN, I_MAX)
        Q = np.clip(Q, Q_MIN, Q_MAX)
        H, W = Y.shape
        yiq = np.stack([Y.ravel(), I.ravel(), Q.ravel()], axis=1)
        rgb = yiq @ YIQ2RGB.T
        rgb = np.clip(rgb, 0.0, 1.0).reshape(H, W, 3)
        return (rgb * 255.0 + 0.5).astype(np.uint8)

    def _apply_sqrt(self, Y):
        # Y' = sqrt(Y) – iluminar
        return np.sqrt(np.clip(Y, 0.0, 1.0))

    def _apply_square(self, Y):
        # Y' = Y^2 – oscurecer
        return np.clip(Y, 0.0, 1.0) ** 2

    def _apply_piecewise(self, Y, ymin, ymax):
        # Lineal a trozos: [ymin, ymax] -> [0,1], saturando fuera
        ymin = float(ymin)
        ymax = float(ymax)
        if ymax <= ymin:
            ymax = ymin + 1e-6
        Yc = np.zeros_like(Y)
        Yc[Y >= ymax] = 1.0
        mask = (Y >= ymin) & (Y < ymax)
        Yc[mask] = (Y[mask] - ymin) / (ymax - ymin)
        return Yc

    def apply(self, force=False):
        if self.img is None:
            return
        arr = np.array(self.img, dtype=np.uint8)
        Y, I, Q = self._rgb_to_yiq_uint8(arr)
        op = self.op_var.get()

        if op == OPS[0]:
            Yp = self._apply_sqrt(Y)
        elif op == OPS[1]:
            Yp = self._apply_square(Y)
        else:
            y0 = float(min(self.ymin.get(), self.ymax.get()))
            y1 = float(max(self.ymin.get(), self.ymax.get()))
            self.ymin.set(y0)
            self.ymax.set(y1)
            Yp = self._apply_piecewise(Y, y0, y1)

        out = self._yiq_to_rgb_u8(Yp, I, Q)
        self.proc = Image.fromarray(out, mode="RGB")
        self._show()

        if op == OPS[2]:
            self.status.set(
                f"Lineal a trozos: Ymin={self.ymin.get():.3f}, Ymax={self.ymax.get():.3f}"
            )
        else:
            self.status.set(f"Filtro aplicado: {op}")

    def auto_percentiles(self):
        """Fija Ymin/Ymax usando percentiles (5%, 95%) de Y de la imagen original."""
        if self.img is None:
            return
        arr = np.array(self.img, dtype=np.uint8)
        Y, _, _ = self._rgb_to_yiq_uint8(arr)
        y0 = float(np.percentile(Y, 5))
        y1 = float(np.percentile(Y, 95))
        if y1 <= y0:
            y1 = min(1.0, y0 + 0.1)
        self.ymin.set(y0)
        self.ymax.set(y1)
        self._maybe_auto()


if __name__ == "__main__":
    app = LuminanciaApp()
    app.update_idletasks()
    app.mainloop()
