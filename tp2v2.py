# -*- coding: utf-8 -*-
"""
PDI – Aritmética de píxeles (YIQ) – v3
- Interpolación convexa de I,Q ponderada por Y.
- Cuasi-suma (promediada/clampeada), cuasi-resta (promediada/clampeada),
  if-lighter, if-darker.
- La primera imagen define el tamaño de referencia; la otra se reescala.

Requisitos:
    pip install pillow numpy
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import os

# ======== Transformaciones YIQ (NTSC) ========
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

OPS = [
    "Cuasi-suma: Promediada (YIQ)",
    "Cuasi-suma: Clampeada (YIQ)",
    "Cuasi-resta: Promediada (YIQ)",
    "Cuasi-resta: Clampeada (YIQ)",
    "If-lighter (YIQ)",
    "If-darker (YIQ)",
]

class AritmeticaPixelesYIQApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDI – Aritmética de píxeles (YIQ) – v3")
        self.geometry("1320x760")
        self.minsize(1100, 640)

        self.imgA = None; self.imgB = None; self.imgC = None
        self.tkA = None;  self.tkB = None;  self.tkC = None

        self.op_var = tk.StringVar(value=OPS[0])
        self.autoupdate = tk.BooleanVar(value=True)
        self.ref_size = None  # (w,h) — la define la primera imagen

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self); top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Button(top, text="📂 Cargar A", command=self.load_A).pack(side=tk.LEFT)
        ttk.Button(top, text="📂 Cargar B", command=self.load_B).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(top, text="   Operación:").pack(side=tk.LEFT, padx=(20, 6))
        cb = ttk.Combobox(top, textvariable=self.op_var, values=OPS, state="readonly", width=36)
        cb.pack(side=tk.LEFT); cb.bind("<<ComboboxSelected>>", lambda e: self.process())

        ttk.Checkbutton(top, text="Actualizar automáticamente", variable=self.autoupdate).pack(side=tk.LEFT, padx=16)
        ttk.Button(top, text="▶ Ejecutar", command=self.process).pack(side=tk.LEFT)
        ttk.Button(top, text="💾 Guardar C", command=self.save_C).pack(side=tk.LEFT, padx=(8, 0))

        mid = ttk.Frame(self); mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        lfA = ttk.LabelFrame(mid, text="Imagen A"); lfA.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.lblA = ttk.Label(lfA, anchor="center"); self.lblA.pack(fill=tk.BOTH, expand=True)

        lfB = ttk.LabelFrame(mid, text="Imagen B"); lfB.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
        self.lblB = ttk.Label(lfB, anchor="center"); self.lblB.pack(fill=tk.BOTH, expand=True)

        lfC = ttk.LabelFrame(mid, text="Resultado C"); lfC.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self.lblC = ttk.Label(lfC, anchor="center"); self.lblC.pack(fill=tk.BOTH, expand=True)

        self.status = tk.StringVar(value="Cargá A y B; la primera imagen define el tamaño de referencia.")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    # ---------- Archivo ----------
    def save_C(self):
        if self.imgC is None:
            messagebox.showinfo("Guardar", "No hay resultado aún."); return
        path = filedialog.asksaveasfilename(
            title="Guardar resultado C", defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg"),("BMP","*.bmp")]
        )
        if not path: return
        try:
            self.imgC.save(path)
            self.status.set(f"Resultado guardado en: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar: {e}")

    # ---------- Carga ----------
    def load_A(self):
        p = filedialog.askopenfilename(title="Seleccionar imagen A",
             filetypes=[("Imágenes","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not p: return
        try: img = Image.open(p).convert("RGB")
        except Exception as e: messagebox.showerror("Error", f"No se pudo abrir A: {e}"); return
        if self.ref_size is None: self.ref_size = img.size
        elif img.size != self.ref_size: img = img.resize(self.ref_size, Image.BICUBIC)
        self.imgA = img; self._show_A(); self._maybe_auto()
        self.status.set(f"A: {os.path.basename(p)} – {self.imgA.size[0]}x{self.imgA.size[1]} px (ref={self.ref_size})")

    def load_B(self):
        p = filedialog.askopenfilename(title="Seleccionar imagen B",
             filetypes=[("Imágenes","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not p: return
        try: img = Image.open(p).convert("RGB")
        except Exception as e: messagebox.showerror("Error", f"No se pudo abrir B: {e}"); return
        if self.ref_size is None: self.ref_size = img.size
        elif img.size != self.ref_size: img = img.resize(self.ref_size, Image.BICUBIC)
        self.imgB = img; self._show_B(); self._maybe_auto()
        self.status.set(f"B: {os.path.basename(p)} – {self.imgB.size[0]}x{self.imgB.size[1]} px (ref={self.ref_size})")

    # ---------- Mostrar ----------
    def _fit(self, pil_img, label):
        lw = label.winfo_width() or 1; lh = label.winfo_height() or 1
        iw, ih = pil_img.size; s = min(lw/iw, lh/ih); s = max(1e-6, min(s, 1.0))
        return pil_img.resize((max(1,int(iw*s)), max(1,int(ih*s))), Image.LANCZOS)

    def _show_A(self):
        if self.imgA is None: return
        x = self._fit(self.imgA, self.lblA); self.tkA = ImageTk.PhotoImage(x); self.lblA.configure(image=self.tkA)

    def _show_B(self):
        if self.imgB is None: return
        x = self._fit(self.imgB, self.lblB); self.tkB = ImageTk.PhotoImage(x); self.lblB.configure(image=self.tkB)

    def _show_C(self):
        if self.imgC is None: return
        x = self._fit(self.imgC, self.lblC); self.tkC = ImageTk.PhotoImage(x); self.lblC.configure(image=self.tkC)

    def _maybe_auto(self):
        if self.autoupdate.get(): self.process()

    # ---------- Helpers YIQ ----------
    @staticmethod
    def _rgb_to_yiq(rgb_u8: np.ndarray):
        rgb = rgb_u8.astype(np.float64) / 255.0
        flat = rgb.reshape(-1, 3)
        yiq = flat @ RGB2YIQ.T
        H, W = rgb.shape[:2]
        return yiq[:,0].reshape(H,W), yiq[:,1].reshape(H,W), yiq[:,2].reshape(H,W)

    @staticmethod
    def _yiq_to_rgb_u8(Y: np.ndarray, I: np.ndarray, Q: np.ndarray) -> np.ndarray:
        Y = np.clip(Y, 0.0, 1.0); I = np.clip(I, I_MIN, I_MAX); Q = np.clip(Q, Q_MIN, Q_MAX)
        H, W = Y.shape
        yiq = np.stack([Y.ravel(), I.ravel(), Q.ravel()], axis=1)
        rgb = yiq @ YIQ2RGB.T
        rgb = np.clip(rgb, 0.0, 1.0).reshape(H, W, 3)
        return (rgb * 255.0 + 0.5).astype(np.uint8)

    @staticmethod
    def _interp_IQ(YA, IA, QA, YB, IB, QB):
        denom = YA + YB
        wA = np.divide(YA, denom, out=np.zeros_like(YA), where=denom!=0)
        wB = 1.0 - wA
        return wA*IA + wB*IB, wA*QA + wB*QB

    # ---------- Operaciones ----------
    def process(self):
        if self.imgA is None or self.imgB is None: return
        op = self.op_var.get()
        A = np.array(self.imgA, dtype=np.uint8)
        B = np.array(self.imgB, dtype=np.uint8)

        if   op == OPS[0]: C = self.cuasi_suma_promediada_yiq(A, B)
        elif op == OPS[1]: C = self.cuasi_suma_clampeada_yiq(A, B)
        elif op == OPS[2]: C = self.cuasi_resta_promediada_yiq(A, B)
        elif op == OPS[3]: C = self.cuasi_resta_clampeada_yiq(A, B)
        elif op == OPS[4]: C = self.if_lighter_yiq(A, B)
        elif op == OPS[5]: C = self.if_darker_yiq(A, B)
        else: C = A.copy()

        self.imgC = Image.fromarray(C, mode="RGB"); self._show_C()

    # Cuasi-sumas
    def cuasi_suma_promediada_yiq(self, A_u8, B_u8):
        YA, IA, QA = self._rgb_to_yiq(A_u8); YB, IB, QB = self._rgb_to_yiq(B_u8)
        YC = 0.5*(YA+YB); IC, QC = self._interp_IQ(YA, IA, QA, YB, IB, QB)
        return self._yiq_to_rgb_u8(YC, IC, QC)

    def cuasi_suma_clampeada_yiq(self, A_u8, B_u8):
        YA, IA, QA = self._rgb_to_yiq(A_u8); YB, IB, QB = self._rgb_to_yiq(B_u8)
        YC = np.clip(YA+YB, 0.0, 1.0); IC, QC = self._interp_IQ(YA, IA, QA, YB, IB, QB)
        return self._yiq_to_rgb_u8(YC, IC, QC)

    # Cuasi-restas
    def cuasi_resta_promediada_yiq(self, A_u8, B_u8):
        YA, IA, QA = self._rgb_to_yiq(A_u8); YB, IB, QB = self._rgb_to_yiq(B_u8)
        YC = np.clip((YA - YB)/2.0 + 0.5, 0.0, 1.0); IC, QC = self._interp_IQ(YA, IA, QA, YB, IB, QB)
        return self._yiq_to_rgb_u8(YC, IC, QC)

    def cuasi_resta_clampeada_yiq(self, A_u8, B_u8):
        YA, IA, QA = self._rgb_to_yiq(A_u8); YB, IB, QB = self._rgb_to_yiq(B_u8)
        YC = np.clip(YA - YB, 0.0, 1.0); IC, QC = self._interp_IQ(YA, IA, QA, YB, IB, QB)
        return self._yiq_to_rgb_u8(YC, IC, QC)

    # If-lighter / If-darker
    def if_lighter_yiq(self, A_u8, B_u8):
        YA, _, _ = self._rgb_to_yiq(A_u8); YB, _, _ = self._rgb_to_yiq(B_u8)
        return np.where((YA >= YB)[..., None], A_u8, B_u8)

    def if_darker_yiq(self, A_u8, B_u8):
        YA, _, _ = self._rgb_to_yiq(A_u8); YB, _, _ = self._rgb_to_yiq(B_u8)
        return np.where((YA <= YB)[..., None], A_u8, B_u8)


if __name__ == "__main__":
    app = AritmeticaPixelesYIQApp()
    app.update_idletasks()
    app.mainloop()
