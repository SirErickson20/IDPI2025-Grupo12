import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import my_functions as mf   # tu archivo con conversiones

# ================= FILTROS =================
def filtro_raiz(Y):
    return np.sqrt(Y)

def filtro_cuadrado(Y):
    return np.power(Y, 2)

def filtro_lineal_trozos(Y, ymin=0.2, ymax=0.8):
    Yp = np.zeros_like(Y)
    Yp[Y < ymin] = 0
    Yp[Y > ymax] = 1
    mask = (Y >= ymin) & (Y <= ymax)
    Yp[mask] = (Y[mask] - ymin) / (ymax - ymin)
    return Yp

# ================= APLICACIÓN =================
class ProcesadorImagen:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Luminancia")
        self.imagen_original = None
        self.imagen_procesada = None
        self.img_array = None

        # Botones y selector
        frame = tk.Frame(root)
        frame.pack()

        tk.Button(frame, text="Abrir Imagen", command=self.abrir_imagen).grid(row=0, column=0)
        tk.Button(frame, text="Procesar", command=self.procesar).grid(row=0, column=1)
        tk.Button(frame, text="Guardar", command=self.guardar).grid(row=0, column=2)

        self.filtro = ttk.Combobox(frame, values=["Raíz", "Cuadrado", "Lineal a Trozos"])
        self.filtro.current(0)
        self.filtro.grid(row=0, column=3)

        # Frames para organizar
        self.frame_imgs = tk.Frame(root)
        self.frame_imgs.pack(side="top", pady=10)

        self.frame_hist = tk.Frame(root)
        self.frame_hist.pack(side="bottom", pady=10)

        # Labels para imágenes
        self.canvas_original = tk.Label(self.frame_imgs)
        self.canvas_original.pack(side="left", padx=10)

        self.canvas_procesada = tk.Label(self.frame_imgs)
        self.canvas_procesada.pack(side="right", padx=10)

        # Canvas para histogramas
        self.hist_original = None
        self.hist_procesada = None

    def mostrar_histograma(self, Y, titulo, pos, bins=10):
        # calcular histograma normalizado en %
        counts, edges = np.histogram(Y.flatten(), bins=bins, range=(0,1))
        counts = counts / counts.sum() * 100   # porcentaje

        # graficar
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(edges[:-1], counts, width=1/bins, color="blue", edgecolor="black", align="edge")
        ax.set_title(titulo)
        ax.set_xlabel("Luminancia")
        ax.set_ylabel("Frec. relativa de aparición (%)")
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, bins+1))
        ax.grid(True)

        # pasar a Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frame_hist)
        canvas.draw()
        widget = canvas.get_tk_widget()
        if pos == "izq":
            if self.hist_original: self.hist_original.destroy()
            self.hist_original = widget
            widget.pack(side="left", padx=10)
        else:
            if self.hist_procesada: self.hist_procesada.destroy()
            self.hist_procesada = widget
            widget.pack(side="right", padx=10)
        plt.close(fig)

    def abrir_imagen(self):
        ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.bmp;*.tiff;*.jpg")])
        if ruta:
            self.imagen_original = Image.open(ruta).convert("RGB")
            self.img_array = np.asarray(self.imagen_original) / 255.0
            self.mostrar_imagen(self.imagen_original, self.canvas_original)

            # Histograma original
            yiq = mf.rgb2yiq(self.img_array)
            Y = yiq[:,:,0]
            self.mostrar_histograma(Y, "Histograma original", "izq")

    def procesar(self):
        if self.imagen_original is None:
            messagebox.showerror("Error", "Primero abre una imagen")
            return

        yiq = mf.rgb2yiq(self.img_array)
        Y, I, Q = yiq[:,:,0], yiq[:,:,1], yiq[:,:,2]

        if self.filtro.get() == "Raíz":
            Yp = filtro_raiz(Y)
        elif self.filtro.get() == "Cuadrado":
            Yp = filtro_cuadrado(Y)
        else:
            Yp = filtro_lineal_trozos(Y)

        yiq_p = np.stack([Yp, I, Q], axis=2)
        rgb_p = np.clip(mf.yiq2rgb(yiq_p), 0, 1)

        self.imagen_procesada = Image.fromarray((rgb_p*255).astype(np.uint8))
        self.mostrar_imagen(self.imagen_procesada, self.canvas_procesada)

        # Histograma procesado
        self.mostrar_histograma(Yp, "Histograma procesado", "der")

    def guardar(self):
        if self.imagen_procesada is None:
            messagebox.showerror("Error", "No hay imagen procesada")
            return
        ruta = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"),
                                                       ("BMP", "*.bmp"),
                                                       ("TIFF", "*.tiff")])
        if ruta:
            self.imagen_procesada.save(ruta)
            messagebox.showinfo("Éxito", f"Imagen guardada en {ruta}")

    def mostrar_imagen(self, img, label):
        img_resized = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img_resized)
        label.config(image=img_tk)
        label.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ProcesadorImagen(root)
    root.mainloop()
