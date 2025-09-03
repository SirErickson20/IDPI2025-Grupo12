import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

class ImageProcessorYIQ:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ajuste de Luminancia y Saturación con YIQ")
        self.root.geometry("1400x700")
        self.root.configure(bg="#0f1419")

        self.imagen_original = None
        self.imagen_procesada = None
        self.Y_img = None
        self.I_img = None
        self.Q_img = None

        self.imagen_tk_izq = None
        self.imagen_tk_der = None

        self.luminancia = tk.DoubleVar(value=1.0)
        self.saturacion = tk.DoubleVar(value=1.0)

        self.setup_ui()
        self.root.mainloop()

    def setup_ui(self):
        # Frames
        self.frame_izq = tk.Frame(self.root, bg="#1a1f29", width=400)
        self.frame_izq.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Scroll para frame_der
        self.frame_der_container = tk.Frame(self.root, bg="#1a1f29", width=400)
        self.frame_der_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.canvas_der = tk.Canvas(self.frame_der_container, bg="#1a1f29")
        self.scrollbar_der = tk.Scrollbar(self.frame_der_container, orient="vertical", command=self.canvas_der.yview)
        self.canvas_der.configure(yscrollcommand=self.scrollbar_der.set)

        self.scrollbar_der.pack(side="right", fill="y")
        self.canvas_der.pack(side="left", fill="both", expand=True)

        self.frame_der = tk.Frame(self.canvas_der, bg="#1a1f29")
        self.canvas_der.create_window((0,0), window=self.frame_der, anchor="nw")

        self.frame_der.bind("<Configure>", lambda e: self.canvas_der.configure(scrollregion=self.canvas_der.bbox("all")))

        self.frame_control = tk.Frame(self.root, bg="#0f1419", width=200)
        self.frame_control.pack(side="left", fill="y", padx=10, pady=10)

        # Labels de imagen
        self.label_izquierda = tk.Label(self.frame_izq, text="Imagen Original", bg="#0f1419", fg="white")
        self.label_izquierda.pack(expand=True)

        self.label_derecha = tk.Label(self.frame_der, text="Imagen Procesada", bg="#0f1419", fg="white")
        self.label_derecha.pack(expand=True, pady=10)

        # NUEVO: Labels para YIQ
        self.label_Y = tk.Label(self.frame_der, text="", bg="#0f1419", fg="white")
        self.label_Y.pack(pady=10)
        self.label_I = tk.Label(self.frame_der, text="", bg="#0f1419", fg="white")
        self.label_I.pack(pady=10)
        self.label_Q = tk.Label(self.frame_der, text="", bg="#0f1419", fg="white")
        self.label_Q.pack(pady=10)

        # Botones
        tk.Button(self.frame_control, text="Cargar Imagen", command=self.cargar_imagen).pack(pady=10, fill="x")
        tk.Button(self.frame_control, text="Guardar Imagen", command=self.guardar_imagen).pack(pady=10, fill="x")
        tk.Button(self.frame_control, text="Aplicar Ajustes", command=self.aplicar_ajustes).pack(pady=10, fill="x")
        tk.Button(self.frame_control, text="Mostrar YIQ", command=self.mostrar_YIQ).pack(pady=10, fill="x")

        # Sliders
        tk.Label(self.frame_control, text="Luminancia (a)", bg="#0f1419", fg="white").pack(pady=(20,0))
        tk.Scale(self.frame_control, from_=0.0, to=2.0, resolution=0.01, orient="horizontal",
                 variable=self.luminancia).pack(fill="x", padx=10)

        tk.Label(self.frame_control, text="Saturación (b)", bg="#0f1419", fg="white").pack(pady=(20,0))
        tk.Scale(self.frame_control, from_=0.0, to=2.0, resolution=0.01, orient="horizontal",
                 variable=self.saturacion).pack(fill="x", padx=10)

    def cargar_imagen(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if file_path:
            self.imagen_original = Image.open(file_path).convert("RGB")
            self.imagen_procesada = self.imagen_original.copy()
            self.mostrar_imagen_izquierda()
            self.mostrar_imagen_derecha()

    def mostrar_imagen_izquierda(self):
        if self.imagen_original:
            img_copy = self.imagen_original.copy()
            img_copy.thumbnail((400, 400))
            self.imagen_tk_izq = ImageTk.PhotoImage(img_copy)
            self.label_izquierda.config(image=self.imagen_tk_izq, text="")

    def mostrar_imagen_derecha(self):
        if self.imagen_procesada:
            img_copy = self.imagen_procesada.copy()
            img_copy.thumbnail((400, 400))
            self.imagen_tk_der = ImageTk.PhotoImage(img_copy)
            self.label_derecha.config(image=self.imagen_tk_der, text="")

    def aplicar_ajustes(self):
        if self.imagen_original is None:
            messagebox.showwarning("Atención", "Primero cargue una imagen")
            return

        a = self.luminancia.get()
        b = self.saturacion.get()

        arr = np.array(self.imagen_original, dtype=float)/255.0

        # Conversion RGB -> YIQ
        R = arr[:,:,0]
        G = arr[:,:,1]
        B = arr[:,:,2]

        Y = 0.299*R + 0.587*G + 0.114*B
        I = 0.596*R - 0.274*G - 0.322*B
        Q = 0.211*R - 0.523*G + 0.312*B

        # Guardar canales normalizados para mostrar
        self.Y_img = Image.fromarray(np.uint8(np.clip(Y*255,0,255)))
        self.I_img = Image.fromarray(np.uint8(np.clip((I+0.5957)/1.1914*255,0,255)))
        self.Q_img = Image.fromarray(np.uint8(np.clip((Q+0.5226)/1.0452*255,0,255)))

        # Ajuste
        Yp = np.clip(a*Y, 0, 1)
        Ip = np.clip(b*I, -0.5957, 0.5957)
        Qp = np.clip(b*Q, -0.5226, 0.5226)

        # Conversion YIQ -> RGB
        Rp = Yp + 0.956*Ip + 0.621*Qp
        Gp = Yp - 0.272*Ip - 0.647*Qp
        Bp = Yp - 1.106*Ip + 1.703*Qp

        rgb_proc = np.stack([Rp, Gp, Bp], axis=2)
        rgb_proc = np.clip(rgb_proc, 0, 1)
        rgb_proc = (rgb_proc*255).astype(np.uint8)

        self.imagen_procesada = Image.fromarray(rgb_proc)
        self.mostrar_imagen_derecha()

    def mostrar_YIQ(self):
        if self.Y_img is None or self.I_img is None or self.Q_img is None:
            messagebox.showwarning("Atención", "Primero cargue una imagen y aplique ajustes")
            return

        # Convertir a PhotoImage
        Y_tk = ImageTk.PhotoImage(self.Y_img.resize((350,350)))
        I_tk = ImageTk.PhotoImage(self.I_img.resize((350,350)))
        Q_tk = ImageTk.PhotoImage(self.Q_img.resize((350,350)))

        self.label_Y.config(text="Y (Luminancia)", image=Y_tk, compound="top")
        self.label_I.config(text="I (In-phase)", image=I_tk, compound="top")
        self.label_Q.config(text="Q (Quadrature)", image=Q_tk, compound="top")

        # Mantener referencias para que no se pierdan
        self.label_Y.image = Y_tk
        self.label_I.image = I_tk
        self.label_Q.image = Q_tk

    def guardar_imagen(self):
        if self.imagen_procesada:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG", "*.png"),
                                                                ("BMP", "*.bmp"),
                                                                ("JPEG", "*.jpg"),
                                                                ("Todos los archivos", "*.*")])
            if file_path:
                self.imagen_procesada.save(file_path)
                messagebox.showinfo("Guardado", "Imagen guardada correctamente")

if __name__ == "__main__":
    ImageProcessorYIQ()