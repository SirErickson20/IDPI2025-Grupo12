import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import my_functions as mf  # librería con rgb2yiq / yiq2rgb

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesamiento de Imágenes")
        self.root.geometry("1200x600")

        # Variables
        self.image1 = None
        self.image2 = None
        self.result = None

        # --- Panel de controles ---
        control_frame = tk.Frame(root, pady=5)
        control_frame.pack(fill="x")

        tk.Button(control_frame, text="Abrir Imagen 1", command=self.load_image1).pack(side="left", padx=5)
        tk.Button(control_frame, text="Abrir Imagen 2", command=self.load_image2).pack(side="left", padx=5)
        tk.Button(control_frame, text="Procesar", command=self.process_images).pack(side="left", padx=5)
        tk.Button(control_frame, text="Guardar", command=self.save_image).pack(side="left", padx=5)
        tk.Button(control_frame, text="Salir", command=root.quit).pack(side="left", padx=5)

        tk.Label(control_frame, text="Operación:").pack(side="left", padx=5)
        self.operation = ttk.Combobox(
            control_frame,
            values=[
                "Cuasi Suma", "Cuasi Resta",
                "Producto", "Cociente",
                "Resta Absoluta",
                "If-Darker", "If-Lighter"
            ],
            state="readonly"
        )
        self.operation.current(0)
        self.operation.pack(side="left", padx=5)

        tk.Label(control_frame, text="Modo:").pack(side="left", padx=5)
        self.mode = ttk.Combobox(control_frame, values=["Clampeada", "Promediada"], state="readonly")
        self.mode.current(0)
        self.mode.pack(side="left", padx=5)

        tk.Label(control_frame, text="Formato:").pack(side="left", padx=5)
        self.format = ttk.Combobox(control_frame, values=["PNG", "BMP", "TIF"], state="readonly")
        self.format.current(0)
        self.format.pack(side="left", padx=5)

        tk.Label(control_frame, text="Espacio:").pack(side="left", padx=5)
        self.space = ttk.Combobox(control_frame, values=["RGB", "YIQ"], state="readonly")
        self.space.current(0)
        self.space.pack(side="left", padx=5)

        # --- Panel de imágenes ---
        self.frame_images = tk.Frame(root)
        self.frame_images.pack(fill="both", expand=True)

        self.label_img1 = tk.Label(self.frame_images, text="Imagen 1", borderwidth=2, relief="groove")
        self.label_img1.pack(side="left", expand=True, fill="both", padx=5, pady=5)

        self.label_img2 = tk.Label(self.frame_images, text="Imagen 2", borderwidth=2, relief="groove")
        self.label_img2.pack(side="left", expand=True, fill="both", padx=5, pady=5)

        self.label_result = tk.Label(self.frame_images, text="Procesada", borderwidth=2, relief="groove")
        self.label_result.pack(side="left", expand=True, fill="both", padx=5, pady=5)

    # --- Funciones de interfaz ---
    def load_image1(self):
        file = filedialog.askopenfilename(filetypes=[("Images", "*.png *.bmp *.tif *.jpg")])
        if file:
            self.image1 = Image.open(file).convert("RGB")
            self.show_image(self.image1, self.label_img1)

    def load_image2(self):
        file = filedialog.askopenfilename(filetypes=[("Images", "*.png *.bmp *.tif *.jpg")])
        if file:
            self.image2 = Image.open(file).convert("RGB")
            self.show_image(self.image2, self.label_img2)

    def show_image(self, img, label):
        img_resized = img.resize((350, 350))
        imgtk = ImageTk.PhotoImage(img_resized)
        label.imgtk = imgtk
        label.config(image=imgtk)

    def process_images(self):
        if self.image1 is None or self.image2 is None:
            messagebox.showerror("Error", "Cargue ambas imágenes primero")
            return

        op = self.operation.get()
        mode = self.mode.get()
        space = self.space.get()

        # Normalizamos a [0,1] para trabajar con floats
        im1 = np.array(self.image1, dtype=np.float32) / 255.0
        im2 = np.array(self.image2, dtype=np.float32) / 255.0

        # Convertir a YIQ si corresponde
        if space == "YIQ" or op in ["If-Darker", "If-Lighter"]:
            im1 = mf.rgb2yiq(im1)
            im2 = mf.rgb2yiq(im2)

        # --- Operaciones ---
        if op in ["Cuasi Suma", "Cuasi Resta"] and space == "RGB":
            if op == "Cuasi Suma":
                result = im1 + im2
            else:
                result = im1 - im2

            if mode == "Promediada":
                result = result / 2

        elif op in ["Cuasi Suma", "Cuasi Resta"] and space == "YIQ":
            YA, IA, QA = im1[:,:,0], im1[:,:,1], im1[:,:,2]
            YB, IB, QB = im2[:,:,0], im2[:,:,1], im2[:,:,2]

            if op == "Cuasi Suma":
                if mode == "Clampeada":
                    YC = np.clip(YA + YB, 0, 1)
                else:  # Promediada
                    YC = (YA + YB) / 2
            else:  # Cuasi Resta
                if mode == "Clampeada":
                    YC = np.clip(YA - YB, 0, 1)
                else:
                    YC = (YA - YB) / 2

            denom = YA + YB + 1e-5
            IC = (YA*IA + YB*IB) / denom
            QC = (YA*QA + YB*QB) / denom
            result = np.dstack([YC, IC, QC])

        elif op == "Producto":
            result = im1 * im2

        elif op == "Cociente":
            result = np.divide(im1, im2 + 1e-5)

        elif op == "Resta Absoluta":
            result = np.abs(im1 - im2)

        elif op == "If-Darker":
            mask = im1[:,:,0] < im2[:,:,0]
            result = np.where(mask[:,:,None], im1, im2)

        elif op == "If-Lighter":
            mask = im1[:,:,0] > im2[:,:,0]
            result = np.where(mask[:,:,None], im1, im2)

        # --- Reconversión ---
        if space == "YIQ" or op in ["If-Darker", "If-Lighter"]:
            result = mf.yiq2rgb(result)

        # Clamp final y convertir a uint8
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

        self.result = Image.fromarray(result)
        self.show_image(self.result, self.label_result)

    def save_image(self):
        if self.result is None:
            messagebox.showerror("Error", "No hay imagen procesada")
            return
        file = filedialog.asksaveasfilename(
            defaultextension=f".{self.format.get().lower()}",
            filetypes=[("Image", f"*.{self.format.get().lower()}")]
        )
        if file:
            self.result.save(file, self.format.get())

# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
