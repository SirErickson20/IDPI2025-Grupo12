import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
from PIL import Image, ImageTk
import io
import math

# ==============================================================================
# 1. Librería "my_functions" (Básica y Convolución con Padding)
# ==============================================================================

MAT_YIQ = np.array([[0.299, 0.595716, 0.211456],
                    [0.587, -0.274453, -0.522591],
                    [0.114, -0.321263, 0.311135]])

def rgb2yiq(_im):
    """Convierte una imagen RGB (numpy, 0-255) a YIQ (float, 0-1)."""
    _im_float = _im.astype(np.float64) / 255.0
    _rgb = _im_float.reshape((-1, 3))
    _yiq = _rgb @ MAT_YIQ 
    _yiq = _yiq.reshape(_im_float.shape)
    return _yiq

def _pad_image_replication(image, kernel_shape):
    """
    Implementa el padding por replicación de bordes ('edge') para mantener 
    el tamaño de la imagen después de la convolución.
    """
    kernel_h, kernel_w = kernel_shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

def _convolution_padded(image_padded, kernel):
    """
    Implementación de convolución manual. 
    Requiere que la imagen de entrada ya esté padddeada por replicación.
    La salida tiene el mismo tamaño que la imagen original.
    """
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    H = image_padded.shape[0] - 2 * pad_h
    W = image_padded.shape[1] - 2 * pad_w
    
    convolved = np.zeros((H, W))

    for x in range(H):
        for y in range(W):
            region = image_padded[x : x + kernel_h, y : y + kernel_w]
            convolved[x, y] = (region * kernel).sum()
            
    return convolved

# ==============================================================================
# 2. Definición de Kernels y Generadores
# ==============================================================================

# Kernels Laplacianos (v4 y v8)
KERNEL_LAPLACIANO_V4 = np.array([[ 0,  1,  0], [ 1, -4,  1], [ 0,  1,  0]])
KERNEL_LAPLACIANO_V8 = np.array([[ 1,  1,  1], [ 1, -8,  1], [ 1,  1,  1]])

def get_gaussian_kernel(size, sigma):
    """Genera un kernel Gaussiano 2D."""
    center = size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()

def create_lowpass_gaussian(size, cutoff_freq):
    """Crea un kernel PasaBajos Gaussiano."""
    if cutoff_freq <= 0: return np.ones((size, size)) / (size * size)
    sigma = 1 / (2 * np.pi * cutoff_freq)
    return get_gaussian_kernel(size, sigma)

def create_highpass_kernel(size, cutoff_freq):
    """Crea un kernel PasaAltos (HPF = Identidad - LPF)."""
    lpf_kernel = create_lowpass_gaussian(size, cutoff_freq)
    identity_kernel = np.zeros((size, size))
    identity_kernel[size // 2, size // 2] = 1
    hpf_kernel = identity_kernel - lpf_kernel
    return hpf_kernel

def create_bandpass_kernel(size_small, size_large, cutoff_freq_small, cutoff_freq_large):
    """Crea un filtro PasaBanda (BPF) como la Diferencia de Gaussianas (DOG)."""
    sigma1 = 1 / (2 * np.pi * cutoff_freq_large)
    lpf1 = get_gaussian_kernel(size_large, sigma1) 
    
    sigma2 = 1 / (2 * np.pi * cutoff_freq_small)
    lpf2 = get_gaussian_kernel(size_small, sigma2) 
    
    pad_size = (size_large - size_small) // 2
    lpf2_padded = np.pad(lpf2, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')

    bpf_kernel = lpf2_padded - lpf1
    bpf_kernel = bpf_kernel / np.abs(bpf_kernel).sum()
    
    return bpf_kernel

# ==============================================================================
# 3. Lógica de Filtrado
# ==============================================================================

def apply_filter_logic(image_np, kernel):
    """
    Aplica el filtro usando la convolución manual con padding de replicación.
    """
    yiq_image = rgb2yiq(image_np)
    y_channel = yiq_image[:, :, 0]

    # Aplicar Padding
    y_padded = _pad_image_replication(y_channel, kernel.shape)
    
    # Aplicar Convolución
    try:
        y_filtered = _convolution_padded(y_padded, kernel)
    except ValueError as e:
        messagebox.showerror("Error de Convolución", str(e))
        return None

    # Normalizar el resultado
    y_min = y_filtered.min()
    y_max = y_filtered.max()

    if y_max != y_min:
        y_normalized = (y_filtered - y_min) / (y_max - y_min)
    else:
        y_normalized = np.zeros_like(y_filtered)

    y_display = (y_normalized * 255).astype(np.uint8)
    return y_display

# ==============================================================================
# 4. Clase de la Interfaz Tkinter
# ==============================================================================

class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("Filtrado por Convolución - PDI UNS")

        # Variables de estado y kernels
        self.original_image_pil = None
        self.filtered_image_pil = None
        self.original_image_np = None
        self.original_image_tk = None
        self.filtered_image_tk_filtered = None 
        
        # Opciones de filtro
        self.fixed_filter_options = {
            "Laplaciano v4": KERNEL_LAPLACIANO_V4,
            "Laplaciano v8": KERNEL_LAPLACIANO_V8,
            "PasaAltos (corte 0.2)": lambda: create_highpass_kernel(size=5, cutoff_freq=0.2),
            "PasaAltos (corte 0.4)": lambda: create_highpass_kernel(size=5, cutoff_freq=0.4),
            "PasaBanda (DOG 5x5/3x3)": lambda: create_bandpass_kernel(size_small=3, size_large=5, cutoff_freq_small=0.4, cutoff_freq_large=0.2)
        }
        self.selected_filter = tk.StringVar(master)
        self.selected_filter.set("Laplaciano v4") 

        # --- Configuración de la Interfaz (Layout) ---
        main_frame = ttk.Frame(master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        
        # Frames de imagen
        img_frame = ttk.Frame(main_frame)
        img_frame.grid(row=0, column=0, columnspan=2, pady=10)
        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=1)
        
        # Etiquetas de imagen
        self.label_original = ttk.Label(img_frame, text="[IMAGEN ORIGINAL]", anchor="center", relief="groove", width=50)
        self.label_original.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.label_filtered = ttk.Label(img_frame, text="[IMAGEN FILTRADA]", anchor="center", relief="groove", width=50)
        self.label_filtered.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        
        # Etiquetas de título
        ttk.Label(img_frame, text="Imagen Original", font=('Arial', 10, 'bold')).grid(row=1, column=0)
        ttk.Label(img_frame, text="Imagen Filtrada (Canal Y)", font=('Arial', 10, 'bold')).grid(row=1, column=1)

        # Frame de Controles Centrales
        control_frame = ttk.Frame(main_frame, padding="10 0 10 0")
        control_frame.grid(row=2, column=0, columnspan=2, pady=10)

        # --- Controles ---
        
        # Cargar
        ttk.Button(control_frame, text="Cargar Imagen", command=self.load_image).grid(row=0, column=0, padx=5)

        # Selector de Filtro
        self.filter_combobox = ttk.Combobox(control_frame, textvariable=self.selected_filter, state="readonly", 
                                            values=list(self.fixed_filter_options.keys()))
        self.filter_combobox.grid(row=0, column=1, padx=5)

        # Filtrar
        ttk.Button(control_frame, text="Aplicar Filtro →", command=self.process_image).grid(row=0, column=2, padx=5)
        
        # Guardar
        self.save_button = ttk.Button(control_frame, text="Guardar Imagen", command=self.save_image, state=tk.DISABLED)
        self.save_button.grid(row=0, column=3, padx=5)
        
    # --- Métodos de la Aplicación ---

    def load_image(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar Imagen RGB",
            filetypes=[("Archivos de Imagen", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if not filepath: return
        try:
            self.original_image_pil = Image.open(filepath).convert("RGB")
            self.original_image_np = np.array(self.original_image_pil)
            self.filtered_image_pil = None
            self.update_image_display()
        except Exception as e:
            messagebox.showerror("Error de Carga", f"No se pudo cargar la imagen. Detalle: {e}")

    def update_image_display(self):
        max_size = 400
        
        # Mostrar Original
        if self.original_image_pil:
            img_disp_orig = self.original_image_pil.copy()
            img_disp_orig.thumbnail((max_size, max_size))
            self.original_image_tk = ImageTk.PhotoImage(img_disp_orig)
            self.label_original.config(image=self.original_image_tk, text="")
        else:
            self.label_original.config(image="", text="[IMAGEN ORIGINAL]")
            
        # Mostrar Filtrada
        if self.filtered_image_pil:
            img_disp_filt = self.filtered_image_pil.copy()
            img_disp_filt.thumbnail((max_size, max_size))
            self.filtered_image_tk_filtered = ImageTk.PhotoImage(img_disp_filt)
            self.label_filtered.config(image=self.filtered_image_tk_filtered, text="")
            self.save_button.config(state=tk.NORMAL)
        else:
            self.label_filtered.config(image="", text="[IMAGEN FILTRADA]")
            self.save_button.config(state=tk.DISABLED)

    def process_image(self):
        if self.original_image_np is None:
            messagebox.showwarning("Advertencia", "Primero debes cargar una imagen.")
            return

        try:
            filter_name = self.selected_filter.get()
            filter_item = self.fixed_filter_options[filter_name]
            
            if callable(filter_item):
                kernel = filter_item() 
            else:
                kernel = filter_item    
            
            y_filtered_array = apply_filter_logic(self.original_image_np, kernel)
            
            if y_filtered_array is None: 
                return

            self.filtered_image_pil = Image.fromarray(y_filtered_array, mode='L')
            
            self.update_image_display()
            
            # Línea eliminada para evitar la notificación:
            # messagebox.showinfo("Éxito", f"Filtro '{filter_name}' aplicado. Tamaño: {y_filtered_array.shape[0]}x{y_filtered_array.shape[1]} (mismo tamaño que el original).")

        except Exception as e:
            messagebox.showerror("Error de Procesamiento", f"Error al aplicar el filtro. Detalle: {e}")

    def save_image(self):
        if self.filtered_image_pil is None:
            messagebox.showwarning("Advertencia", "Aplica un filtro antes de guardar.")
            return

        filetypes = [
            ("PNG (Portable Network Graphics)", "*.png"),
            ("BMP (Bitmap)", "*.bmp"),
            ("TIFF (Tagged Image File Format)", "*.tiff"),
            ("Todos los archivos", "*.*")
        ]
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png", 
            filetypes=filetypes,
            title="Guardar Imagen Filtrada"
        )
        
        if not filepath:
            return

        save_format = filepath.split('.')[-1].upper()
        if save_format == 'TIF': 
             save_format = 'TIFF'
        elif save_format in ('JPG', 'JPEG'):
             save_format = 'JPEG'
        
        try:
            self.filtered_image_pil.save(filepath, format=save_format)
            messagebox.showinfo("Guardado Exitoso", f"Imagen guardada en {filepath} como {save_format}.")
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar la imagen: {e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()