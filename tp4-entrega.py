import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
from PIL import Image, ImageTk
import io
import math

# ==============================================================================
# 1) Librería "my_functions" (Básica y Convolución con Padding)
#    - Conversión RGB→YIQ para trabajar sobre la luminancia (Y)
#    - Padding por replicación de bordes
#    - Convolución manual con tamaño de salida igual a la imagen original
# ==============================================================================

# Matriz de transformación de espacio de color RGB a YIQ
MAT_YIQ = np.array([[0.299, 0.595716, 0.211456],
                    [0.587, -0.274453, -0.522591],
                    [0.114, -0.321263, 0.311135]])

def rgb2yiq(_im):
    """
    Convierte una imagen RGB (numpy, valores 0-255) a YIQ (float, 0-1 aprox).
    - Normaliza a [0,1]
    - Multiplica por la matriz de conversión para obtener (Y, I, Q)
    - Devuelve con la misma forma que la entrada
    """
    _im_float = _im.astype(np.float64) / 255.0       # Normalizar a [0,1]
    _rgb = _im_float.reshape((-1, 3))                # Aplanar a (Npix, 3)
    _yiq = _rgb @ MAT_YIQ                            # Transformación lineal
    _yiq = _yiq.reshape(_im_float.shape)             # Volver a (H, W, 3)
    return _yiq

def _pad_image_replication(image, kernel_shape):
    """
    Aplica padding por replicación de bordes ('edge') para mantener el tamaño
    de la imagen después de la convolución.
    - Calcula cuántos píxeles agregar según tamaño del kernel
    - Usa np.pad con modo 'edge' (replicar el borde)
    """
    kernel_h, kernel_w = kernel_shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

def _convolution_padded(image_padded, kernel):
    """
    Convolución 2D manual sobre una imagen YA padded.
    Devuelve una imagen del MISMO tamaño que la original (sin padding).
    - Recorre cada posición (ventana) del kernel
    - Multiplica elemento a elemento y suma
    """
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # Dimensiones de la imagen original (quitando el padding)
    H = image_padded.shape[0] - 2 * pad_h
    W = image_padded.shape[1] - 2 * pad_w
    
    convolved = np.zeros((H, W))

    # Deslizar la ventana del kernel por toda la imagen (sin padding)
    for x in range(H):
        for y in range(W):
            region = image_padded[x : x + kernel_h, y : y + kernel_w]  # Vecindad
            convolved[x, y] = (region * kernel).sum()                  # Producto y suma
            
    return convolved

# ==============================================================================
# 2) Definición de Kernels y Generadores
#    - Kernels fijos (Laplacianos)
#    - Generadores de filtros Gaussianos (LPF), PasaAltos (HPF) y PasaBanda (DoG)
# ==============================================================================

# Kernels Laplacianos (con vecindad de 4 y de 8)
KERNEL_LAPLACIANO_V4 = np.array([[ 0,  1,  0],
                                 [ 1, -4,  1],
                                 [ 0,  1,  0]])
KERNEL_LAPLACIANO_V8 = np.array([[ 1,  1,  1],
                                 [ 1, -8,  1],
                                 [ 1,  1,  1]])

def get_gaussian_kernel(size, sigma):
    """
    Genera un kernel Gaussiano 2D normalizado de 'size'x'size' con desviación 'sigma'.
    - Crea mallas (x, y) centradas
    - Aplica la ecuación del Gaussiano
    - Normaliza para que la suma sea 1
    """
    center = size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()

def create_lowpass_gaussian(size, cutoff_freq):
    """
    Crea un kernel PasaBajos Gaussiano.
    - Si cutoff_freq <= 0, usa un promedio simple (filtro caja)
    - Si no, aproxima sigma = 1 / (2*pi*fc) y genera el gaussiano
    """
    if cutoff_freq <= 0:
        return np.ones((size, size)) / (size * size)
    sigma = 1 / (2 * np.pi * cutoff_freq)
    return get_gaussian_kernel(size, sigma)

def create_highpass_kernel(size, cutoff_freq):
    """
    Crea un kernel PasaAltos (HPF) como: Identidad - PasaBajos.
    - Genera LPF gaussiano
    - Resta al delta (identidad) para atenuar bajas y resaltar altas
    """
    lpf_kernel = create_lowpass_gaussian(size, cutoff_freq)
    identity_kernel = np.zeros((size, size))
    identity_kernel[size // 2, size // 2] = 1
    hpf_kernel = identity_kernel - lpf_kernel
    return hpf_kernel

def create_bandpass_kernel(size_small, size_large, cutoff_freq_small, cutoff_freq_large):
    """
    Crea un filtro PasaBanda (BPF) con Diferencia de Gaussianas (DoG).
    - Genera dos LPF con distintas sigmas y tamaños
    - Ajusta con padding para poder restarlos
    - Normaliza para que la suma de magnitudes sea 1 (evitar cambios de brillo)
    """
    sigma1 = 1 / (2 * np.pi * cutoff_freq_large)
    lpf1 = get_gaussian_kernel(size_large, sigma1) 
    
    sigma2 = 1 / (2 * np.pi * cutoff_freq_small)
    lpf2 = get_gaussian_kernel(size_small, sigma2) 
    
    # Alinear tamaños: pad del pequeño al grande
    pad_size = (size_large - size_small) // 2
    lpf2_padded = np.pad(lpf2, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')

    bpf_kernel = lpf2_padded - lpf1
    bpf_kernel = bpf_kernel / np.abs(bpf_kernel).sum()  # Normalización de energía
    
    return bpf_kernel

# ==============================================================================
# 3) Lógica de Filtrado (pipeline)
#    - Conversión RGB→YIQ
#    - Toma el canal Y (luminancia)
#    - Padding + Convolución
#    - Normalización del resultado a [0,255] para mostrar/guardar
# ==============================================================================

def apply_filter_logic(image_np, kernel):
    """
    Aplica el filtro al canal Y (luminancia) con padding de replicación.
    Devuelve una imagen en escala de grises (uint8) lista para mostrar.
    """
    yiq_image = rgb2yiq(image_np)          # Convertir a YIQ
    y_channel = yiq_image[:, :, 0]         # Tomar la luminancia

    # Padding de la luminancia para no perder bordes
    y_padded = _pad_image_replication(y_channel, kernel.shape)
    
    # Convolución sobre la imagen padded
    try:
        y_filtered = _convolution_padded(y_padded, kernel)
    except ValueError as e:
        messagebox.showerror("Error de Convolución", str(e))
        return None

    # Normalización min-max a [0,1] para visualizar homogéneo
    y_min = y_filtered.min()
    y_max = y_filtered.max()

    if y_max != y_min:
        y_normalized = (y_filtered - y_min) / (y_max - y_min)
    else:
        y_normalized = np.zeros_like(y_filtered)

    # Escalar a [0,255] y convertir a uint8 (imagen display)
    y_display = (y_normalized * 255).astype(np.uint8)
    return y_display

# ==============================================================================
# 4) Clase de la Interfaz Tkinter
#    - Crea la ventana principal
#    - Maneja carga de imagen, selección y aplicación de filtros, y guardado
#    - Muestra lado a lado imagen original y resultado
# ==============================================================================

class ImageProcessorApp:
    def __init__(self, master):
        # --- Ventana raíz y título ---
        self.master = master
        master.title("Filtrado por Convolución - PDI UNS")

        # --- Estado interno / buffers de imagen ---
        self.original_image_pil = None        # Imagen original (PIL)
        self.filtered_image_pil = None        # Imagen filtrada (PIL, escala de grises)
        self.original_image_np = None         # Imagen original como numpy
        self.original_image_tk = None         # Versión PhotoImage para Tkinter (orig)
        self.filtered_image_tk_filtered = None# Versión PhotoImage para Tkinter (filt)
        
        # --- Diccionario de filtros disponibles en la UI ---
        #     Puede guardar kernels fijos o lambdas que generen kernels
        self.fixed_filter_options = {
            "Laplaciano v4": KERNEL_LAPLACIANO_V4,
            "Laplaciano v8": KERNEL_LAPLACIANO_V8,
            "PasaAltos (corte 0.2)": lambda: create_highpass_kernel(size=5, cutoff_freq=0.2),
            "PasaAltos (corte 0.4)": lambda: create_highpass_kernel(size=5, cutoff_freq=0.4),
            "PasaBanda (DOG 5x5/3x3)": lambda: create_bandpass_kernel(size_small=3, size_large=5,
                                                                       cutoff_freq_small=0.4, cutoff_freq_large=0.2)
        }
        # Variable de Tk para saber qué filtro está seleccionado en el ComboBox
        self.selected_filter = tk.StringVar(master)
        self.selected_filter.set("Laplaciano v4") 

        # --- Layout principal (grid) ---
        main_frame = ttk.Frame(master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        
        # --- Frame que contendrá las 2 vistas de imagen ---
        img_frame = ttk.Frame(main_frame)
        img_frame.grid(row=0, column=0, columnspan=2, pady=10)
        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=1)
        
        # --- Etiquetas/áreas donde se mostrarán las imágenes ---
        self.label_original = ttk.Label(img_frame, text="[IMAGEN ORIGINAL]",
                                        anchor="center", relief="groove", width=50)
        self.label_original.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        self.label_filtered = ttk.Label(img_frame, text="[IMAGEN FILTRADA]",
                                        anchor="center", relief="groove", width=50)
        self.label_filtered.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        
        # --- Títulos debajo de cada imagen ---
        ttk.Label(img_frame, text="Imagen Original",
                  font=('Arial', 10, 'bold')).grid(row=1, column=0)
        ttk.Label(img_frame, text="Imagen Filtrada (Canal Y)",
                  font=('Arial', 10, 'bold')).grid(row=1, column=1)

        # --- Frame de controles (botones y combobox) ---
        control_frame = ttk.Frame(main_frame, padding="10 0 10 0")
        control_frame.grid(row=2, column=0, columnspan=2, pady=10)

        # --- Botón: Cargar imagen ---
        ttk.Button(control_frame, text="Cargar Imagen",
                   command=self.load_image).grid(row=0, column=0, padx=5)

        # --- ComboBox: Selección de filtro ---
        self.filter_combobox = ttk.Combobox(control_frame,
                                            textvariable=self.selected_filter,
                                            state="readonly",
                                            values=list(self.fixed_filter_options.keys()))
        self.filter_combobox.grid(row=0, column=1, padx=5)

        # --- Botón: Aplicar filtro ---
        ttk.Button(control_frame, text="Aplicar Filtro →",
                   command=self.process_image).grid(row=0, column=2, padx=5)
        
        # --- Botón: Guardar imagen filtrada (deshabilitado hasta tener resultado) ---
        self.save_button = ttk.Button(control_frame, text="Guardar Imagen",
                                      command=self.save_image, state=tk.DISABLED)
        self.save_button.grid(row=0, column=3, padx=5)
        
    # --- Métodos de la Aplicación (callbacks de botones y utilidades UI) ---

    def load_image(self):
        """
        Abre un diálogo para elegir archivo de imagen.
        - Carga con PIL, asegura modo RGB
        - Convierte a numpy para el procesamiento
        - Limpia resultado previo y actualiza la UI
        """
        filepath = filedialog.askopenfilename(
            title="Seleccionar Imagen RGB",
            filetypes=[("Archivos de Imagen", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if not filepath:
            return
        try:
            self.original_image_pil = Image.open(filepath).convert("RGB")
            self.original_image_np = np.array(self.original_image_pil)
            self.filtered_image_pil = None  # Reiniciar resultado
            self.update_image_display()
        except Exception as e:
            messagebox.showerror("Error de Carga", f"No se pudo cargar la imagen. Detalle: {e}")

    def update_image_display(self):
        """
        Actualiza las etiquetas que muestran la imagen original y la filtrada.
        - Hace thumbnails para no desbordar la interfaz
        - Habilita/deshabilita botón Guardar según haya resultado
        """
        max_size = 400  # Lado máximo para mostrar en UI
        
        # Mostrar imagen original (si existe)
        if self.original_image_pil:
            img_disp_orig = self.original_image_pil.copy()
            img_disp_orig.thumbnail((max_size, max_size))              # Escalar para UI
            self.original_image_tk = ImageTk.PhotoImage(img_disp_orig)  # Convertir a PhotoImage
            self.label_original.config(image=self.original_image_tk, text="")
        else:
            self.label_original.config(image="", text="[IMAGEN ORIGINAL]")
            
        # Mostrar imagen filtrada (si existe)
        if self.filtered_image_pil:
            img_disp_filt = self.filtered_image_pil.copy()
            img_disp_filt.thumbnail((max_size, max_size))
            self.filtered_image_tk_filtered = ImageTk.PhotoImage(img_disp_filt)
            self.label_filtered.config(image=self.filtered_image_tk_filtered, text="")
            self.save_button.config(state=tk.NORMAL)    # Ya se puede guardar
        else:
            self.label_filtered.config(image="", text="[IMAGEN FILTRADA]")
            self.save_button.config(state=tk.DISABLED)  # No hay nada para guardar

    def process_image(self):
        """
        Aplica el filtro seleccionado a la imagen cargada.
        - Valida que exista imagen
        - Obtiene el kernel (directo o generándolo si es callable)
        - Ejecuta el pipeline de filtrado
        - Actualiza la vista
        """
        if self.original_image_np is None:
            messagebox.showwarning("Advertencia", "Primero debes cargar una imagen.")
            return

        try:
            # Nombre y definición del filtro elegido en el ComboBox
            filter_name = self.selected_filter.get()
            filter_item = self.fixed_filter_options[filter_name]
            
            # Si filter_item es una función (lambda), la ejecutamos para obtener el kernel
            if callable(filter_item):
                kernel = filter_item()
            else:
                kernel = filter_item    
            
            # Procesar imagen → y_filtered_array es escala de grises (uint8)
            y_filtered_array = apply_filter_logic(self.original_image_np, kernel)
            
            if y_filtered_array is None: 
                return

            # Convertir a imagen PIL (modo 'L' = 8-bit grayscale) para poder mostrar/guardar
            self.filtered_image_pil = Image.fromarray(y_filtered_array, mode='L')
            
            # Refrescar UI con el nuevo resultado
            self.update_image_display()
            
            # NOTA: Se dejó sin messagebox de "Éxito" para no interrumpir el flujo.

        except Exception as e:
            messagebox.showerror("Error de Procesamiento", f"Error al aplicar el filtro. Detalle: {e}")

    def save_image(self):
        """
        Guarda en disco la imagen filtrada.
        - Abre diálogo 'Guardar como'
        - Determina el formato según la extensión
        - Maneja errores de escritura
        """
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

        # Deducir formato a partir de la extensión
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


# ==============================================================================
# 5) Punto de entrada
#    - Crea la ventana raíz de Tk
#    - Instancia la app y entra al loop de eventos
# ==============================================================================
if __name__ == '__main__':
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
