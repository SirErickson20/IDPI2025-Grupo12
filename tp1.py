import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import os

class ImageProcessor:
    def __init__(self):
        self.root = tk.Tk()
        self.imagen_original = None
        self.imagen_procesada = None
        self.imagen_tk_izq = None
        self.imagen_tk_der = None
        self.setup_ui()
        
    def setup_ui(self):
        # Configuración principal de la ventana
        self.root.title("Digital de Imágenes")
        self.root.geometry("1200x700")
        self.root.configure(bg="#0f1419")
        self.root.minsize(1000, 600)
        
        # Configurar el grid principal
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Crear header
        self.create_header()
        
        # Crear frames principales
        self.create_left_frame()
        self.create_center_frame()
        self.create_right_frame()
        
        # Crear barra de estado
        self.create_status_bar()
        
    def create_header(self):
        header_frame = tk.Frame(self.root, bg="#1a1f29", height=80)
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 0))
        header_frame.grid_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🖼️ PROCESADOR DIGITAL DE IMÁGENES 2025",
            font=("Segoe UI", 20, "bold"),
            bg="#1a1f29",
            fg="#64ffda"
        )
        title_label.pack(pady=25)
        
    def create_left_frame(self):
        # Frame principal izquierdo
        self.frame_izq = tk.Frame(self.root, bg="#1a1f29", relief="solid", bd=1)
        self.frame_izq.grid(row=1, column=0, padx=(10, 5), pady=10, sticky="nsew")
        
        # Header del frame izquierdo
        header_izq = tk.Frame(self.frame_izq, bg="#252b3a", height=50)
        header_izq.pack(fill="x", padx=2, pady=2)
        header_izq.pack_propagate(False)
        
        tk.Label(
            header_izq,
            text="📥 IMAGEN ORIGINAL",
            font=("Segoe UI", 12, "bold"),
            bg="#252b3a",
            fg="#ffffff"
        ).pack(pady=15)
        
        # Área de imagen
        self.image_area_izq = tk.Frame(self.frame_izq, bg="#0f1419")
        self.image_area_izq.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.label_izquierda = tk.Label(
            self.image_area_izq,
            text="🖼️\n\nArrastre una imagen aquí\no haga clic en 'Cargar Imagen'",
            font=("Segoe UI", 11),
            bg="#0f1419",
            fg="#6b7280",
            justify="center"
        )
        self.label_izquierda.pack(expand=True)
        
        # Botones izquierda
        btn_frame_izq = tk.Frame(self.frame_izq, bg="#1a1f29")
        btn_frame_izq.pack(fill="x", padx=10, pady=(0, 15))
        
        self.btn_cargar = self.create_button(
            btn_frame_izq, "📁 Cargar Imagen", self.cargar_imagen, "#3b82f6"
        )
        self.btn_cargar.pack(fill="x", pady=(0, 8))
        
        self.btn_rgb = self.create_button(
            btn_frame_izq, "📊 Análisis RGB", self.mostrar_curva_rgb, "#8b5cf6", state="disabled"
        )
        self.btn_rgb.pack(fill="x")
        
    def create_center_frame(self):
        # Frame central con controles
        self.frame_centro = tk.Frame(self.root, bg="#1a1f29", width=250)
        self.frame_centro.grid(row=1, column=1, padx=5, pady=10, sticky="ns")
        self.frame_centro.grid_propagate(False)
        
        # Título del panel de control
        tk.Label(
            self.frame_centro,
            text="🎛️ CONTROLES",
            font=("Segoe UI", 12, "bold"),
            bg="#1a1f29",
            fg="#64ffda"
        ).pack(pady=(10, 20))
        
        # Botón de transferencia
        self.btn_transferir = self.create_button(
            self.frame_centro, "➡️ Transferir", self.transferir_imagen, "#10b981", state="disabled"
        )
        self.btn_transferir.pack(fill="x", padx=10, pady=(0, 30))
        
        # Separador
        separator = tk.Frame(self.frame_centro, bg="#374151", height=2)
        separator.pack(fill="x", padx=20, pady=(0, 20))
        
        # Instrucciones para análisis RGB
        tk.Label(
            self.frame_centro,
            text="📏 ANÁLISIS RGB",
            font=("Segoe UI", 11, "bold"),
            bg="#1a1f29",
            fg="#ffffff"
        ).pack(pady=(0, 15))
        
        instruccion_frame = tk.Frame(self.frame_centro, bg="#252b3a", relief="solid", bd=1)
        instruccion_frame.pack(fill="x", padx=10, pady=(0, 20))
        
        tk.Label(
            instruccion_frame,
            text="📝 Instrucciones:",
            font=("Segoe UI", 10, "bold"),
            bg="#252b3a",
            fg="#64ffda"
        ).pack(pady=(10, 5))
        
        tk.Label(
            instruccion_frame,
            text="1. Cargue una imagen\n2. Haga clic en 'Análisis RGB'\n3. Trace una línea horizontal\n4. Vea los valores RGB",
            font=("Segoe UI", 9),
            bg="#252b3a",
            fg="#d1d5db",
            justify="left"
        ).pack(pady=(0, 15), padx=15)
        
    def create_right_frame(self):
        # Frame principal derecho
        self.frame_der = tk.Frame(self.root, bg="#1a1f29", relief="solid", bd=1)
        self.frame_der.grid(row=1, column=2, padx=(5, 10), pady=10, sticky="nsew")
        
        # Header del frame derecho
        header_der = tk.Frame(self.frame_der, bg="#252b3a", height=50)
        header_der.pack(fill="x", padx=2, pady=2)
        header_der.pack_propagate(False)
        
        tk.Label(
            header_der,
            text="📤 IMAGEN PROCESADA",
            font=("Segoe UI", 12, "bold"),
            bg="#252b3a",
            fg="#ffffff"
        ).pack(pady=15)
        
        # Área de imagen
        self.image_area_der = tk.Frame(self.frame_der, bg="#0f1419")
        self.image_area_der.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.label_derecha = tk.Label(
            self.image_area_der,
            text="⏳\n\nEsperando imagen procesada...",
            font=("Segoe UI", 11),
            bg="#0f1419",
            fg="#6b7280",
            justify="center"
        )
        self.label_derecha.pack(expand=True)
        
        # Botón guardar
        btn_frame_der = tk.Frame(self.frame_der, bg="#1a1f29")
        btn_frame_der.pack(fill="x", padx=10, pady=(0, 15))
        
        self.btn_guardar = self.create_button(
            btn_frame_der, "💾 Guardar Imagen", self.guardar_imagen, "#059669", state="disabled"
        )
        self.btn_guardar.pack(fill="x")
        
    def create_status_bar(self):
        self.status_bar = tk.Frame(self.root, bg="#374151", height=30)
        self.status_bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
        self.status_bar.grid_propagate(False)
        
        self.status_label = tk.Label(
            self.status_bar,
            text="✅ Listo para procesar imágenes",
            font=("Segoe UI", 9),
            bg="#374151",
            fg="#d1d5db"
        )
        self.status_label.pack(side="left", padx=10, pady=5)
        
    def create_button(self, parent, text, command, color, state="normal"):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            state=state,
            activebackground=self.lighten_color(color),
            bd=0,
            padx=15,
            pady=8
        )
        
        # Efectos hover
        def on_enter(e):
            if btn['state'] == 'normal':
                btn.config(bg=self.lighten_color(color))
                
        def on_leave(e):
            if btn['state'] == 'normal':
                btn.config(bg=color)
                
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn
        
    def lighten_color(self, color):
        """Hace un color más claro para el efecto hover"""
        color_map = {
            "#3b82f6": "#60a5fa",
            "#8b5cf6": "#a78bfa",
            "#10b981": "#34d399",
            "#6366f1": "#818cf8",
            "#059669": "#10b981"
        }
        return color_map.get(color, color)
        
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update()
        
    def cargar_imagen(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.gif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.update_status("📂 Cargando imagen...")
                self.imagen_original = Image.open(file_path).convert("RGB")
                self.imagen_procesada = self.imagen_original.copy()
                
                # Mostrar imagen en el lado izquierdo
                self.mostrar_imagen_izquierda()
                
                # Habilitar controles
                self.btn_rgb.config(state="normal")
                self.btn_transferir.config(state="normal")
                
                filename = os.path.basename(file_path)
                self.update_status(f"✅ Imagen cargada: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{str(e)}")
                self.update_status("❌ Error al cargar la imagen")
                
    def mostrar_imagen_izquierda(self):
        if self.imagen_original:
            # Calcular tamaño manteniendo proporción
            img_copy = self.imagen_original.copy()
            img_copy.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            self.imagen_tk_izq = ImageTk.PhotoImage(img_copy)
            self.label_izquierda.config(image=self.imagen_tk_izq, text="")
            self.label_izquierda.image = self.imagen_tk_izq
            
    def mostrar_imagen_derecha(self):
        if self.imagen_procesada:
            # Calcular tamaño manteniendo proporción
            img_copy = self.imagen_procesada.copy()
            img_copy.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            self.imagen_tk_der = ImageTk.PhotoImage(img_copy)
            self.label_derecha.config(image=self.imagen_tk_der, text="")
            self.label_derecha.image = self.imagen_tk_der
            
            self.btn_guardar.config(state="normal")
            
    def transferir_imagen(self):
        if self.imagen_procesada:
            self.mostrar_imagen_derecha()
            self.update_status("➡️ Imagen transferida al panel derecho")
            
    def habilitar_controles(self, habilitar):
        # Esta función ya no es necesaria ya que removimos los filtros
        pass
        
    # Funciones de procesamiento removidas - solo mantenemos las necesarias para el análisis RGB
    def transferir_imagen(self):
        if self.imagen_original:
            self.imagen_procesada = self.imagen_original.copy()
            self.mostrar_imagen_derecha()
            self.update_status("➡️ Imagen transferida al panel derecho")
            
    def mostrar_curva_rgb(self):
        if self.imagen_original:
            arr = np.array(self.imagen_original)
            
            plt.style.use('dark_background')
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.fig.patch.set_facecolor('#0f1419')
            self.ax.set_facecolor('#1a1f29')
            
            # Mostrar la imagen
            self.ax.imshow(arr)
            self.ax.set_title("📐 Trace una línea horizontal para análisis RGB", fontsize=14, color='#64ffda', pad=20)
            self.ax.set_xlabel("Haga clic y arrastre para trazar una línea horizontal", fontsize=12, color='white')
            
            # Variables para la línea
            self.start_point = None
            self.end_point = None
            self.line = None
            self.is_drawing = False
            
            # Conectar eventos del mouse
            self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
            self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            
            plt.tight_layout()
            plt.show()
            
            self.update_status("📐 Trace una línea horizontal en la imagen para ver los valores RGB")
    
    def on_mouse_press(self, event):
        if event.inaxes == self.ax:
            self.start_point = (int(event.xdata), int(event.ydata))
            self.is_drawing = True
            
    def on_mouse_move(self, event):
        if self.is_drawing and event.inaxes == self.ax and self.start_point:
            # Limpiar línea anterior si existe
            if self.line:
                self.line.remove()
            
            # Dibujar nueva línea (forzar horizontal)
            y_fixed = self.start_point[1]  # Fijar Y al punto inicial
            self.end_point = (int(event.xdata), y_fixed)
            
            self.line, = self.ax.plot([self.start_point[0], self.end_point[0]], 
                                    [y_fixed, y_fixed], 
                                    color='#64ffda', linewidth=3, alpha=0.8)
            self.fig.canvas.draw()
    
    def on_mouse_release(self, event):
        if self.is_drawing and self.start_point and event.inaxes == self.ax:
            self.is_drawing = False
            
            # Asegurar línea horizontal
            y_fixed = self.start_point[1]
            self.end_point = (int(event.xdata), y_fixed)
            
            # Analizar la línea RGB
            self.analizar_linea_rgb()
    
    def analizar_linea_rgb(self):
        if not (self.start_point and self.end_point and self.imagen_original):
            return
            
        arr = np.array(self.imagen_original)
        
        # Obtener coordenadas de la línea
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # Asegurar que estén dentro de los límites
        h, w = arr.shape[:2]
        x1 = max(0, min(w-1, x1))
        x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1))
        
        # Crear línea horizontal
        if x1 > x2:
            x1, x2 = x2, x1
            
        # Extraer valores RGB a lo largo de la línea
        x_coords = range(x1, x2 + 1)
        rgb_values = []
        
        for x in x_coords:
            r, g, b = arr[y1, x]
            rgb_values.append((r, g, b))
        
        # Crear nueva ventana para mostrar los resultados
        self.mostrar_resultados_rgb(x_coords, rgb_values, y1)
    
    def mostrar_resultados_rgb(self, x_coords, rgb_values, y_line):
        # Crear nueva figura para los resultados
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.patch.set_facecolor('#0f1419')
        
        # Panel superior: Gráfico de valores RGB
        ax1.set_facecolor('#1a1f29')
        
        # Separar valores RGB
        r_values = [rgb[0] for rgb in rgb_values]
        g_values = [rgb[1] for rgb in rgb_values]
        b_values = [rgb[2] for rgb in rgb_values]
        
        # Plotear cada canal
        ax1.plot(x_coords, r_values, color='#ef4444', linewidth=2, label='Canal Rojo', marker='o', markersize=3)
        ax1.plot(x_coords, g_values, color='#22c55e', linewidth=2, label='Canal Verde', marker='s', markersize=3)
        ax1.plot(x_coords, b_values, color='#3b82f6', linewidth=2, label='Canal Azul', marker='^', markersize=3)
        
        ax1.set_title(f"📊 Valores RGB a lo largo de la línea horizontal (Y={y_line})", 
                     fontsize=14, color='#64ffda', pad=15)
        ax1.set_xlabel("Posición X (píxeles)", fontsize=12, color='white')
        ax1.set_ylabel("Valor de intensidad (0-255)", fontsize=12, color='white')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, color='#374151')
        ax1.tick_params(colors='white')
        ax1.set_ylim(0, 255)
        
        # Panel inferior: Tabla de valores
        ax2.set_facecolor('#1a1f29')
        ax2.axis('off')
        
        # Crear tabla con algunos valores de muestra (cada 10 píxeles o máximo 20 valores)
        step = max(1, len(x_coords) // 20)
        sample_indices = range(0, len(x_coords), step)
        
        table_data = []
        headers = ['Posición X', 'Rojo', 'Verde', 'Azul', 'Color RGB']
        
        for i in sample_indices:
            x = x_coords[i]
            r, g, b = rgb_values[i]
            table_data.append([f'{x}', f'{r}', f'{g}', f'{b}', f'({r},{g},{b})'])
        
        # Crear tabla
        table = ax2.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center',
                         bbox=[0.1, 0.1, 0.8, 0.8])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Estilo de la tabla
        for i, key in enumerate(table.get_celld().keys()):
            cell = table.get_celld()[key]
            if key[0] == 0:  # Header
                cell.set_facecolor('#64ffda')
                cell.set_text_props(weight='bold', color='black')
            else:
                cell.set_facecolor('#374151')
                cell.set_text_props(color='white')
            cell.set_edgecolor('#6b7280')
        
        ax2.set_title("📋 Muestra de valores RGB extraídos", fontsize=12, color='#64ffda', pad=20)
        
        plt.tight_layout()
        plt.show()
        
        # Actualizar status con estadísticas
        r_avg = np.mean(r_values)
        g_avg = np.mean(g_values)  
        b_avg = np.mean(b_values)
        
        self.update_status(f"📊 Análisis completado - Promedios: R:{r_avg:.1f} G:{g_avg:.1f} B:{b_avg:.1f}")
            
    def guardar_imagen(self):
        if self.imagen_procesada:
            formatos = [
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tif"),
                ("Todos los archivos", "*.*")
            ]
            
            file_path = filedialog.asksaveasfilename(
                title="Guardar imagen procesada",
                defaultextension=".png",
                filetypes=formatos
            )
            
            if file_path:
                try:
                    self.imagen_procesada.save(file_path, quality=95)
                    filename = os.path.basename(file_path)
                    messagebox.showinfo("✅ Guardado exitoso", f"Imagen guardada como:\n{filename}")
                    self.update_status(f"💾 Imagen guardada: {filename}")
                    
                except Exception as e:
                    messagebox.showerror("❌ Error", f"No se pudo guardar la imagen:\n{str(e)}")
                    self.update_status("❌ Error al guardar la imagen")
                    
    def run(self):
        self.root.mainloop()

# Ejecutar la aplicación
if __name__ == "__main__":
    app = ImageProcessor()
    app.run()