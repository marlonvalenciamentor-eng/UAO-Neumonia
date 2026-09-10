#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
INTERFAZ GRÁFICA DE USUARIO (GUI) - DETECTOR DE NEUMONÍA
ARQUITECTURA: Presentación Desacoplada (Clean Architecture)
RESPONSABILIDAD ÚNICA:
    Interacción con el usuario, visualización de radiografías, despliegue de
    resultados clínicos y exportación de reportes (PDF y CSV).
    
    Delega toda la lógica de lectura a 'src.read_img' y la inferencia
    con explicabilidad Grad-CAM a 'src.integrator'.
=============================================================================
"""

import csv
import cv2
import datetime
from tkinter import Tk, StringVar, Text, END
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING
from PIL import ImageTk, Image, ImageDraw, ImageFont

# Módulos desacoplados del sistema
try:
    from src.read_img import read_file
    from src.integrator import predict
except ModuleNotFoundError:
    from read_img import read_file
    from integrator import predict



class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Sistema de Apoyo al Diagnóstico de Neumonía (IA)")
        self.root.geometry("815x560")
        self.root.resizable(False, False)

        # Fuentes
        font_bold = font.Font(weight="bold")

        # Etiquetas del encabezado y campos
        self.lab_title = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=font_bold,
        )
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=font_bold)
        self.lab2 = ttk.Label(self.root, text="Mapa de Activación (Grad-CAM)", font=font_bold)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=font_bold)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=font_bold)
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=font_bold)

        # Variables de estado
        self.patient_id_var = StringVar()
        self.label = ""
        self.proba = 0.0
        self.array = None
        self.heatmap = None
        self.report_id = 0

        # Cajas de entrada y cuadros de visualización
        self.entry_patient_id = ttk.Entry(self.root, textvariable=self.patient_id_var, width=12)
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text_result = Text(self.root)
        self.text_proba = Text(self.root)

        # Botones de acción
        self.btn_load = ttk.Button(self.root, text="Cargar Imagen", command=self.load_image)
        self.btn_predict = ttk.Button(self.root, text="Predecir", state="disabled", command=self.run_prediction)
        self.btn_save = ttk.Button(self.root, text="Guardar (CSV)", command=self.save_csv)
        self.btn_pdf = ttk.Button(self.root, text="Generar PDF", command=self.create_pdf)
        self.btn_delete = ttk.Button(self.root, text="Borrar", command=self.reset_form)

        # Ubicación geométrica de los widgets
        self.lab_title.place(x=120, y=25)
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        self.lab4.place(x=65, y=350)
        self.entry_patient_id.place(x=200, y=350)

        self.lab3.place(x=500, y=350)
        self.text_result.place(x=610, y=350, width=110, height=30)
        self.lab6.place(x=500, y=400)
        self.text_proba.place(x=610, y=400, width=110, height=30)

        self.btn_load.place(x=70, y=460)
        self.btn_predict.place(x=220, y=460)
        self.btn_save.place(x=370, y=460)
        self.btn_pdf.place(x=520, y=460)
        self.btn_delete.place(x=670, y=460)

        self.entry_patient_id.focus_set()
        self.root.mainloop()

    # ==================== MÉTODOS DE LA APLICACIÓN ====================

    def load_image(self):
        """Carga una imagen radiográfica (.dcm, .jpg, .png) y la proyecta."""
        filepath = filedialog.askopenfilename(
            title="Seleccionar Radiografía",
            filetypes=(
                ("Archivos Médicos", "*.dcm *.jpeg *.jpg *.png"),
                ("DICOM (*.dcm)", "*.dcm"),
                ("JPEG/JPG (*.jpg;*.jpeg)", "*.jpg *.jpeg"),
                ("PNG (*.png)", "*.png"),
            ),
        )
        if filepath:
            # Delegar la lectura al Módulo 1 (read_img)
            self.array, img2show = read_file(filepath)
            
            # Redimensionar para mostrar en pantalla
            pil_thumb = img2show.resize((250, 250), Image.LANCZOS)
            self.img1_tk = ImageTk.PhotoImage(pil_thumb)
            self.text_img1.delete("1.0", END)
            self.text_img1.image_create(END, image=self.img1_tk)
            self.btn_predict["state"] = "normal"

    def run_prediction(self):
        """Ejecuta la inferencia delegando la orquestación a 'integrator.py'."""
        if self.array is None:
            return

        # Llamada de bajo acoplamiento al integrador
        self.label, self.proba, self.heatmap = predict(self.array)

        # Mostrar mapa de calor en la interfaz
        heat_pil = Image.fromarray(self.heatmap).resize((250, 250), Image.LANCZOS)
        self.img2_tk = ImageTk.PhotoImage(heat_pil)
        self.text_img2.delete("1.0", END)
        self.text_img2.image_create(END, image=self.img2_tk)

        # Mostrar valores diagnósticos
        self.text_result.delete("1.0", END)
        self.text_result.insert(END, self.label.upper())

        self.text_proba.delete("1.0", END)
        self.text_proba.insert(END, f"{self.proba:.2f}%")

    def save_csv(self):
        """Almacena el historial diagnóstico en un archivo CSV."""
        if not self.label:
            return
        with open("historial.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([self.entry_patient_id.get(), fecha, self.label, f"{self.proba:.2f}%"])
        showinfo(title="Guardar", message="Los datos del paciente se guardaron con éxito en historial.csv.")

    def create_pdf(self):
        """Genera un reporte médico clínico en formato PDF de forma nativa e independiente del SO."""
        w, h = 1024, 768
        report = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(report)

        # Cargar fuentes del sistema o usar la predeterminada si no están disponibles
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_info = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
            font_labels = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font_title = font_info = font_labels = ImageFont.load_default()

        # Encabezado institucional
        draw.rectangle([(0, 0), (w, 80)], fill=(30, 42, 56))
        draw.text((30, 26), "REPORTE MEDICO - DETECCION DE NEUMONIA ASISTIDA POR IA", fill=(255, 255, 255), font=font_title)

        # Información del paciente
        fecha_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        patient_id = self.entry_patient_id.get().strip() or "S_N"
        draw.text((30, 100), f"ID Paciente: {patient_id}    |    Fecha: {fecha_str}", fill=(20, 20, 20), font=font_info)
        draw.text((30, 130), f"Diagnostico: {self.label.upper()}    |    Certeza Medica: {self.proba:.2f}%", fill=(20, 20, 20), font=font_info)

        # Pegar radiografía original
        if self.array is not None:
            img_orig = self.array.copy()
            if len(img_orig.shape) == 2:
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2RGB)
            elif img_orig.shape[2] == 1:
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2RGB)
            pil_orig = Image.fromarray(img_orig).resize((420, 420))
            report.paste(pil_orig, (50, 180))
            draw.text((50, 615), "Radiografia Original", fill=(50, 50, 50), font=font_labels)

        # Pegar mapa de calor
        if self.heatmap is not None:
            pil_heat = Image.fromarray(self.heatmap).resize((420, 420))
            report.paste(pil_heat, (540, 180))
            draw.text((540, 615), "Explicabilidad Grad-CAM (Region Pulmonar)", fill=(50, 50, 50), font=font_labels)

        pdf_path = f"Reporte_{self.report_id}.pdf"
        report.save(pdf_path)
        self.report_id += 1
        showinfo(title="PDF Generado", message=f"Reporte clínico generado con éxito:\n{pdf_path}")

    def reset_form(self):
        """Limpia todos los campos para un nuevo análisis."""
        if askokcancel(title="Confirmar", message="¿Desea limpiar todos los campos del formulario?"):
            self.entry_patient_id.delete(0, END)
            self.text_result.delete("1.0", END)
            self.text_proba.delete("1.0", END)
            self.text_img1.delete("1.0", END)
            self.text_img2.delete("1.0", END)
            self.array = None
            self.heatmap = None
            self.label = ""
            self.proba = 0.0
            self.btn_predict["state"] = "disabled"


def main():
    App()
    return 0


if __name__ == "__main__":
    main()
