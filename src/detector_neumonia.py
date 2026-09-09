#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk, font, filedialog, Entry

from tkinter.messagebox import askokcancel, showinfo, WARNING
import getpass
from PIL import ImageTk, Image, ImageGrab
import csv
import pyautogui
# import tkcap
import img2pdf
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras import backend as K
import pydicom as dicom


#tf.compat.v1.disable_eager_execution()
#tf.compat.v1.experimental.output_all_intermediates(True)
import cv2


def model_fun():
    # Carga el modelo desde el archivo en la raíz del proyecto
    model_cnn = tf.keras.models.load_model('conv_MLP_84.h5', compile=False)
    return model_cnn


def grad_cam(array):
    img = preprocess(array)
    model = model_fun()
    
    # 1. Creamos un sub-modelo que nos devuelva tanto la salida de la última capa convolucional
    # como la predicción final de la red.
    last_conv_layer = model.get_layer('conv10_thisone')
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # 2. Con tf.GradientTape grabamos las operaciones para calcular el gradiente
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        if isinstance(predictions, list):
            predictions = predictions[0]
            
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 3. Gradiente de la clase predicha con respecto al mapa de características convolucional
    grads = tape.gradient(class_channel, conv_outputs)

    # 4. Promedio espacial de los gradientes (pooled gradients)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Ponderamos los canales de la capa convolucional
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # 6. Generación del mapa de calor (Heatmap)
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    max_val = np.max(heatmap)
    if max_val != 0:
        heatmap /= max_val
        
    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 7. Superposición sobre la imagen original
    img2 = cv2.resize(array, (512, 512))
    if len(img2.shape) == 2:  # Si la imagen original está en escala de grises
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    elif img2.shape[2] == 1:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    hif = 0.8
    transparency = (heatmap * hif).astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2).astype(np.uint8)
    
    return superimposed_img[:, :, ::-1]




def predict(array):
    #   1. call function to pre-process image: it returns image in batch format
    batch_array_img = preprocess(array)
    #   2. call function to load model and predict: it returns predicted class and probability
    model = model_fun()
    # model_cnn = tf.keras.models.load_model('conv_MLP_84.h5')
    prediction = np.argmax(model.predict(batch_array_img))
    proba = np.max(model.predict(batch_array_img)) * 100
    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"
    #   3. call function to generate Grad-CAM: it returns an image with a superimposed heatmap
    heatmap = grad_cam(array)
    return (label, proba, heatmap)


def read_dicom_file(path):
    img = dicom.dcmread(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show


def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show


def preprocess(array):
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        #   BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        #   LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        #   TWO STRING VARIABLES TO CONTAIN ID AND RESULT
        self.ID = StringVar()
        self.result = StringVar()

        #   TWO INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        #   GET ID
        self.ID_content = self.text1.get()

        #   TWO IMAGE INPUT BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        #   BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        #   WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        #   FOCUS ON PATIENT ID
        self.text1.focus_set()

        #  se reconoce como un elemento de la clase
        self.array = None

        #   NUMERO DE IDENTIFICACIÓN PARA GENERAR PDF
        self.reportID = 0

        #   RUN LOOP
        self.root.mainloop()

    #   METHODS
    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if filepath:
            self.array, img2show = read_dicom_file(filepath)
            self.img1 = img2show.resize((250, 250), Image.LANCZOS)
            self.img1 = ImageTk.PhotoImage(self.img1)
            self.text_img1.image_create(END, image=self.img1)
            self.button1["state"] = "enabled"

    def run_model(self):
        self.label, self.proba, self.heatmap = predict(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        print("OK")
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    def save_results_csv(self):
        with open("historial.csv", "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        from PIL import ImageDraw
        import datetime

        # 1. Crear un lienzo blanco para el reporte tamaño 1024x768
        w, h = 1024, 768
        report = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(report)

        # 2. Encabezado del reporte
        draw.rectangle([(0, 0), (w, 80)], fill=(30, 42, 56))
        draw.text((30, 25), "REPORTE MEDICO - DETECCION DE NEUMONIA (IA)", fill=(255, 255, 255))

        # 3. Información del paciente y resultados
        fecha_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_paciente = f"ID Paciente: {self.text1.get()}    |    Fecha: {fecha_str}"
        resultado_ia = f"Diagnostico: {self.label.upper()}    |    Confianza: {self.proba:.2f}%"
        
        draw.text((30, 100), info_paciente, fill=(20, 20, 20))
        draw.text((30, 125), resultado_ia, fill=(20, 20, 20))

        # 4. Pegar la radiografía original (si existe)
        if hasattr(self, "array") and self.array is not None:
            # Asegurar formato RGB para Pillow
            img_orig = self.array.copy()
            if len(img_orig.shape) == 2:
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2RGB)
            elif img_orig.shape[2] == 1:
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2RGB)
            pil_orig = Image.fromarray(img_orig).resize((420, 420))
            report.paste(pil_orig, (50, 180))
            draw.text((50, 615), "Radiografia Original", fill=(50, 50, 50))

        # 5. Pegar el mapa de calor Grad-CAM (si existe)
        if hasattr(self, "heatmap") and self.heatmap is not None:
            pil_heat = Image.fromarray(self.heatmap).resize((420, 420))
            report.paste(pil_heat, (540, 180))
            draw.text((540, 615), "Activacion Grad-CAM (Explicabilidad)", fill=(50, 50, 50))

        # 6. Guardar como PDF
        pdf_path = f"Reporte{self.reportID}.pdf"
        report.save(pdf_path)
        self.reportID += 1
        showinfo(title="PDF", message=f"El PDF fue generado con éxito: {pdf_path}")


    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(self.img1, "end")
            self.text_img2.delete(self.img2, "end")
            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main():
    my_app = App()
    return 0


if __name__ == "__main__":
    main()
