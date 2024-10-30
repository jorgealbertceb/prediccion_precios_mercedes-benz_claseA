# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requerimientos y el resto del código a la imagen
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY final_mercedes_benz_clase_a.csv final_mercedes_benz_clase_a.csv
COPY modelo_gradient_boosting.pkl modelo_gradient_boosting.pkl

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que Gradio correrá
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]