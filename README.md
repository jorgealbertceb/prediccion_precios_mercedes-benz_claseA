# Predicción de Precios de Coches Mercedes-Benz Clase A

Este proyecto utiliza un modelo de Machine Learning para predecir los precios de coches, especialmente del Mercedes-Benz Clase A. La aplicación está configurada para ejecutarse dentro de un contenedor Docker, lo que facilita su despliegue en cualquier entorno compatible con Docker.
El modelo de Machine Learning que se ha utilizado, tras realizar pruebas con otros (Pycaret: modelo ML automático o Decission Tree, por ejemplo), ha sido Gradient Boosting Regression. Esto se debe a que generalmente los resultados de predicción son muy buenos y prácticamente exactos, excepto en algunos (muy pocos) que tiene un pequeño error. 

En el archivo pf_ml.ipynb del notebook, está todo explicado y detallado paso a paso, desde el web-scraping en la web de Autocasión para la obtención de los datos, hasta el despliegue en Gradio.

Los datos que conforman nuestro CSV, ya limpio y listo para utilizar, son los siguientes:
- **Marca y modelo**: Contiene la marca y modelo del vehículo.
- **Potencia**: Contiene la potencia del motor del vehículo.
- **Tipo de vehículo**: Sedán o compacto y Null si no tiene ese valor.
- **Año**: Fecha de fabricación del vehículo en venta.
- **Km**: Kilometraje del vehículo.
- **Tipo de gasolina**: Gasolina, diésel, Híbrido, Híbrido Enchufable o Eléctrico, según cada coche.
- **Ubicación**: Lugar desde donde se vende ese vehículo.
- **Precio**: Precio de venta del vehículo.


## Contenido del Proyecto

- **notebooks**: Directorio que contiene todas las jupyter notebooks (archivos .ipynb) utilizadas para realizar pruebas.
- **data**: Directorio que contiene todos los archivos que forman los datos para realizar el ML (csv).
- **README.md**: Archivo que te encuentras leyendo actualmente.
- **app**: Directorio que contiene:
    - *models*: Directorio que contiene los modelos entrenados.
    - *app.py*: Código de la app de gradio.
    - *Dockerfile*: Archivo Dockerfile para ejecutar la app.
- **requirements.txt**: Archivo que especifica las dependencias necesarias para el proyecto.


## Requisitos

Para ejecutar este proyecto, necesitas:

- Docker (Si no lo tienes puedes descargarlo en este enlace https://docs.docker.com/get-docker/)


### Configuración de la Imagen Docker

El `Dockerfile` está configurado para:

1. Usar una imagen base oficial de Python.
2. Instalar las dependencias especificadas en `requirements.txt`.
3. Copiar el código fuente en el contenedor.
4. Ejecutar el servidor de la API de predicción al iniciar el contenedor.

### Iniciar app

1. **Construir la Imagen Docker**: `docker build -t mercedes-benz-clase-a .`
2. **Ejecutar el Contenedor**: `docker run -d -p 7680:7680 mercedes-benz-clase-a`
3. **Abrir la app en el siguiente enlace**: http://localhost:7680
4. **Hacer una predicción**: Introduce los datos de tu Mercedes-Benz Clase A y... ¡Disfruta de tu predicción!
