import gradio as gr
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

model = joblib.load("modelo_gradient_boosting.pkl")
df = pd.read_csv('final_mercedes_benz_clase_a.csv')

df['Tipo de vehículo'].fillna('Desconocido', inplace=True)

potencia_options = sorted(df['Potencia'].unique())
tipo_vehiculo_options = sorted(df['Tipo de vehículo'].unique())
tipo_gasolina_options = sorted(df['Tipo Gasolina'].unique())
ubicacion_options = sorted(df['Ubicación'].unique())
año_options = sorted(df['Año'].unique())

# Definir las columnas de entrada esperadas por el modelo
expected_columns = [
    'Año', 'Km', 'Potencia_100', 'Potencia_102', 'Potencia_105', 'Potencia_108',
    'Potencia_109', 'Potencia_110', 'Potencia_115', 'Potencia_116', 'Potencia_120',
    'Potencia_122', 'Potencia_135', 'Potencia_136', 'Potencia_140', 'Potencia_150',
    'Potencia_160', 'Potencia_163', 'Potencia_165', 'Potencia_170', 'Potencia_175',
    'Potencia_180', 'Potencia_190', 'Potencia_200', 'Potencia_218', 'Potencia_220',
    'Potencia_250', 'Potencia_305', 'Potencia_360', 'Potencia_380', 'Potencia_420',
    'Potencia_421', 'Tipo de vehículo_Compacto', 'Tipo de vehículo_Desconocido',
    'Tipo de vehículo_Sedán', 'Tipo Gasolina_Diésel', 'Tipo Gasolina_Gasolina',
    'Tipo Gasolina_Híbrido', 'Tipo Gasolina_Híbrido Enchufable', 'Ubicación_Albacete',
    'Ubicación_Alicante', 'Ubicación_Almería', 'Ubicación_Asturias', 'Ubicación_Badajoz',
    'Ubicación_Barcelona', 'Ubicación_Burgos', 'Ubicación_Cantabria', 'Ubicación_Castellón',
    'Ubicación_Ciudad Real', 'Ubicación_Cáceres', 'Ubicación_Cádiz', 'Ubicación_Córdoba',
    'Ubicación_Girona', 'Ubicación_Granada', 'Ubicación_Guipúzcoa', 'Ubicación_Huelva',
    'Ubicación_Huesca', 'Ubicación_Islas Baleares', 'Ubicación_Jaén', 'Ubicación_La Coruña',
    'Ubicación_La Rioja', 'Ubicación_León', 'Ubicación_Lleida', 'Ubicación_Lugo',
    'Ubicación_Madrid', 'Ubicación_Murcia', 'Ubicación_Málaga', 'Ubicación_Navarra',
    'Ubicación_Orense', 'Ubicación_Pontevedra', 'Ubicación_Salamanca', 'Ubicación_Sevilla',
    'Ubicación_Soria', 'Ubicación_Tarragona', 'Ubicación_Toledo', 'Ubicación_Valencia',
    'Ubicación_Valladolid', 'Ubicación_Vizcaya', 'Ubicación_Zamora', 'Ubicación_Zaragoza'
]

def predecir_precio(potencia, tipo_vehiculo, año, km, tipo_gasolina, ubicacion):

    model = joblib.load("modelo_gradient_boosting.pkl")
    
    # Crear un DataFrame con los datos ingresados
    entrada = pd.DataFrame({
        'Año': [año],
        'Km': [km],
        'Potencia': [potencia],
        'Tipo de vehículo': [tipo_vehiculo],
        'Tipo Gasolina': [tipo_gasolina],
        'Ubicación': [ubicacion]
    })

    # Crear variables dummy solo de las columnas específicas
    entrada = pd.get_dummies(entrada, columns=["Potencia", "Tipo de vehículo", "Tipo Gasolina", "Ubicación"])

    # Asegurar que todas las columnas coincidan con el modelo (agregar columnas faltantes)
    for col in expected_columns:
        if col not in entrada.columns:
            entrada[col] = 0
    
    # Reordenar columnas según el orden esperado por el modelo
    entrada = entrada[expected_columns]

    # Realizar la predicción
    precio_predicho = model.predict(entrada)[0]

    # Generar datos para el gráfico de kilometraje
    kilometros = [5, 1000, 20000, 50000, 100000, 150000, 200000, 500000]
    precios = []
    
    for k in kilometros:
        # Simular entrada para calcular precios con diferentes kilometrajes
        sim_entrada = entrada.copy()
        sim_entrada['Km'] = k  # Cambiar el kilometraje
        precio = model.predict(sim_entrada)[0]
        precios.append(precio)

    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(kilometros, precios, marker='o', label='Precio estimado por kilometraje')
    plt.axhline(y=precio_predicho, color='red', linestyle='--', label='Tu coche: €{:.2f}'.format(precio_predicho))
    plt.title('Comparación de Precios por Kilometraje')
    plt.xlabel('Kilometraje (Km)')
    plt.ylabel('Precio (€)')
    plt.legend()
    plt.grid(True)
    plt.savefig("grafico_comparacion.png")
    plt.close()  # Cerrar la figura para liberar memoria

    return f"El precio estimado para el modelo seleccionado es: €{precio_predicho:,.2f}", "grafico_comparacion.png"

with gr.Blocks() as interfaz:
    gr.Image("https://brandemia.org/contenido/subidas/2021/12/07-mercedes-logo-2009-2011-hasta-hoy-1200x670.jpg", width=150, interactive=False, label="Mercedes-Benz Logo", show_label=False)
    gr.Markdown("<h1 style='text-align: center;'>Predice el precio de tu modelo MERCEDES-BENZ</h1>")
    
    with gr.Row():
        modelo = gr.Dropdown(choices=["Clase A"], label="Modelo", value="Clase A", interactive=False)
        potencia = gr.Dropdown(choices=potencia_options, label="Potencia")
        tipo_vehiculo = gr.Dropdown(choices=tipo_vehiculo_options, label="Tipo de vehículo")
    
    with gr.Row():
        año = gr.Dropdown(choices=año_options, label="Año de fabricación")
        km = gr.Number(label="Kilometraje (en km)", value=50000)  # Entrada de texto numérico para Km
        tipo_gasolina = gr.Dropdown(choices=tipo_gasolina_options, label="Tipo de gasolina")
    
    ubicacion = gr.Dropdown(choices=ubicacion_options, label="Ubicación")
    
    boton_predecir = gr.Button("Predecir Precio")
    salida = gr.Textbox(label="Predicción del Precio")
    grafico = gr.Image(label="Gráfico de Comparación")
    
    # Configurar la acción del botón
    boton_predecir.click(predecir_precio, 
                         inputs=[potencia, tipo_vehiculo, año, km, tipo_gasolina, ubicacion], 
                         outputs=[salida, grafico])

interfaz.launch()