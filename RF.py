import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st

# Configuración de Streamlit
st.title('Importancia de las Características en el Modelo de Random Forest')
st.write("""
Esta aplicación permite visualizar la importancia de las características en un modelo de Random Forest para la predicción del volumen de captura.
""")

# Cargar datos
uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Convertir las fechas a formato datetime
    data['Fecha_Faena'] = pd.to_datetime(data['Fecha_Faena'])
    data['Fecha_Desembarque'] = pd.to_datetime(data['Fecha_Desembarque'])

    # Extraer día de la semana y hora del día
    data['Dia_Semana'] = data['Fecha_Faena'].dt.day_name()
    data['Hora_Día'] = data['Fecha_Faena'].dt.hour

    # Seleccionar la especie
    especie_especifica = st.selectbox("Seleccionar la especie", data['Especie'].unique())
    data_especie = data[data['Especie'] == especie_especifica]

    # Convertir las variables categóricas en variables dummy
    data_especie = pd.get_dummies(data_especie, columns=['Dia_Semana', 'Aparejo', 'Origen', 'Embarcacion'])

    # Seleccionar las características
    x = data_especie.drop(columns=['REG', 'Fecha_Faena', 'Fecha_Desembarque', 'Especie', 'Volumen_Kg', 'Talla_cm', 'Precio_Kg', 'Venta'])
    y = data_especie['Volumen_Kg']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = rf.predict(x_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Error cuadrático medio: {mse}')

    # Importancia de las características
    importancia_caracteristicas = pd.Series(rf.feature_importances_, index=x.columns)
    fig, ax = plt.subplots()
    importancia_caracteristicas.nlargest(10).plot(kind='barh', ax=ax)
    ax.set_xlabel('Importancia de la característica')
    ax.set_ylabel('Característica')
    ax.set_title('Importancia de las características en el modelo de Random Forest')
    st.pyplot(fig)
