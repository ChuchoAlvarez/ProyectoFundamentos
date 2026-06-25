import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Configuración inicial de la página web
st.set_page_config(page_title="El Consejo de Sabios", page_icon="🔮", layout="wide")

st.title("🔮 El Consejo de Sabios - Inteligencia Artificial")
st.write("Bienvenido al portal donde el consejo de expertos toma decisiones basadas en datos.")

# --- MENÚ LATERAL DE NAVEGACIÓN ---
opcion = st.sidebar.selectbox(
    "Selecciona el módulo que deseas consultar:",
    ("1. Predicción de Alumnos (Escuela)", "2. Índice de Mortalidad (Pacientes COVID-19)")
)

# =========================================================================
# MÓDULO 1: ALUMNOS
# =========================================================================
if opcion == "1. Predicción de Alumnos (Escuela)":
    st.header("👨‍🎓 Módulo: El Futuro de los Alumnos")
    st.write("Aquí el consejo analiza las horas de estudio, asistencia y tareas de alumnos pasados para predecir el destino de un alumno nuevo.")

    # 1. Datos base
    datos_alumnos = {
        'horas_estudio': [10, 2, 15, 1, 12, 4, 18, 5, 20, 3],
        'asistencia': [90, 40, 95, 20, 80, 50, 100, 60, 90, 30],
        'tareas': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
        'resultado': ['Aprobado', 'Reprobado', 'Aprobado', 'Reprobado', 'Aprobado', 
                      'Reprobado', 'Aprobado', 'Reprobado', 'Aprobado', 'Reprobado']
    }
    df_alumnos = pd.DataFrame(datos_alumnos)

    # Mostrar tabla actual
    with st.expander("Ver bitácora de alumnos históricos (El chisme del pasado)"):
        st.dataframe(df_alumnos)

    # Preparar el modelo
    X = df_alumnos[['horas_estudio', 'asistencia', 'tareas']]
    y = df_alumnos['resultado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sabios_alumnos = RandomForestClassifier(n_estimators=30, random_state=42)
    sabios_alumnos.fit(X_train, y_train)
    importancias = sabios_alumnos.feature_importances_

    # Interfaz para el alumno nuevo
    st.subheader("Evaluar un Alumno Nuevo")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        input_horas = st.number_input("Horas de estudio:", min_value=0, max_value=50, value=23)
    with col2:
        input_asistencia = st.slider("Porcentaje de asistencia:", min_value=0, max_value=100, value=2)
    with col3:
        input_tareas = st.selectbox("¿Cumplió con las tareas?", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]

    if st.button("🔮 Consultar al Consejo de Sabios"):
        alumno_nuevo = pd.DataFrame([[input_horas, input_asistencia, input_tareas]], columns=X.columns)
        prediccion = sabios_alumnos.predict(alumno_nuevo)
        
        # Mostrar resultado llamativo
        st.write("---")
        if prediccion[0] == 'Aprobado':
            st.success(f"### El Consejo de Sabios dice que el alumno será: **{prediccion[0]}**")
        else:
            st.error(f"### El Consejo de Sabios dice que el alumno será: **{prediccion[0]}**")

        # Justificación lógica dinámica
        st.subheader("Justificación del Consejo:")
        if prediccion[0] == 'Reprobado':
            if input_asistencia < 30:
                st.info(f"- Aunque estudió {input_horas} horas, su asistencia es demasiado baja ({input_asistencia}%).")
                st.info(f"- Para los sabios, la asistencia tiene un peso del **{importancias[1]:.2%}**, siendo el factor decisivo.")
            if input_tareas == 0:
                st.info("- Además, el incumplimiento de tareas afectó negativamente el veredicto final.")
        else:
            st.info("- El alumno cumple con los criterios mínimos de asistencia y compromiso que el consejo valora.")

        # Estadísticas de importancia
        st.subheader("📊 Importancia de las pistas (¿Qué miran más los Sabios?)")
        for nombre, valor in zip(X.columns, importancias):
            st.write(f"**{nombre.replace('_', ' ').capitalize()}**: {valor:.2%}")

# =========================================================================
# MÓDULO 2: PACIENTES
# =========================================================================
elif opcion == "2. Índice de Mortalidad (Pacientes COVID-19)":
    st.header("Módulo: Reporte de Predicción de Hospital/Mortalidad")
    st.write("Este módulo analiza variables clínicas para determinar probabilidades de riesgo en pacientes.")

    # Opción de origen de datos para que funcione en web sin tu BD local
    origen_datos = st.radio("Origen de los datos:", ("Usar datos simulados de demostración", "Cargar archivo CSV / Conectar Base de Datos"))

    # Creamos un dataframe de prueba por si no se conecta la BD
    if origen_datos == "Usar datos simulados de demostración":
        df_pacientes = pd.DataFrame({
            'nuevos': ['SI', 'NO', 'SI', 'SI', 'NO', 'SI', 'NO', 'SI', 'SI', 'NO'],
            'intervencion': ['UCI', 'PISO', 'UCI', 'URGENCIAS', 'PISO', 'UCI', 'PISO', 'URGENCIAS', 'UCI', 'PISO'],
            'cantidad': [1, 2, 1, 3, 1, 1, 2, 1, 1, 2],
            'tarifa': [500, 200, 600, 150, 200, 550, 210, 160, 500, 200],
            'dx': ['COVID CONFIRMADO', 'SOSPECHOSO', 'COVID CONFIRMADO', 'NEUMONIA', 'SOSPECHOSO', 'COVID CONFIRMADO', 'SOSPECHOSO', 'NEUMONIA', 'COVID CONFIRMADO', 'SOSPECHOSO'],
            'resclin': ['GRAVE', 'ESTABLE', 'GRAVE', 'MODERADO', 'ESTABLE', 'CRITICO', 'ESTABLE', 'MODERADO', 'CRITICO', 'ESTABLE'],
            'mortalidad': [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
        })
        st.info("💡 Usando base de datos de demostración precargada.")
    else:
        archivo_cargado = st.file_uploader("Sube tu archivo de pacientes (.csv)", type=["csv"])
        if archivo_cargado is not None:
            df_pacientes = pd.read_csv(archivo_cargado)
        else:
            st.warning("Por favor sube un archivo CSV para continuar en este modo. Mientras tanto, puedes usar la opción de simulación de arriba.")
            st.stop()

    if st.checkbox("Ver base de datos actual de Pacientes"):
        st.dataframe(df_pacientes)

    # --- PROCESAMIENTO ---
    le = LabelEncoder()
    df_procesado = df_pacientes.copy()
    
    columnas_texto = ['nuevos', 'intervencion', 'dx', 'resclin']
    for col in columnas_texto:
        if col in df_procesado.columns:
            df_procesado[col] = le.fit_transform(df_procesado[col].astype(str))

    df_procesado['mortalidad'] = df_procesado['mortalidad'].astype(int)

    X = df_procesado[['nuevos', 'intervencion', 'cantidad', 'tarifa', 'dx', 'resclin']]
    y = df_procesado['mortalidad']

    X['cantidad'] = pd.to_numeric(X['cantidad'], errors='coerce').fillna(0)
    X['tarifa'] = pd.to_numeric(X['tarifa'], errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sabios_salud = RandomForestClassifier(n_estimators=100, random_state=42)
    sabios_salud.fit(X_train, y_train)

    predicciones = sabios_salud.predict(X_test)
    probabilidades = sabios_salud.predict_proba(X_test)

    # --- MOSTRAR RESULTADOS EN LA WEB ---
    st.subheader("Reporte de Rendimiento del Modelo")
    
    # Intentar sacar el reporte de clasificación de manera limpia
    reporte_dict = classification_report(y_test, predicciones, target_names=['Vive', 'Fallece'], zero_division=0, output_dict=True)
    df_reporte = pd.DataFrame(reporte_dict).transpose()
    st.table(df_reporte.iloc[:-3, :-1]) # Muestra precisión, recall y f1-score estéticos

    st.subheader("Índice de Riesgo (Primeros 5 casos del examen)")
    cols_pacientes = st.columns(5)
    for i in range(min(5, len(X_test))):
        riesgo = probabilidades[i][1] * 100
        with cols_pacientes[i]:
            st.metric(label=f"Paciente {i+1}", value=f"{riesgo:.2f}%", delta="Riesgo de Fallecimiento", delta_color="inverse")

    st.subheader("🧬 Factores que más influyen en el diagnóstico crítico")
    importancias_salud = sabios_salud.feature_importances_
    df_importancia_salud = pd.DataFrame({
        'Variable': X.columns,
        'Importancia (%)': [valor * 100 for valor in importancias_salud]
    }).sort_values(by='Importancia (%)', ascending=False)
    
    st.bar_chart(data=df_importancia_salud, x='Variable', y='Importancia (%)')