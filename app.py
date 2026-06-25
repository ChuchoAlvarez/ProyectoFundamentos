import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 1. CONFIGURACIÓN E INICIALIZACIÓN DE LA PÁGINA
st.set_page_config(page_title="Modelos Predicitvos", page_icon="🔮", layout="wide")

# Título principal estático que nunca se va a borrar
st.title("Modelos Predicitvos - Fundamentos de Analítica de Datos")
st.write("Bienvenido al portal donde el consejo de expertos toma decisiones basadas en datos.")

# =========================================================================
# 2. PROCESAMIENTO PREVIO DE DATOS 
# =========================================================================

# --- MODELO 1: ALUMNOS ---
datos_alumnos = {
    'horas_estudio': [10, 2, 15, 1, 12, 4, 18, 5, 20, 3],
    'asistencia': [90, 40, 95, 20, 80, 50, 100, 60, 90, 30],
    'tareas': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
    'resultado': ['Aprobado', 'Reprobado', 'Aprobado', 'Reprobado', 'Aprobado', 
                  'Reprobado', 'Aprobado', 'Reprobado', 'Aprobado', 'Reprobado']
}
df_alumnos = pd.DataFrame(datos_alumnos)
X_al = df_alumnos[['horas_estudio', 'asistencia', 'tareas']]
y_al = df_alumnos['resultado']
X_train_al, X_test_al, y_train_al, y_test_al = train_test_split(X_al, y_al, test_size=0.2, random_state=42)
sabios_alumnos = RandomForestClassifier(n_estimators=30, random_state=42)
sabios_alumnos.fit(X_train_al, y_train_al)
importancias_al = sabios_alumnos.feature_importances_


# --- MODELO 2: PACIENTES ---
# Base base por defecto para que la app no dependa de archivos externos al arrancar
df_pacientes = pd.DataFrame({
    'nuevos': ['SI', 'NO', 'SI', 'SI', 'NO', 'SI', 'NO', 'SI', 'SI', 'NO'],
    'intervencion': ['UCI', 'PISO', 'UCI', 'URGENCIAS', 'PISO', 'UCI', 'PISO', 'URGENCIAS', 'UCI', 'PISO'],
    'cantidad': [1, 2, 1, 3, 1, 1, 2, 1, 1, 2],
    'tarifa': [500, 200, 600, 150, 200, 550, 210, 160, 500, 200],
    'dx': ['COVID CONFIRMADO', 'SOSPECHOSO', 'COVID CONFIRMADO', 'NEUMONIA', 'SOSPECHOSO', 'COVID CONFIRMADO', 'SOSPECHOSO', 'NEUMONIA', 'COVID CONFIRMADO', 'SOSPECHOSO'],
    'resclin': ['GRAVE', 'ESTABLE', 'GRAVE', 'MODERADO', 'ESTABLE', 'CRITICO', 'ESTABLE', 'MODERADO', 'CRITICO', 'ESTABLE'],
    'mortalidad': [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
})

# Menú lateral de navegación
st.sidebar.title("Navegación")
opcion = st.sidebar.selectbox(
    "Selecciona el módulo que deseas consultar:",
    ("1. Predicción de Alumnos (Escuela)", "2. Índice de Mortalidad (Pacientes COVID-19)")
)

# Permitir subir CSV solo si se está en el módulo de pacientes
if opcion == "2. Índice de Mortalidad (Pacientes COVID-19)":
    st.sidebar.write("---")
    st.sidebar.subheader("Configuración de Datos")
    origen_datos = st.sidebar.radio("Origen de datos clínicos:", ("Datos de Demostración", "Cargar archivo CSV"))
    if origen_datos == "Cargar archivo CSV":
        archivo_cargado = st.sidebar.file_uploader("Sube tu archivo (.csv)", type=["csv"])
        if archivo_cargado is not None:
            df_pacientes = pd.read_csv(archivo_cargado)

# Procesamiento de encoders para pacientes
dict_encoders = {}
df_procesado = df_pacientes.copy()
columnas_texto = ['nuevos', 'intervencion', 'dx', 'resclin']

for col in columnas_texto:
    le = LabelEncoder()
    df_procesado[col] = df_procesado[col].astype(str)
    le.fit(df_procesado[col])
    df_procesado[col] = le.transform(df_procesado[col])
    dict_encoders[col] = le

df_procesado['mortalidad'] = df_procesado['mortalidad'].astype(int)
X_pa = df_procesado[['nuevos', 'intervencion', 'cantidad', 'tarifa', 'dx', 'resclin']]
y_pa = df_procesado['mortalidad']

X_pa['cantidad'] = pd.to_numeric(X_pa['cantidad'], errors='coerce').fillna(0)
X_pa['tarifa'] = pd.to_numeric(X_pa['tarifa'], errors='coerce').fillna(0)

X_train_pa, X_test_pa, y_train_pa, y_test_pa = train_test_split(X_pa, y_pa, test_size=0.2, random_state=42)
sabios_salud = RandomForestClassifier(n_estimators=100, random_state=42)
sabios_salud.fit(X_train_pa, y_train_pa)


# =========================================================================
# 3. RENDERIZADO DE LAS VISTAS (Pestañas / Módulos)
# =========================================================================

# VISTA 1: ALUMNOS
if opcion == "1. Predicción de Alumnos (Escuela)":
    st.header("Módulo: El Futuro de los Alumnos")
    st.write("Análisis de rendimiento académico basado en el histórico escolar.")

    with st.expander("Ver bitácora de alumnos históricos"):
        st.dataframe(df_alumnos)

    st.subheader("Evaluar un Alumno Nuevo")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_horas = st.number_input("Horas de estudio:", min_value=0, max_value=50, value=23)
    with col2:
        input_asistencia = st.slider("Porcentaje de asistencia:", min_value=0, max_value=100, value=85)
    with col3:
        input_tareas = st.selectbox("¿Cumplió con las tareas?", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]

    if st.button("🔮 Consultar al Consejo de Sabios (Alumnos)"):
        alumno_nuevo = pd.DataFrame([[input_horas, input_asistencia, input_tareas]], columns=X_al.columns)
        prediccion = sabios_alumnos.predict(alumno_nuevo)
        
        st.write("---")
        if prediccion[0] == 'Aprobado':
            st.success(f"### El Consejo de Sabios dice que el alumno será: **{prediccion[0]}**")
        else:
            st.error(f"### El Consejo de Sabios dice que el alumno será: **{prediccion[0]}**")

        st.subheader("Importancia de las variables")
        for nombre, valor in zip(X_al.columns, importancias_al):
            st.write(f"**{nombre.replace('_', ' ').capitalize()}**: {valor:.2%}")

# VISTA 2: PACIENTES
elif opcion == "2. Índice de Mortalidad (Pacientes COVID-19)":
    st.header("Módulo: Reporte de Predicción de Mortalidad Hospitalaria")
    
    if st.checkbox("Ver base de datos activa de Pacientes"):
        st.dataframe(df_pacientes)

    # Reporte de rendimiento técnico
    st.subheader("Reporte de Rendimiento del Modelo")
    predicciones_pa = sabios_salud.predict(X_test_pa)
    reporte_dict = classification_report(y_test_pa, predicciones_pa, target_names=['Vive', 'Fallece'], zero_division=0, output_dict=True)
    df_reporte = pd.DataFrame(reporte_dict).transpose()
    st.table(df_reporte.iloc[:-3, :-1])

    # SIMULADOR INTERACTIVO (Garantizado que aparezca aquí)
    st.write("---")
    st.subheader("Simulador Clínico: Evaluar un Nuevo Paciente")
    st.write("Modifica los siguientes parámetros para interactuar en tiempo real con el modelo Predictivo:")

    # Extracción de clases válidas de los encoders
    opciones_nuevos = dict_encoders['nuevos'].classes_
    opciones_intervencion = dict_encoders['intervencion'].classes_
    opciones_dx = dict_encoders['dx'].classes_
    opciones_resclin = dict_encoders['resclin'].classes_

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        sim_nuevo = st.selectbox("¿Es paciente nuevo?", opciones_nuevos)
        sim_intervencion = st.selectbox("Área de intervención asignada:", opciones_intervencion)
    with col_p2:
        sim_cantidad = st.number_input("Cantidad de servicios médicos:", min_value=1, max_value=100, value=1)
        sim_tarifa = st.number_input("Costo acumulado de atención ($):", min_value=0, max_value=100000, value=500)
    with col_p3:
        sim_dx = st.selectbox("Diagnóstico Principal (Dx):", opciones_dx)
        sim_resclin = st.selectbox("Estado Clínico inicial:", opciones_resclin)

    if st.button("🔮 Calcular Índice de Riesgo Clínico"):
        try:
            # Traducción segura
            val_nuevo = dict_encoders['nuevos'].transform([sim_nuevo])[0]
            val_intervencion = dict_encoders['intervencion'].transform([sim_intervencion])[0]
            val_dx = dict_encoders['dx'].transform([sim_dx])[0]
            val_resclin = dict_encoders['resclin'].transform([sim_resclin])[0]

            paciente_simulado = pd.DataFrame([[
                val_nuevo, val_intervencion, sim_cantidad, sim_tarifa, val_dx, val_resclin
            ]], columns=X_pa.columns)

            prob_resultado = sabios_salud.predict_proba(paciente_simulado)
            riesgo_fallecer = prob_resultado[0][1] * 100
            prob_sobrevivir = prob_resultado[0][0] * 100

            st.write("---")
            st.markdown("### Veredicto de la Inteligencia Médica:")
            
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.metric(label="Porcentaje de Riesgo (Mortalidad)", value=f"{riesgo_fallecer:.2f}%")
            with res_c2:
                st.metric(label="Porcentaje de Recuperación (Supervivencia)", value=f"{prob_sobrevivir:.2f}%")

            if riesgo_fallecer >= 70:
                st.error("🚨 **Prioridad Crítica:** El modelo identifica patrones de riesgo severo. Requiere intervención inmediata.")
            elif 30 <= riesgo_fallecer < 70:
                st.warning("⚠️ **Monitoreo Moderado:** Estado intermedio. Se recomienda observación continua.")
            else:
                st.success("✅ **Bajo Riesgo:** El modelo predice una alta probabilidad de estabilidad clínica.")
        except Exception as e:
            st.error(f"Error al procesar la simulación: {e}")

    # Gráfica de importancia de variables al final
    st.write("---")
    st.subheader("🧬 Factores de Mayor Peso en la Toma de Decisión")
    importancias_pa = sabios_salud.feature_importances_
    df_imp = pd.DataFrame({
        'Variable': X_pa.columns,
        'Importancia (%)': [v * 100 for v in importancias_pa]
    }).sort_values(by='Importancia (%)', ascending=False)
    st.bar_chart(data=df_imp, x='Variable', y='Importancia (%)')
