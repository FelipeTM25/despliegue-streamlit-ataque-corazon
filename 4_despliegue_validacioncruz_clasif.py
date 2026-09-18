# -*- coding: utf-8 -*-
"""Despliegue_ValidacionCruz_Clasif

# Despliegue

- Cargamos el modelo
- Capturamos los datos del paciente con Streamlit
- Preparamos los datos: dummies, reindex
- Aplicamos el modelo para la predicción
"""

import pandas as pd
import streamlit as st
import pickle

# ---------------------------------------------------------------- página
st.set_page_config(
    page_title="Tablero de riesgo cardiovascular",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .stApp { background-color:#f4f6fb; }
    section[data-testid="stSidebar"] { background-color:#12233f; }
    section[data-testid="stSidebar"] * { color:#e8edf6 !important; }
    .kicker { font-size:.78rem; letter-spacing:.14em; text-transform:uppercase;
              color:#5b6b85; margin-bottom:.15rem; }
    .titulo { font-size:1.9rem; font-weight:700; color:#12233f; margin:0 0 .2rem 0; }
    .veredicto { font-size:1.35rem; font-weight:700; margin:.2rem 0 .4rem 0; }
    .alto { color:#b4232a; } .bajo { color:#1d7a54; }
    .barra { height:12px; border-radius:999px; background:#e6eaf2; overflow:hidden; }
    .barra > div { height:100%; border-radius:999px; }
    div[data-testid="stMetricValue"] { color:#12233f; }
    div.stButton > button, div.stFormSubmitButton > button {
        background:#12233f; color:#ffffff; border:1px solid #12233f;
        border-radius:8px; font-weight:600; width:100%;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background:#1d3a66; border-color:#1d3a66; color:#ffffff;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- modelo
@st.cache_resource
def cargar_modelo(filename='modelo-cla.pkl'):
    return pickle.load(open(filename, 'rb'))


modelo, labelencoder, variables, min_max_scaler = cargar_modelo()

# ---------------------------------------------------------------- entradas (sidebar)
with st.sidebar:
    st.markdown("### Ficha del paciente")
    st.caption("Completa los campos y pulsa *Evaluar paciente*.")

    with st.form("ficha"):
        age = st.number_input('Edad (años)', min_value=1, max_value=82, value=45, step=1)
        avg_glucose_level = st.number_input(
            'Glucosa promedio (mg/dL)', min_value=55.0, max_value=272.0, value=100.0, step=0.5
        )
        hypertension = st.radio('Hipertensión', ['No', 'Yes'], horizontal=True)
        heart_disease = st.radio('Enfermedad cardíaca previa', ['No', 'Yes'], horizontal=True)
        ever_married = st.radio('Alguna vez casado/a', ['No', 'Yes'], horizontal=True)
        smoking_status = st.select_slider(
            'Hábito de fumar',
            options=["'never smoked'", "'formerly smoked'", "smokes", "Unknown"],
            value="'never smoked'",
        )
        evaluar = st.form_submit_button('Evaluar paciente')

# ---------------------------------------------------------------- encabezado
st.markdown('<div class="kicker">Modelo Random Forest &middot; despliegue</div>', unsafe_allow_html=True)
st.markdown('<div class="titulo">Tablero de riesgo cardiovascular</div>', unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------- dataframe de entrada
datos = [[age, avg_glucose_level, hypertension, heart_disease, ever_married, smoking_status]]
data = pd.DataFrame(
    datos,
    columns=['age', 'avg_glucose_level', 'hypertension', 'heart_disease',
             'ever_married', 'smoking_status']
)

# ---------------------------------------------------------------- preparación
# En despliegue drop_first=False
data_preparada = pd.get_dummies(
    data.copy(),
    columns=['smoking_status', 'hypertension', 'heart_disease', 'ever_married'],
    drop_first=False, dtype=int
)

# Se adicionan las columnas faltantes
data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

# Como el modelo final es Random Forest (árboles) NO se normaliza.
# data_preparada[['age','avg_glucose_level']] = min_max_scaler.transform(
#     data_preparada[['age','avg_glucose_level']])

# ---------------------------------------------------------------- predicción
prediccion = None
proba = None

if evaluar:
    # Hacemos la predicción con el Random Forest
    Y_pred = modelo.predict(data_preparada)

    # Se convierte 0/1 a la etiqueta original (No/Yes)
    prediccion = labelencoder.inverse_transform(Y_pred)[0]
    data['Prediccion'] = prediccion

    # Probabilidad de la clase positiva, si el modelo la expone
    if hasattr(modelo, "predict_proba"):
        clases = list(labelencoder.inverse_transform(modelo.classes_))
        if 'Yes' in clases:
            proba = float(modelo.predict_proba(data_preparada)[0][clases.index('Yes')])

# ---------------------------------------------------------------- tabs
tab_resultado, tab_datos, tab_modelo = st.tabs(["Resultado", "Datos ingresados", "Modelo"])

with tab_resultado:
    col_izq, col_der = st.columns([3, 2], gap="large")

    with col_izq:
        with st.container(border=True):
            st.markdown('<div class="kicker">Veredicto del modelo</div>', unsafe_allow_html=True)

            if prediccion is None:
                st.markdown("**Sin evaluación todavía.**")
                st.caption("Captura la ficha en el panel izquierdo y pulsa "
                           "*Evaluar paciente* para obtener la predicción.")
            else:
                riesgo_alto = prediccion == 'Yes'
                clase_css = "alto" if riesgo_alto else "bajo"
                texto = ("Riesgo alto de ataque al corazón" if riesgo_alto
                         else "Riesgo bajo de ataque al corazón")

                st.markdown(f'<div class="veredicto {clase_css}">{texto}</div>', unsafe_allow_html=True)

                if proba is not None:
                    pct = round(proba * 100, 1)
                    color = "#b4232a" if riesgo_alto else "#1d7a54"
                    st.markdown(
                        f'<div class="barra"><div style="width:{pct}%;background:{color};"></div></div>'
                        f'<div style="font-size:.85rem;color:#5b6b85;margin-top:.35rem;">'
                        f'Probabilidad estimada de evento: <b>{pct}%</b></div>',
                        unsafe_allow_html=True
                    )

                st.caption("Apoyo estadístico, no un diagnóstico médico. "
                           "El modelo tiene un error aproximado del 8% (mape).")

    with col_der:
        with st.container(border=True):
            st.markdown('<div class="kicker">Resumen de la ficha</div>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Edad", f"{int(age)} años")
            m2.metric("Glucosa", f"{avg_glucose_level:.1f} mg/dL")
            st.markdown(
                "- Hipertensión: **" + hypertension + "**\n"
                "- Enfermedad cardíaca: **" + heart_disease + "**\n"
                "- Casado/a: **" + ever_married + "**\n"
                "- Fumador: **" + smoking_status.replace("'", "") + "**"
            )

with tab_datos:
    st.markdown("**Datos capturados**")
    st.dataframe(data, use_container_width=True, hide_index=True)
    with st.expander("Ver matriz enviada al modelo (dummies + reindex)"):
        st.dataframe(data_preparada, use_container_width=True, hide_index=True)

with tab_modelo:
    st.markdown("**Random Forest de clasificación**")
    st.caption("El modelo tiene un error del 8% (mape: error porcentual). "
               "La predicción es un apoyo estadístico, no un diagnóstico médico.")
    st.caption("Variables esperadas por el modelo: " + str(len(variables)))
