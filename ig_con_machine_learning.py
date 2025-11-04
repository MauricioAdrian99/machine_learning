import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# --- 1. CLASES Y FUNCIONES DE LA LÓGICA CORE (Mantenidas) ---

@st.cache_data
def crear_base_alimentos():
    """Crea la base de datos con la lista original de comidas paraguayas"""
    alimentos_data = {
        'alimento': [
            'Mandioca hervida', 'Mandioca frita', 'Sopa paraguaya', 'Chipa almidón', 'Mbeyú',
            'Caldo de puchero (solo)', 'Caldo de puchero (con carne)', 'Tortilla de harina con huevo',
            'Arroz blanco', 'Arroz carretero', 'Poroto colorado', 'Locro', 'Mbaipy (polenta)', 'Pan casero',
            'Torta frita', 'Empanada frita', 'Lasaña', 'Milanesa de carne', 'Marinera',
            'Pollo al horno', 'Cocido (bebida)', 'Mate', 'Leche', 'Huevo frito',
            'Asado de res magro', 'Asado de costilla de res', 'Chorizo mixto cerdo-vacuno',
            'Cerveza', 'Vino tinto', 'Jugo en caja', 'Gaseosa coca cola',
            'Feijão', 'Farofa', 'Pasta de margarina', 'Pasta de manteca',
            'Coquito de harina', 'Galleta integral', 'Asadito o Espetiño', 'Chipa Guasu'
        ],
        'calorias_100g': [
            112, 165, 280, 320, 250, 15, 85, 285,
            130, 185, 142, 180, 85, 265, 380, 295, 190, 250, 240,
            239, 1, 1, 42, 196, 250, 330, 285,
            43, 85, 54, 37,
            142, 365, 717, 737,
            410, 450, 250, 230
        ],
        'cho_100g': [
            26, 25, 35, 65, 45, 2, 3, 30,
            28, 32, 25, 30, 18, 49, 40, 25, 15, 8, 12,
            0, 0, 0, 4.8, 1, 0, 0, 2,
            3.6, 2.6, 13, 10,
            25, 75, 1, 1,
            80, 70, 1.5, 22.5
        ],
        'ig': [
            70, 85, 65, 75, 70, 15, 20, 70,
            73, 75, 40, 50, 68, 75, 85, 70, 55, 30, 35,
            0, 0, 0, 30, 30, 0, 0, 30,
            110, 16, 55, 53,
            40, 85, 0, 0,
            75, 71, 0, 57.5
        ],
        'alcohol_grados': [
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            4.5, 12, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        ]
    }
    return pd.DataFrame(alimentos_data)

class MLNutritionSystem:
    """Sistema nutricional con Machine Learning"""
    
    def __init__(self):
        self.risk_classifier = None
        self.calorie_predictor = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def generate_synthetic_training_data(self, n_samples=1500):
        np.random.seed(42)
        data = []
        for i in range(n_samples):
            age = np.random.randint(18, 80)
            is_diabetic = np.random.choice([0, 1], p=[0.85, 0.15])
            pa_sistolica = np.random.normal(120 + age*0.5, 15)
            pa_diastolica = np.random.normal(80 + age*0.2, 10)
            fc = np.random.normal(70 + np.random.normal(0, 10), 12)
            sato2 = np.random.normal(97, 2)
            calorias = np.random.normal(2200, 500)
            cho_total = np.random.normal(300, 80)
            carga_glucemica = np.random.normal(60, 20)
            alcohol = np.random.exponential(5) if np.random.random() < 0.3 else 0
            
            risk_score = 0
            if pa_sistolica >= 140 or pa_diastolica >= 90: risk_score += 2
            if pa_sistolica >= 180 or pa_diastolica >= 110: risk_score += 3
            if calorias > 3000: risk_score += 2
            if carga_glucemica > 80: risk_score += 1.5
            if alcohol > 40: risk_score += 2
            if age > 65: risk_score += 0.5
            if is_diabetic and carga_glucemica > 50: risk_score += 2
            if fc >= 60 and fc <= 90 and sato2 >= 95: risk_score -= 0.5

            if risk_score <= 0.5: risk_level = 'NORMAL'
            elif risk_score <= 2: risk_level = 'MODERADO'
            elif risk_score <= 4: risk_level = 'ALTO'
            else: risk_level = 'CRITICO'

            data.append({
                'age': age, 'pa_sistolica': pa_sistolica, 'pa_diastolica': pa_diastolica,
                'fc': fc, 'sato2': sato2, 'is_diabetic': is_diabetic,
                'calorias': calorias, 'cho_total': cho_total, 'carga_glucemica': carga_glucemica,
                'alcohol': alcohol, 'risk_level': risk_level,
                'recommended_calories': max(1200, min(3000, 2000 + np.random.normal(0, 200)))
            })
        return pd.DataFrame(data)

    def train_models(self):
        """Entrena los modelos de ML"""
        df = self.generate_synthetic_training_data(1500)
        feature_cols = ['age', 'pa_sistolica', 'pa_diastolica', 'fc', 'sato2',
                        'is_diabetic', 'calorias', 'cho_total', 'carga_glucemica', 'alcohol']
        X = df[feature_cols]
        y_risk = df['risk_level']
        y_calories = df['recommended_calories']

        X_train, X_test, y_risk_train, y_risk_test, y_cal_train, y_cal_test = train_test_split(
            X, y_risk, y_calories, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.risk_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        self.risk_classifier.fit(X_train_scaled, y_risk_train)

        self.calorie_predictor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        self.calorie_predictor.fit(X_train_scaled, y_cal_train)
        
        self.is_trained = True
        return self

    def predict_risk_ml(self, patient_data):
        """Predice el riesgo y las calorías usando ML"""
        if not self.is_trained:
            return None

        features = np.array([[
            patient_data['age'], patient_data['pa_sistolica'], patient_data['pa_diastolica'],
            patient_data['fc'], patient_data['sato2'], 
            1 if patient_data['is_diabetic'] else 0,
            patient_data['calorias'], patient_data['cho_total'],
            patient_data['carga_glucemica'], patient_data['alcohol']
        ]])

        features_scaled = self.scaler.transform(features)

        risk_pred = self.risk_classifier.predict(features_scaled)[0]
        risk_proba = self.risk_classifier.predict_proba(features_scaled)[0]
        calories_pred = self.calorie_predictor.predict(features_scaled)[0]

        classes = self.risk_classifier.classes_
        risk_probabilities = dict(zip(classes, risk_proba))

        return {
            'predicted_risk': risk_pred,
            'risk_probabilities': risk_probabilities,
            'recommended_calories': calories_pred,
            'confidence': max(risk_proba)
        }

@st.cache_resource
def load_and_train_ml_system():
    """Carga y entrena el sistema ML (se ejecuta una sola vez)"""
    st.info("🤖 **Inicializando y entrenando modelos de Machine Learning...** Esto solo sucede la primera vez.")
    ml_system = MLNutritionSystem()
    ml_system.train_models()
    st.success("✅ **Modelos ML entrenados y listos.**")
    return ml_system

# Funciones de evaluación
def evaluar_riesgo_comida(componentes, base_alimentos):
    if not componentes:
        return { 'calorias_total': 0, 'cho_total': 0, 'carga_total': 0, 'ig_comida': 0, 
                 'alcohol_total_ml': 0, 'indice_alcoholico': 0, 'detalles': [] }
    
    calorias_total = 0
    cho_total = 0
    ig_ponderado = 0
    alcohol_total_ml = 0
    indice_alcoholico = 0
    detalles = []

    for alimento, cantidad in componentes.items():
        fila = base_alimentos[base_alimentos['alimento'] == alimento]
        if not fila.empty:
            fila = fila.iloc[0]
            factor = cantidad / 100 
            
            cals = fila['calorias_100g'] * factor
            cho = fila['cho_100g'] * factor
            ig = fila['ig']

            calorias_total += cals
            cho_total += cho
            if cho > 0: ig_ponderado += ig * cho

            if fila['alcohol_grados'] > 0:
                alcohol_puro = cantidad * fila['alcohol_grados'] * 0.789 / 100
                indice_alcoholico += alcohol_puro
                alcohol_total_ml += cantidad
            
            detalles.append({'alimento': alimento, 'cantidad': cantidad, 'calorias': cals, 'cho': cho, 'ig': ig})

    ig_comida = ig_ponderado / cho_total if cho_total > 0 else 0
    carga_total = ig_comida * cho_total / 100 if cho_total > 0 else 0

    return {
        'calorias_total': calorias_total,
        'cho_total': cho_total,
        'carga_total': carga_total,
        'ig_comida': ig_comida,
        'alcohol_total_ml': alcohol_total_ml,
        'indice_alcoholico': indice_alcoholico,
        'detalles': detalles
    }

def acumular_totales_24h_completo(resultados, total_24h):
    """Acumula los totales de 24 horas"""
    if resultados['cho_total'] > 0:
        total_24h['ig_ponderado_total'] += resultados['ig_comida'] * resultados['cho_total']
        total_24h['cho_total_24h'] += resultados['cho_total']
    
    total_24h['carga_total_24h'] += resultados['carga_total']
    total_24h['calorias_total_24h'] += resultados['calorias_total']
    total_24h['alcohol_total_24h'] += resultados.get('alcohol_total_ml', 0)
    total_24h['indice_alcoholico_24h'] += resultados.get('indice_alcoholico', 0)
    total_24h['detalles_24h'].extend(resultados['detalles'])

def evaluar_signos_vitales(signos):
    """Evalúa los signos vitales usando reglas tradicionales"""
    if not signos: return None
    riesgos = []
    nivel_riesgo = "NORMAL"

    if signos['pa_sistolica'] >= 180 or signos['pa_diastolica'] >= 110:
        riesgos.append("Crisis hipertensiva")
        nivel_riesgo = "CRÍTICO"
    elif signos['pa_sistolica'] >= 140 or signos['pa_diastolica'] >= 90:
        riesgos.append("Hipertensión arterial")
        nivel_riesgo = "ALTO" if nivel_riesgo != "CRÍTICO" else nivel_riesgo
    
    if signos['fc'] > 100:
        riesgos.append("Taquicardia")
        nivel_riesgo = "MODERADO" if nivel_riesgo == "NORMAL" else nivel_riesgo
    elif signos['fc'] < 60:
        riesgos.append("Bradicardia")
        nivel_riesgo = "MODERADO" if nivel_riesgo == "NORMAL" else nivel_riesgo

    if signos['sato2'] < 90:
        riesgos.append("Hipoxemia severa")
        nivel_riesgo = "CRÍTICO"
    elif signos['sato2'] < 95:
        riesgos.append("Hipoxemia leve")
        nivel_riesgo = "ALTO" if nivel_riesgo not in ["CRÍTICO"] else nivel_riesgo

    color_map = {"NORMAL": "🟢", "MODERADO": "🟡", "ALTO": "🟠", "CRÍTICO": "🔴"}
    return {'riesgos': riesgos, 'nivel_riesgo': nivel_riesgo, 'color_riesgo': color_map.get(nivel_riesgo, "⚪")}

def evaluar_riesgos_con_signos_vitales(resultados_24h, signos_vitales, evaluacion_sv):
    """Evalúa riesgos combinando nutrición y signos vitales (función original)"""
    riesgos = []
    recomendaciones = []

    # Riesgos nutricionales
    if resultados_24h['calorias_total'] > 3000:
        riesgos.append("Exceso calórico severo (>3000 kcal/día)")
        recomendaciones.append("Reducir ingesta calórica total")

    if resultados_24h['carga_total'] > 80:
        riesgos.append("Carga glucémica muy elevada")
        recomendaciones.append("Reducir carbohidratos de alto índice glucémico")

    if resultados_24h.get('indice_alcoholico', 0) > 40:
        riesgos.append("Consumo de alcohol peligroso")
        recomendaciones.append("Suspender consumo de alcohol inmediatamente")

    # Riesgos por signos vitales
    if evaluacion_sv and evaluacion_sv['riesgos']:
        riesgos.extend([f"SV: {r}" for r in evaluacion_sv['riesgos']])

    # Riesgos combinados
    if (signos_vitales and signos_vitales.get('is_diabetic') and
        resultados_24h['carga_total'] > 50):
        riesgos.append("Riesgo de descompensación diabética")
        recomendaciones.append("Control glucémico inmediato")

    if not recomendaciones:
        recomendaciones.append("Mantener patrón alimentario actual")

    return {
        'riesgos': riesgos,
        'recomendaciones': recomendaciones
    }

def generar_diagnostico_hipotetico(signos, resultados_nutricionales):
    """Genera diagnósticos hipotéticos educativos"""
    diagnosticos = []
    
    carga_total = resultados_nutricionales.get('carga_total', 0)
    alcohol_index = resultados_nutricionales.get('indice_alcoholico', 0)
    calorias = resultados_nutricionales.get('calorias_total', 0)
    
    if signos['pa_sistolica'] >= 180 or signos['pa_diastolica'] >= 110:
        diagnosticos.append("Crisis hipertensiva - Emergencia médica (criterio tradicional)")

    if signos['is_diabetic'] and carga_total > 50:
        diagnosticos.append("Descompensación diabética potencial por alta carga de CHO")

    if alcohol_index > 40:
        diagnosticos.append("Intoxicación etílica aguda/Consumo de riesgo")

    if signos['pa_sistolica'] > 140 and calorias > 2800:
        diagnosticos.append("Síndrome metabólico (Hipertensión + sobreingesta calórica)")

    if not diagnosticos:
        diagnosticos.append("Parámetros dentro de límites relativamente normales")

    return diagnosticos

# --- 2. CONFIGURACIÓN DE STREAMLIT Y LÓGICA DE LA APP ---

# Inicialización de la aplicación
st.set_page_config(
    page_title="Sistema Nutricional con ML (Py-Streamlit)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar la base de alimentos (solo una vez)
ALIMENTOS_DF = crear_base_alimentos()
ALIMENTOS_LIST = ALIMENTOS_DF['alimento'].tolist()
BEBIDAS = ['Cerveza', 'Vino tinto', 'Jugo en caja', 'Gaseosa coca cola', 'Cocido (bebida)', 'Mate', 'Leche']

# Inicializar y entrenar el sistema ML (solo una vez)
ML_SYSTEM = load_and_train_ml_system()

# Gestión del estado de la sesión (para mantener los datos de la comida)
if 'comidas_dia' not in st.session_state:
    st.session_state.comidas_dia = {
        "Desayuno": {}, "Media Mañana": {}, "Almuerzo": {},
        "Merienda": {}, "Cena": {}, "Desayuno del día siguiente": {}
    }
if 'signos_vitales' not in st.session_state:
    # CORRECCIÓN DE ERROR: Inicialización de valores numéricos como float
    st.session_state.signos_vitales = {
        'age': 40, 
        'pa_sistolica': 120.0,  # Corregido: float
        'pa_diastolica': 80.0,   # Corregido: float
        'fc': 70.0,              # Corregido: float
        'fr': 16.0,              # Corregido: float
        'sato2': 98, 
        'is_diabetic': False
    }

st.title("🇵🇾 Sistema Nutricional y de Riesgo Cardiovascular")
st.markdown("### Evaluación Integral con Machine Learning (Basado en alimentos paraguayos)")

# --- SIDEBAR: Registro de Signos Vitales ---
st.sidebar.header("🩺 Datos Basales del Paciente")
with st.sidebar.form("form_signos_vitales"):
    st.markdown("#### Signos Vitales")
    st.session_state.signos_vitales['age'] = st.slider("Edad (años)", 18, 100, st.session_state.signos_vitales['age'])
    
    # Todos los number_input usan valores y pasos flotantes (X.0)
    st.session_state.signos_vitales['pa_sistolica'] = st.number_input("P.A. Sistólica (mmHg)", 80.0, 250.0, st.session_state.signos_vitales['pa_sistolica'], 1.0)
    st.session_state.signos_vitales['pa_diastolica'] = st.number_input("P.A. Diastólica (mmHg)", 40.0, 150.0, st.session_state.signos_vitales['pa_diastolica'], 1.0)
    st.session_state.signos_vitales['fc'] = st.number_input("Frecuencia Cardíaca (lpm)", 40.0, 150.0, st.session_state.signos_vitales['fc'], 1.0)
    st.session_state.signos_vitales['sato2'] = st.slider("Saturación de Oxígeno (%)", 85, 100, int(st.session_state.signos_vitales['sato2']))
    st.session_state.signos_vitales['is_diabetic'] = st.checkbox("¿El paciente es diabético?", st.session_state.signos_vitales['is_diabetic'])
    
    if st.form_submit_button("Actualizar Signos Vitales"):
        st.sidebar.success("Signos vitales actualizados.")

# Evaluación de signos vitales tradicionales (para mostrar en la sidebar)
evaluacion_sv = evaluar_signos_vitales(st.session_state.signos_vitales)
st.sidebar.markdown("---")
st.sidebar.markdown("#### Evaluación Rápida (Reglas)")
st.sidebar.metric(
    "Nivel de Riesgo Vital", 
    f"{evaluacion_sv['nivel_riesgo']} {evaluacion_sv['color_riesgo']}",
    delta_color="off"
)
if evaluacion_sv['riesgos']:
    st.sidebar.caption("Riesgos detectados:")
    for riesgo in evaluacion_sv['riesgos']:
        st.sidebar.write(f"• *{riesgo}*")


# --- PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3 = st.tabs(["🍽️ Registro de Comidas (24h)", "📊 Resultados ML & Nutrición", "🔬 Base de Alimentos"])

with tab1:
    st.header("1. Registro de Ingesta Nutricional (24 Horas)")
    
    TIPO_COMIDA = st.selectbox("Seleccione la Comida a Registrar:", 
                                list(st.session_state.comidas_dia.keys()))
    
    st.info(f"Registrando alimentos para **{TIPO_COMIDA}**.")
    
    # Interfaz de registro de alimentos (más Streamlit-friendly)
    with st.expander("➕ Agregar/Modificar Alimentos para esta Comida"):
        alimento_seleccionado = st.selectbox(
            "Seleccionar Alimento:",
            ALIMENTOS_LIST
        )
        
        unidad_texto = "gramos"
        alimento_info = ALIMENTOS_DF[ALIMENTOS_DF['alimento'] == alimento_seleccionado].iloc[0]
        
        # Usamos número de pasos de 1 para mantener el float
        if alimento_seleccionado in BEBIDAS:
            unidad_texto = "ml (300ml = 1 vaso)"
            cantidad_input = st.number_input(f"Cantidad ({unidad_texto}):", min_value=0.0, value=300.0, step=1.0)
            st.caption(f"💡 El valor nutricional es por 100ml. Su cantidad total es **{cantidad_input:.0f} ml**.")
        else:
            cantidad_input = st.number_input(f"Cantidad ({unidad_texto}):", min_value=0.0, value=100.0, step=1.0)
            st.caption(f"💡 El valor nutricional es por 100g. Su cantidad total es **{cantidad_input:.0f} gramos**.")
            
        if st.button(f"✅ Agregar/Actualizar {alimento_seleccionado}"):
            if cantidad_input > 0.0:
                st.session_state.comidas_dia[TIPO_COMIDA][alimento_seleccionado] = cantidad_input
                st.success(f"**{alimento_seleccionado}** actualizado con **{cantidad_input:.0f}** {unidad_texto}.")
            else:
                st.session_state.comidas_dia[TIPO_COMIDA].pop(alimento_seleccionado, None)
                st.warning(f"**{alimento_seleccionado}** eliminado de la lista.")

    # Mostrar resumen de la comida actual
    st.markdown("---")
    st.subheader(f"📋 Resumen de {TIPO_COMIDA}")
    if st.session_state.comidas_dia[TIPO_COMIDA]:
        df_comida_actual = pd.DataFrame([
            {'Alimento': k, 'Cantidad': f"{v:.0f}{'ml' if k in BEBIDAS else 'g'}"} 
            for k, v in st.session_state.comidas_dia[TIPO_COMIDA].items()
        ])
        st.dataframe(df_comida_actual, hide_index=True, use_container_width=True)
    else:
        st.info(f"Aún no hay alimentos registrados para {TIPO_COMIDA}.")

with tab2:
    st.header("2. Evaluación Integrada y Predicción ML")
    st.markdown("Aquí se consolidan los datos de las 24h de comida, los signos vitales y se ejecutan las predicciones de Machine Learning.")
    
    # Botón de cálculo y evaluación
    if st.button("🔄 Ejecutar Evaluación Completa (ML + Nutrición)"):
        
        # 1. ACUMULACIÓN DE DATOS NUTRICIONALES
        total_24h = {
            'calorias_total_24h': 0, 'cho_total_24h': 0, 'ig_ponderado_total': 0, 
            'carga_total_24h': 0, 'alcohol_total_24h': 0, 'indice_alcoholico_24h': 0,
            'detalles_24h': []
        }
        
        comidas_evaluadas = {}
        for tipo, componentes in st.session_state.comidas_dia.items():
            resultados = evaluar_riesgo_comida(componentes, ALIMENTOS_DF)
            acumular_totales_24h_completo(resultados, total_24h)
            comidas_evaluadas[tipo] = {'resultados': resultados}
            
        # Calcular el IG promedio de la comida total
        ig_comida_24h = total_24h['ig_ponderado_total'] / total_24h['cho_total_24h'] if total_24h['cho_total_24h'] > 0 else 0

        resultados_24h = {
            'calorias_total': total_24h['calorias_total_24h'],
            'cho_total': total_24h['cho_total_24h'],
            'ig_comida': ig_comida_24h,
            'carga_total': total_24h['carga_total_24h'],
            'alcohol_total_ml': total_24h['alcohol_total_24h'],
            'indice_alcoholico': total_24h['indice_alcoholico_24h']
        }
        
        # 2. PREPARACIÓN DE DATOS PARA ML
        patient_data_ml = {
            'age': st.session_state.signos_vitales['age'],
            'pa_sistolica': st.session_state.signos_vitales['pa_sistolica'],
            'pa_diastolica': st.session_state.signos_vitales['pa_diastolica'],
            'fc': st.session_state.signos_vitales['fc'],
            'sato2': st.session_state.signos_vitales['sato2'],
            'is_diabetic': st.session_state.signos_vitales['is_diabetic'],
            'calorias': resultados_24h['calorias_total'],
            'cho_total': resultados_24h['cho_total'],
            'carga_glucemica': resultados_24h['carga_total'],
            'alcohol': resultados_24h['indice_alcoholico']
        }
        
        # 3. PREDICCIÓN CON MACHINE LEARNING
        ml_result = ML_SYSTEM.predict_risk_ml(patient_data_ml)

        # 4. EVALUACIÓN DE RIESGOS COMBINADOS
        evaluacion_24h_completa = evaluar_riesgos_con_signos_vitales(resultados_24h, st.session_state.signos_vitales, evaluacion_sv)
        diagnosticos_hipoteticos = generar_diagnostico_hipotetico(st.session_state.signos_vitales, resultados_24h)
        
        
        # --- MOSTRAR RESULTADOS EN STREAMLIT ---
        
        st.subheader("🤖 Análisis de Riesgo y Nutrición (Machine Learning)")
        
        col_ml1, col_ml2, col_ml3 = st.columns(3)
        
        # COL 1: Riesgo Predicho por ML
        col_ml1.metric(
            "Riesgo Predicho por ML", 
            ml_result['predicted_risk'], 
            f"Confianza: {ml_result['confidence']:.1%}"
        )
        
        # COL 2: Calorías Recomendadas por ML
        col_ml2.metric(
            "Calorías Recomendadas (ML)", 
            f"{ml_result['recommended_calories']:.0f} kcal",
            f"Consumidas: {resultados_24h['calorias_total']:.0f} kcal"
        )
        
        # COL 3: Carga Glucémica Total
        col_ml3.metric(
            "Carga Glucémica Total", 
            f"{resultados_24h['carga_total']:.1f}",
            f"IG Promedio: {resultados_24h['ig_comida']:.0f}"
        )
        
        st.markdown("---")
        
        # Gráfico de Probabilidades
        st.subheader("📊 Probabilidades de Riesgo (Clasificador ML)")
        risk_df = pd.DataFrame(ml_result['risk_probabilities'].items(), columns=['Nivel de Riesgo', 'Probabilidad'])
        st.bar_chart(risk_df.set_index('Nivel de Riesgo'))
        
        st.markdown("---")
        
        # Resumen Nutricional Detallado
        st.subheader("🍽️ Resumen Nutricional Consolidado (24h)")
        
        col_n1, col_n2, col_n3 = st.columns(3)
        col_n1.metric("Carbohidratos Totales", f"{resultados_24h['cho_total']:.1f} g")
        col_n2.metric("Alcohol Total (puro)", f"{resultados_24h['indice_alcoholico']:.1f} g")
        
        # Mostrar tabla de detalles de todas las comidas
        st.markdown("#### Detalle por Alimento Consumido")
        df_detalles = pd.DataFrame(total_24h['detalles_24h']).round(1)
        st.dataframe(df_detalles, hide_index=True, use_container_width=True)

        st.markdown("---")
        
        # Diagnósticos y Recomendaciones
        col_dx, col_rec = st.columns(2)
        
        with col_dx:
            st.subheader("🏥 Diagnósticos Hipotéticos (Educativo)")
            st.warning("⚠️ **AVISO:** SOLO FINES EDUCATIVOS - NO REEMPLAZA EVALUACIÓN MÉDICA")
            for i, dx in enumerate(diagnosticos_hipoteticos):
                st.write(f"**{i+1}.** {dx}")

        with col_rec:
            st.subheader("💡 Riesgos y Recomendaciones")
            if evaluacion_24h_completa['riesgos']:
                st.markdown("**🚨 Riesgos Identificados:**")
                for riesgo in evaluacion_24h_completa['riesgos']:
                    st.error(f"• {riesgo}")
            else:
                st.success("🟢 No se detectaron riesgos severos.")

            st.markdown("**✅ Recomendaciones:**")
            for rec in evaluacion_24h_completa['recomendaciones']:
                st.info(f"• {rec}")
    else:
        st.warning("⚠️ Presione el botón **'Ejecutar Evaluación Completa'** para analizar los datos.")

with tab3:
    st.header("3. Base de Datos de Alimentos")
    st.markdown("Datos nutricionales de referencia por 100g/100ml utilizados en el cálculo de la ingesta.")
    st.dataframe(ALIMENTOS_DF, use_container_width=True)

