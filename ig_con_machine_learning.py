import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Crear base de datos de alimentos (EXACTAMENTE la lista original del usuario)
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

    def generate_synthetic_training_data(self, n_samples=1000):
        """Genera datos sintéticos para entrenamiento del modelo"""
        np.random.seed(42)

        # Generar datos sintéticos realistas
        data = []

        for i in range(n_samples):
            # Variables básicas
            age = np.random.randint(18, 80)
            is_diabetic = np.random.choice([0, 1], p=[0.85, 0.15])

            # Signos vitales correlacionados con edad y estado
            pa_sistolica = np.random.normal(120 + age*0.5, 15)
            pa_diastolica = np.random.normal(80 + age*0.2, 10)
            fc = np.random.normal(70 + np.random.normal(0, 10), 12)
            sato2 = np.random.normal(97, 2)

            # Datos nutricionales
            calorias = np.random.normal(2200, 500)
            cho_total = np.random.normal(300, 80)
            carga_glucemica = np.random.normal(60, 20)
            alcohol = np.random.exponential(5) if np.random.random() < 0.3 else 0

            # LÓGICA PARA CLASIFICAR RIESGO (más sofisticada)
            risk_score = 0

            # Factores de riesgo cardiovascular
            if pa_sistolica >= 140 or pa_diastolica >= 90:
                risk_score += 2
            if pa_sistolica >= 180 or pa_diastolica >= 110:
                risk_score += 3

            # Factores nutricionales
            if calorias > 3000:
                risk_score += 2
            if carga_glucemica > 80:
                risk_score += 1.5
            if alcohol > 40:
                risk_score += 2

            # Factores de edad y diabetes
            if age > 65:
                risk_score += 0.5
            if is_diabetic and carga_glucemica > 50:
                risk_score += 2

            # Factores protectores
            if fc >= 60 and fc <= 90 and sato2 >= 95:
                risk_score -= 0.5

            # Clasificación final
            if risk_score <= 0.5:
                risk_level = 'NORMAL'
            elif risk_score <= 2:
                risk_level = 'MODERADO'
            elif risk_score <= 4:
                risk_level = 'ALTO'
            else:
                risk_level = 'CRITICO'

            data.append({
                'age': age,
                'pa_sistolica': pa_sistolica,
                'pa_diastolica': pa_diastolica,
                'fc': fc,
                'sato2': sato2,
                'is_diabetic': is_diabetic,
                'calorias': calorias,
                'cho_total': cho_total,
                'carga_glucemica': carga_glucemica,
                'alcohol': alcohol,
                'risk_level': risk_level,
                'recommended_calories': max(1200, min(3000, 2000 + np.random.normal(0, 200)))
            })

        return pd.DataFrame(data)

    def train_models(self):
        """Entrena los modelos de ML"""
        print("🤖 Generando datos de entrenamiento sintéticos...")
        df = self.generate_synthetic_training_data(1500)

        # Preparar features
        feature_cols = ['age', 'pa_sistolica', 'pa_diastolica', 'fc', 'sato2',
                       'is_diabetic', 'calorias', 'cho_total', 'carga_glucemica', 'alcohol']

        X = df[feature_cols]
        y_risk = df['risk_level']
        y_calories = df['recommended_calories']

        # Dividir datos
        X_train, X_test, y_risk_train, y_risk_test, y_cal_train, y_cal_test = train_test_split(
            X, y_risk, y_calories, test_size=0.2, random_state=42
        )

        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Entrenar clasificador de riesgo
        print("🎯 Entrenando clasificador de riesgo cardiovascular...")
        self.risk_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.risk_classifier.fit(X_train_scaled, y_risk_train)

        # Entrenar predictor de calorías
        print("📊 Entrenando predictor de calorías recomendadas...")
        self.calorie_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.calorie_predictor.fit(X_train_scaled, y_cal_train)

        # Evaluación
        y_risk_pred = self.risk_classifier.predict(X_test_scaled)
        y_cal_pred = self.calorie_predictor.predict(X_test_scaled)

        print("\n📈 EVALUACIÓN DEL MODELO:")
        print("="*40)
        print("🎯 Clasificación de Riesgo:")
        print(classification_report(y_risk_test, y_risk_pred))

        cal_mse = mean_squared_error(y_cal_test, y_cal_pred)
        print(f"📊 Predicción de Calorías - MSE: {cal_mse:.2f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.risk_classifier.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n🏆 IMPORTANCIA DE VARIABLES (para clasificación de riesgo):")
        for _, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")

        self.is_trained = True
        print("\n✅ Modelos entrenados exitosamente!")

    def predict_risk_ml(self, patient_data):
        """Predice el riesgo usando ML"""
        if not self.is_trained:
            print("⚠️ Modelos no entrenados. Entrenando ahora...")
            self.train_models()

        # Preparar datos del paciente
        features = np.array([[
            patient_data['age'],
            patient_data['pa_sistolica'],
            patient_data['pa_diastolica'],
            patient_data['fc'],
            patient_data['sato2'],
            1 if patient_data['is_diabetic'] else 0,
            patient_data['calorias'],
            patient_data['cho_total'],
            patient_data['carga_glucemica'],
            patient_data['alcohol']
        ]])

        # Escalar
        features_scaled = self.scaler.transform(features)

        # Predicciones
        risk_pred = self.risk_classifier.predict(features_scaled)[0]
        risk_proba = self.risk_classifier.predict_proba(features_scaled)[0]
        calories_pred = self.calorie_predictor.predict(features_scaled)[0]

        # Obtener probabilidades por clase
        classes = self.risk_classifier.classes_
        risk_probabilities = dict(zip(classes, risk_proba))

        return {
            'predicted_risk': risk_pred,
            'risk_probabilities': risk_probabilities,
            'recommended_calories': calories_pred,
            'confidence': max(risk_proba)
        }

    def explain_prediction(self, patient_data, ml_result):
        """Explica la predicción del modelo"""
        print(f"\n🤖 ANÁLISIS CON MACHINE LEARNING:")
        print("="*45)

        print(f"🎯 Riesgo Predicho: {ml_result['predicted_risk']}")
        print(f"🎲 Confianza: {ml_result['confidence']:.1%}")
        print(f"🍽️ Calorías Recomendadas: {ml_result['recommended_calories']:.0f} kcal")

        print(f"\n📊 Probabilidades por Categoría:")
        for risk_level, prob in sorted(ml_result['risk_probabilities'].items()):
            bar = "█" * int(prob * 20)
            print(f"   {risk_level:8}: {prob:.1%} {bar}")

        # Análisis de factores críticos
        critical_factors = []
        if patient_data['pa_sistolica'] >= 140:
            critical_factors.append("Presión sistólica elevada")
        if patient_data['carga_glucemica'] > 80:
            critical_factors.append("Carga glucémica muy alta")
        if patient_data['alcohol'] > 40:
            critical_factors.append("Consumo excesivo de alcohol")
        if patient_data['is_diabetic'] and patient_data['carga_glucemica'] > 50:
            critical_factors.append("Diabetes + alta carga glucémica")

        if critical_factors:
            print(f"\n⚠️ Factores de Riesgo Detectados por ML:")
            for factor in critical_factors:
                print(f"   • {factor}")

        return ml_result

class MLSignosVitales:
    """Sistema de signos vitales con ML integrado"""

    def __init__(self):
        self.ml_system = MLNutritionSystem()

    def registrar_signos_vitales_ml(self):
        """Registra signos vitales para análisis ML"""
        print("\n🩺 REGISTRO DE SIGNOS VITALES BASALES")
        print("=" * 40)

        try:
            edad = int(input("Edad del paciente: "))
            pa_sistolica = float(input("Presión arterial sistólica (mmHg): "))
            pa_diastolica = float(input("Presión arterial diastólica (mmHg): "))
            fc = float(input("Frecuencia cardíaca (lpm): "))
            fr = float(input("Frecuencia respiratoria (rpm): "))
            sato2 = float(input("Saturación de oxígeno (%): "))

            es_diabetico = input("¿Es diabético? (s/n): ").lower().strip() in ['s', 'si', 'sí', 'yes']

            return {
                'age': edad,
                'edad': edad,
                'PA_sistolica': pa_sistolica,
                'pa_sistolica': pa_sistolica,
                'PA_diastolica': pa_diastolica,
                'pa_diastolica': pa_diastolica,
                'FC': fc,
                'fc': fc,
                'FR': fr,
                'SatO2': sato2,
                'sato2': sato2,
                'is_diabetic': es_diabetico,
                'es_diabetico': es_diabetico,
                'timestamp': datetime.now()
            }
        except ValueError:
            print("❌ Error en el ingreso de datos.")
            return None

    def evaluar_signos_vitales(self, signos):
        """Evalúa los signos vitales usando reglas tradicionales"""
        if not signos:
            return None

        riesgos = []
        nivel_riesgo = "NORMAL"

        # Evaluar presión arterial
        if signos['PA_sistolica'] >= 180 or signos['PA_diastolica'] >= 110:
            riesgos.append("Crisis hipertensiva")
            nivel_riesgo = "CRÍTICO"
        elif signos['PA_sistolica'] >= 140 or signos['PA_diastolica'] >= 90:
            riesgos.append("Hipertensión arterial")
            nivel_riesgo = "ALTO" if nivel_riesgo != "CRÍTICO" else nivel_riesgo

        # Evaluar frecuencia cardíaca
        if signos['FC'] > 100:
            riesgos.append("Taquicardia")
            nivel_riesgo = "MODERADO" if nivel_riesgo == "NORMAL" else nivel_riesgo
        elif signos['FC'] < 60:
            riesgos.append("Bradicardia")
            nivel_riesgo = "MODERADO" if nivel_riesgo == "NORMAL" else nivel_riesgo

        # Evaluar saturación
        if signos['SatO2'] < 90:
            riesgos.append("Hipoxemia severa")
            nivel_riesgo = "CRÍTICO"
        elif signos['SatO2'] < 95:
            riesgos.append("Hipoxemia leve")
            nivel_riesgo = "ALTO" if nivel_riesgo not in ["CRÍTICO"] else nivel_riesgo

        color_map = {
            "NORMAL": "🟢",
            "MODERADO": "🟡",
            "ALTO": "🟠",
            "CRÍTICO": "🔴"
        }

        return {
            'riesgos': riesgos,
            'nivel_riesgo': nivel_riesgo,
            'color_riesgo': color_map.get(nivel_riesgo, "⚪")
        }

    def generar_diagnostico_hipotetico(self, signos, evaluacion, resultados_nutricionales):
        """Genera diagnósticos hipotéticos educativos"""
        diagnosticos = []

        if evaluacion['nivel_riesgo'] == "CRÍTICO":
            diagnosticos.append("Crisis hipertensiva - Emergencia médica")

        if signos['es_diabetico'] and resultados_nutricionales['carga_total'] > 50:
            diagnosticos.append("Descompensación diabética por sobrecarga de CHO")

        if resultados_nutricionales.get('indice_alcoholico', 0) > 30:
            diagnosticos.append("Intoxicación etílica aguda")

        if signos['PA_sistolica'] > 140 and resultados_nutricionales['calorias_total'] > 2800:
            diagnosticos.append("Síndrome metabólico")

        if not diagnosticos:
            diagnosticos.append("Parámetros dentro de límites relativamente normales")

        return diagnosticos

def evaluar_riesgo_comida(componentes, base_alimentos):
    """Evalúa el riesgo nutricional de una comida"""
    if not componentes:
        return {
            'calorias_total': 0,
            'cho_total': 0,
            'carga_total': 0,
            'ig_comida': 0,
            'alcohol_total_ml': 0,
            'indice_alcoholico': 0,
            'detalles': []
        }

    calorias_total = 0
    cho_total = 0
    ig_ponderado = 0
    alcohol_total_ml = 0
    indice_alcoholico = 0
    detalles = []

    for alimento, cantidad in componentes.items():
        # Buscar el alimento en la base
        fila = base_alimentos[base_alimentos['alimento'] == alimento]

        if not fila.empty:
            fila = fila.iloc[0]

            # Calcular por porción
            if alimento in ['Cerveza', 'Vino tinto', 'Jugo en caja', 'Gaseosa coca cola',
                           'Cocido (bebida)', 'Mate', 'Leche']:
                # Para líquidos, cantidad está en ml
                factor = cantidad / 100  # ml a 100ml
            else:
                # Para sólidos, cantidad está en gramos
                factor = cantidad / 100  # gramos a 100g

            cals = fila['calorias_100g'] * factor
            cho = fila['cho_100g'] * factor
            ig = fila['ig']

            calorias_total += cals
            cho_total += cho

            if cho > 0:
                ig_ponderado += ig * cho

            # Calcular alcohol
            if fila['alcohol_grados'] > 0:
                # Fórmula: ml × grados × 0.789 (densidad etanol) / 100
                alcohol_puro = cantidad * fila['alcohol_grados'] * 0.789 / 100
                indice_alcoholico += alcohol_puro
                alcohol_total_ml += cantidad

            detalles.append({
                'alimento': alimento,
                'cantidad': cantidad,
                'calorias': cals,
                'cho': cho,
                'ig': ig
            })

    # Calcular IG promedio de la comida
    ig_comida = ig_ponderado / cho_total if cho_total > 0 else 0

    # Calcular carga glucémica
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

def registrar_comida_especifica_24h(base_alimentos, tipo_comida):
    """Registra alimentos para comida específica en seguimiento 24h"""
    componentes = {}
    bebidas = ['Cerveza', 'Vino tinto', 'Jugo en caja', 'Gaseosa coca cola', 'Cocido (bebida)', 'Mate', 'Leche']

    print(f"\n📋 REGISTRO DE ALIMENTOS PARA {tipo_comida.upper()}:")
    primera_vez = True

    while True:
        if primera_vez:
            print("💡 Escriba el NÚMERO del alimento + ENTER")
            print("🔍 Escriba 'buscar' para buscar por nombre")
            print("📝 Escriba 'n' o 'no' para terminar")

            # Mostrar TODA la lista completa siempre
            print(f"\n📋 TODOS LOS ALIMENTOS DISPONIBLES ({len(base_alimentos)} total):")
            print("=" * 60)
            for i, alimento in enumerate(base_alimentos['alimento'], 1):
                calorias = base_alimentos.iloc[i-1]['calorias_100g']
                cho = base_alimentos.iloc[i-1]['cho_100g']
                ig = base_alimentos.iloc[i-1]['ig']
                if alimento in bebidas:
                    unidad = "(vasos 300ml)"
                else:
                    unidad = "(gramos)"
                print(f"{i:2d}. {alimento:<25} {unidad:<15} Cal:{calorias:3.0f} CHO:{cho:4.1f}g IG:{ig:2.0f}")
            print("=" * 60)
            entrada = input(f"\nAlimento para {tipo_comida} (n=no, 'buscar'): ").strip()
            primera_vez = False
        else:
            entrada = input(f"\n¿Consumió algo más en {tipo_comida}? (n=no, 'buscar'): ").strip()

        # Opción de buscar
        if entrada.lower() == 'buscar':
            termino = input("Escriba el nombre del alimento a buscar: ").strip().lower()
            coincidencias = []

            for idx, alimento in enumerate(base_alimentos['alimento']):
                if termino in alimento.lower():
                    coincidencias.append((idx + 1, alimento))

            if coincidencias:
                print("🔍 Alimentos encontrados:")
                for num, nombre in coincidencias:
                    if nombre in bebidas:
                        print(f"   {num}. {nombre} (vasos de 300ml)")
                    else:
                        print(f"   {num}. {nombre} (gramos)")
            else:
                print("❌ No se encontraron coincidencias")
            continue

        # Terminar
        if entrada.lower() in ['n', 'no']:
            break

        try:
            seleccion = int(entrada)
            if 1 <= seleccion <= len(base_alimentos):
                alimento = base_alimentos.iloc[seleccion-1]['alimento']

                if alimento in bebidas:
                    print(f"\n🥤 {alimento} - Ingreso por VASOS:")
                    print("💡 1 vaso = 300ml (porción estándar)")
                    vasos = float(input(f"Número de vasos de {alimento}: "))
                    cantidad = vasos * 300  # Convertir a ml
                    print(f"📊 Equivale a {cantidad:.0f}ml")
                else:
                    cantidad = float(input(f"Cantidad en gramos de {alimento}: "))

                if cantidad > 0:
                    if alimento in componentes:
                        print(f"⚠️ {alimento} ya está registrado con {componentes[alimento]:.0f}")
                        respuesta = input("¿Desea sumar a la cantidad existente? (s/n): ").lower().strip()
                        if respuesta in ['s', 'si', 'sí', 'yes', 'y']:
                            componentes[alimento] += cantidad
                            print(f"✅ Cantidad actualizada: {componentes[alimento]:.0f}")
                        else:
                            componentes[alimento] = cantidad
                            print(f"✅ Cantidad reemplazada: {cantidad:.0f}")
                    else:
                        componentes[alimento] = cantidad

                    if alimento in bebidas:
                        total_vasos = componentes[alimento] / 300
                        print(f"✅ Total registrado: {total_vasos:.1f} vasos ({componentes[alimento]:.0f}ml) de {alimento}")
                    else:
                        print(f"✅ Total registrado: {componentes[alimento]:.0f}g de {alimento}")
                else:
                    print("❌ La cantidad debe ser mayor a 0")
            else:
                print(f"❌ Número fuera del rango (1-{len(base_alimentos)})")

        except ValueError:
            print("❌ Ingrese un número válido, 'buscar' o 'n'")

    # Mostrar resumen final de la comida
    if componentes:
        print(f"\n📋 RESUMEN DE {tipo_comida.upper()}:")
        print("=" * 40)
        for alimento, cantidad in componentes.items():
            if alimento in bebidas:
                vasos = cantidad / 300
                print(f"   • {alimento}: {vasos:.1f} vasos ({cantidad:.0f}ml)")
            else:
                print(f"   • {alimento}: {cantidad:.0f}g")
        print("=" * 40)
    else:
        print(f"⚠️ No se registraron alimentos para {tipo_comida}")

    return componentes

def acumular_totales_24h_completo(resultados, total_24h):
    """Acumula los totales de 24 horas incluyendo índice alcohólico"""

    # Acumular CHO e IG ponderado
    if resultados['cho_total'] > 0:
        total_24h['ig_ponderado_total'] += resultados['ig_comida'] * resultados['cho_total']
        total_24h['cho_total_24h'] += resultados['cho_total']

    total_24h['carga_total_24h'] += resultados['carga_total']
    total_24h['calorias_total_24h'] += resultados['calorias_total']
    total_24h['alcohol_total_24h'] += resultados.get('alcohol_total_ml', 0)
    total_24h['indice_alcoholico_24h'] += resultados.get('indice_alcoholico', 0)

    # Agregar todos los detalles
    total_24h['detalles_24h'].extend(resultados['detalles'])

def evaluar_riesgos_con_signos_vitales(resultados_24h, signos_vitales, evaluacion_sv):
    """Evalúa riesgos combinando nutrición y signos vitales"""
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
    if (signos_vitales and signos_vitales.get('es_diabetico') and
        resultados_24h['carga_total'] > 50):
        riesgos.append("Riesgo de descompensación diabética")
        recomendaciones.append("Control glucémico inmediato")

    if not recomendaciones:
        recomendaciones.append("Mantener patrón alimentario actual")

    return {
        'riesgos': riesgos,
        'recomendaciones': recomendaciones
    }

def mostrar_evaluacion_24h_completa_con_ml(comidas_dia, resultados_24h, evaluacion_24h,
                                          signos_vitales=None, evaluacion_sv=None,
                                          diagnosticos=None, ml_result=None):
    """Muestra evaluación completa de 24 horas con ML y signos vitales"""

    print("\n" + "="*70)
    print("⏰ EVALUACIÓN MÉDICA INTEGRAL CON MACHINE LEARNING - 24 HORAS")
    print("="*70)

    # SECCIÓN 1: RESUMEN NUTRICIONAL 24H
    print(f"\n📊 RESUMEN NUTRICIONAL 24H:")
    print(f"   Total de comidas registradas: {len(comidas_dia)}")
    print(f"   Calorías totales: {resultados_24h['calorias_total']:.0f} kcal")
    print(f"   Carbohidratos totales: {resultados_24h['cho_total']:.1f}g")
    print(f"   IG promedio 24h: {resultados_24h['ig_comida']:.0f}")
    print(f"   Carga glucémica total: {resultados_24h['carga_total']:.1f}")
    if resultados_24h.get('indice_alcoholico', 0) > 0:
        print(f"   Índice alcohólico 24h: {resultados_24h['indice_alcoholico']:.1f}g (alcohol puro)")
        print(f"   Alcohol total consumido: {resultados_24h.get('alcohol_total_ml', 0):.0f}ml")

    # SECCIÓN 2: DISTRIBUCIÓN POR COMIDAS
    print(f"\n🍽️ DISTRIBUCIÓN POR COMIDAS:")
    emojis = {"Desayuno": "🌅", "Media Mañana": "☕", "Almuerzo": "🍽️",
              "Merienda": "🧉", "Cena": "🌙", "Desayuno del día siguiente": "🌅+1"}

    for tipo, datos in comidas_dia.items():
        emoji = emojis.get(tipo, "🍽️")
        res = datos['resultados']
        print(f"   {emoji} {tipo}:")
        print(f"      Cal: {res['calorias_total']:.0f} | CHO: {res['cho_total']:.1f}g | CG: {res['carga_total']:.1f}")
        if res.get('indice_alcoholico', 0) > 0:
            print(f"      Alcohol: {res['indice_alcoholico']:.1f}g puro")

    # SECCIÓN 3: SIGNOS VITALES
    if signos_vitales:
        print(f"\n🩺 SIGNOS VITALES BASALES:")
        print(f"   PA: {signos_vitales['PA_sistolica']:.0f}/{signos_vitales['PA_diastolica']:.0f} mmHg")
        print(f"   FC: {signos_vitales['FC']:.0f} lpm")
        print(f"   FR: {signos_vitales['FR']:.0f} rpm")
        print(f"   SatO2: {signos_vitales['SatO2']:.0f}%")

        if signos_vitales.get('edad'):
            print(f"   Edad: {signos_vitales['edad']} años")
        if signos_vitales.get('es_diabetico'):
            print(f"   Diabético: {'Sí' if signos_vitales['es_diabetico'] else 'No'}")

        if evaluacion_sv:
            print(f"   Estado vital: {evaluacion_sv['nivel_riesgo']} {evaluacion_sv['color_riesgo']}")

    # SECCIÓN 4: ANÁLISIS CON MACHINE LEARNING
    if ml_result:
        print(f"\n🤖 ANÁLISIS CON MACHINE LEARNING:")
        print("="*45)

        print(f"🎯 Riesgo Predicho por ML: {ml_result['predicted_risk']}")
        print(f"🎲 Confianza del Modelo: {ml_result['confidence']:.1%}")
        print(f"🍽️ Calorías Recomendadas por ML: {ml_result['recommended_calories']:.0f} kcal")

        print(f"\n📊 Probabilidades por Categoría de Riesgo:")
        for risk_level, prob in sorted(ml_result['risk_probabilities'].items()):
            bar = "█" * int(prob * 20)
            print(f"   {risk_level:8}: {prob:.1%} {bar}")

    # SECCIÓN 5: COMPARACIÓN ML vs REGLAS TRADICIONALES
    if ml_result and evaluacion_sv:
        print(f"\n🔄 COMPARACIÓN: ML vs EVALUACIÓN TRADICIONAL")
        print("="*50)

        # Evaluación tradicional simplificada
        if signos_vitales['PA_sistolica'] >= 140 or signos_vitales['PA_diastolica'] >= 90:
            traditional_risk = "ALTO"
        elif resultados_24h['carga_total'] > 80:
            traditional_risk = "MODERADO"
        else:
            traditional_risk = "NORMAL"

        print(f"   Sistema Tradicional: {traditional_risk}")
        print(f"   Machine Learning: {ml_result['predicted_risk']} (conf: {ml_result['confidence']:.1%})")

        if ml_result['predicted_risk'] != traditional_risk:
            print(f"   ⚡ ML detectó patrones que las reglas tradicionales no capturaron")
        else:
            print(f"   ✅ Ambos sistemas coinciden en la evaluación")

    # SECCIÓN 6: ANÁLISIS DETALLADO POR PARÁMETROS
    print(f"\n📈 ANÁLISIS DETALLADO:")

    # Análisis calórico
    if resultados_24h['calorias_total'] > 3000:
        print(f"   🔴 CALORÍAS: Exceso severo ({resultados_24h['calorias_total']:.0f} kcal)")
    elif resultados_24h['calorias_total'] > 2500:
        print(f"   🟡 CALORÍAS: Exceso moderado ({resultados_24h['calorias_total']:.0f} kcal)")
    elif resultados_24h['calorias_total'] < 1200:
        print(f"   🟡 CALORÍAS: Posible déficit ({resultados_24h['calorias_total']:.0f} kcal)")
    else:
        print(f"   🟢 CALORÍAS: Dentro del rango ({resultados_24h['calorias_total']:.0f} kcal)")

    # Análisis de carbohidratos
    if resultados_24h['cho_total'] > 300:
        print(f"   🔴 CHO: Exceso ({resultados_24h['cho_total']:.1f}g)")
    elif resultados_24h['cho_total'] < 130:
        print(f"   🟡 CHO: Bajo ({resultados_24h['cho_total']:.1f}g)")
    else:
        print(f"   🟢 CHO: Adecuado ({resultados_24h['cho_total']:.1f}g)")

    # Análisis de carga glucémica
    if resultados_24h['carga_total'] > 80:
        print(f"   🔴 CG: Muy alta ({resultados_24h['carga_total']:.1f})")
    elif resultados_24h['carga_total'] > 50:
        print(f"   🟡 CG: Alta ({resultados_24h['carga_total']:.1f})")
    else:
        print(f"   🟢 CG: Aceptable ({resultados_24h['carga_total']:.1f})")

    # Análisis alcohólico
    if resultados_24h.get('indice_alcoholico', 0) > 0:
        alcohol_gramos = resultados_24h['indice_alcoholico']
        if alcohol_gramos > 40:  # >40g = riesgo alto
            print(f"   🔴 ALCOHOL: Consumo peligroso ({alcohol_gramos:.1f}g puro)")
        elif alcohol_gramos > 20:  # 20-40g = moderado
            print(f"   🟡 ALCOHOL: Consumo moderado-alto ({alcohol_gramos:.1f}g puro)")
        else:
            print(f"   🟢 ALCOHOL: Consumo bajo ({alcohol_gramos:.1f}g puro)")

    # SECCIÓN 7: DIAGNÓSTICOS HIPOTÉTICOS
    if diagnosticos:
        print(f"\n🏥 DIAGNÓSTICOS DIFERENCIALES (EDUCATIVOS):")
        print("   ⚠️ SOLO FINES EDUCATIVOS - NO REEMPLAZA EVALUACIÓN MÉDICA")
        for i, dx in enumerate(diagnosticos, 1):
            print(f"   {i}. {dx}")

    # SECCIÓN 8: RIESGOS Y RECOMENDACIONES
    if evaluacion_24h['riesgos']:
        print(f"\n🚨 RIESGOS IDENTIFICADOS EN 24H:")
        for riesgo in evaluacion_24h['riesgos']:
            print(f"   • {riesgo}")

    if evaluacion_24h['recomendaciones']:
        print(f"\n💡 RECOMENDACIONES:")
        for rec in evaluacion_24h['recomendaciones']:
            print(f"   ✓ {rec}")

    # SECCIÓN 9: EVALUACIÓN FINAL INTEGRAL
    print(f"\n🏥 EVALUACIÓN MÉDICA FINAL INTEGRADA:")

    # Determinar riesgo general combinando ML y reglas
    riesgo_nutricional_alto = (resultados_24h['calorias_total'] > 3000 or
                              resultados_24h['carga_total'] > 80 or
                              resultados_24h.get('indice_alcoholico', 0) > 40)
    riesgo_vital_critico = evaluacion_sv and evaluacion_sv['nivel_riesgo'] == 'CRÍTICO'
    riesgo_vital_alto = evaluacion_sv and evaluacion_sv['nivel_riesgo'] == 'ALTO'
    riesgo_ml_critico = ml_result and ml_result['predicted_risk'] == 'CRITICO'
    riesgo_ml_alto = ml_result and ml_result['predicted_risk'] == 'ALTO'

    if riesgo_vital_critico or riesgo_ml_critico:
        print("   🚨 RIESGO CRÍTICO: Requiere atención médica INMEDIATA")
        print("   📞 Activar protocolo de emergencia")
        print("   🚑 Considerar traslado urgente a centro médico")
    elif (riesgo_vital_alto or riesgo_ml_alto or
          (riesgo_nutricional_alto and signos_vitales and signos_vitales.get('es_diabetico'))):
        print("   🔴 ALTO RIESGO: Supervisión médica urgente requerida")
        print("   📋 Monitoreo continuo de signos vitales cada 2-4 horas")
        print("   💊 Evaluar necesidad de medicación o ajuste de dosis")
    elif (riesgo_nutricional_alto or (evaluacion_sv and evaluacion_sv['nivel_riesgo'] in ['MODERADO', 'ALTO']) or
          (ml_result and ml_result['predicted_risk'] == 'MODERADO')):
        print("   🟡 RIESGO MODERADO: Control médico recomendado en 24-48h")
        print("   📊 Seguimiento de glucemia, PA y peso")
    else:
        print("   🟢 RIESGO BAJO: Parámetros dentro de límites aceptables")
        print("   📝 Mantener controles de rutina")

    # Recomendaciones específicas de ML
    if ml_result:
        print(f"\n🤖 RECOMENDACIONES ESPECÍFICAS DEL ML:")
        calorie_diff = ml_result['recommended_calories'] - resultados_24h['calorias_total']
        if abs(calorie_diff) > 200:
            if calorie_diff > 0:
                print(f"   📈 Considerar aumentar ingesta calórica en {calorie_diff:.0f} kcal")
            else:
                print(f"   📉 Considerar reducir ingesta calórica en {abs(calorie_diff):.0f} kcal")
        else:
            print(f"   ✅ Ingesta calórica dentro del rango recomendado por ML")

    # Proyección de eliminación de alcohol
    if resultados_24h.get('indice_alcoholico', 0) > 0:
        horas_eliminacion = resultados_24h['indice_alcoholico'] / 7  # ~7g/hora eliminación
        print(f"\n🍷 METABOLISMO DEL ALCOHOL:")
        print(f"   Tiempo estimado de eliminación: {horas_eliminacion:.1f} horas")
        if horas_eliminacion > 12:
            print("   ⚠️ Eliminación prolongada - Riesgo de efectos residuales")

    print(f"\n📚 RECORDATORIO IMPORTANTE:")
    print(f"   Esta evaluación es EXCLUSIVAMENTE para fines educativos")
    print(f"   El análisis ML es un modelo experimental de demostración")
    print(f"   Ante cualquier síntoma, malestar o duda, consulte inmediatamente")
    print(f"   con un profesional médico calificado")

def seguimiento_24_horas_con_ml(base_alimentos, sistema_ml):
    """Seguimiento completo de 24 horas con evaluación ML"""

    print("\n⏰ SEGUIMIENTO INTEGRAL DE 24 HORAS CON MACHINE LEARNING")
    print("=" * 60)
    print("📝 Registro completo: alimentación + signos vitales + análisis ML")
    print("🤖 Evaluación médica con inteligencia artificial")
    print("⏱️  Seguimiento de 24 horas reales: desde desayuno hasta desayuno siguiente")

    # Entrenar modelos ML si es necesario
    if not sistema_ml.ml_system.is_trained:
        print("\n🤖 Entrenando modelos de Machine Learning...")
        sistema_ml.ml_system.train_models()

    # Registrar signos vitales al inicio
    print("\n🩺 SIGNOS VITALES BASALES")
    signos_vitales = sistema_ml.registrar_signos_vitales_ml()
    evaluacion_sv = None
    if signos_vitales:
        evaluacion_sv = sistema_ml.evaluar_signos_vitales(signos_vitales)

    # SEGUIMIENTO NUTRICIONAL DE 24 HORAS COMPLETAS
    comidas_dia = {}
    total_24h = {
        'ig_ponderado_total': 0,
        'cho_total_24h': 0,
        'carga_total_24h': 0,
        'calorias_total_24h': 0,
        'alcohol_total_24h': 0,
        'indice_alcoholico_24h': 0,
        'detalles_24h': []
    }

    # TODAS LAS COMIDAS EN 24 HORAS
    tipos_comida = [
        ("Desayuno", "🌅", "6:00-9:00"),
        ("Media Mañana", "☕", "9:30-11:00"),
        ("Almuerzo", "🍽️", "12:00-14:00"),
        ("Merienda", "🧉", "15:00-17:00"),
        ("Cena", "🌙", "19:00-21:00"),
        ("Desayuno del día siguiente", "🌅+1", "6:00-9:00 (+24h)")
    ]

    print(f"\n📋 REGISTRO DE COMIDAS EN 24 HORAS:")
    print("=" * 45)

    for tipo, emoji, horario in tipos_comida:
        print(f"\n{emoji} {tipo.upper()} ({horario})")
        print("=" * 40)

        respuesta = input(f"¿Consumió algo en {tipo.lower()}? (s/n): ").lower().strip()
        if respuesta not in ['s', 'si', 'sí', 'yes', 'y']:
            print(f"⏭️ Saltando {tipo.lower()}")
            continue

        componentes = registrar_comida_especifica_24h(base_alimentos, tipo)

        if componentes:
            resultados = evaluar_riesgo_comida(componentes, base_alimentos)
            comidas_dia[tipo] = {'componentes': componentes, 'resultados': resultados}
            acumular_totales_24h_completo(resultados, total_24h)

            print(f"✅ {emoji} {tipo} registrado:")
            print(f"    Calorías: {resultados['calorias_total']:.0f} kcal")
            print(f"    CHO: {resultados['cho_total']:.1f}g")
            print(f"    CG: {resultados['carga_total']:.1f}")
            if resultados['indice_alcoholico'] > 0:
                print(f"    Alcohol: {resultados['indice_alcoholico']:.1f}g (puro)")
        else:
            print(f"⚠️ No se registraron alimentos para {tipo}")

    if comidas_dia:
        # Calcular totales 24h
        ig_24h = total_24h['ig_ponderado_total'] / total_24h['cho_total_24h'] if total_24h['cho_total_24h'] > 0 else 0

        resultados_24h = {
            'ig_comida': ig_24h,
            'cho_total': total_24h['cho_total_24h'],
            'carga_total': total_24h['carga_total_24h'],
            'calorias_total': total_24h['calorias_total_24h'],
            'alcohol_total_ml': total_24h['alcohol_total_24h'],
            'indice_alcoholico': total_24h['indice_alcoholico_24h'],
            'detalles': total_24h['detalles_24h']
        }

        # Evaluación con signos vitales tradicional
        evaluacion_24h = evaluar_riesgos_con_signos_vitales(resultados_24h, signos_vitales, evaluacion_sv)

        # ANÁLISIS CON MACHINE LEARNING
        ml_result = None
        if signos_vitales:
            print(f"\n🤖 PROCESANDO DATOS CON MACHINE LEARNING...")

            # Preparar datos del paciente para ML
            patient_data = {
                'age': signos_vitales['age'],
                'pa_sistolica': signos_vitales['pa_sistolica'],
                'pa_diastolica': signos_vitales['pa_diastolica'],
                'fc': signos_vitales['fc'],
                'sato2': signos_vitales['sato2'],
                'is_diabetic': signos_vitales['is_diabetic'],
                'calorias': resultados_24h['calorias_total'],
                'cho_total': resultados_24h['cho_total'],
                'carga_glucemica': resultados_24h['carga_total'],
                'alcohol': resultados_24h.get('indice_alcoholico', 0)
            }

            # Predicción con ML
            ml_result = sistema_ml.ml_system.predict_risk_ml(patient_data)

        # Diagnósticos hipotéticos
        diagnosticos = None
        if signos_vitales and evaluacion_sv:
            diagnosticos = sistema_ml.generar_diagnostico_hipotetico(signos_vitales, evaluacion_sv, resultados_24h)

        # Mostrar evaluación completa con ML
        mostrar_evaluacion_24h_completa_con_ml(comidas_dia, resultados_24h, evaluacion_24h,
                                             signos_vitales, evaluacion_sv, diagnosticos, ml_result)
    else:
        print("❌ No se registraron comidas en las 24 horas")

# PROGRAMA PRINCIPAL CON ML
def main():
    """Función principal del programa con ML"""
    print("🤖 SISTEMA DE SEGUIMIENTO NUTRICIONAL Y MÉDICO 24H CON ML")
    print("="*60)
    print("⚠️  SOLO PARA FINES EDUCATIVOS")
    print("📚 NO REEMPLAZA CONSULTA MÉDICA PROFESIONAL")
    print("🤖 INCLUYE ANÁLISIS CON INTELIGENCIA ARTIFICIAL")
    print("="*60)

    # Crear base de datos y sistema
    base_alimentos = crear_base_alimentos()
    sistema_ml = MLSignosVitales()

    # Menú principal
    while True:
        print("\n📋 MENÚ PRINCIPAL:")
        print("1. 🔍 Ver alimentos disponibles")
        print("2. ⏰ Iniciar seguimiento de 24 horas CON ML")
        print("3. 🤖 Entrenar/Ver información de modelos ML")
        print("4. 🚪 Salir")

        try:
            opcion = input("\nSeleccione una opción (1-4): ").strip()

            if opcion == "1":
                print("\n📋 ALIMENTOS DISPONIBLES EN LA BASE:")
                print("="*40)
                for i, alimento in enumerate(base_alimentos['alimento'], 1):
                    calorias = base_alimentos.iloc[i-1]['calorias_100g']
                    cho = base_alimentos.iloc[i-1]['cho_100g']
                    ig = base_alimentos.iloc[i-1]['ig']
                    print(f"{i:2d}. {alimento:<25} | Cal: {calorias:3.0f} | CHO: {cho:4.1f}g | IG: {ig:2.0f}")

                input("\nPresione ENTER para continuar...")

            elif opcion == "2":
                seguimiento_24_horas_con_ml(base_alimentos, sistema_ml)
                input("\nPresione ENTER para continuar...")

            elif opcion == "3":
                print("\n🤖 INFORMACIÓN DEL SISTEMA DE MACHINE LEARNING:")
                print("="*50)
                print("📊 Modelos utilizados:")
                print("   • Random Forest Classifier (clasificación de riesgo)")
                print("   • Gradient Boosting Regressor (predicción de calorías)")
                print("📈 Variables de entrada:")
                print("   • Edad, PA sistólica, PA diastólica, FC, SatO2")
                print("   • Estado diabético, calorías, CHO, carga glucémica, alcohol")
                print("🎯 Salidas del modelo:")
                print("   • Clasificación de riesgo: NORMAL, MODERADO, ALTO, CRÍTICO")
                print("   • Calorías recomendadas personalizadas")
                print("   • Probabilidades y confianza de predicción")

                if not sistema_ml.ml_system.is_trained:
                    entrenar = input("\n¿Desea entrenar los modelos ahora? (s/n): ").lower().strip()
                    if entrenar in ['s', 'si', 'sí', 'yes', 'y']:
                        sistema_ml.ml_system.train_models()
                else:
                    print("\n✅ Modelos ya entrenados y listos para usar")

                input("\nPresione ENTER para continuar...")

            elif opcion == "4":
                print("\n👋 ¡Gracias por usar el sistema!")
                print("🏥 Recuerde: ante cualquier duda médica, consulte con un profesional")
                print("🤖 El ML es solo una herramienta de apoyo educativo")
                break

            else:
                print("❌ Opción inválida. Por favor seleccione 1, 2, 3 o 4.")

        except KeyboardInterrupt:
            print("\n\n👋 Programa interrumpido por el usuario")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            print("🔄 Reiniciando menú...")

# Ejecutar el programa
if __name__ == "__main__":
    main()