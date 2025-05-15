import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Crear carpeta si no existe
os.makedirs("datasets/datos_procesados", exist_ok=True)

# === Parte 1: Preprocesamiento del archivo original ===
df = pd.read_csv("datasets/datos.csv")
df.columns = df.columns.str.strip().str.lower()

if 'entries' not in df.columns:
    print("❌ La columna 'entries' no existe.")
    print("Columnas disponibles:", df.columns.tolist())
    exit()

# Función para procesar cada fila
def procesar_fila(entry_str):
    try:
        partes = re.findall(r"\{[^}]*\}", entry_str)
        if len(partes) < 2:
            return None
        user_part = partes[0]
        assistant_part = partes[1]

        texto_usuario = re.search(r"Text: '([^']+)'", user_part)
        if not texto_usuario:
            return None

        caracteristicas = dict(re.findall(r"(\w+): ([^,]+)", texto_usuario.group(1)))
        etiqueta_match = re.search(r"'content': '(\w+)'", assistant_part)
        if not etiqueta_match:
            return None

        caracteristicas['target'] = etiqueta_match.group(1).lower()
        return caracteristicas
    except Exception as e:
        print(f"Error: {e}")
        return None

# Procesar datos
datos_procesados = df['entries'].apply(procesar_fila).dropna()
df_modelo = pd.DataFrame(datos_procesados)

# Guardar datos procesados
df_modelo.to_csv("datasets/datos_procesados.csv", index=False)

# === Parte 2: Limpieza y transformación para ML ===
df_modelo = df_modelo.dropna()

# Codificar variables categóricas
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseInducedAngina', 'ThalassemiaStatus']

# Aplicar LabelEncoder
le = LabelEncoder()
for col in categorical_columns:
    if col in df_modelo.columns:
        df_modelo[col] = le.fit_transform(df_modelo[col])

# Escalar variables numéricas
numerical_columns = ['Age', 'RestingBloodPressure', 'Cholesterol', 'MaxHeartRate', 'OldPeak', 'NumberOfVesselsColored']
scaler = StandardScaler()
df_modelo[numerical_columns] = scaler.fit_transform(df_modelo[numerical_columns])

# Separar características y etiquetas
X = df_modelo.drop(columns=['target'])
y = df_modelo['target']

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Parte 3: Guardar datasets divididos ===
X_train.to_csv("datasets/divididos/X_train.csv", index=False)
X_test.to_csv("datasets/divididos/X_test.csv", index=False)
y_train.to_csv("datasets/divididos/y_train.csv", index=False)
y_test.to_csv("datasets/divididos/y_test.csv", index=False)

print("✅ Datos divididos y guardados en 'datasets/divididos/'")
print(df_modelo.columns)
