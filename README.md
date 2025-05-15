## Conversión de Conversaciones a Formato Tabular

###  Objetivo

Transformar un conjunto de conversaciones tipo ChatGPT (lista de mensajes con roles `"user"` y `"assistant"`) a un formato más estructurado o plano para su análisis, almacenamiento o entrenamiento clásico de modelos de ML.

------

### Proceso realizado

1. **Entrada original** (conversaciones en texto tipo string que representan listas de diccionarios):

   ```python
   python
   
   
   CopiarEditar
   "[{'content': 'Predict whether the patient has heart disease...Text: Age: 58.0, ...Answer:', 'role': 'user'}, {'content': 'unhealthy', 'role': 'assistant'}]"
   ```

2. **Conversión a estructura de datos real**:

   - Se deserializó la cadena usando `ast.literal_eval` para convertirla en una lista de diccionarios válida.
   - Se extrajo el contenido relevante del mensaje del `"user"` (usualmente el texto entre `Text:` y `Answer:`).
   - Se extrajo la respuesta del `"assistant"` como la etiqueta.

   Ejemplo de código:

   ```python
   pythonCopiarEditarimport ast
   
   raw_entry = "[{'content': 'Predict...Text: Age: 58.0, ...Answer:', 'role': 'user'}, {'content': 'unhealthy', 'role': 'assistant'}]"
   messages = ast.literal_eval(raw_entry)
   
   user_prompt = messages[0]['content']
   text = user_prompt.split("Text:")[1].split("Answer:")[0].strip()
   label = messages[1]['content'].strip()
   
   result = {"text": text, "label": label}
   ```

3. **Resultado final (formato plano)**:

   ```python
   jsonCopiarEditar{
     "text": "Age: 58.0000, Sex: male, ChestPainType: asymptomatic, RestingBloodPressure: 125.0000, Cholesterol: 300.0000, ...",
     "label": "unhealthy"
   }
   ```

# Modelo Predictivo de Enfermedades Cardíacas

Este proyecto consiste en construir un modelo de machine learning capaz de predecir enfermedades cardíacas a partir de características clínicas de pacientes. Los datos provienen de un dataset estructurado en conversaciones tipo `chat`, que han sido procesadas y transformadas para entrenamiento supervisado.

------

## 1. Preprocesamiento de los Datos

El dataset original contiene columnas con entradas conversacionales en formato string. Cada entrada tiene la estructura de una lista de diccionarios con los roles `user` (que incluye las características del paciente) y `assistant` (que da el diagnóstico).

### Conversión del texto a estructura tabular

```
pythonCopiarEditarimport pandas as pd
import ast

# Leer el CSV
df = pd.read_csv('datos.csv')

# Parsear las entradas (que están en formato de cadena)
entries = []
for entry in df['entries']:
    # Convertir la cadena a una lista de diccionarios
    data = ast.literal_eval(entry)
    
    # Extraer características del texto del usuario
    user_content = data[0]['content']
    
    # Extraer la etiqueta del asistente
    label = data[1]['content']
    
    # Parsear las características
    features = {}
    parts = user_content.split("Text:")[1].split("Answer:")[0].strip().strip("' .").split(', ')
    for part in parts:
        key, value = part.split(': ')
        features[key.strip()] = value.strip()
    
    features['target'] = 1 if label == 'unhealthy' else 0  # 1 = enfermo, 0 = sano
    entries.append(features)

# Convertir a DataFrame
data = pd.DataFrame(entries)
```

------

### Limpieza de datos

- Conversión de tipos numéricos (`Age`, `Cholesterol`, etc.)
- Manejo de valores faltantes (`norecord → NaN`)
- Codificación de variables categóricas

------

## 2. Análisis Exploratorio de Datos (EDA)

### Distribución de la variable objetivo

```
pythonCopiarEditarimport matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='target', data=data)
plt.title('Distribución de enfermedades cardíacas')
plt.show()
```

### Matriz de correlación

```
pythonCopiarEditarnumeric_cols = ['Age', 'RestingBloodPressure', 'Cholesterol', 'MaxHeartRate', 'OldPeak']
sns.heatmap(data[numeric_cols].corr(), annot=True)
plt.title('Matriz de correlación entre variables numéricas')
plt.show()
```

------

##  3. Preparación para el modelo

### Codificación y limpieza de columnas

```
pythonCopiarEditarfrom sklearn.model_selection import train_test_split

# One-hot encoding de variables categóricas
data = pd.get_dummies(
    data,
    columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseInducedAngina', 'PeakExerciseSTSegmentSlope']
)

# Eliminar columnas no útiles si son irrelevantes
data.drop(columns=['ThalassemiaStatus', 'NumberOfVesselsColored'], inplace=True, errors='ignore')

# Separar X e y
X = data.drop('target', axis=1)
y = data['target']

# División de los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

------

##  4. Entrenamiento del Modelo

###  Random Forest

```
pythonCopiarEditarfrom sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Evaluación
y_pred = model_rf.predict(X_test)
print("Precisión:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Regresión Logística

```
pythonCopiarEditarfrom sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
print("Precisión (Regresión Logística):", accuracy_score(y_test, y_pred_lr))
```

------

##  5. Optimización del Modelo

### GridSearchCV para Random Forest

```
pythonCopiarEditarfrom sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Mejores parámetros:", grid_search.best_params_)

best_model = grid_search.best_estimator_
```

------

##  6. Interpretación del Modelo

### Importancia de características (Random Forest)

```
pythonCopiarEditarfeature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)
```

### Coeficientes (Regresión Logística)

```
pythonCopiarEditarpd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model_lr.coef_[0]
}).sort_values('Coefficient', ascending=False)
```

------

##  7. Guardado del Modelo

```
pythonCopiarEditarimport joblib

joblib.dump(best_model, 'heart_disease_model.pkl')
```



![Texto alternativo](https://i.pinimg.com/1200x/e7/be/04/e7be04147b6a305f0ea37497f684adff.jpg)