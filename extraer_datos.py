import pandas as pd
import re

# Cargar y limpiar el dataset
df = pd.read_csv("datasets/datos.csv")
df.columns = df.columns.str.strip().str.lower()

# Verifica que 'entries' existe ahora
if 'entries' not in df.columns:
    print("❌ La columna 'entries' no existe en el archivo CSV.")
    print("Columnas disponibles:", df.columns.tolist())
    exit()

# Función para extraer características y etiqueta
def procesar_fila(entry_str):
    try:
        # Ajuste para mejor separación de partes
        partes = re.findall(r"\{[^}]*\}", entry_str)
        
        if len(partes) < 2:
            return None
        
        user_part = partes[0]
        assistant_part = partes[1]
        
        # Extraer las características del usuario
        texto_usuario = re.search(r"Text: '([^']+)'", user_part)
        if not texto_usuario:
            return None
        
        caracteristicas = dict(re.findall(r"(\w+): ([^,]+)", texto_usuario.group(1)))
        
        # Extraer la etiqueta del asistente
        etiqueta_match = re.search(r"'content': '(\w+)'", assistant_part)
        if not etiqueta_match:
            return None
        
        caracteristicas['target'] = etiqueta_match.group(1).lower()
        
        return caracteristicas
    except Exception as e:
        print(f"Error: {e}")
        return None

# Procesar y mostrar
datos_procesados = df['entries'].apply(procesar_fila).dropna()
df_modelo = pd.DataFrame(datos_procesados)
print(df_modelo.head())

df_modelo.to_csv("datasets/datos_procesados.csv", index=False)
print("Archivo CSV guardado como 'datos_procesados.csv'")