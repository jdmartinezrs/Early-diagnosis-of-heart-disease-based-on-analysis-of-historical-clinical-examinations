import pandas as pd

# Cargar el archivo Parquet
df = pd.read_parquet("hf://datasets/TheFinAI/FL-med-syn0-cleveland-instruction/data/train-00000-of-00001.parquet")

# Guardar el DataFrame como CSV en el directorio actual
df.to_csv("datos.csv", index=False)
print("Archivo CSV guardado como 'datos.csv' en el directorio actual.")