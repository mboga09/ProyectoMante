import pandas as pd
import numpy as np

# Ruta del archivo CSV separado por espacios
ruta_archivo = "train.txt"

# Leer el archivo usando separador de espacio
# sensors_nums = list(range(21))
# sensors_nums = [x+1 for x in sensors_nums]
columnas = ["id","cycles","s1","s2","s3"]
for i in range(21):
    columnas.append("sensor"+str(i+1))
df = pd.read_csv(ruta_archivo, sep=r"\s+", engine="python", names=columnas)
#["id","cycles","s1","s2","s3","sensor "+str(sensors_nums)]

# Mostrar las primeras filas
print("Contenido del DataFrame:")
print(df)

# Convertir el DataFrame a un arreglo de NumPy
arr = df.to_numpy()

print("\nArreglo NumPy:")
print(arr)


train_df = df[df.id==1]

print(train_df.describe())

stds = train_df.std(numeric_only=True)

columnas_std_cero = stds[stds <= 0.1].index.tolist()

print("Columnas con std = 0:")
print(columnas_std_cero)
