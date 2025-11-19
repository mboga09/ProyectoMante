import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ruta del archivo CSV separado por espacios
ruta_archivo = "train.txt"

# Leer el archivo usando separador de espacio
# sensors_nums = list(range(21))
# sensors_nums = [x+1 for x in sensors_nums]
columnas = ["id","cycles","config1","config2","config3"]
for i in range(21):
    columnas.append("sensor"+str(i+1))
df = pd.read_csv(ruta_archivo, sep=r"\s+", engine="python", names=columnas)
#["id","cycles","s1","s2","s3","sensor "+str(sensors_nums)]

# Mostrar las primeras filas
print("Contenido del DataFrame:")
print(df)

# # Convertir el DataFrame a un arreglo de NumPy
# arr = df.to_numpy()

# print("\nArreglo NumPy:")
# print(arr)


train_df = df[df.id==1]

print(train_df.describe())

numeroDeMotores = np.max(df["id"])

# Separación en modos de operación hecha según publicación 1 mencionada en la Explicación y Guía de Uso
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x = df.config1
y = df.config2
z = df.config3
        
ax.scatter(x,y,z)
ax.set_xlabel("Config op 1")
ax.set_ylabel("Config op 2")
ax.set_zlabel("Config op 3")
ax.view_init(elev=0, azim=-90, roll=0)
# plt.show()

modosOp = []
for rowID in range(len(df)):
    row = df.iloc[rowID]
    modoOperacion = 0
    if row.config1 <= 0:
        modoOperacion = 1
    elif row.config1 <= 15:
        modoOperacion = 2
    elif row.config1 <= 21:
        modoOperacion = 3
    elif row.config1 <= 30:
        modoOperacion = 4
    elif row.config1 <= 38:
        modoOperacion = 5
    else:
        modoOperacion = 6
    modosOp.append(modoOperacion)
df.insert(2, "opSetting", modosOp)
df = df.drop(columns=['config1', 'config2', 'config3'])
print(df.describe())


# Estandarizar el número de ciclos
numCiclosMotores = []
for i in range(numeroDeMotores):
    numCiclosMotor = len(df[df.id==i+1])
    numCiclosMotores.extend([numCiclosMotor]*numCiclosMotor)

    # for j in range(numCiclosMotor):
    #     numCiclosMotores.append(numCiclosMotor)

df.cycles = df.cycles - numCiclosMotores

    # for j in range(numCiclosMotor):
    #     ciclo = df.cycles.iloc[i*numCiclosMotor+j]
    #     df.cycles.iloc[i*numCiclosMotor+j] = ciclo - numCiclosMotor

plt.figure()
plt.plot(df.cycles,df.sensor2)
plt.show()



# stds = train_df.std(numeric_only=True)

# columnas_std_cero = stds[stds <= 0.1].index.tolist()

# print("Columnas con std = 0:")
# print(columnas_std_cero)
