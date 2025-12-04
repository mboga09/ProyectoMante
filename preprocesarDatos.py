import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ruta del archivo CSV separado por espacios
ruta_archivo = "train.txt"

# Definir las etiquetas para cada una de las columnas
columnas = ["id","cycles","config1","config2","config3"] 
for i in range(21):
    columnas.append("sensor"+str(i+1))
# Leer el archivo usando separador de espacio
df = pd.read_csv(ruta_archivo, sep=r"\s+", engine="python", names=columnas)


# Separación en modos de operación hecha según publicación 1 mencionada en la Explicación y Guía de Uso
# (se puede comentar para mejorar tiempo de ejecución)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# x = df.config1
# y = df.config2
# z = df.config3
        
# ax.scatter(x,y,z)
# ax.set_xlabel("Config op 1")
# ax.set_ylabel("Config op 2")
# ax.set_zlabel("Config op 3")
# ax.view_init(elev=0, azim=-90, roll=0)
# plt.show()


# Categorización por modo de operación de cada una de las filas de datos
modosOp = []
for rowID in range(len(df)): # para cada fila de la base de datos
    row = df.iloc[rowID]
    modoOperacion = 0
    # Se categoriza utilizando condiciones
    if row.config1 <= 5:
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
# Se cambian las variables de configuración operacional por una sola que va de 1 a 6 (inclusive)
df.insert(2, "opSetting", modosOp)
df = df.drop(columns=['config1', 'config2', 'config3'])


# Se obtiene el número de motores que hay en la base de datos
numeroDeMotores = np.max(df["id"])

# Estandarizar el número de ciclos (para poder comparar entre motores distintos)
numCiclosMotores = []
for i in range(numeroDeMotores):
    numCiclosMotor = len(df[df.id==i+1])
    numCiclosMotores.extend([numCiclosMotor]*numCiclosMotor)
# El número de ciclos ahora será el número de ciclo menos la vida total que tuvo ese motor
df.cycles = df.cycles - numCiclosMotores


# Para generar visualización del comportamiento de cierto sensor en cada modo de operación por sepearado
# (se puede comentar para mejorar tiempo ejecución)
for modoOp in range(6):
    datosModoOp = df[df.opSetting==modoOp+1]
    print("Modo: ", modoOp)
    print(datosModoOp)
    plt.figure()
    plt.scatter(datosModoOp.cycles,datosModoOp.sensor21)
    plt.pause(0.001)



# Sensores descartados: 1, 5, 6 (por std despreciable), 8 (porque hacia el final algunos datos agarran para arriba y otros para abajo), 9 (por alta varianza en el final), 10
#(por baja std), 13 (en varios modosOP da alta varianza al final), 14 (alta varianza), 16, 17 (valores independientes de modoOP pero organizados), 18, 19, 20 (alta variabilidad),
#21 (tendencia poco clara por muy alta variabilidad)
# -- Los anteriores se descartaron por ser constantes para cada modo de operación (o por la razón entre paréntesis)

# Se eliminan las columnas de los sensores mencionados
df = df.drop(columns=['sensor1', 'sensor5', 'sensor6', 'sensor8', 'sensor9', 'sensor10', 'sensor13', 'sensor14', 'sensor16', 'sensor17', 'sensor18', 'sensor19', 'sensor20', 'sensor21'])
print(df.describe())

# Opino que se podría hacer que una vez se haga la normalización y eliminación de ruido se podría guardar el resultado en otro csv ya procesado, y que el código de la red 
# neuronal lo que lea sea ese

input("Presionar Enter para cerrar plots.")
plt.close('all') 
