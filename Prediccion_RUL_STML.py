import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import seaborn as sns
import math

# Ruta del archivo CSV separado por espacios
ruta_archivo = 'C:\\Users\\mboga\\OneDrive\\Documentos\\Mante\\train.txt'

# Definir las etiquetas para cada una de las columnas
columnas = ["id", "cycles", "config1", "config2", "config3"]
for i in range(21):
    columnas.append("sensor" + str(i + 1))
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
for rowID in range(len(df)):  # para cada fila de la base de datos
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
    numCiclosMotor = len(df[df.id == i + 1])
    numCiclosMotores.extend([numCiclosMotor] * numCiclosMotor)
# El número de ciclos ahora será el número de ciclo menos la vida total que tuvo ese motor
df.cycles = df.cycles - numCiclosMotores

# Para generar visualización del comportamiento de cierto sensor en cada modo de operación por sepearado
# (se puede comentar para mejorar tiempo ejecución)
for modoOp in range(6):
    datosModoOp = df[df.opSetting == modoOp + 1]
    print("Modo: ", modoOp)
    print(datosModoOp)
    plt.figure()
    plt.scatter(datosModoOp.cycles, datosModoOp.sensor21)
    plt.pause(0.001)

# Sensores descartados: 1, 5, 6 (por std despreciable), 8 (porque hacia el final algunos datos agarran para arriba y otros para abajo), 9 (por alta varianza en el final), 10
# (por baja std), 13 (en varios modosOP da alta varianza al final), 14 (alta varianza), 16, 17 (valores independientes de modoOP pero organizados), 18, 19, 20 (alta variabilidad),
# 21 (tendencia poco clara por muy alta variabilidad)
# -- Los anteriores se descartaron por ser constantes para cada modo de operación (o por la razón entre paréntesis)

# Se eliminan las columnas de los sensores mencionados
df = df.drop(
    columns=['sensor1', 'sensor5', 'sensor6', 'sensor8', 'sensor9', 'sensor10', 'sensor13', 'sensor14', 'sensor16',
             'sensor17', 'sensor18', 'sensor19', 'sensor20', 'sensor21'])
print(df.describe())

# Opino que se podría hacer que una vez se haga la normalización y eliminación de ruido se podría guardar el resultado en otro csv ya procesado, y que el código de la red
# neuronal lo que lea sea ese

input("Presionar Enter para cerrar plots.")
plt.close('all')

# Tratamiento de ruido

print("\n=== INICIANDO TRATAMIENTO DE RUIDO ===")


# Esta función marca valores muy extremos por IQR y los reemplaza por la mediana del grupo
def eliminar_outliers_IQR(df_grupo, columna, factor=1.5):
    Q1 = df_grupo[columna].quantile(0.25)
    Q3 = df_grupo[columna].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - factor * IQR
    limite_superior = Q3 + factor * IQR

    mediana = df_grupo[columna].median()
    df_grupo.loc[
        (df_grupo[columna] < limite_inferior) |
        (df_grupo[columna] > limite_superior),
        columna
    ] = mediana
    return df_grupo


# Aquí defino qué sensores voy a filtrar (todos los que quedaron después de Carlos)
sensores_a_filtrar = [col for col in df.columns if 'sensor' in col]
df_filtrado = df.copy()

print("Aplicando filtros de ruido por motor y por modo de operación...")
for motor_id in range(1, numeroDeMotores + 1):
    for modo in range(1, 7):
        # Me quedo solo con las filas de este motor en este modo
        mask = (df_filtrado['id'] == motor_id) & (df_filtrado['opSetting'] == modo)
        if mask.sum() < 5:
            # Si casi no hay datos para este caso, prefiero no tocarlo
            continue

        indices = df_filtrado[mask].index

        # Primero trabajo cada sensor con Savitzky-Golay para quitar ruido de alta frecuencia
        for sensor in sensores_a_filtrar:
            datos_sensor = df_filtrado.loc[mask, sensor].values

            if len(datos_sensor) >= 5:
                window_length = min(7, len(datos_sensor))
                if window_length % 2 == 0:
                    window_length -= 1
                if window_length >= 3:
                    datos_suavizados = savgol_filter(
                        datos_sensor,
                        window_length=window_length,
                        polyorder=2
                    )
                    df_filtrado.loc[indices, sensor] = datos_suavizados

        # Después del suavizado, reviso outliers por IQR y los pego a la mediana local
        for sensor in sensores_a_filtrar:
            df_grupo = df_filtrado.loc[mask, ['id', 'cycles', 'opSetting', sensor]].copy()
            df_grupo = eliminar_outliers_IQR(df_grupo, sensor, factor=2.0)
            df_filtrado.loc[mask, sensor] = df_grupo[sensor].values

print("Filtrado de ruido completado.")

# Normalización de datos (Max–Min estilo artículo, pero hecha a mano)
print("\n=== INICIANDO NORMALIZACIÓN MAX–MIN (RANGO [-1,1]) ===")

# Trabajo sobre una copia ya filtrada para no perder el original de Carlos
df_normalizado = df_filtrado.copy()

# Calculo min y max por sensor
min_vals = df_filtrado[sensores_a_filtrar].min()
max_vals = df_filtrado[sensores_a_filtrar].max()
rango = (max_vals - min_vals).replace(0, 1.0)

# Escalo cada sensor al rango [-1, 1]
df_normalizado[sensores_a_filtrar] = -1 + 2 * (
        (df_filtrado[sensores_a_filtrar] - min_vals) / rango
)

print("Normalización Max–Min completada.")

# Verificación, visualización y guardado
print("\n=== ESTADÍSTICAS DESPUÉS DEL PROCESAMIENTO ===")
print(df_normalizado[sensores_a_filtrar].describe())

# Aquí solo se muestra un ejemplo antes/después para un motor y un sensor para tener una idea visual
# Si María quiere lo puede comentar o quitar para hacerlo más rápido
motor_ejemplo = 1
sensor_ejemplo = 'sensor2'

plt.figure(figsize=(12, 5))

# Datos originales después de la parte de Carlos
plt.subplot(1, 2, 1)
datos_originales = df[df['id'] == motor_ejemplo]
plt.scatter(datos_originales['cycles'], datos_originales[sensor_ejemplo],
            c=datos_originales['opSetting'], cmap='viridis', alpha=0.6)
plt.title(f'Motor {motor_ejemplo} - {sensor_ejemplo} (Antes de mi procesamiento)')
plt.xlabel('Ciclos')
plt.ylabel('Valor del sensor')
plt.colorbar(label='Modo de operación')

# Datos ya filtrados y normalizados
plt.subplot(1, 2, 2)
datos_procesados = df_normalizado[df_normalizado['id'] == motor_ejemplo]
plt.scatter(datos_procesados['cycles'], datos_procesados[sensor_ejemplo],
            c=datos_procesados['opSetting'], cmap='viridis', alpha=0.6)
plt.title(f'Motor {motor_ejemplo} - {sensor_ejemplo} (Filtrado + Normalizado)')
plt.xlabel('Ciclos')
plt.ylabel('Valor normalizado')
plt.colorbar(label='Modo de operación')

plt.tight_layout()
plt.show()

# Guardado del DataFrame procesado a un nuevo CSV
df_normalizado.to_csv('train_procesado.csv', index=False)
print("Datos guardados en 'train_procesado.csv'.")

# Preparación de datos para el modelo (features y target)
print("\n=== PREPARACIÓN FINAL DE DATOS PARA LA RED ===")

# RUL como vida útil remanente: cycles ya está en negativo por el ajuste de Carlos
df_normalizado['RUL'] = -df_normalizado['cycles']

#   Implementación de RUL Clipping (usado en el paper)
RUL_MAX = 130
print(f"Aplicando RUL Clipping: RUL máximo establecido a {RUL_MAX}")
# Se asegura que ningún valor de RUL en el conjunto de entrenamiento sea mayor que RUL_MAX
df_normalizado['RUL'] = df_normalizado['RUL'].clip(upper=RUL_MAX)
# ===================================================

# Features: modos de operación + sensores procesados
features_cols = ['opSetting'] + sensores_a_filtrar
X = df_normalizado[features_cols].values
y = df_normalizado['RUL'].values
ids_motores = df_normalizado['id'].values

print(f"Dimensiones de X (características): {X.shape}")
print(f"Dimensiones de y (RUL): {y.shape}")
print(f"Rango de RUL: [{y.min()}, {y.max()}]")

#   Crear secuencias (lookback)
def crear_secuencias(X, y, ids, lookback):
    X_seq, y_seq = [], []

    for motor in np.unique(ids):
        idx = np.where(ids == motor)[0]
        X_m = X[idx]
        y_m = y[idx]

        # generar secuencias válidas
        for i in range(len(X_m) - lookback):
            X_seq.append(X_m[i:i + lookback])
            y_seq.append(y_m[i + lookback])

    return np.array(X_seq), np.array(y_seq), X.shape[1]

LOOKBACK = 30 # Lookback usado en el paper

print(f"\n=== GENERANDO SECUENCIAS LOOKBACK={LOOKBACK} ===")
X_seq, y_seq, n_features = crear_secuencias(X, y, ids_motores, LOOKBACK)


#   Separar 80% train / 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

#   Tensores
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

#   Crear IDs para secuencias
ids_seq = []
for motor in np.unique(ids_motores):
    idx = np.where(ids_motores == motor)[0]
    n = len(idx)
    ids_seq.extend([motor] * (n - LOOKBACK))

ids_seq = np.array(ids_seq)

train_ids_seq = ids_seq[:len(X_train)]
test_ids_seq = ids_seq[len(X_train):]

#   Modelo LSTM
class LSTM_RUL(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        # --- Capa 1: LSTM (50 unidades) ---
        # En el artículo se usa L2 Regularization
        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=50,  # 50 unidades
            num_layers=1,
            batch_first=True,
        )

        # --- Dropout 1 (0.4) ---
        self.dropout1 = nn.Dropout(0.4)  # Primer Dropout

        # --- Capa 2: LSTM (25 unidades) ---
        # La entrada es la salida de lstm1 (hidden_size=50)
        self.lstm2 = nn.LSTM(
            input_size=50,
            hidden_size=25,  # 25 unidades
            num_layers=1,
            batch_first=True,
        )

        # --- Dropout 2 (0.4) ---
        self.dropout2 = nn.Dropout(0.4)  # Segundo Dropout

        # --- Capa de Salida (1 neurona) ---
        # La entrada es la salida de lstm2 (hidden_size=25)
        self.fc = nn.Linear(25, 1)

    def forward(self, x):
        # Pasar por la primera LSTM
        out, _ = self.lstm1(x)
        # Aplicar Dropout 1
        out = self.dropout1(out)

        # Pasar por la segunda LSTM
        out, _ = self.lstm2(out)
        # Aplicar Dropout 2
        out = self.dropout2(out)

        # Tomar el último paso de tiempo para la predicción RUL
        return self.fc(out[:, -1, :])


model = LSTM_RUL(n_features)
criterion = nn.MSELoss()

#   Optimizador (L2 Regularization)
# En el artículo usa L2 Regularization con un factor de 0.01.
L2_FACTOR = 0.01
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=L2_FACTOR  # Implementación de L2 Regularization
)

#   Entrenamiento guardando el mejor modelo
train_losses = []  # MSE por secuencia
val_losses = []  # MSE por secuencia
# Listas para el RMSE
train_rmse_list = []
val_rmse_list = []
best_val_rmse = float('inf')
best_epoch = 0
EPOCHS = 75 #En el paper usan 100 pero se decide dejar en 75 porque no generaba mayor diferencia

# Función auxiliar para obtener las predicciones completas
def get_predictions(model, X_data_t):
    model.eval()
    with torch.no_grad():
        return model(X_data_t).cpu().numpy().reshape(-1)


for epoch in range(EPOCHS):
    model.train()
    batch_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_loss_epoch = np.mean(batch_losses)
    train_losses.append(train_loss_epoch)

    # Calcular train RMES
    train_rmse = math.sqrt(train_loss_epoch)
    train_rmse_list.append(train_rmse)

    # Validación
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        val_loss = criterion(preds, y_test_t).item()
        val_losses.append(val_loss)

    # Calcular validation RMES
    val_rmse = math.sqrt(val_loss)
    val_rmse_list.append(val_rmse)

    # Guardar mejor modelo
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_epoch = epoch + 1
        # Guardar una copia del modelo
        best_model_state = model.state_dict().copy()
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss={train_loss_epoch:.4f} | Val Loss={val_loss:.4f} | Train RMSE={train_rmse:.4f} | Val RMSE={val_rmse:.4f} *** NUEVO MEJOR MODELO ***")
    else:
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss={train_loss_epoch:.4f} | Val Loss={val_loss:.4f} | Train RMSE={train_rmse:.4f} | Val RMSE={val_rmse:.4f}")


#   Predicciones finales (con el mejor modelo)
# Cargar el mejor modelo encontrado para las métricas finales
model.load_state_dict(best_model_state)

# Predicciones y valores reales
train_pred_final = get_predictions(model, X_train_t)
test_pred_final = get_predictions(model, X_test_t)

train_y_final = y_train
test_y_final = y_test

#   Métricas
print("\n=== Métricas Test ===")
print("MSE:", mean_squared_error(test_y_final, test_pred_final))
print("MAE:", mean_absolute_error(test_y_final, test_pred_final))

#   Gráfica de curvas de apredizaje
train_rmse = train_rmse_list
val_rmse = val_rmse_list

# Encontrar el mejor epoch
best_epoch = np.argmin(val_rmse) + 1
best_val_rmse = min(val_rmse)

# Crear la gráfica
plt.figure(figsize=(10, 6))
epochs_range = range(1, EPOCHS + 1)

# Plotear curvas
plt.plot(epochs_range, train_rmse, 'b-', linewidth=2, label='RMSE de Entrenamiento')
plt.plot(epochs_range, val_rmse, 'r-', linewidth=2, label='RMSE de Validación')

# Marcar el mejor epoch
plt.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5,
            label=f'Mejor época ({best_epoch})')
plt.plot(best_epoch, best_val_rmse, 'go', markersize=10)

# Etiquetas y título
plt.xlabel('Época', fontsize=12, fontweight='bold')
plt.ylabel('Raíz del Error Cuadrático Medio (RMSE)', fontsize=12, fontweight='bold')
plt.title('Curvas de Aprendizaje: Rendimiento del Modelo por época', fontsize=14, fontweight='bold')

# Leyenda y grid
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

# Ajustar márgenes
plt.tight_layout()

# Guardar y mostrar
plt.savefig('learning_curves_standard_rmse.png', dpi=300, bbox_inches='tight')
print(f"Mejor época: {best_epoch} con Validation RMSE: {best_val_rmse:.4f}")

plt.show()


#   Gráfica de RUL real vs predicho
import matplotlib.pyplot as plt
import numpy as np

# Asegurar que el límite del RUL sea el máximo usado en el clipping (130)
RUL_MAX_LIM = 130

# Crear la figura
plt.figure(figsize=(8, 8))

# 1. Graficar la Línea Ideal (Predicted = Actual)
# Se usa el RUL_MAX_LIM como referencia para la línea diagonal
plt.plot([0, RUL_MAX_LIM], [0, RUL_MAX_LIM],
         'r--', linewidth=4, label='Línea Ideal (Predicho = Real)')

# 2. Plot de las Predicciones (Model prediction)
plt.scatter(test_y_final, test_pred_final,
            color='blue', alpha=0.6, label='Predicción del Modelo')

# Etiquetas y Título
plt.title('RUL Real vs Predicho', fontsize=14, fontweight='bold')
plt.xlabel('RUL Real (ciclos)', fontsize=12, fontweight='bold')
plt.ylabel('RUL Predicho (ciclos)', fontsize=12, fontweight='bold')

# Límites del Eje
plt.xlim([0, RUL_MAX_LIM * 1.05])
plt.ylim([0, RUL_MAX_LIM * 1.05])

# Leyenda y Grid
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)

# Ajustar márgenes
plt.tight_layout()
plt.savefig('actual_vs_predicted_rul.png', dpi=300, bbox_inches='tight')

plt.show()


# RUL  predicho por motor
print("\n=== ÚLTIMO RUL POR MOTOR ===\n")

# Obtener los motores presentes en el conjunto test
unique_motors = np.unique(test_ids_seq)

for motor in unique_motors:
    # Índices donde aparece este motor en el conjunto test
    idx = np.where(test_ids_seq == motor)[0]

    if len(idx) == 0:
        continue

    # Último índice disponible para ese motor (última secuencia)
    last_idx = idx[-1]

    real_rul = test_y_final[last_idx]
    pred_rul = test_pred_final[last_idx]

    print(f"Motor {motor}:  RUL predicho = {pred_rul:.2f}")
