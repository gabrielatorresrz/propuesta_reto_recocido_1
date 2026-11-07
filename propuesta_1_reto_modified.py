# Simulación del recocido en lote de bobinas de acero (modelo 2D en r, z)
# Bibliotecas:
#   numpy      -> arreglos y operaciones vectorizadas sobre la malla
#   matplotlib -> gráficas y mapas de color
#   pandas     -> tablas de resultados y exportación a Excel
#   os         -> creación de la carpeta de resultados
# Método numérico: diferencias finitas explícitas (esquema hacia adelante en el
# tiempo y centrado en el espacio) para la ecuación de conducción en coordenadas
# cilíndricas, con condición de convección en las cuatro caras de la bobina.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# PASO 0: parámetros modificables
# Se pueden cambiar los diámetros y las temperaturas objetivo y volver a ejecutar
# el programa; el resto del análisis se recalcula automáticamente
D_int = 0.508 #m, diámetro interno
D_ext = 1.829 #m, diámetro externo
T_obj = [650, 654] #°C, en orden ascendente y menores que T_gas

# PASO 1: geometría de los rollos de ternium
r_int = D_int / 2
r_ext = D_ext / 2
H = 1.200
S = 1.0e-3 

# PASO 2: seleccion de material y propiedades
# Acero 1: SAE 1010 (α = 1.27e-5 m^2/s, α_min)
# Acero 2: SAE 1006 (α = 1.49e-5 m^2/s, α_max)
lambda1_s = 48
lambda2_s = 54
cm1 = 480
cm2 = 460
rho = 7850 #kg/m^3

# PASO 2.1: condiciones de proceso
T_i =  25.0 #°C
T_gas = 685 #°C, temperatura de saturación del gas (T_sat)
h_eq = 120 #W/m^2K

# validación de los parámetros modificables
if D_int >= D_ext:
    raise ValueError("D_int debe ser menor que D_ext")
if T_obj != sorted(T_obj):
    raise ValueError("Las temperaturas objetivo deben ir en orden ascendente")
if max(T_obj) >= T_gas:
    raise ValueError("Las temperaturas objetivo deben ser menores que T_gas")

# PASO 2.2: parámetros para calcular la función
P = 8 #MPa
H_V = 1133.86
eps = 0.08554
sigma_P = 1.5e-6 #rugosidad en m
b_O = 10e-6 #oxido en m
cte_SB = 5.67e-8 #W/m^2K^4
lambda_O = 2
tan_theta = 0.1
K = 273.15 # para convertir de celsius a kelvin

# PASO 3: tamaño de la malla
M = 50
N = 40 # num de intervalos
delta_r = (r_ext - r_int) / M
delta_z = H / N
r = np.linspace(r_int, r_ext, M+1)
z = np.linspace(0, H, N+1)

# indices del cold spot
m_cold = round(0.4 *M) # radial
n_cold = round(N/2) # axial

# PASO 4: red de resistencias 
def calcular_kr(T, lambda_s, P=P):
    phi = P/(H_V + P)
    b = 42.7e-6 * np.exp(-0.05 * P)
    T_K = T + K
    lambda_g = 0.18 + 3.86e-4 * (T_K - 300)

    radiacion = 4*(1-phi)*eps*cte_SB*T_K**3 / (2-eps)
    gas = (1-phi)*lambda_g / b
    contacto = 1.13 * lambda_s*tan_theta / (sigma_P) * phi**0.94

    lambda_r = (S+b+b_O) / (S/lambda_s + b_O/lambda_O + 1/(radiacion + gas + contacto))
    return lambda_r

# PASO 5: paso temporal critico
def calcular_dt(rho, cm, lambda_s, P=P):
    k_r = calcular_kr(T_gas, lambda_s, P)
    k_z = lambda_s

    denominador = (2*k_r/delta_r**2 + 2*k_z/delta_z**2
                   + 2*h_eq/delta_r + 2*h_eq/delta_z)
    dt_crit = rho*cm / denominador
    return 0.9 * dt_crit

# PASO 6: fijar la matriz de temperaturas iniciales
dt1 = calcular_dt(rho, cm1, lambda1_s)
dt2 = calcular_dt(rho, cm2, lambda2_s)
T = np.full((M+1, N+1), T_i)  # condicion inicial

# PASO 7: bucle temporal
def avanzar(T, dt, cm, lambda_s, P=P):
    k_r = calcular_kr(T, lambda_s, P)
    k_z = lambda_s

    R = np.zeros_like(T) # radial
    Z = np.zeros_like(T) # axial

    # término radial por filas m
    r_m = r[1:-1, None]
    Tc = T[1:-1, :]
    R[1:-1, :] = k_r[1:-1, :] / (r_m*delta_r**2) * ((r_m + delta_r/2)*(T[2:, :] - Tc) + (r_m - delta_r/2)*(T[:-2, :] - Tc))

    # cara interna m = 0: vecino m = 1, cociente r_{1/2}/r_0
    R[0, :] = 2/delta_r * (h_eq*(T_gas - T[0, :]) + k_r[0, :]*((r[0] + delta_r/2)/r[0])*(T[1, :] - T[0, :])/delta_r)
    # cara externa m = M: vecino m = M-1, cociente r_{M-1/2}/r_M
    R[-1, :] = 2/delta_r * (h_eq*(T_gas - T[-1, :]) + k_r[-1, :]*((r[-1] - delta_r/2)/r[-1])*(T[-2, :] - T[-1, :])/delta_r)

    # término axial por columnas n
    Z[:, 1:-1] = k_z * (T[:, 2:] - 2*T[:, 1:-1] + T[:, :-2]) / delta_z**2

    # cara inferior n = 0: vecino n = 1
    Z[:, 0] = 2/delta_z * (h_eq*(T_gas - T[:, 0]) + k_z*(T[:, 1] - T[:, 0])/delta_z)
    # cara superior n = N: vecino n = N-1
    Z[:, -1] = 2/delta_z * (h_eq*(T_gas - T[:, -1]) + k_z*(T[:, -2] - T[:, -1])/delta_z)

    return T + dt/(rho*cm) * (R + Z)

# PASO 8: criterio de parada 
def simular(dt, cm, lambda_s, objetivos, t_max=100*3600, P=P, mostrar=True):
    T = np.full((M+1, N+1), T_i)
    tiempo = 0.0
    tiempos_obj = []
    hist_t = [0.0]
    hist_Tcold = [T[m_cold, n_cold]]

    while len(tiempos_obj) < len(objetivos) and tiempo < t_max:
        T = avanzar(T, dt, cm, lambda_s, P)
        tiempo += dt
        hist_t.append(tiempo)
        hist_Tcold.append(T[m_cold, n_cold])

        T_sig = objetivos[len(tiempos_obj)]
        if T[m_cold, n_cold] >= T_sig:
            tiempos_obj.append(tiempo/3600)
            if mostrar:
                print(f"{T_sig} °C alcanzados en {tiempo/3600:.2f} h")

    if len(tiempos_obj) < len(objetivos):
        print("Aviso: se llegó a t_max sin alcanzar todos los objetivos")
    return tiempos_obj, hist_t, hist_Tcold, T

# PASO 9: correr la simulación para ambos aceros
print("Rollo 1: ")
t1, hist_t1, hist_T1, T_final1 = simular(dt1, cm1, lambda1_s, T_obj)

print("Rollo 2: ")
t2, hist_t2, hist_T2, T_final2 = simular(dt2, cm2, lambda2_s, T_obj)

# PASO 10: analisis de sensibilidad
def sensibilidad(valores_P, T_objetivo=T_obj[0]):
    aceros = [("Acero 1", lambda1_s, cm1), ("Acero 2", lambda2_s, cm2)]
    resultados = {}

    print(f"\nSensibilidad a la presión P (objetivo: {T_objetivo} °C)")
    for nombre, lambda_s, cm in aceros:
        resultados[nombre] = []
        print(f"\n{nombre}")
        print(f"{'P (MPa)':>8} {f'k_r({T_gas}) (W/mK)':>16} {'dt (s)':>8} {'t (h)':>7}")
        print("-" * 42)
        for P_i in valores_P:
            dt = calcular_dt(rho, cm, lambda_s, P_i)    # Δτ cambia con P
            t, _, _, _ = simular(dt, cm, lambda_s, [T_objetivo], P=P_i, mostrar=False)
            resultados[nombre].append(t[0])
            k_r = calcular_kr(T_gas, lambda_s, P_i)
            print(f"{P_i:>8} {k_r:>16.2f} {dt:>8.2f} {t[0]:>7.2f}")
    return resultados

valores_P = [4, 6, 8, 10, 12]
res_sens = sensibilidad(valores_P)

# PASO 11: resultados y limites
print("\nBrecha de tiempos entre aceros: Δτ_rango = τ(α_min) − τ(α_max)")
print(f"{'T obj (°C)':>10} {'t1 (h)':>8} {'t2 (h)':>8} {'Δτ (h)':>8} {'Δτ (%)':>8}")
print("-" * 46)
for T_ob, ta, tb in zip(T_obj, t1, t2):
    print(f"{T_ob:>10} {ta:>8.2f} {tb:>8.2f} {ta - tb:>8.2f} {100*(ta - tb)/ta:>8.1f}")

AZUL, NARANJA = "#2a78d6", "#eb6834"

# carpeta "resultados" junto a este archivo .py
carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
os.makedirs(carpeta, exist_ok=True)


# Gráfica 1: curva de calentamiento del cold spot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.array(hist_t1)/3600, hist_T1, color=AZUL, lw=2, label="Acero 1: SAE 1010 (α_min)")
ax.plot(np.array(hist_t2)/3600, hist_T2, color=NARANJA, lw=2, label="Acero 2: SAE 1006 (α_max)")
for T_ob in T_obj:
    ax.axhline(T_ob, color="gray", ls="--", lw=0.8)
    ax.text(1.01, T_ob, f"{T_ob} °C", color="gray", fontsize=8, va="center",
            transform=ax.get_yaxis_transform())
ax.axhline(T_gas, color="black", ls=":", lw=1)
ax.text(0.3, T_gas + 8, f"T_gas = {T_gas} °C", fontsize=8)
ax.set_xlabel("Tiempo (h)")
ax.set_ylabel("Temperatura del cold spot (°C)")
ax.set_title("Calentamiento del cold spot")
ax.set_ylim(0, T_gas + 40)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(carpeta, "1_calentamiento_cold_spot.png"), dpi=200)

# Gráfica 2: mapa T(r, z) cuando el cold spot del acero 1 llega a la primera temperatura objetivo
_, _, _, T_mapa = simular(dt1, cm1, lambda1_s, [T_obj[0]], mostrar=False)
fig, ax = plt.subplots(figsize=(6, 6))
mapa = ax.pcolormesh(r, z, T_mapa.T, shading="auto", cmap="Oranges")
fig.colorbar(mapa, ax=ax, label="T (°C)")
ax.plot(r[m_cold], z[n_cold], "kx", ms=10, mew=2)
ax.set_xlabel("r (m)")
ax.set_ylabel("z (m)")
ax.set_title(f"Acero 1: T(r, z) cuando el cold spot llega a {T_obj[0]} °C")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(carpeta, "2_mapa_T_rz.png"), dpi=200)

# Gráfica 3: sensibilidad del tiempo a P
fig, ax = plt.subplots(figsize=(7, 4.5))
for (nombre, tiempos), color in zip(res_sens.items(), [AZUL, NARANJA]):
    ax.plot(valores_P, tiempos, "o-", color=color, lw=2, ms=7, label=nombre)
ax.set_xlabel("Presión entre espiras P (MPa)")
ax.set_ylabel(f"Tiempo hasta {T_obj[0]} °C (h)")
ax.set_title("Sensibilidad del tiempo de recocido a P")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(carpeta, "3_sensibilidad_P.png"), dpi=200)



# PASO 12: guardar resultados
# tabla 1: brecha entre aceros
brecha = pd.DataFrame({
    "T objetivo (°C)": T_obj,
    "t Acero 1 (h)": t1,
    "t Acero 2 (h)": t2,
})
brecha["Δτ (h)"] = brecha["t Acero 1 (h)"] - brecha["t Acero 2 (h)"]
brecha["Δτ (%)"] = 100 * brecha["Δτ (h)"] / brecha["t Acero 1 (h)"]

# tabla 2: sensibilidad a P (una fila por acero y por P)
filas = []
for nombre, lambda_s, cm in [("Acero 1", lambda1_s, cm1), ("Acero 2", lambda2_s, cm2)]:
    for P_i, t_P in zip(valores_P, res_sens[nombre]):
        filas.append({"Acero": nombre, "P (MPa)": P_i,
                      f"k_r({T_gas}) (W/mK)": calcular_kr(T_gas, lambda_s, P_i),
                      "dt (s)": calcular_dt(rho, cm, lambda_s, P_i),
                      f"t hasta {T_obj[0]} °C (h)": t_P})
sens = pd.DataFrame(filas)

# tabla 3: historia del cold spot (para graficar en Excel si se quiere)
hist = pd.concat([
    pd.DataFrame({"t Acero 1 (h)": np.array(hist_t1)/3600, "T_cold Acero 1 (°C)": hist_T1}),
    pd.DataFrame({"t Acero 2 (h)": np.array(hist_t2)/3600, "T_cold Acero 2 (°C)": hist_T2}),
], axis=1)

# un solo Excel, una hoja por tabla
ruta_excel = os.path.join(carpeta, "resultados_recocido.xlsx")
with pd.ExcelWriter(ruta_excel) as writer:
    brecha.to_excel(writer, sheet_name="Brecha", index=False)
    sens.to_excel(writer, sheet_name="Sensibilidad", index=False)
    hist.to_excel(writer, sheet_name="Historia cold spot", index=False)
print("Excel guardado en:", ruta_excel)

plt.show()