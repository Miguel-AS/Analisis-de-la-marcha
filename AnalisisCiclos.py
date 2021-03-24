import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_valores(archivo, hoja, inv=1):
    # CARGAMOS LA BASE DE DATOS
    # data = pd.read_excel("datosmarchacopia.xlsx", 0, header=160, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11])
    data = pd.read_excel(archivo, hoja, header=8, usecols='A:U')
    df = pd.DataFrame(data)
    df = df.dropna()
    filas = df.shape[0]
    # DECLARAMOS VARIABLES INTERNAS
    Mx = df['x.2'] - df['x.1']
    My = df['y.2'] - df['y.1']
    M = []
    Tx = df['x.4'] - df['x.3']
    Ty = df['y.4'] - df['y.3']
    T = []
    Px = df['x.6'] - df['x.5']
    Py = df['y.6'] - df['y.5']
    P = []
    Xx = np.ones(filas)
    Xy = np.zeros(filas)
    X = []
    # CALCULAMOS LOS MODULOS:
    for i in range(filas):
        M.append(math.sqrt(Mx[i] ** 2 + My[i] ** 2))
        T.append(math.sqrt(Tx[i] ** 2 + Ty[i] ** 2))
        P.append(math.sqrt(Px[i] ** 2 + Py[i] ** 2))
        X.append(math.sqrt(Xx[i] ** 2 + Xy[i] ** 2))

    df2 = pd.DataFrame(list(zip(Mx, My, M, Tx, Ty, T, Px, Py, P, Xx, Xy, X)), columns=['Mx', 'My', 'M', 'Tx', 'Ty', 'T', 'Px', 'Py', 'P', 'Xx', 'Xy', 'X'])
    filas2 = df2.shape[0]
    ejex = np.linspace(0, 100, filas2)
    # modulos = df['|X|'] * df['|M|']
    tita = []
    alfa = []
    phi = []
    beta = []
    psi = []
    delta = []
    for i in range(filas2):
        # angulo.append(math.cos(math.radians(modulos[i])))
        tita.append(math.acos((df2['Xx'][i]*df2['Mx'][i])/(df2['X'][i] * df2['M'][i])))
        alfa.append((90 - tita[i]*(180/math.pi)) * inv)
        phi.append(math.acos((df2['Xx'][i]*df2['Tx'][i])/(df2['X'][i] * df2['T'][i])))
        beta.append(((phi[i] - tita[i])*(180 / math.pi)) * inv)
        psi.append(math.acos((df2['Tx'][i]*df2['Px'][i] + df2['Ty'][i]*df2['Py'][i])/(df2['T'][i] * df2['P'][i])))
        delta.append((90 - psi[i] * (180 / math.pi)) * inv)

    return ejex, alfa, beta, delta

# DECLARAMOS VARIABLE EXTERNAS
num_ciclos = int(input('ingrese el numero de cilos a analizar '))
datos_normales = pd.read_excel('TobilloNormal.xlsx')
ejex_comun = np.linspace(0, 100, 300)
ejex = []
alfa = []
beta = []
delta = []
ejex_der = []
alfa_der = []
beta_der = []
delta_der = []
alfa_reg = []
beta_reg = []
delta_reg = []
alfa_reg_der = []
beta_reg_der = []
delta_reg_der = []
alfa_nor = []  # son los parametros que tienen la misma cantidad de valores
beta_nor = []
delta_nor = []
alfa_nor_der = []
beta_nor_der = []
delta_nor_der = []
prom_alfa = 0
prom_beta = 0
prom_delta = 0
prom_alfa_der = 0
prom_beta_der = 0
prom_delta_der = 0
# INICIO DEL PROGRAMA:
def graficar(lado):

    fig, axs = plt.subplots(2, 2)
    plt.tight_layout(pad=1.5, w_pad=0.5, h_pad=1.5)

    axs[0, 0].axhline(0, lw=2, dashes=[5, 5], color='black')
    #axs[0, 0].axvline(60, lw=2, dashes=[5, 5], color='black')
    #axs[0, 0].axvline(59, lw=2, dashes=[5, 5], color='black')
    axs[0, 0].axvline(65, lw=2, dashes=[5, 5], color='black')
    axs[0, 0].set_title('Angulos de la cadera', fontsize=14, color='blue')
    axs[0, 0].legend([f'{lado}'])
    axs[0, 0].set(ylabel='Angulo')
    axs[0, 0].set(xlabel='% Ciclo')
    axs[0, 0].grid(True)

    axs[0, 1].axhline(0, lw=2, dashes=[5, 5], color='black')
    #axs[0, 1].axvline(60, lw=2, dashes=[5, 5], color='black')
    #axs[0, 1].axvline(59, lw=2, dashes=[5, 5], color='black')
    axs[0, 1].axvline(65, lw=2, dashes=[5, 5], color='black')
    axs[0, 1].set_title('Angulos de la rodilla', fontsize=14, color='blue')
    axs[0, 1].legend([f'{lado}'])
    axs[0, 1].set(ylabel='Angulo')
    axs[0, 1].set(xlabel='% Ciclo')
    axs[0, 1].grid(True)

    axs[1, 0].axhline(0, lw=2, dashes=[5, 5], color='black')
    #axs[1, 0].axvline(60, lw=2, dashes=[5, 5], color='black')
    #axs[1, 0].axvline(59, lw=2, dashes=[5, 5], color='black')
    axs[1, 0].axvline(65, lw=2, dashes=[5, 5], color='black')
    axs[1, 0].set_title('Angulos del tobillo', fontsize=14, color='blue')
    axs[1, 0].legend([f'{lado}'])
    axs[1, 0].set(ylabel='Angulo')
    axs[1, 0].set(xlabel='% Ciclo')
    axs[1, 0].grid(True)
    return fig, axs


# REALIZA LAS REGRESIONES

for i in range(num_ciclos):
    # ejex[i][:], alfa[i][:], beta[i][:], delta[i][:] = cargar_valores(i)
    a, b, c, d = cargar_valores("ciclos_izquierdos.xlsx", i, inv=-1)

    ejex.insert(i, a)
    alfa.insert(i, b)
    beta.insert(i, c)
    delta.insert(i, d)

    alfa_reg.insert(i, np.poly1d(np.polyfit(ejex[i], alfa[i], 12)))
    beta_reg.insert(i, np.poly1d(np.polyfit(ejex[i], beta[i], 12)))
    delta_reg.insert(i, np.poly1d(np.polyfit(ejex[i], delta[i], 12)))

    alfa_nor.insert(i, alfa_reg[i](ejex_comun))
    beta_nor.insert(i, beta_reg[i](ejex_comun))
    delta_nor.insert(i, delta_reg[i](ejex_comun))

for i in range(num_ciclos):

    a, b, c, d = cargar_valores("ciclos_derechos.xlsx", i)
    d = np.array(d)
    d = d * -1  # este paso se realiza porque los datos vienen invertidos
    ejex_der.insert(i, a)
    alfa_der.insert(i, b)
    beta_der.insert(i, c)
    delta_der.insert(i, d)

    alfa_reg_der.insert(i, np.poly1d(np.polyfit(ejex_der[i], alfa_der[i], 12)))
    beta_reg_der.insert(i, np.poly1d(np.polyfit(ejex_der[i], beta_der[i], 12)))
    delta_reg_der.insert(i, np.poly1d(np.polyfit(ejex_der[i], delta_der[i], 12)))

    alfa_nor_der.insert(i, alfa_reg_der[i](ejex_comun))
    beta_nor_der.insert(i, beta_reg_der[i](ejex_comun))
    delta_nor_der.insert(i, delta_reg_der[i](ejex_comun))

def calcular_promedio():

    for i in range(num_ciclos):
        global prom_alfa, prom_beta, prom_delta, prom_alfa_der, prom_beta_der, prom_delta_der
        prom_alfa += alfa_nor[i] / num_ciclos
        prom_beta += beta_nor[i] / num_ciclos
        prom_delta += delta_nor[i] / num_ciclos
        prom_alfa_der += alfa_nor_der[i] / num_ciclos
        prom_beta_der += beta_nor_der[i] / num_ciclos
        prom_delta_der += delta_nor_der[i] / num_ciclos

    # GRAFICAS:

'''    fig, axs = graficar(lado='Miembro izquierdo')
    for i in range(num_ciclos):
        axs[0, 0].plot(ejex[i], alfa[i])
        axs[0, 1].plot(ejex[i], beta[i])
        axs[1, 0].plot(ejex[i], delta[i])
        axs[0, 0].plot(ejex_comun, alfa_nor[i])
        axs[0, 1].plot(ejex_comun, beta_nor[i])
        axs[1, 0].plot(ejex_comun, delta_nor[i])
    plt.show()
    plt.close('all')

    fig, axs = graficar(lado='Miembro izquierdo regresion')
    for i in range(num_ciclos):
        axs[0, 0].plot(ejex_comun, alfa_nor[i])
        axs[0, 1].plot(ejex_comun, beta_nor[i])
        axs[1, 0].plot(ejex_comun, delta_nor[i])
    plt.show()

    fig, axs = graficar(lado='Miembro izquierdo regresion')
    for i in range(num_ciclos):
        axs[0, 0].plot(ejex_comun, prom_alfa)
        axs[0, 1].plot(ejex_comun, prom_beta)
        axs[1, 0].plot(ejex_comun, prom_delta)
    plt.show()

    fig, axs = graficar(lado='Miembro derecho').
    for i in range(num_ciclos):
        axs[0, 0].plot(ejex_der[i], alfa_der[i])
        axs[0, 1].plot(ejex_der[i], beta_der[i])
        axs[1, 0].plot(ejex_der[i], delta_der[i])
        axs[0, 0].plot(ejex_comun, alfa_nor_der[i])
        axs[0, 1].plot(ejex_comun, beta_nor_der[i])
        axs[1, 0].plot(ejex_comun, delta_nor_der[i])
    plt.show()
'''
def main():
    # GRAFICOS FINALES:
    calcular_promedio()
    #leyenda = ['Rodilla izquierda', 'Rodilla derecha', 'Normal']
    plt.axhline(0, lw=2, dashes=[5, 5], color='black')
    plt.axvline(65, lw=2, dashes=[5, 5], color='blue')
    plt.axvline(59, lw=2, dashes=[5, 5], color='orange')
    plt.axvline(60, lw=2, dashes=[5, 5], color='green')
    plt.title('Angulos de la cadera', fontsize=14, color='blue')
    plt.grid(True)
    plt.plot(ejex_comun, prom_delta, label='Tobillo izquierdo')
    plt.plot(ejex_comun, prom_delta_der, label='Tobillo derecho')
    sns.lineplot(data=datos_normales, x='% ciclo', y='angulo', label='Normal')
    plt.show()

if __name__ == '__main__':
    main()
