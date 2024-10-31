import numpy as np
import matplotlib.pyplot as plt

def plot_temporal(valores, title, outpath): 
    # Calcular la media y la desviación estándar
    media = np.mean(valores)
    std_dev = np.std(valores)

    # Calcular los límites de ±2SD
    limite_superior = media + 2 * std_dev
    limite_inferior = media - 2 * std_dev

    # Crear los índices para el eje X
    indices = range(1, len(valores) + 1)

    # Crear la figura y el eje
    fig, ax = plt.subplots()

    # Graficar los valores
    ax.plot(indices, valores, marker='o', linestyle='-', color='b')

    # Añadir la línea de la media
    ax.axhline(media, color='g', linestyle='-')
    ax.text(len(indices) + 0.1, media + 0.05, f'Mean: {media:.2f}', color='g', verticalalignment='bottom', fontsize=12)

    # Añadir las líneas de media ± 2SD
    ax.axhline(limite_superior, color='r', linestyle='--')
    ax.text(len(indices) + 0.1, limite_superior - 0.05, f'Mean + 2SD: {limite_superior:.2f}', color='r', verticalalignment='top', fontsize=12)

    ax.axhline(limite_inferior, color='r', linestyle='--')
    ax.text(len(indices) + 0.1, limite_inferior + 0.05, f'Mean - 2SD: {limite_inferior:.2f}', color='r', verticalalignment='bottom', fontsize=12)

    # Añadir números de muestras a cada punto
    for i, valor in enumerate(valores):
        ax.annotate(f'{indices[i]}', (indices[i], valores[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)

    # Ajustar los límites del eje X para dar espacio a las etiquetas
    ax.set_xlim(1, len(valores) + 2)

    # Añadir título y etiquetas de los ejes
    ax.set_xlabel('ASL pairs', fontsize=12)
    ax.set_ylabel('Intensity values (a.u.)', fontsize=12)

    # ax.set_title(title, fontsize=14)
    # Guardar la gráfica como PNG
    # plt.savefig(outpath + title + '.png')
    plt.savefig(outpath + title + '.eps', format='eps')
    # Mostrar el gráfico
    plt.show()




