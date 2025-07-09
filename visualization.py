import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import gudhi

def load_topology(filepath):
    """Carga archivo de topología generado por topology.py"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def visualize_all(filepath, output_dir="visualizations-combined", figures=[True, True, True]):
    """Genera una única imagen con los tres gráficos en columnas: Euler, espectro, persistencia"""
    os.makedirs(output_dir, exist_ok=True)
    data = load_topology(filepath)

    ind = 0
    
    # Preparar figura con 3 subplots en una fila
    fig, axs = plt.subplots(1, sum(figures), figsize=(18, 5))
    
    
    print(min(data['euler_characteristic'].keys(), key=lambda k: abs(k - 0.03076923)))
    

    # -------- Gráfico 1: Característica de Euler --------
    if(figures[0]):
        chi = data['euler_characteristic']
        thresholds = sorted(chi.keys())
        values = [chi[t] for t in thresholds]

        axs[ind].plot(thresholds, values, label='Característica de Euler')
        #axs[ind].set_ylim(bottom=-800, top=128)
        axs[ind].set_xlabel("Umbral")
        axs[ind].set_ylabel(r"$\chi$")
        axs[ind].set_title("Característica de Euler")
        axs[ind].grid(True)
        ind += 1

    # -------- Gráfico 2: Espectro del Laplaciano --------
    if(figures[1]):
        spectra = data['laplacian_spectra']
        thresholds = sorted(spectra.keys())
        values = [spectra[t] for t in thresholds]
        max_len = max(len(v) for v in values)
        #max_len = 128
        matrix = np.full((len(thresholds), max_len), 0.0)
        for i, row in enumerate(values):
            matrix[i, :len(row)] = row
        #print(matrix.shape)
        im = axs[ind].imshow(matrix.T, aspect='auto', origin='lower',
                        extent=[thresholds[0], thresholds[-1], 0, max_len],
                        cmap='viridis', vmin=0)
        #axs[ind].set_ylim(top=128)
        axs[ind].set_xlabel("Umbral")
        axs[ind].set_ylabel("Índice espectral")
        axs[ind].set_title("Espectro del Laplaciano")
        fig.colorbar(im, ax=axs[1], orientation='vertical', shrink=0.7)
        ind += 1
    
    # -------- Gráfico 3: Diagrama de persistencia --------
    if(figures[2]):
        diagram = data['persistence_diagram']
        gudhi.plot_persistence_diagram(diagram,axes=axs[2])

        axs[ind].set_title("Diagrama de persistencia")
    
    

    filepath2 = filepath.split("/")[-1]
    # Ajustar y guardar
    plt.tight_layout()
    plt.savefig(output_dir+"/"+filepath2.replace("_topology.pkl", ".png"))
    plt.close()
    return output_dir+"/"+filepath2.replace("_topology.pkl", ".png")

def visualize_separated(filepath, output_dir="visualizations-split", figures=[True, True, True]):
    """Genera una imagen separada de los tres gráficos: Euler, espectro, persistencia"""
    os.makedirs(output_dir, exist_ok=True)
    data = load_topology(filepath)
    filepath2 = filepath.split("/")[-1]    

    # -------- Gráfico 1: Característica de Euler --------
    if(figures[0]):
        plt.figure(figsize=(6, 6))
        chi = data['euler_characteristic']
        thresholds = sorted(chi.keys())
        values = [chi[t] for t in thresholds]

        plt.plot(thresholds, values, label='Característica de Euler')
        plt.xlabel("Umbral")
        plt.ylabel(r"$\chi$")
        plt.title("Característica de Euler")
        plt.grid(True)
        plt.savefig(output_dir+"/"+filepath2.replace("_topology.pkl", "_euler.png"))
        plt.close()

    # -------- Gráfico 2: Espectro del Laplaciano --------
    if(figures[1]):
        plt.figure(figsize=(6, 6))
        spectra = data['laplacian_spectra']
        thresholds = sorted(spectra.keys())
        values = [spectra[t] for t in thresholds]
        max_len = max(len(v) for v in values)
        matrix = np.full((len(thresholds), max_len), np.nan)
        for i, row in enumerate(values):
            matrix[i, :len(row)] = row

        im = plt.imshow(matrix.T, aspect='auto', origin='lower',
                        extent=[thresholds[0], thresholds[-1], 0, max_len],
                        cmap='viridis')
        plt.xlabel("Umbral")
        plt.ylabel("Índice espectral")
        plt.title("Espectro del Laplaciano")
        plt.colorbar(label='Autovalor')
        plt.savefig(output_dir+"/"+filepath2.replace("_topology.pkl", "_laplaciano.png"))
        plt.close()
    
    # -------- Gráfico 3: Diagrama de persistencia --------
    if(figures[2]):
        plt.figure(figsize=(6, 6))
        diagram = data['persistence_diagram']
        gudhi.plot_persistence_diagram(diagram)

        plt.title("Diagrama de persistencia")
        plt.savefig(output_dir+"/"+filepath2.replace("_topology.pkl", "_persistence.png"))
        plt.close()
