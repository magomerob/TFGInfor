import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import gudhi
from gudhi.representations import Entropy

def load_topology(filepath):
    """Carga archivo de topología generado por topology.py"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

data = []
for filename in sorted(os.listdir('topolong')):
    temp = int(filename.split("_")[0])
    topo = load_topology('topolong/'+filename)
    data.append((temp, topo['euler_characteristic'][np.float64(0.05263157894736842)*3]))

data.sort(key=lambda t: t[0])
temp, entropies = zip(*data)
x = np.array(temp)
y = np.array(entropies)

plt.plot(x, y, label='Característica de Euler')
plt.xlabel("Iteración")
plt.ylabel(r"$\chi$")
plt.title("Característica de Euler en el umbral 0.15")
plt.grid(True)
plt.savefig("eulervsiter3.png")
plt.close()