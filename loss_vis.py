import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def loaddata(filepath):
    """Carga archivo de topología generado por topology.py"""
    checkpoint = torch.load(os.path.join("largedata", filename), map_location="cpu")
    iter_num = checkpoint['iter_num']
    loss = checkpoint['loss']
    return (iter_num, loss)

data = []
for filename in sorted(os.listdir('largedata')):  
    data.append(loaddata(filename))


data.sort(key=lambda t: t[0])
temp, losses = zip(*data)
x = np.array(temp)
y = np.array(losses)

plt.plot(x, y, label='Valor función de pérdida')
plt.xlabel("Iteración")
plt.ylabel("Pérdida")
plt.title("Función de pérdida por iteración")	
plt.grid(True)
plt.savefig("loss_plot.png")
plt.close()