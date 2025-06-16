import torch
from model import GPTConfig, GPT
import os
import os.path
from pathlib import Path
import pickle
import numpy as np
import torch.nn.functional as F

def extract(filename, prompt="BRUTUS:",input_dir='out-analisis' ,output_dir="matrices", head=0, layer=0, text_dir="text", temperature=0.0, evolution=False, timesblock=1):
    """
    Reconstruye un modelo GPT desde un checkpoint y extrae la matriz de atención 
    de una cabeza y capa específicas a partir de un prompt. Permite guardar la evolución 
    durante la generación si se desea.
    
    Args:
        filename (str): Nombre del checkpoint.
        prompt (str): Entrada inicial del modelo.
        input_dir (str): Carpeta de entrada de checkpoints.
        output_dir (str): Carpeta para guardar matrices.
        head (int): Cabeza de atención a extraer.
        layer (int): Capa de atención a extraer.
        text_dir (str): Carpeta para guardar texto generado.
        temperature (float): Controla aleatoriedad en generación.
        evolution (bool): Guarda todas las matrices generadas si True.
        timesblock (int): Número de bloques de generación.
    Returns:
        list: Lista de ficheros .npy generados.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    i = filename.split('_')[0]  # Identificador simple del fichero
    checkpoint = torch.load(os.path.join(input_dir, filename), map_location="cpu")
    print(checkpoint['iter_num'])   
    print(input_dir+'/'+filename)

    # Reconstrucción del modelo desde el checkpoint
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Carga del diccionario del dataset
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Codificación del prompt
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device="cpu")[None, ...]

    if evolution:
        retMatrices = []
        retText = []

    # Generación autoregresiva de tokens hasta completar el bloque
    while x.shape[1] < checkpoint['config']['block_size'] * timesblock:
        with torch.no_grad():
            logits, _, attn_matrix = model(x[:, -checkpoint['config']['block_size']:])
            if evolution:
                retMatrices.append(attn_matrix[head].squeeze(1).cpu().numpy()[layer])

        # Selección del siguiente token
        if temperature <= 0.0:
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        else:
            scaled = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, next_token), dim=1)

        if evolution:
            retText.append(decode(x[0].tolist()))

    outtext = decode(x[0].tolist())

    # Segunda pasada para extraer atención final
    with torch.no_grad():
        logits, _, attn_matrices = model(x[:, -checkpoint['config']['block_size']:])

    if attn_matrices is not None:
        attn_matrix = attn_matrices[head].squeeze(1)
        attn_np = attn_matrix.cpu().numpy()

        if evolution:
            retMatrices.append(attn_np[layer])
            fileList = []
            for j, m in enumerate(retMatrices):
                path = os.path.join(output_dir, f"{i}_evolution_{j}.npy")
                np.save(path, m)
                fileList.append(path)
            for j, t in enumerate(retText):
                with open(os.path.join(text_dir, f"{i}_evolution_{j}.txt"), "w") as text_file:
                    text_file.write(t)
            return fileList

        # Guardado de matriz de atención final y texto generado
        np.save(os.path.join(output_dir, f"{i}.npy"), attn_np[layer])
        with open(os.path.join(text_dir, f"{i}.txt"), "w") as text_file:
            text_file.write(outtext)

        return [os.path.join(output_dir, f"{i}.npy")]