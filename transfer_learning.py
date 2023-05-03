import numpy as np
import pandas as pd
import argparse

import tensorflow.keras as k
from tensorflow.keras import backend as K

from predfull import getmod, fastmass, embed
from train_model import readmgf, spectrum2vector, asnp32

max_charge = 30
BIN_SIZE = 0.1
Alist = list('ACDEFGHIKLMNPQRSTVWYZ')
oh_dim = len(Alist) + 3
x_dim = oh_dim + 2 + 3
meta_shape = (3, 30) # (charge, ftype, other(mass, nce))
mz_scale = 20000.0
xshape = [-1, x_dim]

# preprocess function for inputs
def preprocessor(batch):
    batch_size = len(batch)
    embedding = np.zeros((batch_size, *xshape), dtype='float32')
    meta = np.zeros((batch_size, *meta_shape), dtype='float32')

    for i, sp in enumerate(batch):
        pep = sp['pep']

        embed(sp, embedding=embedding[i])
        meta[i][0][sp['charge'] - 1] = 1 # charge
        meta[i][1][sp['type']] = 1 # ftype
        meta[i][2][0] = fastmass(pep, ion_type='M', charge=1) / mz_scale         

        if not 'nce' in sp or sp['nce'] == 0:
            meta[i][2][-1] = 0.25
        else:
            meta[i][2][-1] = sp['nce'] / 100.0
        
    return (embedding, meta)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mgf', type=str,
                        help='train mgf', default='data/train.mgf')
    parser.add_argument('--tsv', type=str,
                        help='train psm', default='data/train.tsv')
    parser.add_argument('--model', type=str,
                        help='model path', default='pm.h5')
    parser.add_argument('--out', type=str,
                        help='filename to save the trained model', default='new.h5')
    parser.add_argument('--epochs', type=int, 
                        help='train epochs', default = 20)
    args = parser.parse_args()

    K.clear_session()
    pm = k.models.load_model(args.model, compile=0)
    pm.compile(optimizer=k.optimizers.Adam(lr=0.0003), loss='cosine_similarity')

    types = {'un': 0, 'cid': 1, 'etd': 2, 'hcd': 3, 'ethcd': 4, 'etcid': 5}

    # read inputs
    inputs, outputs = [], []
    df = pd.read_csv(args.tsv, sep='\t')
    for item in df.itertuples():
        if item.Charge < 1 or item.Charge > max_charge:
            print("input", item.Peptide, 'exceed max charge of', max_charge, ", ignored")
            continue

        pep, mod, nterm_mod = getmod(item.Peptide)

        if nterm_mod != 0:
            print("input", item.Peptide, 'has N-term modification, ignored')
            continue

        if np.any(mod != 0) and set(mod) != set([0, 1]):
            print("Only Oxidation modification is supported, ignored", item.Peptide)
            continue

        inputs.append({'pep': pep, 'mod': mod, 'charge': item.Charge, 'title': item.Peptide,
                    'nce': item.NCE, 'type': types[item.Type.lower()],
                    'mass': fastmass(pep, 'M', item.Charge, mod=mod)})
                    
        xshape[0] = max(xshape[0], len(pep) + 2) # update xshape to match max input peptide

    embedding, meta = preprocessor(inputs)
    y = []
    spectra = readmgf(args.mgf)
    for sp in spectra:
        y.append(spectrum2vector(sp['mz'], sp['it'], sp['mass'], BIN_SIZE, sp['charge']))
    y = np.array(y)

    assert len(embedding) == len(y)
    idx_all = np.arange(len(y))
    np.random.shuffle(idx_all)
    train_num = int(len(y) * 0.85)
    train_idx, test_idx = idx_all[:train_num], idx_all[train_num:]
    train_X, train_y = (embedding[train_idx], meta[train_idx]), y[train_idx]
    test_X, test_y = (embedding[test_idx], meta[test_idx]), y[test_idx]
    print('Train size', len(train_y), 'Valid size', len(test_y))
    best_loss = 0
    for i in range(args.epochs):
        pm.fit(x=train_X, y=asnp32(train_y), verbose=0)
        val_loss = pm.evaluate(x=test_X, y = asnp32(test_y), verbose=1)
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Save model at epoch {i} with loss {best_loss}")
            pm.save(args.out)
    