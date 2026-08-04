from warnings import filterwarnings

import scanpy as sc
from pathlib import Path
import muon as mu
import pandas as pd
import scanpy as sc


filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

sc.set_figure_params(fontsize=18, figsize=[6, 6])

def print_size_in_MB(x):
    print(f"{x.__sizeof__() / 1e6:.3} MB")



#adata = sc.read_h5ad("/Users/saberhq/code/vcc/vcc_data/adata_Training.h5ad", backed="r").to_memory()
#adata = adata[:50000].copy()  # adjust down
#adata.write("/Users/saberhq/code/vcc/vcc_data/adata_Training_subset.h5ad")


adata = sc.read_h5ad("/Users/saberhq/code/vcc/vcc_data/adata_Training.h5ad", backed=True)
print_size_in_MB(adata)
adata.isbacked

sc.pl.umap(adata, color=["louvain", "CD4"], wspace=0.6)