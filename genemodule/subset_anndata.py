import argparse
import random,os
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc
#from load import *

adata = sc.read_h5ad("/w5home/bmoore/scRNAseq/GAMM/GAMM_S2/output_20230830_155530/GAMM_S2_clabeled-clusters_0.5.h5ad")
set(adata.obs["CellType_manual"])
conedata = adata[adata.obs.CellType_manual == "Cones"]
conedata
# weird seurat to anndata thing- anndata doesn not like a column named _index
conedata.__dict__['_raw'].__dict__['_var'] = conedata.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})

# write
conedata.write("/w5home/bmoore/scRNAseq/GAMM/GAMM_S2/output_20230830_155530/GAMM_S2_clabeled-clusters_0.5_cones.h5ad")