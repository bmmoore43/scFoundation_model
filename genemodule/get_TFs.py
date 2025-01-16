'''
Download human motifs (from pySCENIC github):
wget https://resources.aertslab.org/cistarget/motif2tf/motifs-v9-nr.hgnc-m0.001-o0.0.tbl

'''
# import libraries
import os
import pandas as pd

df_motifs_hgnc = pd.read_csv('./data/motifs-v9-nr.hgnc-m0.001-o0.0.tbl',sep='\t')
hs_tfs = df_motifs_hgnc.gene_name.unique()
with open('./data/hs_hgnc_tfs.txt', 'wt') as f:
    f.write('\n'.join(hs_tfs) + '\n')
len(hs_tfs)