# map human gene names to pig adata object
import scanpy as sc
import pandas as pd
adata = sc.read_h5ad("/w5home/bmoore/scRNAseq/GAMM/GAMM_S2/output_20230830_155530/GAMM_S2_clabeled-clusters_0.5_cones.h5ad")


# Assuming your mapping file is a CSV/TSV with columns like 'old_name' and 'new_name'
# Load the mapping file
gene_map = pd.read_csv('/w5home/bmoore/scRNAseq_library/sc_pipeline/metadata/Human_Pig_Biomart_Filtered_mod.txt', sep="\t")  # or pd.read_csv('file.tsv', sep='\t')

# Create a dictionary for faster lookup
name_dict = dict(zip(gene_map['pig.gene.name'], gene_map['human.gene.name']))

# Method 1: Using rename
adata.var_names = adata.var_names.map(lambda x: name_dict.get(x, x))

# Method 2: Alternative approach if you want to keep track of old names
# This preserves old names in a column called 'old_names'
adata.var['old_names'] = adata.var.index
adata.var_names = [name_dict.get(x, x) for x in adata.var_names]

# Verify the changes
print("First few gene names after update:")
print(adata.var_names[:5])

# Check for any genes that weren't in the mapping
unmapped = [x for x in adata.var_names if x not in name_dict.values()]
if unmapped:
    print(f"\nNumber of unmapped genes: {len(unmapped)}")
    print("First few unmapped genes:")
    print(unmapped[:5])
    
# keep only mapped genes
mapped_genes = set(gene_map['pig.gene.name'])

# Create boolean mask for genes that were in the mapping file
genes_to_keep = [gene in mapped_genes for gene in adata.var['old_names']]
 # subset
adata_subset = adata[:, genes_to_keep]
# Print some stats to verify
print(f"Original number of genes: {adata.n_vars}")
print(f"Number of genes after subsetting: {adata_subset.n_vars}")
print(f"Number of genes in mapping file: {len(mapped_genes)}")
# Optional: Check if any genes from mapping file are missing in your data
genes_not_found = mapped_genes - set(adata.var['old_names']) 

# write out
