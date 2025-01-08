# copy files over
cp /staging/bmoore22/scFoundation.tar.gz ./
cp /staging/bmoore22/scfound2.tar.gz ./
cp /staging/bmoore22/models.ckpt ./
cp /staging/bmoore22/model_example.zip ./
cp /staging/bmoore22/GAMM_S2_clabeled-clusters_0.5_cones_orthologs.h5ad ./
# set environment
# set environment name
ENVNAME=scfound2
export ENVDIR=$ENVNAME
# set up the environment
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate
# unzip input files
chmod 777 scFoundation.tar.gz
tar -xzvf scFoundation.tar.gz
unzip model_example.zip
# move model to models folder
mv models.ckpt scFoundation/model/models/
# test environment
python scfound_test.py
# mv modified script
mv get_embedding.py scFoundation/model/
# get cell embeddings
cd scFoundation/model
### Cell embedding
python get_embedding.py --task_name Baron --input_type singlecell --output_type cell \
--pool_type all --tgthighres a5 --data_path ../../examples/enhancement/Baron_enhancement.csv \
--save_path ../../examples/enhancement/ --pre_normalized F --version rde
# for Gamm cell embeddings
python get_embedding.py --task_name Gamm --input_type singlecell --output_type cell \
--pool_type all --tgthighres a5 --data_path ../../GAMM_S2_clabeled-clusters_0.5.h5ad \
--save_path ../../ --pre_normalized A --version ce
# for gene embeddings
python get_embedding.py --task_name genemodule --input_type singlecell --output_type gene \
--pool_type all --tgthighres f1 --data_path ../../examples/genemodule/zheng_subset_cd8t_b_mono.csv \
--save_path ../../examples/genemodule/ --pre_normalized F --demo
# Gamm gene embeddings
python get_embedding.py --task_name Gamm --input_type singlecell --output_type gene \
--pool_type all --tgthighres a5 --data_path ../../GAMM_S2_clabeled-clusters_0.5_cones_orthologs.h5ad \
--save_path ../../ --pre_normalized A --version ce

cd ../../
rm scfound2.tar.gz
rm scFoundation.tar.gz