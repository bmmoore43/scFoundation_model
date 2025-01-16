# copy files over
cp /staging/bmoore22/scFoundation.tar.gz ./
cp /staging/bmoore22/scfound2.tar.gz ./
cp /staging/bmoore22/models.ckpt ./
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
# move model to models folder
mv models.ckpt scFoundation/model/models/
# test environment
python scfound_test.py
# mv modified script
mv get_embedding.py scFoundation/model/
# cd to model folder
cd scFoundation/model
# for gene embeddings
python get_embedding.py --task_name Gamm --input_type singlecell --output_type gene \
--pool_type all --tgthighres a5 --data_path ../../GAMM_S2_clabeled-clusters_0.5_cones_orthologs.h5ad \
--save_path ../../ --pre_normalized A --version ce
