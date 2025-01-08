# get scfound environment
conda create -n scfound2 python=3.12
conda activate scfound2
# install packages
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install pandas scipy scanpy
pip install argparse
pip install einops
pip install local-attention
# check installed packages then deactivate
conda list
conda deactivate
# pack environment
conda install -c conda-forge conda-pack
conda pack -n scfound2 --dest-prefix='$ENVDIR'