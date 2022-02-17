conda create -n py37 python=3.7
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda activate py37
pip install hdf5storage
pip install Ninja
pip install tensorboard
pip install h5py
pip install tqdm
pip install matplotlib
pip install plyfile
