'''
Author: fuchy@stu.pku.edu.cn
LastEditors: FCY
Description: Network parameters and helper functions
FilePath: /compression/networkTool.py
'''

import torch
import os,random
import numpy as np
# torch.set_default_tensor_type(torch.DoubleTensor)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network parameters
bptt = 1024 # Context window length
expName = './Exp/Kitti'
DataRoot = './Data/Lidar'

checkpointPath = expName+'/checkpoint'
levelNumK = 4

trainDataRoot = DataRoot+"/train/*.mat" # DON'T FORGET RUN ImageFolder.calcdataLenPerFile() FIRST
expComment = 'OctAttention, trained on SemanticKITTI 1~12 level. 2021/12. All rights reserved.'

MAX_OCTREE_LEVEL = 12
# Random seed
seed=2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
os.environ["H5PY_DEFAULT_READONLY"] = "1"
# Tool functions
def save(index, saveDict,modelDir='checkpoint',pthType='epoch'):
    if os.path.dirname(modelDir)!='' and not os.path.exists(os.path.dirname(modelDir)):
        os.makedirs(os.path.dirname(modelDir))
    torch.save(saveDict, modelDir+'/encoder_{}_{:08d}.pth'.format(pthType, index))
        
def reload(checkpoint,modelDir='checkpoint',pthType='epoch',print=print,multiGPU=False):
    try:
        if checkpoint is not None:
            saveDict = torch.load(modelDir+'/encoder_{}_{:08d}.pth'.format(pthType, checkpoint),map_location=device)
            pth = modelDir+'/encoder_{}_{:08d}.pth'.format(pthType, checkpoint)
        if checkpoint is None:
            saveDict = torch.load(modelDir,map_location=device)
            pth = modelDir
        saveDict['path'] = pth
        # print('load: ',pth)
        if multiGPU:
            from collections import OrderedDict
            state_dict = OrderedDict()
            new_state_dict = OrderedDict()
            for k, v in saveDict['encoder'].items():
                name = k[7:]  # remove `module.`
                state_dict[name] = v
            saveDict['encoder'] = state_dict
        return saveDict
    except Exception as e:
        print('**warning**',e,' start from initial model')
        # saveDict['path'] = e
    return None

class CPrintl():
    def __init__(self,logName) -> None:
        self.log_file = logName
        if os.path.dirname(logName)!='' and not os.path.exists(os.path.dirname(logName)):
            os.makedirs(os.path.dirname(logName))
    def __call__(self, *args):
        print(*args)
        print(*args, file=open(self.log_file, 'a'))

def model_structure(model,print=print):
    print('-'*120)
    print('|'+' '*30+'weight name'+' '*31+'|' \
            +' '*10+'weight shape'+' '*10+'|' \
            +' '*3+'number'+' '*3+'|')
    print('-'*120)
    num_para = 0
    for _, (key, w_variable) in enumerate(model.named_parameters()):
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para

    
        print('| {:70s} | {:30s} | {:10d} |'.format(key, str(w_variable.shape), each_para))
    print('-'*120)
    print('The total number of parameters: ' + str(num_para))
    print('-'*120)