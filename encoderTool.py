'''
Author: fuchy@stu.pku.edu.cn
Description: The encoder helper
FilePath: /compression/encoderTool.py
'''

#%%
import numpy as np 
import torch
import time
import os
from networkTool import device,bptt,expName,levelNumK,MAX_OCTREE_LEVEL
from dataset import default_loader as matloader
import numpyAc
import tqdm



def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
#%%
 
'''
description: Rearrange data for batch processing
'''
def batchify(oct_seq,bptt,oct_len):
    oct_seq[:-1,0:-1,:] = oct_seq[1:,0:-1,:]
    oct_seq[:-1,-1,1:3] = oct_seq[1:,-1,1:3]  
    oct_seq[:,:,0] = oct_seq[:,:,0] - 1
    pad_len = bptt#int(np.ceil(len(oct_seq)/bptt)*bptt - len(oct_seq))
    oct_seq = torch.Tensor(np.r_[np.zeros((bptt,*oct_seq.shape[1:])),oct_seq,np.zeros((pad_len,*oct_seq.shape[1:]))])
    dataID = torch.LongTensor(np.r_[np.ones((bptt))*-1,np.arange(oct_len),np.ones((pad_len))*-1])
    return dataID.unsqueeze(1),oct_seq.unsqueeze(1)

def encodeNode(pro,octvalue):
    assert octvalue<=255 and octvalue>=1
    pre = np.argmax(pro)+1
    return -np.log2(pro[octvalue-1]+1e-07),int(octvalue==pre)
'''
description: compress function
param {N[n treepoints]*K[k ancestors]*C[oct code,level,octant,position(xyz)] array; Octree data sequence} oct_data_seq
param {str;bin file name} outputfile
param {model} model
param {bool;Determines whether to perform entropy coding} actualcode
return {float;estimated/true bin size (in bit)} binsz
return {int;oct length of all octree} oct_len
return {float;total foward time of the model} elapsed
return {float list;estimated bin size (in bit) of depth 8~maxlevel data} binszList
return {int list;oct length of 8~maxlevel octree} octNumList
'''
def compress(oct_data_seq,outputfile,model,actualcode = True,print=print,showRelut=False):
    model.eval()
    levelID = oct_data_seq[:,-1,1].copy()
    oct_data_seq = oct_data_seq.copy()

    if levelID.max()>MAX_OCTREE_LEVEL:
        print('**warning!!**,to clip the level>{:d}!'.format(MAX_OCTREE_LEVEL))
        
    oct_seq = oct_data_seq[:,-1:,0].astype(int)   
    oct_len = len(oct_seq)
 
    batch_size =1  # 1 for safety encoder

    assert(batch_size*bptt<oct_len)
    
    #%%
    dataID,padingdata = batchify(oct_data_seq,bptt,oct_len)
    MAX_GPU_MEM_It = 2**13 # you can change this according to the GPU memory size (2**12 for 24G)
    MAX_GPU_MEM = min(bptt*MAX_GPU_MEM_It,dataID.max())+2  #  bptt <= MAX_GPU_MEM -1 < min(MAX_GPU,dataID)
    pro = torch.zeros((MAX_GPU_MEM,255)).to(device)
    padingLength = padingdata.shape[0]
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    padingdata = padingdata
    elapsed = 0
    proBit = []
    offset = 0
    if not showRelut:
        trange = range
    else:
        trange = tqdm.trange
    with torch.no_grad():
        for n,i in enumerate(trange(0, padingLength-bptt , bptt)):
            input = padingdata[i:i+bptt].long().to(device)   #input torch.Size([256, 32, 4, 3]) bptt,batch_sz,kparent,[oct,level,octant]
            nodeID = dataID[i+1:i+bptt+1].squeeze(0) - offset
            nodeID[nodeID<0] = -1
            start_time = time.time()
            output = model(input,src_mask,[])
            elapsed =elapsed+ time.time() - start_time
            output = output.reshape(-1,255)
            nodeID = nodeID.reshape(-1)
            p  = torch.softmax(output,1)
            pro[nodeID,:] = p
            if( (n % MAX_GPU_MEM_It==0 and n>0) or n == padingLength//bptt-1):
                proBit.append(pro[:nodeID.max()+1].detach().cpu().numpy())
                offset = offset + nodeID.max() +1

    del pro,input,src_mask
    torch.cuda.empty_cache()
    proBit = np.vstack(proBit)
    #%%
 
    bit = 0
    acc = 0
    templevel = 1
    binszList = []
    octNumList = []
    if True:
        # Estimate the bitrate at each level
        for i in range(oct_len):
            octvalue = int(oct_seq[i,-1])
            bit0,acc0 =encodeNode(proBit[i],octvalue)
            bit+=bit0
            acc+=acc0
            if templevel!=levelID[i]:
                templevel = levelID[i]
                binszList.append(bit)
                octNumList.append(i+1)
        binszList.append(bit)
        octNumList.append(i+1)
        binsz = bit # estimated bin size

        if actualcode:
            codec = numpyAc.arithmeticCoding()
            if not os.path.exists(os.path.dirname(outputfile)):
                os.makedirs(os.path.dirname(outputfile))
            _,binsz = codec.encode(proBit[:oct_len,:], oct_seq.astype(np.int16).squeeze(-1)-1,outputfile)
       
        if len(binszList)<=7:
            return binsz,oct_len,elapsed,np.array(binszList),np.array(octNumList)  
        return binsz,oct_len,elapsed ,np.array(binszList[7:]),np.array(octNumList[7:])  
 # %%
def main(fileName,model,actualcode = True,showRelut=True,printl = print):
    
    matDataPath = fileName
    octDataPath = matDataPath
    cell,mat = matloader(matDataPath)
    FeatDim = levelNumK                          
    oct_data_seq = np.transpose(mat[cell[0,0]]).astype(int)[:,-FeatDim:,0:6] 

    p = np.transpose(mat[cell[1,0]]['Location'])
    ptNum = p.shape[0]
    ptName = os.path.basename(matDataPath)
    outputfile = expName+"/data/"+ptName[:-4]+".bin"
    binsz,oct_len,elapsed,binszList,octNumList = compress(oct_data_seq,outputfile,model,actualcode,printl,showRelut)
    if showRelut:
        printl("ptName: ",ptName)
        printl("time(s):",elapsed)
        printl("ori file",octDataPath)
        printl("ptNum:",ptNum)
        printl("binsize(b):",binsz)
        printl("bpip:",binsz/ptNum)

        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        printl("pre sz(b) from Q8:",(binszList))
        printl("pre bit per oct from Q8:",(binszList/octNumList))
        printl('octNum',octNumList)
        printl("bit per oct:",binsz/oct_len)
        printl("oct len",oct_len)
 
    return binsz/oct_len
