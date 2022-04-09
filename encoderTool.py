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
bpttRepeatTime = 1


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
#%%
 
'''
description: Rearrange data for batch processing
'''
def dataPreProcess(oct_seq,bptt,batch_size,oct_len):
    oct_seq[:,:,0] = oct_seq[:,:,0] - 1
    oct_seq = torch.Tensor(oct_seq).long()    # [1,255]->[0,254]. shape [n,K]
    FeatDim = oct_seq.shape[-1]
    padingdata = torch.vstack((torch.zeros((bptt,levelNumK,FeatDim)),oct_seq)) # to be the context of oct[0]
    padingsize = batch_size - padingdata.shape[0]%batch_size            
    padingdata = torch.vstack((padingdata,torch.zeros(padingsize,levelNumK,FeatDim))).reshape((batch_size,-1,levelNumK,FeatDim)).permute(1,0,2,3)          #[bptt,batch_size,K]
    dataID = torch.hstack((torch.ones(bptt)*-1,torch.Tensor(list(range(oct_len))),torch.ones((padingsize))*-1)).reshape((batch_size,-1)).long().permute(1,0)
    padingdata = torch.vstack((padingdata, padingdata[0:bptt,list(range(1,batch_size,1))+[0]])).long()
    dataID = torch.vstack((dataID, dataID[0:bptt,list(range(1,batch_size,1))+[0]])).long()
    return dataID,padingdata

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
    oct_data_seq[:-1,0:-1,:] = oct_data_seq[1:,0:-1,:]
    oct_data_seq[:-1,-1,1:3] = oct_data_seq[1:,-1,1:3]     
    oct_len = len(oct_seq)
 
    batch_size =1  # 1 for safety encoder

    assert(batch_size*bptt<oct_len)
    
    #%%
    dataID,padingdata = dataPreProcess(oct_data_seq,bptt,batch_size,oct_len)
    MAX_GPU_MEM_It = 2**13 # you can change this according to the GUP memory size (2**12 for 24G)
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
        for n,i in enumerate(trange(0, padingLength-bptt , bptt//bpttRepeatTime)):
            seq_len = min(bptt, padingLength - 1 - i)
            input = padingdata[i:i+seq_len].long().to(device)   #input torch.Size([256, 32, 4, 3]) bptt,batch_sz,kparent,[oct,level,octant]
 
            if( n % MAX_GPU_MEM_It==0 and n):
                proBit.append(pro[:nodeID.max()+1].detach().cpu().numpy())
                offset = offset + nodeID.max() +1
            nodeID = dataID[i+seq_len - bptt//bpttRepeatTime+1:i+seq_len+1].squeeze(0) - offset
            nodeID[nodeID<0] = -1 #for padding data
            if input.size(0) != bptt:
                src_mask = generate_square_subsequent_mask(input.size(0))
            start_time = time.time()
            output = model(input,src_mask,[])
            elapsed =elapsed+ time.time() - start_time

            output = output[bptt-bptt//bpttRepeatTime:].reshape(-1,255)
            nodeID = nodeID.reshape(-1)
            p  = torch.softmax(output,1)
            pro[nodeID,:] = p

    if not proBit:# all data is in the MAX_GPU_MEM
        proBit.append(pro[:dataID.max()+1].detach().cpu().numpy())
    else:
        proBit.append(pro[:nodeID.max()+1].detach().cpu().numpy())
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
