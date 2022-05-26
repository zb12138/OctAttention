import os
import os.path
import numpy as np
import glob
import torch.utils.data as data
from PIL import Image
import glob
# import scipy.io as scio
import h5py
from networkTool import trainDataRoot,levelNumK
IMG_EXTENSIONS = [
    'Kitti'
]


def is_image_file(filename):
    return any(extension in filename for extension in IMG_EXTENSIONS)


def default_loader(path):
    mat = h5py.File(path)
    # data = scio.loadmat(path)
    cell = mat['patchFile']
    return cell,mat

class DataFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, TreePoint,dataLenPerFile, transform=None ,loader=default_loader): 
         
        # dataLenPerFile is the number of all octnodes in one 'mat' file on average
        
        dataNames = []
        for filename in sorted(glob.glob(root)):
            if is_image_file(filename):
                dataNames.append('{}'.format(filename))
        self.root = root
        self.dataNames =sorted(dataNames)
        self.transform = transform
        self.loader = loader
        self.index = 0
        self.datalen = 0
        self.dataBuffer = []
        self.fileIndx = 0
        self.TreePoint = TreePoint
        self.fileLen = len(self.dataNames)
        assert self.fileLen>0,'no file found!'
        # self.dataLenPerFile = dataLenPerFile # you can replace 'dataLenPerFile' with the certain number in the 'calcdataLenPerFile'
        self.dataLenPerFile = self.calcdataLenPerFile() # you can comment this line after you ran the 'calcdataLenPerFile'
        
    def calcdataLenPerFile(self):
        dataLenPerFile = 0
        for filename in self.dataNames:
            cell,mat = self.loader(filename)
            for i in range(cell.shape[1]):
                dataLenPerFile+= mat[cell[0,i]].shape[2]
        dataLenPerFile = dataLenPerFile/self.fileLen
        print('**'*40)
        print('dataLenPerFile:',dataLenPerFile,'you just use this function for the first time')
        print('**'*40)
        return dataLenPerFile

    def __getitem__(self, index):
        while(self.index+self.TreePoint>self.datalen):
            filename = self.dataNames[self.fileIndx]
            # print(filename)
            if self.dataBuffer:
                a = [self.dataBuffer[0][self.index:].copy()]
            else:
                a=[]
                
            cell,mat = self.loader(filename)
            for i in range(cell.shape[1]):
                data = np.transpose(mat[cell[0,i]]) #shape[ptNum,Kparent, Seq[1],Level[1],Octant[1],Pos[3] ] e.g 123456*7*6
                data[:,:,0] = data[:,:,0] - 1
                a.append(data[:,-levelNumK:,:])# only take levelNumK level feats
                
            self.dataBuffer = []
            self.dataBuffer.append(np.vstack(tuple(a)))

            self.datalen = self.dataBuffer[0].shape[0]
            self.fileIndx+=1  # shuffle step = 1, will load continuous mat
            self.index = 0
            if(self.fileIndx>=self.fileLen):
                self.fileIndx=index%self.fileLen
        # try read
        img = []
        img.append(self.dataBuffer[0][self.index:self.index+self.TreePoint])

        self.index+=self.TreePoint

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return int(self.dataLenPerFile*self.fileLen/self.TreePoint) # dataLen = octlen in total/TreePoint
