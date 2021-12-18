'''
LastEditors: fcy
'''
from Octree import GenOctree,GenKparentSeq
import pt as pointCloud
import numpy as np
import os
import hdf5storage

def dataPrepare(fileName,saveMatDir='Data',qs=1,ptNamePrefix='',offset='min',qlevel=None,rotation=False,normalize=False):
    if not os.path.exists(saveMatDir):
        os.makedirs(saveMatDir)
    ptName = ptNamePrefix+os.path.splitext(os.path.basename(fileName))[0] 
    p = pointCloud.ptread(fileName)
    ptcloud = {'Location':p}
    
    refPt = p
    if normalize is True: # normalize pc to [-1,1]^3
        p = p - np.mean(p,axis=0)
        p = p/abs(p).max()
        refPt = p

    if rotation:
        p = p[:,[0,2,1]]
        p[:,2] = - p[:,2]

    if offset is 'min':
        offset = np.min(p,0)

    points = p - offset

    if qlevel is not None:
        qs = (points.max() - points.min())/(2**qlevel-1)

    pt = np.round(points/qs)
    pt,idx = np.unique(pt,axis=0,return_index=True)
    pt = pt.astype(int)
    # pointCloud.write_ply_data('pori.ply',np.hstack((pt,c)),attributeName=['reflectance'],attriType=['uint16'])
    code,Octree,QLevel = GenOctree(pt)
    DataSturct = GenKparentSeq(Octree,4)
    Info = {'qs':qs,'offset':offset,'Lmax':QLevel,'name':ptName,'levelSID':np.array([Octreelevel.node[-1].nodeid for Octreelevel in Octree])}
    patchFile = {'patchFile':(np.concatenate((np.expand_dims(DataSturct['Seq'],2),DataSturct['Level'],DataSturct['Pos']),2), ptcloud, Info)}
    hdf5storage.savemat(os.path.join(saveMatDir,ptName+'.mat'), patchFile, format='7.3', oned_as='row', store_python_metadata=True)
    DQpt = (pt*qs+offset) 
    return os.path.join(saveMatDir,ptName+'.mat'),DQpt,refPt