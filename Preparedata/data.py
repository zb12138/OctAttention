'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-17 23:30:48
LastEditTime: 2021-12-02 22:18:56
LastEditors: FCY
Description: dataPrepare helper
FilePath: /compression/Preparedata/data.py
All rights reserved.
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
    
    refPt = p
    if normalize is True: # normalize pc to [-1,1]^3
        p = p - np.mean(p,axis=0)
        p = p/abs(p).max()
        refPt = p

    if rotation:
        refPt = refPt[:,[0,2,1]]
        refPt[:,2] = - refPt[:,2]

    if offset is 'min':
        offset = np.min(refPt,0)

    points = refPt - offset

    if qlevel is not None:
        qs = (points.max() - points.min())/(2**qlevel-1)

    pt = np.round(points/qs)
    pt,idx = np.unique(pt,axis=0,return_index=True)
    pt = pt.astype(int)
    # pointCloud.write_ply_data('pori.ply',np.hstack((pt,c)),attributeName=['reflectance'],attriType=['uint16'])
    code,Octree,QLevel = GenOctree(pt)
    DataSturct = GenKparentSeq(Octree,4)
    
    ptcloud = {'Location':refPt}
    Info = {'qs':qs,'offset':offset,'Lmax':QLevel,'name':ptName,'levelSID':np.array([Octreelevel.node[-1].nodeid for Octreelevel in Octree])}
    patchFile = {'patchFile':(np.concatenate((np.expand_dims(DataSturct['Seq'],2),DataSturct['Level'],DataSturct['Pos']),2), ptcloud, Info)}
    hdf5storage.savemat(os.path.join(saveMatDir,ptName+'.mat'), patchFile, format='7.3', oned_as='row', store_python_metadata=True)
    DQpt = (pt*qs+offset) 
    return os.path.join(saveMatDir,ptName+'.mat'),DQpt,refPt