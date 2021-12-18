'''
Author: fuchy@stu.pku.edu.cn
LastEditors: Please set LastEditors
Description: this file encodes MAT Files
FilePath: /compression/encoderTool.py
'''
import datetime,os
import pt as pointCloud
import numpy as np

###########LiDar##############
from subprocess import Popen
GPCC_MULTIPLE = 2**20 # Times the input to make it appropriate for TMC13 
list_orifile = ['file/Ply/11_000000.bin']
if __name__=="__main__":
    print("****TMC V14****")
    print(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'))
    for oriFile in list_orifile:
        print(oriFile)
        for qlevel in [12]:
            p = pointCloud.ptread(oriFile)
            p = p - np.mean(p,axis=0)
            normalizePt = p/abs(p).max()
            pointCloud.write_ply_data('./temp/tmc/normalizePt.ply',normalizePt*GPCC_MULTIPLE)
            qstmc =  1/(2/(2**qlevel-1))/GPCC_MULTIPLE
            print('_'*50,'encode','_'*50)
            cmd = "./file/tmc13v14 --mode=0 -c ./file/kitti.cfg --uncompressedDataPath=./temp/tmc/normalizePt.ply --compressedStreamPath=./temp/tmc/tmc.bin --mergeDuplicatedPoints=1 --positionBaseQp=4 --positionQuantizationScale="+str(qstmc)
            p = Popen(cmd, shell=True).wait()
            print('ptNum ',normalizePt.shape[0])      
            print('tmc bpip ',os.path.getsize('./temp/tmc/tmc.bin')*8/normalizePt.shape[0])
            
            print('_'*50,'decode','_'*50)
            cmd = "./file/tmc13v14  --mode=1 --reconstructedDataPath=./temp/tmc/recPt.ply --compressedStreamPath=./temp/tmc/tmc.bin"
            Popen(cmd, shell=True).wait()

            print('_'*50,'pc_error','_'*50)
            pointCloud.pcerror(normalizePt,pointCloud.ptread('./temp/tmc/recPt.ply')/GPCC_MULTIPLE,None,'-r 1',None).wait()