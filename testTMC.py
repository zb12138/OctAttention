'''
Author: fuchy@stu.pku.edu.cn
LastEditors: FCY
Description: this file encodes MAT Files
FilePath: /compression/encoderTool.py
'''
import datetime,os
import pt as pointCloud
import numpy as np

###########Objct##############
from subprocess import Popen
GPCC_MULTIPLE = 1
list_orifile = ['file/Ply/2851.ply']
if __name__=="__main__":
    print("****TMC V14****")
    print(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'))
    for oriFile in list_orifile:
        print(oriFile)

        p = pointCloud.ptread(oriFile)

        pointCloud.write_ply_data('./temp/tmc/p.ply',p*GPCC_MULTIPLE)
        qstmc =  1.0
        print('_'*50,'encode','_'*50)
        cmd = "./file/tmc13v14 --mode=0 -c ./file/redandblack.cfg --uncompressedDataPath=./temp/tmc/p.ply --compressedStreamPath=./temp/tmc/tmc.bin --mergeDuplicatedPoints=1 --positionBaseQp=4 --positionQuantizationScale="+str(qstmc)
        Popen(cmd, shell=True).wait()
        print('ptNum: ',p.shape[0]) 
        bz = os.path.getsize('./temp/tmc/tmc.bin')*8
        print('binsize(b): ',bz)    
        print('tmc bpip: ',bz/p.shape[0])
        
        print('_'*50,'decode','_'*50)
        cmd = "./file/tmc13v14  --mode=1 --reconstructedDataPath=./temp/tmc/recPt.ply --compressedStreamPath=./temp/tmc/tmc.bin"
        Popen(cmd, shell=True).wait()

        print('_'*50,'pc_error','_'*50)
        pointCloud.pcerror(p,pointCloud.ptread('./temp/tmc/recPt.ply')/GPCC_MULTIPLE,None,'-r 1023',None).wait()