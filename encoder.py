'''
Author: fuchy@stu.pku.edu.cn
Description: this file encodes point cloud
FilePath: /compression/encoder.py
All rights reserved.
'''
from Preparedata.data import dataPrepare
from encoderTool import main
from networkTool import reload,CPrintl,expName,device
from octAttention import model
import glob,datetime,os
import pt as pointCloud
############## warning ###############
## decoder.py and test.py rely on this model here
## do not move this lines to somewhere else
model = model.to(device)
saveDic = reload(None,'modelsave/lidar/encoder_epoch_00801460.pth',multiGPU=False)
model.load_state_dict(saveDic['encoder'])

###########LiDar##############
GPCC_MULTIPLE = 2**20
list_orifile = ['file/Ply/11_000000.bin']
if __name__=="__main__":
    printl = CPrintl(expName+'/encoderPLY.txt')
    printl('_'*50,'OctAttention V0.4','_'*50)
    printl(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'))
    printl('load checkpoint', saveDic['path'])
    for oriFile in list_orifile:
        printl(oriFile)
        ptName = os.path.splitext(os.path.basename(oriFile))[0] 
        for qlevel in [12]:
            matFile,DQpt,normalizePt = dataPrepare(oriFile,saveMatDir='./Data/testPly',offset='min', qs=2/(2**qlevel-1),rotation=False,normalize=True)
            main(matFile,model,actualcode=True,printl =printl) # actualcode=False: bin file will not be generated
            print('_'*50,'pc_error','_'*50)
            pointCloud.pcerror(normalizePt,DQpt,None,'-r 1',None).wait()
            print('cd %e'%pointCloud.distChamfer(normalizePt,DQpt))
