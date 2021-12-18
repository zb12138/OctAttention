import numpy as np
import h5py
import os
from plyfile import PlyData
import subprocess

PCERRORPATH = "file/pc_error"
TEMPPATH = "temp/data/"

def pcerror(pcRefer,pc,pcReferNorm,pcerror_cfg_params, pcerror_result,pcerror_path=PCERRORPATH):
  '''
  Options: 
          --help=0            This help text
    -a,   --fileA=""          Input file 1, original version
    -b,   --fileB=""          Input file 2, processed version
    -n,   --inputNorm=""      File name to import the normals of original point
                              cloud, if different from original file 1n
    -s,   --singlePass=0      Force running a single pass, where the loop is
                              over the original point cloud
    -d,   --hausdorff=0       Send the Haursdorff metric as well
    -c,   --color=0           Check color distortion as well
    -l,   --lidar=0           Check lidar reflectance as well
    -r,   --resolution=0      Specify the intrinsic resolution
          --dropdups=2        0(detect), 1(drop), 2(average) subsequent points
                              with same coordinates
          --neighborsProc=1   0(undefined), 1(average), 2(weighted average),
                              3(min), 4(max) neighbors with same geometric
                              distance
          --averageNormals=1  0(undefined), 1(average normal based on neighbors
                              with same geometric distance)
          --mseSpace=1        Colour space used for PSNR calculation
                              0: none (identity) 1: ITU-R BT.709 8: YCgCo-R
          --nbThreads=1       Number of threads used for parallel processing
  '''
  if pcerror_result is not None:
    pcLabel =os.path.basename(pcerror_result).split(".")[0]
  else:
    pcLabel = "pt0"
  if type(pc) is not str:
    write_ply_data(TEMPPATH + pcLabel + "pc.ply",pc)
    pc = TEMPPATH +pcLabel + "pc.ply"
  if type(pcRefer) is not str:
    write_ply_data(TEMPPATH+pcLabel+"pcRefer.ply",pcRefer)
    pcRefer = TEMPPATH + pcLabel + "pcRefer.ply"
  if pcerror_result is not None:
    f = open(pcerror_result, 'a+')
  else:
    import sys
    f = sys.stdout
  if type(pcerror_cfg_params) is str:
    pcerror_cfg_params = pcerror_cfg_params.split(' ')
  if pcReferNorm==None:
    return subprocess.Popen([pcerror_path,
            '-a', pcRefer, '-b', pc] + pcerror_cfg_params,
            stdout=f, stderr=f)
  return subprocess.Popen([pcerror_path,
                '-a', pcRefer, '-b', pc, '-n', pcReferNorm] + pcerror_cfg_params,
                stdout=f, stderr=f)

def loadply2(path,color_format='rgb'):
    plydata = PlyData.read(path)
    
    data = plydata.elements[0].data
    points = np.asarray([data['x'],data['y'],data['z']]).T 
    if color_format!='geometry':
      if len(data.dtype)>=6 and color_format=='rgb':
          colors =  np.asarray([data['red'],data['green'],data['blue']]).T
      else:
          colors = []
          for properties in color_format:
            colors.append(data[properties])
          colors = np.array(colors).T
    else:
      colors = None
    return points,colors

def write_ply_data(filename, points,attributeName=[],attriType=[]): 
    '''
    write data to ply file.
    e.g pt.write_ply_data('ScanNet_{:5d}.ply'.format(idx), np.hstack((point,np.expand_dims(label,1) )) , attributeName=['intensity'], attriType =['uint16'])
    '''
    # if os.path.exists(filename):
    #   os.system('rm '+filename)
    if type(points) is list:
      points = np.array(points)

    attrNum = len(attributeName)
    assert points.shape[1]>=(attrNum+3)

    if os.path.dirname(filename)!='' and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename)) 

    plyheader = ['ply\n','format ascii 1.0\n'] + ['element vertex '+str(points.shape[0])+'\n'] + ['property float x\n','property float y\n','property float z\n']
    for aN,attrName in enumerate(attributeName):
      plyheader.extend(['property '+attriType[aN]+' '+ attrName+'\n'])
    plyheader.append('end_header')
    typeList = {'uint16':"%d",'float':"float",'uchar':"%d"}

    np.savetxt(filename, points, newline="\n",fmt=["%f","%f","%f"]+[typeList[t] for t in attriType],header=''.join(plyheader),comments='')

    return

def h5toPly(h5Path,plyPath):
    pt,_ = pcread(h5Path)
    write_ply_data(plyPath,pt)
    return pt

def ptread(path):
  """
  Load ptcloud
  Returns: coords.
  """
  pt,_ = pcread(path,'geometry')
  return pt

def pcread(path,color_format = 'rgb'):
  """
  Load ptcloud
  Returns: coords & feats.
  """
  if not os.path.exists(path):
    raise Exception("no such file:"+path)

  if path.endswith(".ply"):
      try:
        return loadply(path,color_format)
      except:
        return loadply2(path,color_format)
  elif path.endswith(".h5"):
      return loadh5(path,color_format)
  elif path.endswith(".bin"):
      return loadbin(path)

def loadbin(file): # for KITTI
  points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
  return points[:,0:3],points[:,3:4]


def loadh5(filedir, color_format='rgb'):
  """Load coords & feats from h5 file.

  Arguments: file direction

  Returns: coords & feats.
  """
  pc = h5py.File(filedir, 'r')['data'][:]
  if pc.shape[1] == 3:
    color_format = 'geometry'

  coords = pc[:,0:3].astype('float32')

  if color_format == 'rgb':
    feats = pc[:,3:6]//255. 
  elif color_format == 'yuv':
    R, G, B = pc[:, 3:4], pc[:, 4:5], pc[:, 5:6]
    Y = 0.257*R + 0.504*G + 0.098*B + 16
    Cb = -0.148*R - 0.291*G + 0.439*B + 128
    Cr = 0.439*R - 0.368*G - 0.071*B + 128
    feats = np.concatenate((Y,Cb,Cr), -1)//256.
  elif color_format == 'geometry':
    feats = np.expand_dims(np.ones(coords.shape[0]), 1)
    
  feats = feats.astype('float32')

  return coords, feats

def loadply(filedir, color_format='rgb'):
  """Load coords & feats from ply file.
  
  Arguments: file direction.
  
  Returns: coords & feats.
  """

  files = open(filedir)
  coords = []
  feats = []
  for line in files:
    wordslist = line.split(' ')
    try:
      x, y, z, = float(wordslist[0]),float(wordslist[1]),float(wordslist[2])
      if color_format != 'geometry':
        try:
            r, g, b  = float(wordslist[3]),float(wordslist[4]),float(wordslist[5])
        except IndexError:
            color_format = 'geometry'    
    except ValueError:
      continue
    coords.append([x,y,z])
    if color_format != 'geometry':
        feats.append([r,g,b])

  coords = np.array(coords).astype('float32')
  if color_format != 'geometry':
    feats = np.array(feats).astype('float32')
  
  if color_format == 'rgb':
    pass
    # feats = feats//255.
  elif color_format == 'yuv':
    R, G, B = feats[:, 0:1], feats[:, 1:2], feats[:, 2:3]
    Y = 0.257*R + 0.504*G + 0.098*B + 16
    Cb = -0.148*R - 0.291*G + 0.439*B + 128
    Cr = 0.439*R - 0.368*G - 0.071*B + 128
    feats = np.concatenate((Y,Cb,Cr), -1)//256.
    
  elif color_format=='geometry':
    # feats = np.expand_dims(np.ones(coords.shape[0]), 1)
    return coords, None
  
  feats = feats.astype('float32')
  return coords, feats