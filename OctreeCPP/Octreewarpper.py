'''
Author: fuchy@stu.pku.edu.cn
LastEditors: Please set LastEditors
Description: 
'''
from ctypes import *
import numpy as np
import os
import numpy.ctypeslib as npct


class Node(Structure):
    _fields_ = [
        ('nodeid',c_uint),
        ('octant',c_uint),
        ('parent',c_uint),
        ('oct',c_uint8),
        # ('pointIdx',c_void_p),
        ('pos',c_uint*3)
    ]
c_double_p = POINTER(c_double)
c_uint16_p = POINTER(c_uint16)
lib = cdll.LoadLibrary(os.path.dirname(os.path.abspath(__file__))+'/Octree_python_lib.so') # class level loading lib
lib.new_vector.restype = c_void_p
lib.new_vector.argtypes = []
lib.delete_vector.restype = None
lib.delete_vector.argtypes = [c_void_p]
lib.vector_size.restype = c_int
lib.vector_size.argtypes = [c_void_p]
lib.vector_get.restype = c_void_p
lib.vector_get.argtypes = [c_void_p, c_int]
lib.vector_push_back.restype = None
lib.vector_push_back.argtypes = [c_void_p, c_int]
lib.genOctreeInterface.restype = c_void_p
lib.genOctreeInterface.argtypes = [c_void_p ,c_double_p,c_int]
lib.Nodes_get.argtypes = [c_void_p,c_int]
lib.Nodes_get.restype = POINTER(Node)
lib.Nodes_size.restype = c_int
lib.Nodes_size.argtypes = [c_void_p]

lib.int_size.restype = c_int
lib.int_size.argtypes = [c_void_p]

lib.int_get.restype = c_int
lib.int_get.argtypes = [c_void_p,c_int]


class COctree(object):

    def __init__(self):
        self.vector = lib.new_vector()  # octree pointer to new vector
        self.code = None 
    def __del__(self):  # when reference count hits 0 in Python,
        lib.delete_vector(self.vector)  # call C++ vector destructor

    def __len__(self):
        return lib.vector_size(self.vector)

    def __getitem__(self, i):  # access elements in vector at index
        L = self.__len__()
        if i>=L or i<-L:
            raise IndexError('Vector index out of range')
        if i<0:
            i += L
        return Level(lib.vector_get(self.vector, c_int(i)),i)
        
    def __repr__(self):
        return '[{}]'.format(', '.join(str(self[i]) for i in range(len(self))))

    def push(self, i):  # push calls vector's push_back
        lib.vector_push_back(self.vector, c_int(i))

    def genOctree(self, p):  # foo in Python calls foo in C++
        
        data = np.ascontiguousarray(p).astype(np.double)
        data_p = data.ctypes.data_as(c_double_p)
        self.code = OctCode(lib.genOctreeInterface(self.vector,data_p,data.shape[0]))

class OctCode():
    def __init__(self,Adr) -> None:
        self.nodeAdr = Adr
        self.Len = lib.int_size(Adr)
    def __repr__(self):
        return '[{}]'.format(', '.join(str(self[i]) for i in range(len(self))))
    def __getitem__(self, i):
        L = self.Len
        if i>=L or i<-L:
            raise IndexError('Vector index out of range')
        if i<0:
            i += L
        return lib.int_get(self.nodeAdr,i)
    def __len__(self):
        return lib.int_size(self.nodeAdr)

class Level():
    def __init__(self, Adr,i) -> None:
        self.Adr = Adr
        self.node = Node(Adr)
        self.level = i+1
        self.Len = lib.Nodes_size(self.Adr)
    def __getitem__(self, i):
        L = self.Len
        if i>=L or i<-L:
            raise IndexError('Vector index out of range')
        if i<0:
            i += L
        return lib.Nodes_get(self.Adr,i).contents
    def __len__(self):
        return lib.Nodes_size(self.Adr)

class Node():
    def __init__(self,Adr) -> None:
        self.nodeAdr = Adr
        self.Len = lib.Nodes_size(Adr)
    def __getitem__(self, i):
        L = self.Len
        if i>=L or i<-L:
            raise IndexError('Vector index out of range')
        if i<0:
            i += L
        return lib.Nodes_get(self.nodeAdr,i).contents
    def __len__(self):
        return lib.Nodes_size(self.nodeAdr)

def GenOctree(points):
    Octree = COctree()
    Octree.genOctree(points)
    return list(Octree.code),Octree,len(Octree)