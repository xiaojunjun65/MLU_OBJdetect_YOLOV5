from ctypes import *
from numpy.ctypeslib import ndpointer
import numpy as np
class InferParam(Structure):
     _fields_ = [("inputData", POINTER(ndpointer(c_float))),
                 ("outputData", POINTER(ndpointer(c_float)))]

class DataShape(Structure):
     _fields_ = [("N", c_int),
                 ("C", c_int),
                 ("W", c_int),
                 ("H", c_int)]

class ModelOffline():
    def __init__(self, offline_model_path, device_id):            
        libc = cdll.LoadLibrary("/usr/local/lib/libinferEngine.so")
        loadCambriconModel =  libc.loadCambriconModel
        loadCambriconModel.argtypes = [c_char_p, c_uint]
        loadCambriconModel.restype = c_int
        loadCambriconModel(bytes(offline_model_path, encoding='utf-8'), device_id)

        self.applyModelBatchWiseStruct =  libc.applyModelBatchWiseStruct
        self.applyModelBatchWiseStruct.argtypes = [POINTER(InferParam)]
        self.applyModelBatchWiseStruct.restype = c_int

        self.getModelInputShape =  libc.getModelInputShape
        self.getModelInputShape.argtypes = [c_int]
        self.getModelInputShape.restype = DataShape

        self.getModelOutputShape =  libc.getModelOutputShape
        self.getModelOutputShape.argtypes = [c_int]
        self.getModelOutputShape.restype = DataShape

        self.getModelInputNum =  libc.getModelInputNum
        self.getModelInputNum.restype = c_int

        self.getModelOutputNum =  libc.getModelOutputNum
        self.getModelOutputNum.restype = c_int

        self.forward = self.ctypes_forward
        
    def __call__(self, img):
        output_list = self.forward(img)
        if len(output_list) == 1:
            return output_list[0]
        return output_list

    def ctypes_forward(self, img):
        if isinstance(img, np.ndarray):
            input_list = [img]
        else:
            input_list = img
        output_list = self.genOuputStroreArr()
        ctype_param_p = self.prepareParam(input_list, output_list)
        self.applyModelBatchWiseStruct(ctype_param_p)
        return output_list

    def pybind_forward(self, img):
        output_list = self.model.forward(img)
        return output_list

    def getInputShape(self, dim = 0):
        datashape = self.getModelInputShape(dim)
        return datashape.N, datashape.C, datashape.W, datashape.H

    def getOutputShape(self, dim = 0):
        datashape = self.getModelOutputShape(dim)
        return datashape.N, datashape.C, datashape.W, datashape.H

    def getInputNum(self):
        return self.getModelInputNum()

    def getOutputNum(self):
        return self.getModelOutputNum()

    def prepareParam(self, input_list, output_list):
        np_input_pointer = ndpointer(c_float)*len(input_list)
        param_list = [input_data.ctypes.data_as(ndpointer(c_float)) for input_data in input_list]
        np_input_pointer_obj = np_input_pointer(*param_list)

        np_output_pointer = ndpointer(c_float)*len(output_list)
        param_list = [output_data.ctypes.data_as(ndpointer(c_float)) for output_data in output_list]
        np_output_pointer_obj = np_output_pointer(*param_list)
        param = [np_input_pointer_obj, np_output_pointer_obj]
        infer_param = InferParam(*param)
        return pointer(infer_param) 

    def genInputStroreArr(self):
        input_list = []
        for i in range(self.getInputNum()):
            n,c,w,h = self.getInputShape(i)
            input_list.append(np.asanyarray(np.zeros((n, w, h, c), dtype=np.float32)))
        return input_list

    def genOuputStroreArr(self):
        output_list = []
        for i in range(self.getOutputNum()):
            n,c,w,h = self.getOutputShape(i)
            output_list.append(np.asanyarray(np.zeros((n, w, h, c), dtype=np.float32)))
        return output_list




#    print(odata.shape)
