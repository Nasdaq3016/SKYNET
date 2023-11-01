# -*- coding: utf-8 -*-

import numpy as np

class LHS_BANK(object):
    def __init__(self):
        self.memory = np.zeros([256, 4]) # 4 words * 224 
        self.write_pointer = 0
        self.read_pointer = 0
    
    def write_row(self, data):
        self.memory[self.write_pointer, 0] = data
        self.write_pointer += 1
    
    def write_col(self, data):
        self.memory[self.write_pointer] = data
        self.write_pointer += 1

    def push_rows(self):
        self.memory[:, 1:] = self.memory[:, :-1]
        self.reset_write_pointer()
        
        
    def reset_write_pointer(self, addr=0):
        self.write_pointer = addr
    
    def reset_read_pointer(self, addr=0):
        self.read_pointer = addr
    
    def set_read_pointer(self, addr):
        self.read_pointer = addr
    
    
    def read_single(self):
        out = self.memory[self.read_pointer]
        self.read_pointer += 1
        return out

    def read(self, read_start, read_end):
        return self.memory[read_start:read_end]
    

class Dispatcher(object):
    def __init__(self):
        self.bank_list = None
        self.buffer = np.zeros([4])
        return
    
    def set_read_banks(self, bank_list):
        self.bank_list = bank_list
    
    def dispatch_data(self, in_data):
        send_data =self.buffer
        self.buffer = in_data
        return send_data


class RHS_BANK(object):
    def __init__(self):
        self.use_8bW = False
        self.memory = np.zeros([4608])
    
    def load_data(self, read_pointer):
        return self.memory[read_pointer]
        
    def write_data(self, write_pointer, data):
        self.memory[write_pointer] = data
    
    def mode_change(self, mode='use_8b'):
        if mode=='use_8b':
            self.use_8bW=True
            self.memory = np.zeros([4608, 2])
        else:
            self.use_8bW = False
            self.memory = np.zeros([4608])
            

        


class PE(object):
    def __init__(self):
        self.weights = np.zeros([3])
        self.scratchpad = np.zeros([512])
        self.write_pointer = 0
        self.read_pointer = 0
        #print('hi')
        return
    
    def matmul(self, data0, data1, data2, data3):
        p0 = np.matmul(self.weights, data0)
        p1 = np.matmul(self.weights, data1)
        p2 = np.matmul(self.weights, data2)
        p3 = np.matmul(self.weights, data3)
        return p0, p1, p2, p3
    
    def load_scratchpad(self, data):
        self.scratchpad[self.write_pointer] = data
        self.write_pointer += 1
    
    def w_from_scratchpad(self, n=3):
        assert n < 4
        self.weights[:n] = self.scratchpad[self.read_pointer:self.read_pointer+n]
        self.read_pointer += n
    def reset_write_pointer(self, addr):
        self.write_pointer = addr

    def reset_read_pointer(self, addr):
        self.read_pointer = addr

    def reset_pointer(self):
        self.write_pointer = 0
        self.read_pointer = 0
    #%%