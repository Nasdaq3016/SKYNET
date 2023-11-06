# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:20:56 2019

@author: Jeongwoo Park
"""
import numpy as np

class LHS_BANK(object):
    def __init__(self):
        self.memory = np.zeros([224, 4]) # 4 words * 224 
        self.write_pointer = 0
    
    def write_row(self, data):
        self.memory[self.write_pointer, 0] = data
        self.write_pointer += 1
    
    def push_rows(self):
        self.memory[:, 1:] = self.memory[:, :-1]
        
        
    def reset_write_pointer(self):
        self.write_pointer = 0
    
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
    
    def w_from_scratchpad(self):
        self.weights = self.scratchpad[self.read_pointer:self.read_pointer+3]
        self.read_pointer += 3
    
    def reset_pointer(self):
        self.write_pointer = 0
        self.read_pointer = 0
    
class PSUM(object):
    def __init__(self):
        self.memory = np.zeros([128,4])
        self.use_8bW = False
        self.write_pointer = 0
    
    def psum_save(self, PSUMs):
        assert PSUMS.shape==(16, 4)
        p0_sum = np.sum(p0s, axis=0)
        p1_sum = np.sum(p1s, axis=0)
        p2_sum = np.sum(p2s, axis=0)
        p3_sum = np.sum(p3s, axis=0)
        self.memory[self.write_pointer] += np.stack([p0_sum, p1_sum, p2_sum, p3_sum])
        self.write_pointer += 1
    
    def reset_write_pointer(self):
        self.write_pointer = 0
        
    def mode_change(self, mode='use_8b'):
        if mode=='use_8b':
            self.use_8bW = True
            self.memory = np.zeros([128, 4, 2])
        else:
            self.use_8bW = False
            self.memory = np.zeros([128, 4])
            
    def psum_export(self, export_pointer):
        readout_data = self.memory[export_pointer].copy()
        self.memory[export_pointer] = 0
        return readout_data
    
def pad(arr, pad):
    assert len(arr.shape)==4
    padded = np.zeros(arr)
    
    
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

data_base = torch.randn(4, 64, 32, 32)
conv_baselayer = nn.Conv2d(64, 64, 5, 1, padding=2, bias=False)
PAD = 2
DATA = data_base.numpy()
CONV_W = conv_baselayer.weight.detach()

ROW_LENGTH = 32 + 2 * PAD
ROW_DEPTH = 32 + 2 *PAD
K_H = 5
K_W = 5
BATCH_SIZE = 4
IN_C_SIZE = 64
OUT_C_SIZE = 64

OUT_ROW_LENGTH = 32

#%%
LHS_BANKS = [LHS_BANK() for i in range(32)]
RHS_BANKS = [RHS_BANK() for i in range(16)]
Dispatchers = [Dispatcher() for i in range(24)]

PE_array = [[PE() for i in range(16)] for j in range(8)]


#%%
# INITIAL LHS LOADING
DATA_FLATTENED = F.pad(data_base,(PAD,PAD,PAD,PAD)).permute(0,2,3,1).numpy().reshape(-1) # Batch, H, W, INC
CONVW_FLATTENED = CONV_W.permute(2,3,1,0).numpy().reshape(-1) # This shaping ensures KH, KW, inC, outC shape
#CONVW_FLATTENED = CONV_W.permute(2,1,3,0).numpy().reshape(-1) # This shaping ensures KH,  inC, KW, outC shape
#%%
# Parameters for LHS Loading
Start_Index = 0
Run_Length = 8
Offset = IN_C_SIZE
RunTimes = ROW_LENGTH
TargetGroup = 0

Group_Decode = {
        0 : LHS_BANKS[0:8],
        1 : LHS_BANKS[8:16],
        2 : LHS_BANKS[16:24],
        3 : LHS_BANKS[24:32]
        } # To be added (Groups A~G) 

LoadLHS_ff_init = [[[(8*k + j * (IN_C_SIZE * ROW_LENGTH) + 32 * i ,8, IN_C_SIZE, ROW_LENGTH, k) for i in range(2)] for j in range(4)] for k in range(4)]
LoadLHS_ff = [[[[(batch * IN_C_SIZE * ROW_LENGTH * ROW_DEPTH + 8*k + j * (IN_C_SIZE * ROW_LENGTH) + 32 * i ,8, IN_C_SIZE, ROW_LENGTH, k) for i in range(2)]
 for k in range(4)] for j in range(ROW_DEPTH)] for batch in range(BATCH_SIZE)]

# 32 for Channel-Split (0/1/2/3/32/33/34/35)
# 4 Determined from 4 INC Consequetive in LHS loading
#LoadRHS_ff = [[[[(gA * OUT_C_SIZE * K_W * 4 + gB * 8 + inc_split * 32 * OUT_C_SIZE * K_W + outc_split * 16, # Start Index Offset
#                K_W * OUT_C_SIZE, 4, OUT_C_SIZE, gA, gB) for inc_split in range(2)] for outc_split in range(4)] for gA in range(8)] for gB in range(2)]  # 4 is determined by the kernel size

def RHS_OFFSET_GEN(gA, gB, k_num, inc_num, k_H_num, OC_num, in_channels=64, out_channels=64, k_size=5):
    idx = 3 * gA + k_num
    ch_group = idx // k_size
    kernel_idx = idx % k_size
    assert k_size in [1,3,5,7] # Channels_per_LHS_BUFFER_SET=16
    in_c = ch_group * 8 + inc_num % 8 + 32 * (inc_num // 8)
    inc_offset = out_channels * in_c
    outc_offset = OC_num * 16 + 8 * gB
    kH_offset = in_channels * out_channels * k_size * k_H_num
    kW_offset = in_channels * out_channels * kernel_idx
    OFFSET = inc_offset + outc_offset + kH_offset + kW_offset
    return OFFSET

LoadRHS_FF_1 = [[[[[[(RHS_OFFSET_GEN(gA, gB, k_num, inc_num, k_H_num, OC_num), gA, gB) for k_num in range(3)] for inc_num in range(16)]
     for k_H_num in range(K_H)] for OC_num in range(2)] for gB in range(2)] for gA in range(7)]

LoadRHS_FF_2 = [[[[[[(RHS_OFFSET_GEN(gA, gB, k_num, inc_num, k_H_num, OC_num), gA, gB) for k_num in range(3)] for inc_num in range(16)]
     for k_H_num in range(K_H)] for OC_num in range(2,4)] for gB in range(2)] for gA in range(7)]
#LoadRHS_ff = [[[[[ (RHS_PATTERN(gA, i) + gB * 8 + inc_split * 32 * OUT_C_SIZE + outc_split * 16) for inc_split in range(2)] for outc_split in range(4)] for gB in range(2)] for i in range(3)] for gA in range(8)]
#LoadLHS_group0 = [[(j * (IN_C_SIZE * ROW_LENGTH) + 8*i, 8, IN_C_SIZE, ROW_LENGTH, 0) for i in range(6)] for j in range(4)]
#LoadLHS_group1 = [[(j * (IN_C_SIZE * ROW_LENGTH) + 8*i, 8, IN_C_SIZE, ROW_LENGTH, 0) for i in range(6)] for j in range(4)]
#LoadLHS_group2 = [[(j * (IN_C_SIZE * ROW_LENGTH) + 8*i, 8, IN_C_SIZE, ROW_LENGTH, 0) for i in range(6)] for j in range(4)]
#LoadLHS_group3 = [[(j * (IN_C_SIZE * ROW_LENGTH) + 8*i, 8, IN_C_SIZE, ROW_LENGTH, 0) for i in range(6)] for j in range(4)]

#%%
def LoadLHS(Start_Index, Run_Length, Offset, RunTimes, TargetGroup):
    for i in range(RunTimes):
        total_offset = Start_Index + i * Offset
        loaded_lhs = DATA_FLATTENED[total_offset : total_offset + Run_Length]
        for k, lhs_bank in enumerate(Group_Decode[TargetGroup]):
            lhs_bank.write_row(loaded_lhs[k])

def PushRows(TargetGroup):
    for lhs_bank in Group_Decode[TargetGroup]:
        lhs_bank.push_rows()
        lhs_bank.reset_write_pointer()
        #%%
def LoadRHS(Start_Index, GroupA, GroupB):
    loaded_rhs = CONVW_FLATTENED[Start_Index : Start_Index + 8]
    for k, pe in enumerate(PE_array[GroupA][8*GroupB:8*(GroupB+1)]):
        pe.load_scratchpad(loaded_rhs[k])
#%%
# LOAD-LHS (Initial)
# Validated that these work as intended
for LoadLHS_ff_row in LoadLHS_ff[0][:4]:
    for group in range(4):
        PushRows(group)
    for LoadLHS_ff_group_r in LoadLHS_ff_row:
        for load_lhs in LoadLHS_ff_group_r:
            LoadLHS(*load_lhs)
            #%%
print('(IN_C 63, Row 1)')
print(LHS_BANKS[31].memory.transpose()[0][36:72]) # Should store IN-CHANNEL 63's row 1
print(data_base[0][63][1])
print('(IN_C 20, Row 0)')
print(LHS_BANKS[20].memory.transpose()[1][:36]) # Should store IN-CHANNEL 20's row 2
print(data_base[0][20][0])


for LoadRHS_ff_gA in LoadRHS_FF_1:
    for LoadRHS_ff_gB in LoadRHS_ff_gA:
        for LoadRHS_oc in LoadRHS_ff_gB:
            for L_kh in LoadRHS_oc:
                for L_inc in L_kh:
                    for L_k in L_inc:
                        LoadRHS(*L_k)


    
#%%
PSUM_Acc = np.zeros([128, 16, 4])

"""
for PE_row in PE_array:
    for pe in PE_row:
        pe.w_from_scratchpad()


ch0_r = LHS_BANKS[0].read(0, 36)
ch8_r = LHS_BANKS[8].read(0, 36)
ch16_r = LHS_BANKS[16].read(0, 36)
ch24_r = LHS_BANKS[24].read(0, 36)


PE_line0_datas = np.flip(ch0_r[0:3].transpose(), 0)
PE_line1_datas = np.flip(np.concatenate([ch0_r[3:5], ch8_r[0:1]]).transpose() ,0)
PE_line2_datas = np.flip(ch8_r[1:4].transpose() ,0)
PE_line3_datas = np.flip(np.concatenate([ch8_r[4:5], ch16_r[0:2]]).transpose() ,0)
PE_line4_datas = np.flip(ch16_r[2:5].transpose() ,0)
PE_line5_datas = np.flip(ch24_r[0:3].transpose() ,0)
PE_line6_datas = np.flip(np.concatenate([ch24_r[3:5], np.zeros([1,4])]).transpose() ,0)

Ps = np.zeros([16, 4])

for i, pe in enumerate(PE_array[0]):
    Ps[i] += pe.matmul(*PE_line0_datas)
for i, pe in enumerate(PE_array[1]):
    Ps[i] += pe.matmul(*PE_line1_datas)
for i, pe in enumerate(PE_array[2]):
    Ps[i] += pe.matmul(*PE_line2_datas)
for i, pe in enumerate(PE_array[3]):
    Ps[i] += pe.matmul(*PE_line3_datas)
for i, pe in enumerate(PE_array[4]):
    Ps[i] += pe.matmul(*PE_line4_datas)
for i, pe in enumerate(PE_array[5]):
    Ps[i] += pe.matmul(*PE_line5_datas)
for i, pe in enumerate(PE_array[6]):
    Ps[i] += pe.matmul(*PE_line6_datas)

PSUM_Acc[0] += Ps
"""
#%%
for k_idx in range(K_H):
    for ch_grp in range(2):
        for bank_select in range(8):
            for PE_row in PE_array:
                for pe in PE_row:
                    pe.w_from_scratchpad()
            ch0_r = LHS_BANKS[0 + bank_select].read(36*ch_grp, 36*(ch_grp + 1))
            ch8_r = LHS_BANKS[8 + bank_select].read(36*ch_grp, 36*(ch_grp + 1))
            ch16_r = LHS_BANKS[16 + bank_select].read(36*ch_grp, 36*(ch_grp + 1))
            ch24_r = LHS_BANKS[24 + bank_select].read(36*ch_grp, 36*(ch_grp + 1))

            for PIXEL in range(OUT_ROW_LENGTH):
                PE_line0_datas = np.flip(ch0_r[PIXEL:PIXEL+3].transpose(), 0)
                PE_line1_datas = np.flip(np.concatenate([ch0_r[PIXEL+3:PIXEL+5], ch8_r[PIXEL:PIXEL+1]]).transpose() ,0)
                PE_line2_datas = np.flip(ch8_r[PIXEL+1:PIXEL+4].transpose() ,0)
                PE_line3_datas = np.flip(np.concatenate([ch8_r[PIXEL+4:PIXEL+5], ch16_r[PIXEL:PIXEL+2]]).transpose() ,0)
                PE_line4_datas = np.flip(ch16_r[PIXEL+2:PIXEL+5].transpose() ,0)
                PE_line5_datas = np.flip(ch24_r[PIXEL:PIXEL+3].transpose() ,0)
                PE_line6_datas = np.flip(np.concatenate([ch24_r[PIXEL+3:PIXEL+5], np.zeros([1,4])]).transpose() ,0)

                Ps = np.zeros([16, 4])

                for i, pe in enumerate(PE_array[0]):
                    Ps[i] += pe.matmul(*PE_line0_datas)
                for i, pe in enumerate(PE_array[1]):
                    Ps[i] += pe.matmul(*PE_line1_datas)
                for i, pe in enumerate(PE_array[2]):
                    Ps[i] += pe.matmul(*PE_line2_datas)
                for i, pe in enumerate(PE_array[3]):
                    Ps[i] += pe.matmul(*PE_line3_datas)
                for i, pe in enumerate(PE_array[4]):
                    Ps[i] += pe.matmul(*PE_line4_datas)
                for i, pe in enumerate(PE_array[5]):
                    Ps[i] += pe.matmul(*PE_line5_datas)
                for i, pe in enumerate(PE_array[6]):
                    Ps[i] += pe.matmul(*PE_line6_datas)

                PSUM_Acc[PIXEL] += Ps
                
    for group in range(4):
        PushRows(group)
    for LoadLHS_ff_grp in LoadLHS_ff[0][4+k_idx]:
        for load_lhs in LoadLHS_ff_grp:
            LoadLHS(*load_lhs)
    

        
#%%
ch1_r = LHS_BANKS[1].read(0, 36)
ch9_r = LHS_BANKS[9].read(0, 36)
ch17_r = LHS_BANKS[17].read(0, 36)
ch25_r = LHS_BANKS[25].read(0, 36)


PE_line0_datas = np.flip(ch1_r[0:3].transpose(), 0)
PE_line1_datas = np.flip(np.concatenate([ch0_r[3:5], ch8_r[0:1]]).transpose() ,0)
PE_line2_datas = np.flip(ch8_r[1:4].transpose() ,0)
PE_line3_datas = np.flip(np.concatenate([ch8_r[4:5], ch16_r[0:2]]).transpose() ,0)
PE_line4_datas = np.flip(ch16_r[2:5].transpose() ,0)
PE_line5_datas = np.flip(ch24_r[0:3].transpose() ,0)
PE_line6_datas = np.flip(np.concatenate([ch24_r[3:5], np.zeros([1,4])]).transpose() ,0)

Ps = np.zeros([16, 4])

for i, pe in enumerate(PE_array[0]):
    Ps[i] += pe.matmul(*PE_line0_datas)
for i, pe in enumerate(PE_array[1]):
    Ps[i] += pe.matmul(*PE_line1_datas)
for i, pe in enumerate(PE_array[2]):
    Ps[i] += pe.matmul(*PE_line2_datas)
for i, pe in enumerate(PE_array[3]):
    Ps[i] += pe.matmul(*PE_line3_datas)
for i, pe in enumerate(PE_array[4]):
    Ps[i] += pe.matmul(*PE_line4_datas)
for i, pe in enumerate(PE_array[5]):
    Ps[i] += pe.matmul(*PE_line5_datas)
for i, pe in enumerate(PE_array[6]):
    Ps[i] += pe.matmul(*PE_line6_datas)

PSUM_Acc[0] += Ps

























#%%
