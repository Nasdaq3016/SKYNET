 # -*- coding: utf-8 -*-


from modules import PE, LHS_BANK
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
#
IN_C_SIZE = 32
IN_F_SIZE = 28
OUT_C_SIZE = 32
OUT_F_SIZE = 28
BATCH_SIZE = 4

K_SIZE = 3
PAD = 1
STRIDE = 1

PADDED_SIZE = IN_F_SIZE + 2 * PAD


LHS_BANKS = [LHS_BANK() for i in range(32)]
PE_array = [[PE() for i in range(16)] for j in range(8)]
PSUM_acc = np.zeros([128, 16, 4])

PE_DECODE = [{0 : list([PE_array[0][i]]*3) + list([PE_array[1][i]]*3) + list([PE_array[2][i]]*2),
              1 : list([PE_array[2][i]]*1) + list([PE_array[3][i]]*3) + list([PE_array[4][i]]*3) + list([PE_array[5][i]]),
              2 : list([PE_array[5][i]]*2) + list([PE_array[6][i]]*3) + list([PE_array[7][i]]*3)} for i in range(16)]


# Baseline Data, reordered
in_features = torch.randn(BATCH_SIZE, IN_C_SIZE, IN_F_SIZE, IN_F_SIZE)
conv_baselayer = nn.Conv2d(IN_C_SIZE, OUT_C_SIZE, K_SIZE, stride=STRIDE, padding=PAD, bias=False)
grad_out = torch.randn(BATCH_SIZE, OUT_C_SIZE, OUT_F_SIZE, OUT_F_SIZE)
s_go = get_scale(grad_out, 8)
grad_out = quantize(grad_out, s_go, 8).float()
convw = conv_baselayer.weight.detach()
s_convw = get_scale(convw, 8)
convw = quantize(convw, s_convw, 8).float()
# Padded & Flattened data, with order (Batch, H, W, INC)
in_features_reordered = F.pad(in_features, (PAD, PAD, PAD, PAD)).permute(0, 2, 3, 1).numpy().reshape(-1)
# Weight order (H, W, InC, OutC)
weight_reordered = convw.permute(2, 3, 1, 0).numpy().reshape(-1)
grad_out_reordered = grad_out.permute(0,2,3,1).numpy().reshape(-1)
#%%

def LoadLHS(start, run_length, run_offset, run_times, target_group, lhs_banks=LHS_BANKS, ext_data=grad_out_reordered):
    for i in range(run_times):
        total_offset = start + i * run_offset
        loaded_data = ext_data[total_offset:total_offset+run_length]
        for j, lhs_bank in enumerate(lhs_banks[8*target_group:8*(target_group+1)]):
            lhs_bank.write_row(loaded_data[j])

def LoadRHS(start, groupA, groupB, ext_data=weight_reordered):
    loaded_data = ext_data[start:start+8]
    PEs = PE_DECODE[groupB][groupA] # A: 0~2 B:0~15
    for i, pe in enumerate(PEs):
        pe.load_scratchpad(loaded_data[i])
#%%
"""
Load Groups  : Groups used for loading (Always grouped by 8 in BP/FF pass) 
Access Groups: Groups used for actual data access for PE array

The number of groups of LHS that will be used in BP pass is fixed to 3. (3 * 8 = 24 Load Banks Utilized)
First, // 24 should be distributed to each LHS group
Remaining channels should be distributed (as close as possible) evenly

"""

#%%
"""
Collect W0, W1, W2 | W0, W1, W2 | W0, W1 || W2 | W0, W1, W2 | W0, W1, W2 |
i th row in PE:
    Get from Bank #(3i) ~ #(3i+2)

After accumulating from all channels & batches, switch to next row in kernel.

"""

#max_fit = 512 // (3 * math.ceil(OUT_C_SIZE/24))
#W_SPLIT = IN_C_SIZE // max_fit
#W_REMAIN = IN_C_SIZE % max_fit

inc_split = 16 // K_SIZE
n_base_split_rhs = IN_C_SIZE // inc_split
remain_rhs = IN_C_SIZE % inc_split
extra_rhs = remain_rhs * K_SIZE # For PE array with 0~extra_rhs, has extra INC to compute.

def generate_LoadRHS_instructions_CONVBP(OChannels, IChannels, GroupA, GroupB, kH, kW, k_size = K_SIZE, in_c_size=IN_C_SIZE, out_c_size=OUT_C_SIZE):
    # OChannels : First of OutCs (8)
    och_offset = OChannels
    ich_offset = IChannels * out_c_size
    kW_offset = kW * in_c_size * out_c_size
    kH_offset = kH * k_size * in_c_size * out_c_size
    start = och_offset + ich_offset + kW_offset + kH_offset
    return (start, GroupA, GroupB)
    

LoadRHS_instructions = []

for inc_base in range(n_base_split_rhs):
    for i in range(inc_split):
        for kW in range(K_SIZE):
            inc = inc_split * inc_base + i
            gB = K_SIZE * i + kW
            for kH in range(K_SIZE):
                for c_base in range(n_base_c):
                    for g in range(3):
                        outc = 24 * c_base + 8 * g
                        LoadRHS_instructions.append(generate_LoadRHS_instructions_CONVBP(outc, inc, g, gB, kH, kW))
                        
                for extra in range(extra_groups):
                    outc = 24 * n_base_c + 8 * extra
                    LoadRHS_instructions.append(generate_LoadRHS_instructions_CONVBP(outc, inc, extra, gB, kH, kW))
for extra in range(extra_rhs):
    gB = extra
    inc = n_base_split_rhs * inc_split + extra // K_SIZE # 
    kW = extra % K_SIZE
    for kH in range(K_SIZE):
        for c_base in range(n_base_c):
            for g in range(3):
                outc = 24 * c_base + 8 * g
                LoadRHS_instructions.append(generate_LoadRHS_instructions_CONVBP(outc, inc, g, gB, kH, kW))
                
        for extra in range(extra_groups):
            outc = 24 * n_base_c + 8 * extra
            LoadRHS_instructions.append(generate_LoadRHS_instructions_CONVBP(outc, inc, extra, gB, kH, kW)) 
                
                #%%
n_inst_per_rhs_loop = K_SIZE * (n_base_c * 3 + extra_groups) * K_SIZE * inc_split
for i in range(len(LoadRHS_instructions)):
    LoadRHS(*LoadRHS_instructions[i])

#%%
assert OUT_C_SIZE % 8 == 0, "Only channel sizes with channels divisible by 8 are allowed!"
n_base_c = OUT_C_SIZE // 24
remain = OUT_C_SIZE % 24
extra_groups = remain // 8 # Extra groups (0/1/2. If 0:No extra channels(balanced) 1: 1 Extra group)


def generate_LoadLHS_instructions_CONVBP(Batch, Channels, RowNum, TargetGroups, LHS_BANKS=LHS_BANKS, outf_size=IN_F_SIZE, outc_size=OUT_C_SIZE):
    #assert len(Channels)==8
    ch_offset = Channels
    row_offset = outf_size * outc_size * RowNum
    batch_offset = Batch * outf_size * outf_size * outc_size
    start = ch_offset + row_offset + batch_offset
    run_times = outf_size # Row Size, Row-Wise Loading for LoadLHS
    run_length = 8
    run_offset = outc_size
    
    #lhs_banks = LHS_BANKS[8*TargetGroups:8*(TargetGroups+1)]
    return (start, run_length, run_offset, run_times, TargetGroups)
LoadLHS_instructions = []
for b in range(BATCH_SIZE):
    for r in range(OUT_F_SIZE):
        for c_base in range(n_base_c):
            for g in range(3):
                LoadLHS_instructions.append(generate_LoadLHS_instructions_CONVBP(b, 24 * c_base + 8 * g, r, g)) # Start channel in Load Group
        for extra in range(extra_groups):
            LoadLHS_instructions.append(generate_LoadLHS_instructions_CONVBP(b, 24*n_base_c + 8*extra, r, extra))



#[generate_LoadLHS_instructions_CONVBP(b, c, r, g, Grad_Out) for g in range(3)] for c in 
#%%
inst_per_global_row = n_base_c * 3 + extra_groups
for i in range(4):
    for lhs_bank in LHS_BANKS:
        lhs_bank.push_rows()
    for j in range(inst_per_global_row):
        LoadLHS(*LoadLHS_instructions[i*inst_per_global_row + j])
    #for lhs_bank in LHS_BANKS:
    #    lhs_bank.push_rows()
    #%%
print(LHS_BANKS[15].memory[64:96].transpose()[3])
print(grad_out[0][63][0])

print(LHS_BANKS[20].memory[32:64].transpose()[2])
print(grad_out[0][44][1])
#%%
P_ACCUMULATE = np.zeros((16, OUT_F_SIZE + K_SIZE - 1, 4))
#
 #[[np.zeros(4) for i in range(K_SIZE)] for j in range(inc_split)]

# Computing a column
for i in range(n_base_c):
    p_buffers = np.zeros((inc_split, K_SIZE, 4))
    for PE_row in PE_array:
        for pe in PE_row:
            pe.w_from_scratchpad()
    for p in range(OUT_F_SIZE):
        pixel = OUT_F_SIZE * i + p
        data_line = np.zeros((8,3,4))
        psums = np.zeros((inc_split*K_SIZE, 4))
        for b in range(8):
            data_0 = LHS_BANKS[3*b].read(pixel, pixel+1)
            data_1 = LHS_BANKS[3*b+1].read(pixel, pixel+1)
            data_2 = LHS_BANKS[3*b+2].read(pixel, pixel+1)
            # Data line assignment
            data_line[b] = np.flip(np.stack([data_0, data_1, data_2], axis=0).reshape(3,4),1)
            #p_buffers = p_buffers.reshape(K_SIZE * inc_split, 4)
            for psum, pe in zip(psums, PE_array[b][:K_SIZE * inc_split]):
                psum += pe.matmul(*data_line[b].transpose())
            #p_buffers = p_buffers.reshape(inc_split, K_SIZE, 4)
        psums = psums.reshape(inc_split, K_SIZE, 4)
        for inc in range(inc_split):
            # Shift & Add outputs
            for k in range(K_SIZE-1):
                p_buffers[inc][k] = psums[inc][k] + p_buffers[inc][k+1]
            P_ACCUMULATE[inc][p] = P_ACCUMULATE[inc][p] + p_buffers[inc][0]
            p_buffers[inc][K_SIZE-1] = psums[inc][K_SIZE-1]
    # Finish storing extra pixels
    for inc in range(inc_split):
        for k in range(1, K_SIZE):
            P_ACCUMULATE[inc][OUT_F_SIZE+k-1] = P_ACCUMULATE[inc][OUT_F_SIZE+k-1] + p_buffers[inc][k]
#%%
            
# Probably a error here somewhere. -> Now fixed. Error from automatically copying data.
# Works perfectly if this step is not required, (OUTC: divisible by 3)
#
FULL_ROWS = remain // 3
REMAIN_ROWS = remain % 3
p_buffers = np.zeros((inc_split, K_SIZE, 4))
# For extra remaining OUT_CHANNELS
# Load weights
for i in range(FULL_ROWS):
    for pe in PE_array[i]:
        pe.w_from_scratchpad()
for pe in PE_array[FULL_ROWS]:
    pe.w_from_scratchpad(REMAIN_ROWS)
#%%
# A. Load data from LHS banks
for p in range(OUT_F_SIZE):
    pixel = n_base_c * OUT_F_SIZE + p
    data_line = np.zeros((FULL_ROWS+1, 3, 4))
    psums = np.zeros((inc_split*K_SIZE, 4))
    for b in range(FULL_ROWS):
        data_0 = LHS_BANKS[3*b].read(pixel, pixel+1)
        data_1 = LHS_BANKS[3*b+1].read(pixel, pixel+1)
        data_2 = LHS_BANKS[3*b+2].read(pixel, pixel+1)
        
        data_line[b] = np.flip(np.stack([data_0, data_1, data_2], axis=0).reshape(3,4),1)

    
    data_ = []
    for r in range(REMAIN_ROWS):
        data_.append(LHS_BANKS[3*FULL_ROWS+r].read(pixel, pixel+1))
    for r in range(3-REMAIN_ROWS):
        data_.append(np.zeros((1,4)))
    data_line[-1] = np.flip(np.stack(data_, axis=0).reshape(3, 4), 1)
    
    
    # MATMUL OPERATIONS
    for b in range(FULL_ROWS+1):
        for psum, pe in zip(psums, PE_array[b][:K_SIZE * inc_split]):
            psum += pe.matmul(*data_line[b].transpose())
    
    psums = psums.reshape(inc_split, K_SIZE, 4)
    for inc in range(inc_split):
        # Shift & Add outputs
        for k in range(K_SIZE-1):
            p_buffers[inc][k] = psums[inc][k] + p_buffers[inc][k+1]
        P_ACCUMULATE[inc][p] = P_ACCUMULATE[inc][p] + p_buffers[inc][0]
        p_buffers[inc][K_SIZE-1] = psums[inc][K_SIZE-1]
    # Finish storing extra pixels
for inc in range(inc_split):
    for k in range(1, K_SIZE):
        P_ACCUMULATE[inc][OUT_F_SIZE+k-1] = P_ACCUMULATE[inc][OUT_F_SIZE+k-1] + p_buffers[inc][k]
#%%
#out_ref = F.conv_transpose2d(grad_out, convw)
out_ref = nn.grad.conv2d_input((BATCH_SIZE, IN_C_SIZE, IN_F_SIZE, IN_F_SIZE), convw, grad_out, padding=PAD, stride=STRIDE)
GRAD_IN = np.zeros((BATCH_SIZE, IN_C_SIZE, PADDED_SIZE, PADDED_SIZE))

P_ACCUMULATE = np.zeros((IN_C_SIZE, OUT_F_SIZE + K_SIZE - 1, 4))
#
 #[[np.zeros(4) for i in range(K_SIZE)] for j in range(inc_split)]
 # Need some update on cases when GRAD_OUT_F_SIZE % 8 != 0
 # Extra loop for Unfinished RHS_SPLIT
# Should add case for when rhs splits are not even

for SPLIT_NUM in range(1): # 1 Should be number of splits
    for batch in range(BATCH_SIZE):
        P_ACCUMULATE = np.zeros((IN_C_SIZE, OUT_F_SIZE + K_SIZE - 1, 4))
        for go_split in range(OUT_F_SIZE//4):
            grad_out_row = 4 * go_split
            inst_per_global_row = n_base_c * 3 + extra_groups
            for i in range(4*go_split, 4*(go_split+1)):
                for lhs_bank in LHS_BANKS:
                    lhs_bank.push_rows()
                for j in range(inst_per_global_row):
                    LoadLHS(*LoadLHS_instructions[batch * OUT_F_SIZE * inst_per_global_row + i*inst_per_global_row + j])

            # GRAD_IN_CHANNEL change
            for inc_base in range(n_base_split_rhs * (SPLIT_NUM), n_base_split_rhs * (SPLIT_NUM+1)):
                for kw in range(K_SIZE): 
                    # Repeat for Kernel Height size
                    # Target feature row
                    grad_in_row = kw + grad_out_row
                    # Computing a column
                    for i in range(n_base_c):   # Output channel split
                        p_buffers = np.zeros((inc_split, K_SIZE, 4))
                        for PE_row in PE_array:
                            for pe in PE_row:
                                pe.w_from_scratchpad()
                        for p in range(OUT_F_SIZE):
                            pixel = OUT_F_SIZE * i + p
                            data_line = np.zeros((8,3,4))
                            psums = np.zeros((inc_split*K_SIZE, 4))
                            for b in range(8):
                                data_0 = LHS_BANKS[3*b].read(pixel, pixel+1)
                                data_1 = LHS_BANKS[3*b+1].read(pixel, pixel+1)
                                data_2 = LHS_BANKS[3*b+2].read(pixel, pixel+1)
                                # Data line assignment 
                                data_line[b] = np.flip(np.stack([data_0, data_1, data_2], axis=0).reshape(3,4),1)
                                #p_buffers = p_buffers.reshape(K_SIZE * inc_split, 4)
                                for psum, pe in zip(psums, PE_array[b][:K_SIZE * inc_split]):
                                    psum += pe.matmul(*data_line[b].transpose())
                                #p_buffers = p_buffers.reshape(inc_split, K_SIZE, 4)
                            psums = psums.reshape(inc_split, K_SIZE, 4)
                            for inc in range(inc_split):
                                in_channel = inc_base * inc_split + inc + SPLIT_NUM * inc_split * n_base_split_rhs
                                # Shift & Add outputs
                                for k in range(K_SIZE-1):
                                    p_buffers[inc][k] = psums[inc][k] + p_buffers[inc][k+1]
                                if K_SIZE > 1:
                                    P_ACCUMULATE[in_channel][p] = P_ACCUMULATE[in_channel][p] + p_buffers[inc][0]
                                    p_buffers[inc][K_SIZE-1] = psums[inc][K_SIZE-1]
                                else:
                                    p_buffers[inc][K_SIZE-1] = psums[inc][K_SIZE-1]
                                    P_ACCUMULATE[in_channel][p] = P_ACCUMULATE[in_channel][p] + p_buffers[inc][0]
                        # Finish storing extra pixels
                        for inc in range(inc_split):
                            in_channel = inc_base * inc_split + inc + SPLIT_NUM * inc_split * n_base_split_rhs
                            for k in range(1, K_SIZE):
                                P_ACCUMULATE[in_channel][OUT_F_SIZE+k-1] = P_ACCUMULATE[in_channel][OUT_F_SIZE+k-1] + p_buffers[inc][k]
                                
                    # Probably a error here somewhere. -> Now fixed. Error from automatically copying data.
                    # Works perfectly if this step is not required, (OUTC: divisible by 3)
                    #
                    FULL_ROWS = remain // 3
                    REMAIN_ROWS = remain % 3
                    p_buffers = np.zeros((inc_split, K_SIZE, 4))
                    # For extra remaining OUT_CHANNELS
                    # Load weights
                    for i in range(FULL_ROWS):
                        for pe in PE_array[i]:
                            pe.w_from_scratchpad()
                    for pe in PE_array[FULL_ROWS]:
                        pe.w_from_scratchpad(REMAIN_ROWS)

                    # A. Load data from LHS banks
                    for p in range(OUT_F_SIZE):
                        pixel = n_base_c * OUT_F_SIZE + p
                        data_line = np.zeros((FULL_ROWS+1, 3, 4))
                        psums = np.zeros((inc_split*K_SIZE, 4))
                        for b in range(FULL_ROWS):
                            data_0 = LHS_BANKS[3*b].read(pixel, pixel+1)
                            data_1 = LHS_BANKS[3*b+1].read(pixel, pixel+1)
                            data_2 = LHS_BANKS[3*b+2].read(pixel, pixel+1)
                            
                            data_line[b] = np.flip(np.stack([data_0, data_1, data_2], axis=0).reshape(3,4),1)

                        
                        data_ = []
                        for r in range(REMAIN_ROWS):
                            data_.append(LHS_BANKS[3*FULL_ROWS+r].read(pixel, pixel+1))
                        for r in range(3-REMAIN_ROWS):
                            data_.append(np.zeros((1,4)))
                        data_line[-1] = np.flip(np.stack(data_, axis=0).reshape(3, 4), 1)
                        
                        
                        # MATMUL OPERATIONS
                        for b in range(FULL_ROWS+1):
                            for psum, pe in zip(psums, PE_array[b][:K_SIZE * inc_split]):
                                psum += pe.matmul(*data_line[b].transpose())
                        
                        psums = psums.reshape(inc_split, K_SIZE, 4)
                        for inc in range(inc_split):
                            in_channel = inc_base * inc_split + inc + SPLIT_NUM * inc_split * n_base_split_rhs
                            # Shift & Add outputs
                            for k in range(K_SIZE-1):
                                p_buffers[inc][k] = psums[inc][k] + p_buffers[inc][k+1]
                            if K_SIZE > 1:
                                P_ACCUMULATE[in_channel][p] = P_ACCUMULATE[in_channel][p] + p_buffers[inc][0]
                                p_buffers[inc][K_SIZE-1] = psums[inc][K_SIZE-1]
                            else:
                                p_buffers[inc][K_SIZE-1] = psums[inc][K_SIZE-1]
                                P_ACCUMULATE[in_channel][p] = P_ACCUMULATE[in_channel][p] + p_buffers[inc][0]
                        # Finish storing extra pixels
                    for inc in range(inc_split):
                        in_channel = inc_base * inc_split + inc + SPLIT_NUM * inc_split * n_base_split_rhs
                        for k in range(1, K_SIZE):
                            P_ACCUMULATE[in_channel][OUT_F_SIZE+k-1] = P_ACCUMULATE[in_channel][OUT_F_SIZE+k-1] + p_buffers[inc][k]


                    # Shift Rows, Write Finished Data if not last
                    if kw < (K_SIZE - 1):
                        for inc in range(inc_split):
                            in_channel = inc_base * inc_split + inc + SPLIT_NUM * inc_split * n_base_split_rhs
                            GRAD_IN[batch][in_channel][grad_in_row] = P_ACCUMULATE[in_channel].transpose()[0]
                            for p_ac in P_ACCUMULATE[in_channel]:
                                p_ac[:3] = p_ac[1:]
                                p_ac[3] = 0
                    else:
                        for ext_shift in range(1 + 4 - K_SIZE): # Push remainder of rows in PSUM_SPLIT
                            for inc in range(inc_split):
                                in_channel = inc_base * inc_split + inc + SPLIT_NUM * inc_split * n_base_split_rhs
                                GRAD_IN[batch][in_channel][grad_in_row+ext_shift] = P_ACCUMULATE[in_channel].transpose()[0]
                                for p_ac in P_ACCUMULATE[in_channel]:
                                    p_ac[:3] = p_ac[1:]
                                    p_ac[3] = 0


            """ CASE FOR WHEN INC has some leftovers"""
            if (remain_rhs > 0) and SPLIT_NUM==(1-1):
                for kw in range(K_SIZE):
                    # Repeat for Kernel Height size
                    # Target feature row
                    grad_in_row = kw + grad_out_row
                    # Computing a column
                    for i in range(n_base_c):
                        p_buffers = np.zeros((remain_rhs, K_SIZE, 4))
                        for PE_row in PE_array:
                            for pe in PE_row[:K_SIZE*remain_rhs]:
                                pe.w_from_scratchpad()
                        for p in range(OUT_F_SIZE):
                            pixel = OUT_F_SIZE * i + p
                            data_line = np.zeros((8,3,4))
                            psums = np.zeros((remain_rhs*K_SIZE, 4))
                            for b in range(8):
                                data_0 = LHS_BANKS[3*b].read(pixel, pixel+1)
                                data_1 = LHS_BANKS[3*b+1].read(pixel, pixel+1)
                                data_2 = LHS_BANKS[3*b+2].read(pixel, pixel+1)
                                # Data line assignment 
                                data_line[b] = np.flip(np.stack([data_0, data_1, data_2], axis=0).reshape(3,4),1)
                                #p_buffers = p_buffers.reshape(K_SIZE * remain_rhs, 4)
                                for psum, pe in zip(psums, PE_array[b][:K_SIZE * remain_rhs]):
                                    psum += pe.matmul(*data_line[b].transpose())
                                #p_buffers = p_buffers.reshape(remain_rhs, K_SIZE, 4)
                            psums = psums.reshape(remain_rhs, K_SIZE, 4)
                            for inc in range(remain_rhs):
                                in_channel = n_base_split_rhs * inc_split + inc + SPLIT_NUM * inc_split * n_base_split_rhs
                                # Shift & Add outputs
                                for k in range(K_SIZE-1):
                                    p_buffers[inc][k] = psums[inc][k] + p_buffers[inc][k+1]
                                if K_SIZE > 1:
                                    P_ACCUMULATE[in_channel][p] = P_ACCUMULATE[in_channel][p] + p_buffers[inc][0]
                                    p_buffers[inc][K_SIZE-1] = psums[inc][K_SIZE-1]
                                else:
                                    p_buffers[inc][K_SIZE-1] = psums[inc][K_SIZE-1]
                                    P_ACCUMULATE[in_channel][p] = P_ACCUMULATE[in_channel][p] + p_buffers[inc][0]
                        # Finish storing extra pixels
                        for inc in range(remain_rhs):
                            in_channel = n_base_split_rhs * inc_split + inc + SPLIT_NUM * inc_split * n_base_split_rhs
                            for k in range(1, K_SIZE):
                                P_ACCUMULATE[in_channel][OUT_F_SIZE+k-1] = P_ACCUMULATE[in_channel][OUT_F_SIZE+k-1] + p_buffers[inc][k]
                                
                    # Probably a error here somewhere. -> Now fixed. Error from automatically copying data.
                    # Works perfectly if this step is not required, (OUTC: divisible by 3)
                    #
                    FULL_ROWS = remain // 3
                    REMAIN_ROWS = remain % 3
                    p_buffers = np.zeros((remain_rhs, K_SIZE, 4))
                    # For extra remaining OUT_CHANNELS
                    # Load weights
                    for i in range(FULL_ROWS):
                        for pe in PE_array[i][:K_SIZE*remain_rhs]:
                            pe.w_from_scratchpad()
                    for pe in PE_array[FULL_ROWS][:K_SIZE*remain_rhs]:
                        pe.w_from_scratchpad(REMAIN_ROWS)

                    # A. Load data from LHS banks
                    for p in range(OUT_F_SIZE):
                        pixel = n_base_c * OUT_F_SIZE + p
                        data_line = np.zeros((FULL_ROWS+1, 3, 4))
                        psums = np.zeros((remain_rhs*K_SIZE, 4))
                        for b in range(FULL_ROWS):
                            data_0 = LHS_BANKS[3*b].read(pixel, pixel+1)
                            data_1 = LHS_BANKS[3*b+1].read(pixel, pixel+1)
                            data_2 = LHS_BANKS[3*b+2].read(pixel, pixel+1)
                            
                            data_line[b] = np.flip(np.stack([data_0, data_1, data_2], axis=0).reshape(3,4),1)

                        
                        data_ = []
                        for r in range(REMAIN_ROWS):
                            data_.append(LHS_BANKS[3*FULL_ROWS+r].read(pixel, pixel+1))
                        for r in range(3-REMAIN_ROWS):
                            data_.append(np.zeros((1,4)))
                        data_line[-1] = np.flip(np.stack(data_, axis=0).reshape(3, 4), 1)
                        
                        
                        # MATMUL OPERATIONS
                        for b in range(FULL_ROWS+1):
                            for psum, pe in zip(psums, PE_array[b][:K_SIZE * remain_rhs]):
                                psum += pe.matmul(*data_line[b].transpose())
                        
                        psums = psums.reshape(remain_rhs, K_SIZE, 4)
                        for inc in range(remain_rhs):
                            in_channel = n_base_split_rhs * inc_split + inc + SPLIT_NUM * inc_split * n_base_split_rhs
                            # Shift & Add outputs
                            for k in range(K_SIZE-1):
                                p_buffers[inc][k] = psums[inc][k] + p_buffers[inc][k+1]
                            if K_SIZE > 1:
                                P_ACCUMULATE[in_channel][p] = P_ACCUMULATE[in_channel][p] + p_buffers[inc][0]
                                p_buffers[inc][K_SIZE-1] = psums[inc][K_SIZE-1]
                            else:
                                p_buffers[inc][K_SIZE-1] = psums[inc][K_SIZE-1]
                                P_ACCUMULATE[in_channel][p] = P_ACCUMULATE[in_channel][p] + p_buffers[inc][0]
                        # Finish storing extra pixels
                    for inc in range(remain_rhs):
                        in_channel = n_base_split_rhs * inc_split + inc + SPLIT_NUM * inc_split * n_base_split_rhs
                        for k in range(1, K_SIZE):
                            P_ACCUMULATE[in_channel][OUT_F_SIZE+k-1] = P_ACCUMULATE[in_channel][OUT_F_SIZE+k-1] + p_buffers[inc][k]


                    # Shift Rows, Write Finished Data if not last
                    if kw < (K_SIZE - 1):
                        for inc in range(remain_rhs):
                            in_channel = inc_split * n_base_split_rhs + inc + SPLIT_NUM * inc_split * n_base_split_rhs
                            GRAD_IN[batch][in_channel][grad_in_row] = P_ACCUMULATE[in_channel].transpose()[0]
                            for p_ac in P_ACCUMULATE[in_channel]:
                                p_ac[:3] = p_ac[1:]
                                p_ac[3] = 0
                    else:
                        for ext_shift in range(1 + 4 - K_SIZE):
                            for inc in range(remain_rhs):
                                in_channel = inc_split * n_base_split_rhs + inc + SPLIT_NUM * inc_split * n_base_split_rhs
                                GRAD_IN[batch][in_channel][grad_in_row+ext_shift] = P_ACCUMULATE[in_channel].transpose()[0]
                                for p_ac in P_ACCUMULATE[in_channel]:
                                    p_ac[:3] = p_ac[1:]
                                    p_ac[3] = 0
            # Reset all read pointers in pe
            for pe_row in PE_array:
                for pe in pe_row:
                    pe.read_pointer = 0

        for in_channel in range(IN_C_SIZE):
            for leftover in range(K_SIZE-1):
                # Leftover shifts 
                GRAD_IN[batch][in_channel][PADDED_SIZE-K_SIZE+1+leftover] = P_ACCUMULATE[in_channel].transpose()[leftover]
if PAD > 0:      
    print((out_ref.numpy() - GRAD_IN[:,:,PAD:-PAD,PAD:-PAD]).max())
else:
    print((out_ref.numpy() - GRAD_IN).max())