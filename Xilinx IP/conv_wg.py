from modules import PE, LHS_BANK
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
#
# Basic HW configurations
n_lhs_banks = 32
n_pe_row = 8
n_pe_col = 16
n_way_mac = 3
n_parallel_mac = 4

#
IN_C_SIZE = 256
IN_F_SIZE = 28
OUT_C_SIZE = 256
OUT_F_SIZE = 28
BATCH_SIZE = 4

K_SIZE = 3
KH = 3
KW = 3
PAD = 1
STRIDE = 1

PADDED_SIZE_H = PADDED_SIZE_W = IN_F_SIZE + 2 * PAD
OUT_SIZE_H = OUT_SIZE_W = OUT_F_SIZE
IN_SIZE_H = IN_SIZE_W = IN_F_SIZE

UTIL_H = PADDED_SIZE_H # TODO LOGIC

LHS_BANKS = [LHS_BANK() for i in range(n_lhs_banks)]
PE_array = [[PE() for i in range(n_pe_col)] for j in range(n_pe_row)]

in_features = torch.randn(BATCH_SIZE, IN_C_SIZE, IN_F_SIZE, IN_F_SIZE)
in_features_reordered = F.pad(in_features, (PAD, PAD, PAD, PAD)).permute(0, 2, 3, 1).numpy().reshape(-1)

grad_output = torch.randn(BATCH_SIZE, OUT_C_SIZE, OUT_F_SIZE, OUT_F_SIZE)
grad_output_reordered = grad_output.permute(0, 2, 3, 1).numpy().reshape(-1)

def LoadLHS_WG(start, run_length, run_offset, run_times, target_group, lhs_banks=LHS_BANKS, ext_data=in_features_reordered):
    for i in range(run_times):
        total_offset = start + i * run_offset
        loaded_data = ext_data[total_offset:total_offset+run_length]
        if target_group % 2 == 0:
            lhs_banks[target_group].write_col(loaded_data[0:4])
            lhs_banks[target_group+1].write_col(loaded_data[4:8])
        else:
            lhs_banks[target_group].write_col(loaded_data[0:4])
            lhs_banks[target_group-1].write_col(loaded_data[4:8])

"""
def LoadRHS(start, groupA, groupB, run_times, run_offset, ext_data=grad_output_reordered):
    for i in range(run_times):
        total_offset = start + i * run_offset
        loaded_data = ext_data[total_offset:total_offset+8]
        PEs = PE_array[groupA][8 * groupB: 8*(groupB+1)] # A: 0~7 B:0~1
        for i, pe in enumerate(PEs):
            pe.load_scratchpad(loaded_data[i])
"""
def LoadRHS(start, groupA, groupB, ext_data=grad_output_reordered):
    loaded_data = ext_data[start:start+8]
    PEs = PE_array[groupA][8 * groupB: 8*(groupB+1)] # A: 0~7 B:0~1
    for i, pe in enumerate(PEs):
        pe.load_scratchpad(loaded_data[i])

def generate_LoadLHS_instructions_CONVWG(Batch, Channels, RowNum, TargetGroups, LHS_BANKS=LHS_BANKS, inf_size_H=PADDED_SIZE_H, inf_size_W=PADDED_SIZE_W, inc_size=IN_C_SIZE):
    batch_offset = inc_size * inf_size_H * inf_size_W * Batch
    row_offset = inc_size * inf_size_W * RowNum # RowNum : 0 ~ INF_SIZE_H-1
    ch_offset = Channels
    PIXEL_START = TargetGroups % 4 # Should be one of 0~3. 
    pixel_offset = inc_size * PIXEL_START

    start = ch_offset + row_offset + batch_offset + pixel_offset

    run_times = int(math.ceil((inf_size_W - PIXEL_START) / 4))
    run_length = 8
    run_offset = inc_size * 4  # Offset of 4 pixels
    return (start, run_length, run_offset, run_times, TargetGroups)


assert IN_C_SIZE%8==0
n_inc_split = IN_C_SIZE // 8
n_rows_split = PADDED_SIZE_H // 8 + bool(PADDED_SIZE_H % 8)

LoadLHS_Instructions = []
for inc_split in range(n_inc_split):
    inc_start = 8 * inc_split
    # Should reset (1,3) Bank pointers at this point.
    for batch in range(BATCH_SIZE):
        REMAIN_ROWS = UTIL_H
        for row_split in range(n_rows_split):
            n_rows = min(REMAIN_ROWS, 8)
            for group in range(n_rows):
                for bank in [0,2]:
                    target_group = group * 4 + bank
                    rownum = row_split * 8 + group
                    instruction = generate_LoadLHS_instructions_CONVWG(batch, inc_start, rownum, target_group)
                    LoadLHS_Instructions.append(instruction)
            REMAIN_ROWS -= n_rows
    # Should reset (0,2) Bank write pointers at this point.
    for batch in range(BATCH_SIZE):
        REMAIN_ROWS = UTIL_H
        for row_split in range(n_rows_split):
            n_rows = min(REMAIN_ROWS, 8)
            for group in range(n_rows):
                for bank in [1,3]:
                    target_group = group * 4 + bank
                    rownum = row_split * 8 + group
                    instruction = generate_LoadLHS_instructions_CONVWG(batch, inc_start, rownum, target_group)
                    LoadLHS_Instructions.append(instruction)
            REMAIN_ROWS -= n_rows

n_half_load_instructions = int(len(LoadLHS_Instructions) / (2*n_inc_split))
#%%
def generate_LoadRHS_instructions_CONVWG(batch, outc_start, rownum, pixel_start, gB, outf_size_H=OUT_SIZE_H, outf_size_W=OUT_SIZE_W, outc_size=IN_C_SIZE):
     row_rank = rownum % 6
     pixel_rank = pixel_start % 4
     gA = (row_rank * 4 + pixel_rank) // 3
     batch_offset = outc_size * outf_size_H * outf_size_W * batch
     row_offset = outc_size * outf_size_W * rownum
     ch_offset = outc_start
     pixel_offset = outc_size * pixel_start

     start = ch_offset + pixel_offset + row_offset + batch_offset

     return (start, gA, gB)


# TODO : Add H-split in LHS
# Should happen when pixels_per_oc > 1024 (MAX_PE). Happens when output pixel sizes > 
pixels_per_oc = BATCH_SIZE * int(math.ceil(OUT_SIZE_W/4.0) *3 ) * int(math.ceil(OUT_SIZE_H / 6.0))
n_oc_per_PE = max(1024 // pixels_per_oc, 1) # should be > 1
if pixels_per_oc < 1024:
    oh_split =  .
n_oc_per_PE_row = n_oc_per_PE * 16
n_oc_split = OUT_C_SIZE // n_oc_per_PE_row + bool(OUT_C_SIZE % n_oc_per_PE_row)
n_out_rows_split = OUT_SIZE_H // 6 + bool(OUT_SIZE_H % 6)

n_pixel_split = OUT_SIZE_W // 4 + bool(OUT_SIZE_W % 4)

remain_outc = OUT_C_SIZE
LoadRHS_Instructions = []
for oc_split in range(n_oc_split):
    oc_per_PE = min(remain_outc // 16, n_oc_per_PE)
    for oc_idx in range(n_oc_per_PE):
        oc_base = oc_split * n_oc_per_PE_row + 16 * oc_idx
        for gB in range(2):
            oc_start = 16 * oc_base + 8 * gB
            for batch in range(BATCH_SIZE):
                REMAIN_ROWS = OUT_SIZE_H
                for row_split in range(n_out_rows_split):
                    n_rows = min(REMAIN_ROWS, 6)
                    REMAIN_PIXELS = OUT_SIZE_W
                    for pixel_split in range(n_pixel_split):
                        for row in range(n_rows):
                            rownum = 6 * row_split + row
                            n_pixels = min(REMAIN_PIXELS, 4)
                            for pixel in range(n_pixels):
                                pixel_start = pixel_split * 4 + pixel
                                instruction = generate_LoadRHS_instructions_CONVWG(batch, oc_start, rownum, pixel_start, gB)
                                LoadRHS_Instructions.append(instruction)
                        REMAIN_PIXELS -= n_pixels
                    REMAIN_ROWS -= n_rows

    remain_outc -= 16

for i in range(960):
    LoadRHS(*LoadRHS_Instructions[i])
    
#%%
# TEST
lhs_instr_count = 0
# Reset WRITE POINTERS
for i in range(8):
    for j in [1,3]:
        n_w = int(math.ceil((PADDED_SIZE_W - j)/ 4.0))
        n_h = int(math.ceil((UTIL_H - i) / 8.0))
        reset_addr = BATCH_SIZE * n_w * n_h
        LHS_BANKS[4*i+j].reset_write_pointer(reset_addr)
    for j in [0,2]:
        LHS_BANKS[4*i+j].reset_write_pointer()
for k in range(n_half_load_instructions):
    LoadLHS_WG(*LoadLHS_Instructions[lhs_instr_count])
    lhs_instr_count += 1

# Reset WRITE POINTERS & LOAD
for i in range(8):
    for j in [0,2]:
        n_w = int(math.ceil((PADDED_SIZE_W - j)/ 4.0))
        n_h = int(math.ceil((UTIL_H - i) / 8.0))
        reset_addr = BATCH_SIZE * n_w * n_h
        LHS_BANKS[4*i+j].reset_write_pointer(reset_addr)
    for j in [1,3]:
        LHS_BANKS[4*i+j].reset_write_pointer()
for k in range(n_half_load_instructions):
    LoadLHS_WG(*LoadLHS_Instructions[lhs_instr_count])
    lhs_instr_count += 1

#%%

# OC_SPLIT determination
RHS_per_batch = int(3*math.ceil(OUT_SIZE_H/6)*math.ceil(OUT_SIZE_W/4))
RHS_per_PE = BATCH_SIZE * RHS_per_batch
fittable_oc_size = min(512 // RHS_per_PE, 1)
util_oc_size = min(OUT_C_SIZE // 16, fittable_oc_size)

n_oc_split = int(math.ceil(OUT_C_SIZE /(16 * util_oc_size)))

# Required storage in RHS/LHS
LHS_per_batch = int(2*math.ceil(IN_SIZE_H/8)*math.ceil(IN_SIZE_W/4))    # LHS must satisfy C_PER_BANK >=8
assert RHS_per_batch <= 512, "IMAGE TOO LARGE FOR RHS"
maxbatch_fit_RHS = 512 // RHS_per_batch if util_oc_size <= 2 else BATCH_SIZE
maxbatch_fit_LHS = 256 // LHS_per_batch 

fittable_batch_size = min(maxbatch_fit_LHS, maxbatch_fit_RHS)
util_batch_size = min(BATCH_SIZE, fittable_batch_size)

n_batch_split = int(math.ceil(BATCH_SIZE / float(util_batch_size)))

# PSUM_SPLIT determination
# Must satisfy (8 * n * util_oc_size * KH * KW) < 4 * 128 (PSUM_DEPTH)
# Where n is the step size for P_SPLIT 
assert util_oc_size * KH * KW <= 64, "KERNEL TOO LARGE FOR PSUMS"
util_psumstep_size = 128 // (2*util_oc_size*KH*KW)

n_psum_split = int(math.ceil(IN_C_SIZE / float(util_psumstep_size * 8)))

# assert INC_SIZE % 8 == 0
n_inc_split = IN_C_SIZE // 8 

# OUTPUT H SPLIT
n_oh_split = int(math.ceil(OUT_SIZE_H/6.0))

# OUTPUT W SPLIT
n_ow_split = int(math.ceil(OUT_SIZE_W/float(KW)))


REMAIN_OC = OUT_C_SIZE
for oc_idx in range(n_oc_split):
    n_oc = min(16 * util_oc_size, REMAIN_OC)        # Actual oc in use for this loop
    assert n_oc % 16 == 0
    REMAIN_B = BATCH_SIZE                       
    for b_idx in range(n_batch_split): 
        n_batch = min(util_batch_size, REMAIN_B)    # Actual batch in use for this loop
        # Load RHS (output gradients) to PE array
        # TODO
        REMAIN_INC = IN_C_SIZE
        for psum_split in range(n_psum_split):
            if b_idx > 0:
                # Load PSUM logic should be here
                # TODO
                PSUM_acc = stored_PSUM

            for b in range(n_batch):
                batch_num = b_idx * util_batch_size + b
                # Set RHS starting point (read address)

                step_size = min(util_psumstep_size, REMAIN_INC // 8) # 8 inc per step size
                for inc_idx in range(step_size):
                    
                    # Load LHS (input features) to LHS_BANK
                    # Make sure to add inverted loading logic
                    # TODO
                    # Reset pointers


                    for inc_ in range(2):
                        inc_start = inc_idx * 8 + psum_split * util_psumstep_size * 8 + inc_ * 4
                        use_inverted_pixels = bool(inc_) #0~3 non-invert 4~7 invert
                        for pe_row in PE_array:
                            for pe in pe_row:
                                pe.reset_pointer()
                        routing_select = 0
                        REMAIN_OH = OUT_SIZE_H
                        for oh_split in range(n_oh_split):
                            n_oh = min(6, REMAIN_OH)

                            # Used for determining address reset for pe
                            start_addr_det = [int(math.ceil((OUT_SIZE_W-i)/4.0)) for i in range(4)]
                            n_pixels_pe_row = [OUT_SIZE_W - start_addr_det[-(i+1)%4] for i in range(8)]
                            
                            # Full read 
                            util_pe_row = (n_oh * 4) // 3
                            # if read < 3 
                            remain_pe_row = (n_oh * 4) % 3
                            for kh in range(KH):
                                # PE should reset to start of current row
                                for rownum, pe_row in enumerate(PE_array[:util_pe_row+bool(remain_pe_row)]):
                                    for pe in pe_row:
                                        n_pixels = oh_split * n_pixels_pe_row[rownum]    # Start address for each PE rows
                                        pe.reset_read_pointer(n_pixels)

                                select_LHS_GROUPS = [LHS_BANKS[4*j:4*(j+1)] for j in (np.arange(n_oh)+routing_select)%8]

                                remain_W = OUT_SIZE_W
                                for ow_split in range(n_ow_split):
                                    n_ow = min(4, REMAIN_W)
                                    n_valid_rd_cycles = KW + n_ow - 5
                                    rd_cycles_count = n_valid_rd_cycles
                                    # Load from scratchpad in PE array
                                    for rownum in range(8):
                                        n_load = 0
                                        for idx in range(3):
                                            h = (rownum * 3 + idx) // 4
                                            w = (rownum * 3 + idx) % 4
                                            if h < n_oh and w < n_ow:
                                                n_load += 1
                                        for pe in PE_array[rownum]:
                                            pe.w_from_scratchpad(n_load)

                                        
                                    
                                    for kw in range(KW):
                                        # Data loading
                                        PE_dataline = np.zeros(6, 4, 4) # ROW/PIXEL/CH, reshape to (8,3,4)
                                        PSUMS = np.zeros(16, 4)
                                        if (kw==0):
                                            # Dispatcher initialization
                                            # May be optimized later (to only load all values when kw==0 & ow_split==0)
                                            for linenum, bank_group in enumerate(select_LHS_GROUPS):
                                                for pixelnum, bank in enumerate(bank_group[:n_ow+KW-1]):   # A constraint on pixelnum?
                                                    # Consider inverted loading here
                                                    (use_inverted_pixels) * (-2*(pixelnum%2))
                                                    PE_dataline[linenum][pixelnum] = bank.read_single()     # Inverted & Invalid data comes in? -> ignored in PE_dataline MAC operations
                                        else: 
                                            # Pixel shifts
                                            PE_dataline[:,:3,:] = PE_dataline[:,1:,:]
                                            # New line load
                                            if rd_cycles_count < n_valid_rd_cycles:
                                                for linenum, bank_group in enumerate(select_LHS_GROUPS):
                                                    PE_dataline[linenum][3] = bank_group[(kw-1)%4].read_single()
                                            else:
                                                for linenum in range(6):
                                                    PE_dataline[linenum][3] = 0
                                        data_patch = (input_padded[batch_num, inc_start:inc_start+4, oh_split*6:oh_split*6+n_oh, ow_split*4:ow_split*4+n_ow]).transpose(2,0,1) # CH/ROW/PIXEL -> Row/pixel/ch
                                        assert (PE_dataline[:n_oh, :n_ow, :] - data_patch).abs().max() < 1e-6 
                                        PE_dataline = PE_dataline.reshape(8, 3, 4) # Now PE/data/CH

                                        # MAC operations
                                        for idx ,dataline in enumerate(PE_dataline):
                                            for col, pe in enumerate(PE_array[idx]):
                                                PSUMS[col] += pe.matmul(*PE_dataline[idx])
                                    REMAIN_W -= n_ow
                                    # End of a row processing for single 
                                # RESET read pixel start location in LHS BANKS
                                for groupnum in range(routing_select+1, routing_select+6):
                                    j = groupnum % 8
                                    for bank in LHS_BANKS[4*j:4*(j+1)]:
                                        start_addr = 
                                        
                                # Set read pixel start location to the next row for these banks
                                for groupnum in range()

                                routing_select += 1 
                                # End of 
                            
                            routing_select += (6-KH)
                            # End of 
                        routing_select = 0
                                    
                                    # PE load next datas
                                    for pe_row in PE_array[:util_pe_row]:
                                        for pe in pe_row:
                                            pe.w_from_scratchpad(3)
                                    if remain_pe_row:
                                        for pe in PE_array[util_pe_row]:
                                            pe.w_from_scratchpad(remain_pe_row)
                                        
                                        
                                        

                                routing_select += 1




                        

                        
                        


                    REMAIN_OH -= n_oh
                assert REMAIN_OH==0, "OUTPUT ROW not finished! CURRENT ROW: %i"%(OUT_SIZE_H-REMAIN_OH-1)
                 
            REMAIN_INC -= 8 * step_size

        REMAIN_B -= n_batch
    
    REMAIN_OC -= n_oc
