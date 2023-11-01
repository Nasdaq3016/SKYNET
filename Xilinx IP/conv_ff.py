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
IN_F_SIZE = 14
OUT_C_SIZE = 256
OUT_F_SIZE = 14
BATCH_SIZE = 4

K_SIZE = 3
KH = 3
KW = 3
PAD = 1
STRIDE = 1

PADDED_SIZE_H = PADDED_SIZE_W = IN_F_SIZE + 2 * PAD
OUT_SIZE_H = OUT_SIZE_W = OUT_F_SIZE

LHS_BANKS = [LHS_BANK() for i in range(n_lhs_banks)]
PE_array = [[PE() for i in range(n_pe_col)] for j in range(n_pe_row)]



# Baseline Data, reordered
in_features = torch.randn(BATCH_SIZE, IN_C_SIZE, IN_F_SIZE, IN_F_SIZE)
conv_baselayer = nn.Conv2d(IN_C_SIZE, OUT_C_SIZE, K_SIZE, stride=STRIDE, padding=PAD, bias=False)

#s_in = get_scale(grad_out, 8)
#in_features = quantize(in_features, s_in, 8).float()
convw = conv_baselayer.weight.detach()
#s_convw = get_scale(convw, 8)
#convw = quantize(convw, s_convw, 8).float()

# Padded & Flattened data, with order (Batch, H, W, INC)
in_features_reordered = F.pad(in_features, (PAD, PAD, PAD, PAD)).permute(0, 2, 3, 1).numpy().reshape(-1)
# Weight order (H, W, InC, OutC)
weight_reordered = convw.permute(2, 3, 1, 0).numpy().reshape(-1)

#%%
def LoadLHS(start, run_length, run_offset, run_times, target_group, lhs_banks=LHS_BANKS, ext_data=in_features_reordered):
    for i in range(run_times):
        total_offset = start + i * run_offset
        loaded_data = ext_data[total_offset:total_offset+run_length]
        for j, lhs_bank in enumerate(lhs_banks[8*target_group:8*(target_group+1)]):
            lhs_bank.write_row(loaded_data[j])

def LoadRHS(start, groupA, groupB, ext_data=weight_reordered):
    loaded_data = ext_data[start:start+8]
    PEs = PE_array[groupA][8 * groupB: 8*(groupB+1)] # A: 0~7 B:0~1
    for i, pe in enumerate(PEs):
        pe.load_scratchpad(loaded_data[i])

#%%
def generate_LoadLHS_instructions_CONVFF(Batch, Channels, RowNum, TargetGroups, LHS_BANKS=LHS_BANKS, inf_size_H=PADDED_SIZE_H, inf_size_W=PADDED_SIZE_W, inc_size=IN_C_SIZE):
    batch_offset = inc_size * inf_size_H * inf_size_W * Batch
    row_offset = inc_size * inf_size_W * RowNum # RowNum : 0 ~ INF_SIZE_H-1
    ch_offset = Channels
    start = ch_offset + row_offset + batch_offset
    run_times = inf_size_W # Each row should run for inf_size_W
    run_length = 8
    run_offset = inc_size
    return (start, run_length, run_offset, run_times, TargetGroups)

assert IN_C_SIZE%8==0, "Only channel sizes with channels divisible by 8 are allowed!"

n_util_rows = ( 24 // KW ) * KW
n_util_banks = 24 if (KW in [1,2,4]) else 32
n_lhs_bank_groups = n_util_banks // 8
n_inc_split = IN_C_SIZE // n_util_banks + bool(IN_C_SIZE % n_util_banks)


loadlhs_instructions = []
for b in range(BATCH_SIZE):
    for r in range(PADDED_SIZE_H):
        remain_inc = IN_C_SIZE
        for idx in range(n_inc_split):
            for g in range(min(remain_inc//8, n_lhs_bank_groups)):
                inc = idx * n_util_banks + g * 8
                instruction = generate_LoadLHS_instructions_CONVFF(b, inc, r, g)
                loadlhs_instructions.append(instruction)
            remain_inc -= n_util_banks # Full bank rotation

n_inst_per_row = IN_C_SIZE // 8



#%%
def generate_LoadRHS_instructions_CONVFF(outc_start, inc, kh, kw, gA, gB, KW=KW, KH=KH, IN_C_SIZE=IN_C_SIZE, OUT_C_SIZE=OUT_C_SIZE):
    inc_offset = OUT_C_SIZE * inc
    KW_offset = OUT_C_SIZE * IN_C_SIZE * kw
    KH_offset = OUT_C_SIZE * IN_C_SIZE * KW * kh
    total_offset = outc_start + inc_offset + KW_offset + KH_offset
    return (total_offset, gA, gB)
    
kg_decode = {1:24, 2:12, 3:8, 4:6, 5:4, 7:3}
bpg_decode = {1:1, 2:2, 3:4, 4:4, 5:8, 7:8}

n_kernel_group = kg_decode[KW]
n_bank_per_group = bpg_decode[KW]

k_per_oc = 3 * KH * n_inc_split * n_bank_per_group
max_oc_per_PE = 512 // k_per_oc
# First fit as much OutChannel as possible. If we could fit it all, equals 0.
oc_per_PE_row = 16 * max_oc_per_PE
n_split_outc = OUT_C_SIZE // (oc_per_PE_row) + bool(OUT_C_SIZE % oc_per_PE_row)



remain_outc = OUT_C_SIZE
n_instr_per_split = []
loadrhs_instructions = []
for split in range(n_split_outc):
    n_instr = 0
    oc_per_PE = min(remain_outc // 16, max_oc_per_PE)
    #for oc_idx in range(oc_per_PE):
    for kh in range(KH):
        for oc_idx in range(oc_per_PE):
            oc_base = split * oc_per_PE_row + 16 * oc_idx #  base out-channel: use this as offset output channel
        #for kh in range(KH):
            for gB in range(2):
                oc_offset = oc_base + 8*gB
                remain_inc = IN_C_SIZE
                for inc_rot in range(n_inc_split):
                    n_util_banks_ = min(remain_inc//8, n_lhs_bank_groups) * 8
                    n_util_inc = (n_util_banks_ // n_bank_per_group)
                    n_util_kg = n_util_inc * KW
                    n_util_gA = n_util_kg // 3 #+ bool(n_util_kg % 3)
                    for bank_rot in range(n_bank_per_group):
                        for gA in range(n_util_gA):
                            for k in range(3):
                                kG = (3 * gA + k) // (KW)
                                kw = (3 * gA + k) % KW
                                inc_offset = inc_rot * n_util_banks + bank_rot + kG * n_bank_per_group
                                instruction = generate_LoadRHS_instructions_CONVFF(oc_offset, inc_offset, kh, kw, gA, gB)
                                loadrhs_instructions.append(instruction)
                                n_instr += 1
                                

                        for k in range(n_util_kg % 3):
                            kG = (3 * n_util_gA + k) // KW
                            kw = (3 * n_util_gA + k) % KW
                            inc_offset = inc_rot * n_util_banks + bank_rot + kG * n_bank_per_group
                            instruction = generate_LoadRHS_instructions_CONVFF(oc_offset, inc_offset, kh, kw, n_util_gA, gB)
                            loadrhs_instructions.append(instruction)
                            n_instr += 1
                    remain_inc = remain_inc - n_util_banks_
    n_instr_per_split.append(n_instr)
    remain_outc -= 16 * oc_per_PE
    # Now working correctly
                #%%

    
    #%%
    
baseline = conv_baselayer(in_features).detach().numpy()
loadrhs_instr_count = 0
outc_count = 0
remain_outc = OUT_C_SIZE
CONV_OUT = np.zeros([BATCH_SIZE, OUT_C_SIZE, OUT_SIZE_H, OUT_SIZE_W])
MAC_CYCLES = 0

for oc_split in range(n_split_outc):
    oc_per_PE = min(remain_outc // 16, max_oc_per_PE)
    
    # Reset loadlhs instructions
    loadlhs_instr_count = 0
    for pe_row in PE_array:
        for pe in pe_row:
            pe.reset_pointer()
    # Weight loading
    for i in range(n_instr_per_split[oc_split]):
        LoadRHS(*loadrhs_instructions[i + loadrhs_instr_count])
    loadrhs_instr_count += n_instr_per_split[oc_split]
    
    for batch in range(BATCH_SIZE):

        row_pointer = 0 # Next row to load
        for r in range(4):
            for lhs_bank in LHS_BANKS:
                lhs_bank.push_rows()
            for i in range(n_inst_per_row):
                LoadLHS(*loadlhs_instructions[loadlhs_instr_count + i])
            loadlhs_instr_count += n_inst_per_row
            row_pointer += 1
        
        #for r in range(PADDED_SIZE_H - 3):
        n_row_fullblocks = (PADDED_SIZE_H - KH + 1) // 4  # Number of full blocks (w/o padding)
        n_row_blocks = int(math.ceil((PADDED_SIZE_H - KH + 1)/4)) # Total number of blocks
         
        for row_block_num in range(n_row_blocks):
            # Reset PE read & write pointers
            
            PSUM_Acc = np.zeros([OUT_SIZE_W, 16*oc_per_PE, 4])
            for PE_row in PE_array:
                for pe in PE_row:
                    pe.reset_pointer()
            
            for kh in range(KH):
                # OUTC-SPLIT
                for oc in range(oc_per_PE):
                    # INC-SPLIT
                    remain_inc = IN_C_SIZE
                    for inc_split in range(n_inc_split):
                        n_util_banks_ = min(remain_inc//8, n_lhs_bank_groups) * 8
                        n_util_inc = (n_util_banks_ // n_bank_per_group)
                        n_util_kg = n_util_inc * KW
                        n_util_gA = n_util_kg // 3 #+ bool(n_util_kg % 3)
        
                        # Datas loading logic
                        for bank_select in range(n_bank_per_group):
                            # Load Weight Logic
                            for PE_rows in PE_array[:n_util_gA]:
                                for pe in PE_rows:
                                    pe.w_from_scratchpad(3)
                            if (n_util_kg%3):
                                for pe in PE_array[n_util_gA]:
                                    pe.w_from_scratchpad(n_util_kg%3)
                            
                            # Load ROWS logic
                            ch_rows = np.zeros((n_util_inc, PADDED_SIZE_W, 4))
                            for i in range(n_util_inc):
                                ch_rows[i] = LHS_BANKS[n_bank_per_group*i + bank_select].read(PADDED_SIZE_W * (inc_split), PADDED_SIZE_W * (inc_split+1))
                            PE_line_datas = np.zeros((n_util_gA+bool(n_util_kg % 3), 4, 3))
                            
                            for PIXEL in range(PADDED_SIZE_W - KW + 1):
                                # Load PIXELs logic
                                for gA in range(n_util_gA):
                                    for k in range(3):
                                        kG = (3 * gA + k) // (KW)
                                        kw = (3 * gA + k) % (KW)
                                        PE_line_datas[gA, :, k] = np.flip(ch_rows[kG][PIXEL+kw])
                                if (n_util_kg % 3):
                                    for k in range(n_util_kg % 3):
                                        kG = (3 * n_util_gA + k) // (KW)
                                        kw = (3 * n_util_gA + k) % (KW)
                                        PE_line_datas[n_util_gA, :, k] = np.flip(ch_rows[kG][PIXEL+kw])
                                
                                # GET PSUMS & ACCUMULATE
                                Ps = np.zeros([16, 4])
                                MAC_CYCLES += 1
                                for i, pe_line in enumerate(PE_line_datas):
                                    for j, pe in enumerate(PE_array[i]):
                                        Ps[j] += pe.matmul(*pe_line)
                                PSUM_Acc[PIXEL][16*oc:16*(oc+1)] += Ps

                        remain_inc -= n_util_banks_
                
                # Push rows after finishing a row processing.
                # Determine if zero-masks are necessary.
                if (kh < KH-1):
                    for lhs_bank in LHS_BANKS:
                        lhs_bank.push_rows()
                    if row_pointer < PADDED_SIZE_H:
                        for i in range(n_inst_per_row):
                            LoadLHS(*loadlhs_instructions[loadlhs_instr_count + i])
                        loadlhs_instr_count += n_inst_per_row
                        row_pointer += 1
                else:
                    for i in range(1+4-KH):
                        for lhs_bank in LHS_BANKS:
                            lhs_bank.push_rows()
                        if row_pointer < PADDED_SIZE_H:
                            for i in range(n_inst_per_row):
                                LoadLHS(*loadlhs_instructions[loadlhs_instr_count + i])
                            loadlhs_instr_count += n_inst_per_row
                            row_pointer += 1
            print(row_block_num)
            # Partial sum store
            CONV_OUT[batch, outc_count:outc_count+16*oc_per_PE, 4*row_block_num:min(4*(row_block_num+1),OUT_SIZE_W)] = PSUM_Acc.transpose([1,2,0])[:,:min(OUT_SIZE_W - 4*row_block_num, 4), :]
            
        for lhs_bank in LHS_BANKS:
            lhs_bank.reset_write_pointer()
    outc_count += 16*oc_per_PE
    remain_outc -= 16 * oc_per_PE

print((CONV_OUT - baseline).max())
              #%%          
                        
                    
                
                
                
                """
                if ((kh < KH-1) and (i<n_row_fullblocks)):  
                    for lhs_bank in LHS_BANKS:
                        lhs_bank.push_rows()
                    for i in range(n_inst_per_row):
                        LoadLHS(*loadlhs_instructions[loadlhs_instr_count + i])
                    loadlhs_instr_count += n_inst_per_row
                else:
                    if (i<n_row_fullblocks):
                        for i in range(1+4-KH):
                            for lhs_bank in LHS_BANKS:
                                lhs_bank.push_rows()
                            for i in range(n_inst_per_row):
                                LoadLHS(*loadlhs_instructions[loadlhs_instr_count + i])
                            loadlhs_instr_count += n_inst_per_row
                            """




