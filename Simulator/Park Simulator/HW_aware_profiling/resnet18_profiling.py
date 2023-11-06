# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:40:21 2019

@author: Jeongwoo Park
"""
import math
#import matplotlib.pyplot as plt

def total_MACs(inc, inf, k_size, outc, outf):
    N = inc*k_size
    M = outf
    L = outc
    return N*M*L

def datamax_DRAMACCESS(inc, inf, k_size, outc, outf, Nc=4, Dp=128, batch_size=1):
    in_features_size = inc*batch_size * inf
    weight_size = inc * outc * k_size
    out_features_size = outc*batch_size * outf
    repeat = math.ceil(batch_size * outf / (Nc * Dp))
    return in_features_size + weight_size * repeat + out_features_size 

def weightmax_DRAMACCESS(inc, inf, k_size, outc, outf, Nc=4, Dp=128, batch_size=1, weight_buffer_size=3*8*2048):
    in_features_size = inc*batch_size * inf
    weight_size = inc * outc * k_size
    out_features_size = outc*batch_size * outf
    repeat = math.ceil(weight_size / weight_buffer_size)
    return in_features_size * repeat + weight_size + out_features_size     

def psplit_DRAMACCESS(inc, inf, k_size, outc, outf, Nc=4, Dp=128, batch_size=1, weight_buffer_size=3*8*2048):
    in_features_size = inc*batch_size * inf
    weight_size = inc * outc * k_size
    out_features_size = outc * batch_size * outf
    repeat = math.ceil(weight_size / weight_buffer_size)
    return in_features_size + weight_size  + out_features_size + out_features_size * (repeat - 1) * 3
    #%%
def get_output_split(inc, k_width, k_height, outc, RAM_DEPTH=512):
    outc_max = math.floor(3*RAM_DEPTH/(math.ceil(inc*k_width/8)*k_height))
    return math.ceil(outc/(16*outc_max))

def get_ic_split(outc, k_width, k_height, inc, RAM_DEPTH=512):
    ic_per_col = 16 // k_width
    oc_per_pe = math.ceil(outc/24)
    oc_word_per_pe = k_height * oc_per_pe;
    inc_max = math.floor(3*RAM_DEPTH/oc_word_per_pe)    #Maximum allowable input channels
    max_ic_per_split = inc_max * ic_per_col    
    return math.ceil(inc/max_ic_per_split)

#def get_wg_split(batch_size, outc, ow, oh, RAM_DEPTH=512):
    
    
    
#%%
# ResNet18 Configurations-ImageNet
patterns = [(3, 224*224 , 7*7, 64, 112*112),
            (64, 56*56, 3*3, 64, 56*56),
            (64, 56*56, 3*3, 64, 56*56),
            (64, 56*56, 3*3, 64, 56*56),
            (64, 56*56, 3*3, 64, 56*56),
            (64, 56*56, 3*3, 128, 28*28),
            (128, 28*28, 3*3, 128, 28*28),
            (128, 28*28, 3*3, 128, 28*28),
            (128, 28*28, 3*3, 128, 28*28),
            (64, 56*56, 1*1, 128, 28*28),
            (128, 28*28, 3*3, 256, 14*14),
            (256, 14*14, 3*3, 256, 14*14),
            (256, 14*14, 3*3, 256, 14*14),
            (256, 14*14, 3*3, 256, 14*14),
            (128, 28*28, 1*1, 256, 14*14),
            (256, 14*14, 3*3, 512, 7*7),
            (512, 7*7, 3*3, 512, 7*7) ,
            (512, 7*7, 3*3, 512, 7*7) ,
            (512, 7*7, 3*3, 512, 7*7) ,
            (256, 14*14, 1*1, 512, 7*7)
            ]
#%%
patterns = [(3, 224*224/49 , 7*7, 64, 112*112/49),
            (64, 56*56/49, 3*3, 64, 56*56/49),
            (64, 56*56/49, 3*3, 64, 56*56/49),
            (64, 56*56/49, 3*3, 64, 56*56/49),
            (64, 56*56/49, 3*3, 64, 56*56/49),
            (64, 56*56/49, 3*3, 128, 28*28/49),
            (128, 28*28/49, 3*3, 128, 28*28/49),
            (128, 28*28/49, 3*3, 128, 28*28/49),
            (128, 28*28/49, 3*3, 128, 28*28/49),
            (64, 56*56/49, 1*1, 128, 28*28/49),
            (128, 28*28/49, 3*3, 256, 14*14/49),
            (256, 14*14/49, 3*3, 256, 14*14/49),
            (256, 14*14/49, 3*3, 256, 14*14/49),
            (256, 14*14/49, 3*3, 256, 14*14/49),
            (128, 28*28/49, 1*1, 256, 14*14/49),
            (256, 14*14/49, 3*3, 512, 7*7/49),
            (512, 7*7/49, 3*3, 512, 7*7/49) ,
            (512, 7*7/49, 3*3, 512, 7*7/49) ,
            (512, 7*7/49, 3*3, 512, 7*7/49) ,
            (256, 14*14/49, 1*1, 512, 7*7)
            ]
#%%
# VGG-16 Configurations
patterns = [(3, 224*224, 3*3, 64, 224*224),
            (64, 224*224, 3*3, 64, 224*224),
            (64, 112*112, 3*3, 128, 112*112),
            (128, 112*112, 3*3, 128, 112*112),
            (256, 56*56, 3*3, 256, 56*56),
            (256, 56*56, 3*3, 256, 56*56),
            (256, 28*28, 3*3, 512, 28*28),
            (512, 28*28, 3*3, 512, 28*28),
            (512, 28*28, 3*3, 512, 28*28),
            (512, 28*28, 3*3, 512, 14*14),
            (512, 14*14, 3*3, 512, 14*14),
            (512, 14*14, 3*3, 512, 14*14),
            (512, 14*14, 3*3, 512, 14*14)
            ]
#%%
# AlexNet Configuration
patterns = [(3, 224*224, 11*11, 64, 55*55),
            (64, 27*27, 5*5, 192, 27*27 ),
            (192, 13*13, 3*3, 384, 13*13),
            (384, 13*13, 3*3, 256, 13*13),
            (256, 13*13, 3*3, 256, 13*13)
            ]
#%%
dram_footprint=0
total_mac = 0
tot_instr = 0
# Total Cache for Conv-Activations
for pattern in patterns:
    oc_split = get_output_split(pattern[0], math.sqrt(pattern[2]), math.sqrt(pattern[2]), pattern[3], 512)
    tot_instr += oc_split
    #print(oc_split)
    dram_footprint += (4*oc_split*pattern[0] * pattern[1] + 4*pattern[3]*pattern[4] + pattern[0]*pattern[2]*pattern[3])
    total_mac += total_MACs(*pattern)

dram_bp = 0
for pattern in patterns[1:]:
    ic_split = get_ic_split(pattern[3], math.sqrt(pattern[2]), math.sqrt(pattern[2]), pattern[0], 512)
    dram_bp += (4*ic_split*pattern[3]*pattern[4] + 4*pattern[0]*pattern[1] + pattern[0]*pattern[2]*pattern[3])

dram_wg = 0
for pattern in patterns:
    

# Total Cache for Conv-indexes in pooling
    
# Total Cache for BNorm-Inputs
    

    # This is only considering for CONV-layers.
# Calc. for Forward phase 
print('Total Memory Footprint : %.3fMB'%((dram_footprint+dram_bp)/(1024 * 1024)))
print('Total MACs : %.3fGMAC'%(total_mac / (1024*1024*1024)))
#%%
# Testing against various HW configurations
# Analysis for forward phase
#weight_buffer_size = [3*8*2048, 3*8*]
R = 8#8 
C = 32#24
Nc = 4 
Nw = 3 
Dw = 256 
Dp = 16 
B = 2
freq = 200*1e6 # 200 MHz
words_per_cycle = 16 
weight_buffer_size = C * Nw * R * Dw # Number of words inside 
mux_per_PE = Nc * 3 * 16 / B
print('Partial Sums Buffer Size : %.3fKB'%(C*6*Nc*Dp/1024.0))
print('Weight Buffer Size : %.3fKB'%(C*2*Nw*R*Dw/1024.0))
print('GOPS (8b-8b) : %.3fGOPS'%(4*C*R*Nc*Nw*freq/(B*1e9)))
print('GOPS (16b-16b) : %.3fGOPS'%(C*R*Nc*Nw*freq/(B*1e9)))

MAC_per_cycle = 4*C*R*Nc*Nw / B
# Testing against various batch sizes
for batch_size in [1,4,16,64,256]:
    n_MAC = []
    #print('DataMax Strategy')
    datamax = []
    weightmax = []
    psplitmax = []
    balancedmax = []
    cycles_required = []
    latencies = []
    memory_bound = []
    dram_footprint=0
    for i, pattern in enumerate(patterns):
        dram_footprint += (2*batch_size*pattern[0] * pattern[1] + 2*pattern[0]*pattern[2]*pattern[3])
        da_d = int(datamax_DRAMACCESS(*pattern, Nc=Nc, Dp=Dp, batch_size=batch_size))
        datamax.append(da_d)
        da_w = int(weightmax_DRAMACCESS(*pattern, Nc=Nc, Dp=Dp, weight_buffer_size=weight_buffer_size, batch_size=batch_size))
        weightmax.append(da_w)
        da_p = int(psplit_DRAMACCESS(*pattern, Nc=Nc, Dp=Dp, weight_buffer_size=weight_buffer_size, batch_size=batch_size))
        psplitmax.append(da_p)
        da = min(da_d, da_w, da_p)
        balancedmax.append(da)
        numb_mac = batch_size * total_MACs(*pattern)
        n_MAC.append(numb_mac)
        # Latency in terms of cycles
        cycles_required.append((math.ceil(da/words_per_cycle), numb_mac/MAC_per_cycle)) # 8b Activations : 16 words per cycle
        memory_bound.append(math.ceil(da/words_per_cycle) > numb_mac/MAC_per_cycle)
        latency = max(math.ceil(da/words_per_cycle), numb_mac/MAC_per_cycle)
        latencies.append(latency)
        
    print('---------BATCH_SIZE=%i-------------'%(batch_size))
    print('Total DRAM Access for DataMax:\t\t%i'%(sum(datamax)))
    print('Total DRAM Access for WeightMax:\t%i'%(sum(weightmax)))
    print('Total DRAM Access for Psplit:\t\t%i'%(sum(psplitmax)))
    print('Total DRAM Access for Balanced:\t\t%i'%(sum(balancedmax)))
    print('Total Latency for inference : %i(cycles) | %.3fms'%(sum(latencies), sum(latencies) * 1e3 / freq) )
    print('DRAM Access per image(forward) :%.3f MB'%(sum(balancedmax)/(1024*1024*batch_size)))
    print('DRAM FootPrint for Training: %.3fMB'%(dram_footprint/(1024*1024)))
    print('# of MAC for forward pass : %.3fGOPs'%(sum(n_MAC)/(1024*1024*1024*batch_size)))
