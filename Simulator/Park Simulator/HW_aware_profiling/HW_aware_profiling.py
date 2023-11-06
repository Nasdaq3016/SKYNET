# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt
#%%
"""
def total_MACs(inc, inf, k_size, outc, outf):
    N = inc*k_size
    M = outf
    L = outc
    return N*M*L

def total_MACs_b(inc, inf, k_size, outc, outf):
    N = outf
    M = outc
    L = inc * k_size
    return N * M * L

def total_MACs_w(inc, inf, k_size, outc, outf):
    N = outc
    M = outf
    L = inc * k_size
    return N * M * L
    

def datamax_DRAMACCESS(inc, inf, k_size, outc, outf, Nc=4, Dp=128, batch_size=1):
    in_features_size = inc*batch_size * outf
    weight_size = inc * outc * k_size
    out_features_size = outc*batch_size * outf
    repeat = math.ceil(batch_size * outf / (Nc * Dp))
    return in_features_size + weight_size * repeat + out_features_size 

def weightmax_DRAMACCESS(inc, inf, k_size, outc, outf, Nc=4, Dp=128, batch_size=1, weight_buffer_size=3*8*2048):
    in_features_size = inc*batch_size * outf
    weight_size = inc * outc * k_size
    out_features_size = outc*batch_size * outf
    repeat = math.ceil(weight_size / weight_buffer_size)
    return in_features_size * repeat + weight_size + out_features_size     

def psplit_DRAMACCESS(inc, inf, k_size, outc, outf, Nc=4, Dp=128, batch_size=1, weight_buffer_size=3*8*2048):
    in_features_size = inc*batch_size * outf
    weight_size = inc * outc * k_size
    out_features_size = outc * batch_size * outf
    repeat = math.ceil(weight_size / weight_buffer_size)
    return in_features_size + weight_size  + out_features_size + out_features_size * (repeat - 1) * 3

def datamax_DRAMACCESS_b(inc, inf, k_size, outc, outf, Nc=4, Dp=128, batch_size=1):
    in_features_size = outf * batch_size * outc
    weight_size = outc  * inc * k_size
    out_features_size = outf * k_size * inc * batch_size
    repeat = math.ceil(outc / (Nc * Dp))
    return in_features_size + weight_size * repeat + out_features_size 

def weightmax_DRAMACCESS_b(inc, inf, k_size, outc, outf, Nc=4, Dp=128, batch_size=1, weight_buffer_size=3*8*2048):
    in_features_size = outf * batch_size * outc
    weight_size = outc  * inc * k_size
    out_features_size = outf * k_size * inc * batch_size
    repeat = math.ceil(weight_size / weight_buffer_size)
    return in_features_size * repeat + weight_size + out_features_size     

def psplit_DRAMACCESS_b(inc, inf, k_size, outc, outf, Nc=4, Dp=128, batch_size=1, weight_buffer_size=3*8*2048):
    in_features_size = outf * batch_size * outc
    weight_size = outc  * inc * k_size
    out_features_size = outf * k_size * inc * batch_size
    repeat = math.ceil(weight_size / weight_buffer_size)
    return in_features_size + weight_size  + out_features_size + out_features_size * (repeat - 1) * 3

def datamax_DRAMACCESS_w(inc, inf, k_size, outc, outf, Nc=4, Dp=128, batch_size=1):
    in_features_size = outc * outf * batch_size
    weight_size = outf * batch_size * inc 
    out_features_size = outc * inc * k_size
    repeat = math.ceil(batch_size * outf / (Nc * Dp))
    return in_features_size + weight_size * repeat + out_features_size 

def weightmax_DRAMACCESS_w(inc, inf, k_size, outc, outf, Nc=4, Dp=128, batch_size=1, weight_buffer_size=3*8*2048):
    in_features_size = outc * outf * batch_size
    weight_size = outf * batch_size * inc 
    out_features_size = outc * inc * k_size
    repeat = math.ceil(weight_size / weight_buffer_size)
    return in_features_size * repeat + weight_size + out_features_size     

def psplit_DRAMACCESS_w(inc, inf, k_size, outc, outf, Nc=4, Dp=128, batch_size=1, weight_buffer_size=3*8*2048):
    in_features_size = outc * outf * batch_size
    weight_size = outf * batch_size * inc 
    out_features_size = outc * inc * k_size
    repeat = math.ceil(weight_size / weight_buffer_size)
    return in_features_size + weight_size  + out_features_size + out_features_size * (repeat - 1) * 3
"""
#%%

def get_analysis(inc, inf, k_size, outc, outf, batch_size,  Nc=4, Dp=128, R=12, C=12, B=8, Nw=3, Dw=256, k_size=9, bit_width=16, io=128):
    lhs = [outf*batch_size, inc*k_size]
    rhs = [inc*k_size, ]
    
    total_cycles, ema = get_MAC_cycles(lhs, rhs, elhs, Nc, Dp, R, C, B, Nw, Dw, k_size, bit_width, io)
    

def get_MAC_cycles(lhs, rhs, elhs=None, Nc=4, Dp=128, R=12, C=12, B=8, Nw=3, Dw=256, k_size=9, bit_width=16, io=128):
    if elhs is None:
        elhs = lhs
    mac_wpc = bit_width // B
    io_wpc = io // bit_width # wpc = Word Per Cycle
    req_wpc = R * Nc * Nw // mac_wpc
    
    eff_io_wpc = io_wpc * k_size * elhs[0] / lhs[0]
    
    
    
    n_full_blocks = math.floor(lhs[0]/(Nc*Dp)) * math.ceil(lhs[1]/(Nw*R)) * math.ceil(rhs[1]/C)
    n_small_blocks = math.ceil(lhs[1]/(3*R)) * math.ceil(rhs[1]/C)
    small_block_size = math.ceil((lhs[0] - math.floor(lhs[0]/(Nc*Dp)) * Nc * Dp)/Nc)
    
    load_cycles = math.ceil(rhs[0] * rhs[1] / io_wpc)
    mac_cycles = math.ceil(req_wpc / eff_io_wpc) * mac_wpc * (n_full_blocks * Dp + n_small_blocks * small_block_size)
    
    w_buf_word_size = Dw * R * C * Nw * (16 // bit_width)
    lhs_repeat = math.ceil(rhs[0] * rhs[1] / w_buf_word_size)
    ema = elhs[0] * elhs[1] * lhs_repeat + rhs[0]*rhs[1] + lhs[0] * rhs[1]
    
    return load_cycles+mac_cycles, ema
    
    #%%
    
hw_config = {
        'Nw' : 3,
        'Nc' : 4,
        'R'  : 8,
        'C'  : 16,
        'Dp' : 128,
        'Dw' : 512,
        'lhs_buf' : 32*1024,
        'io' : 8,
        'B' : 8
        }
hw_config['rhs_buf'] = hw_config['Nw']*hw_config['R']*hw_config['C']*hw_config['Dw']
class Analysis_v01(object):
    def __init__(self, patterns, hw_config):
        self.patterns = patterns
        self.hw_config = hw_config
        self.analysis = {}
        self.analysis['ff'] = lambda batch_size, i : self.get_cycles_ff(self.patterns[i], batch_size, 8)
        self.analysis['bp'] = lambda batch_size, i : self.get_cycles_bp(self.patterns[i], batch_size, 8)
        self.analysis['wg'] = lambda batch_size, i : self.get_cycles_wg(self.patterns[i], batch_size, 16)
        
        
        #for i, pattern in enumerate(patterns):
        #    self.analysis['conv%i'%i] = {}
        #    self.analysis['conv%i'%i]['ff'] = lambda batch_size : self.get_cycles_ff(patterns[i], batch_size, 16)
        #    self.analysis['conv%i'%i]['bp'] = lambda batch_size : self.get_cycles_bp(patterns[i], batch_size, 16)
        #    self.analysis['conv%i'%i]['wg'] = lambda batch_size : self.get_cycles_wg(patterns[i], batch_size, 16)
        self.analysis['total'] = lambda batch_size : self.total_analysis(batch_size)
            #for batch_size in batch_sizes:
            #    self.analysis['conv%i'%i][batch_size] = {}
            #    self.analysis['conv%i'%i][batch_size]['ff'] = self.get_cycles_ff(pattern, batch_size, 16)
            #    self.analysis['conv%i'%i][batch_size]['bp'] = self.get_cycles_bp(pattern, batch_size, 16)
            #    self.analysis['conv%i'%i][batch_size]['wg'] = self.get_cycles_wg(pattern, batch_size, 16)
    
    def total_analysis(self, batch_size):
        template = {
                'layer_time' : 0,
                'layer_cycles' : 0,
                'mac_cycles' : 0,
                'w_load_cycles' : 0,
                'p_store_cycles' : 0,
                'dram_access' : 0
                }
        ff = template.copy()
        bp = template.copy()
        wg = template.copy()
        for i in range(len(self.patterns)):
            for key in template.keys():
                ff[key] += self.analysis['ff'](batch_size, i)[key]
                bp[key] += self.analysis['bp'](batch_size, i)[key]
                wg[key] += self.analysis['wg'](batch_size, i)[key]
        
        #for i in range(15, len(self.patterns)):
        #    for key in template.keys():
        #        bp[key] += self.analysis['bp'](batch_size, i)[key]
        #        wg[key] += self.analysis['wg'](batch_size, i)[key]
        
        ff['fps'] = 1. / ff['layer_time']
        bp['fps'] = 1. / bp['layer_time']
        wg['fps'] = 1. / wg['layer_time']
        
        total = {
                'ff' : ff,
                'bp' : bp,
                'wg' : wg,
                'fps' : 1. / (ff['layer_time'] + bp['layer_time'] + wg['layer_time'])
                }
        return total
            
    def lhs_buffer(self, lhs, rhs, loading_factor=1, bitwidth=16):
        """
        loading_factor: k^2 / s^2. if >1 than faster effective loading
        io_factor :     determines whether the layer will be bound by i/o. 
                        if == 1, i/o does not degrade performance
                        if > 1,  i/o degrades performance
        """
        bw_mod = 16 // bitwidth
        N = min(math.floor(bw_mod * self.hw_config['rhs_buf'] / rhs[0]), rhs[1])
        run_depth = min(math.floor(bw_mod * self.hw_config['lhs_buf'] * loading_factor / lhs[1]), self.hw_config['Dp'])
        eff_io = bw_mod * self.hw_config['io'] * N * loading_factor
        req_io = bw_mod * self.hw_config['R'] * self.hw_config['Nw'] * self.hw_config['Nc'] / (16 /(self.hw_config['B']))
        io_factor = max(1, req_io / eff_io)
        return N, run_depth, io_factor
    
    def get_cycles_ff(self, pattern, batch_size, bitwidth=16):
        # Data loads are "hidden" for ff path
        bw_mod = 16 // bitwidth
        inc, inf, k_size, outc, outf = pattern
        # (outf*batch_size, inc*kernel_size)
        lhs = (outf * batch_size, inc * k_size)
        # (inc * kernel_size, outc)
        rhs = (inc * k_size, outc)
        weight_transpose = False
        #if rhs[1] > lhs[0]:
        #    loading_factor = 1
        #    rhs, lhs = (lhs[1], lhs[0]), (rhs[1], rhs[0])
        #    N, run_depth, io_factor = self.lhs_buffer(lhs, rhs, loading_factor, bitwidth)
        #    weight_transpose = True
        #else:
        loading_factor = k_size * outf / inf
        N, run_depth, io_factor = self.lhs_buffer(lhs, rhs, loading_factor)
        n_lhs_row_full = math.floor(lhs[0] / run_depth)
        n_rhs_row = math.ceil(rhs[0] / (self.hw_config['R']*self.hw_config['Nw']))
        n_rhs_col = math.ceil(rhs[1] / self.hw_config['C'])
        
        lhs_residue = lhs[0] - n_lhs_row_full * run_depth
        mac_cycles = io_factor * (1/bw_mod) * (run_depth * n_lhs_row_full / self.hw_config['Nc'] + math.ceil(lhs_residue/self.hw_config['Nc'])) * n_rhs_row * n_rhs_col
        w_load_cycles = rhs[0] * rhs[1] * (1/bw_mod) / self.hw_config['io']
        p_store_cycles = lhs[0] * rhs[1] * (1/bw_mod) / self.hw_config['io']
        
        #total_cycles = mac_cycles + w_load_cycles + p_store_cycles
        total_cycles = mac_cycles + w_load_cycles
        
        lhs_repeat = max(math.ceil(rhs[0] * rhs[1] / (self.hw_config['rhs_buf'] * loading_factor)), 1)
        #lhs_da = lhs_repeat * lhs[0] * lhs[1]
        lhs_da = inc * inf * batch_size #* lhs_repeat
        rhs_da = rhs[0] * rhs[1]
        out_da = lhs[0] * rhs[1]
        dram_accesses = lhs_da + rhs_da + out_da
        
        analysis = {
                'layer_time' : total_cycles / (2*1e8*batch_size),
                'layer_cycles' : total_cycles,
                'mac_cycles' : mac_cycles,
                'w_load_cycles' : w_load_cycles,
                'p_store_cycles' : p_store_cycles,
                'io_factor' : io_factor,
                'transposed' : weight_transpose,
                'dram_access':  dram_accesses,
                'lhs_da' : lhs_da,
                'rhs_da' : rhs_da,
                'out_da' : out_da,
                'lhs_repeat' : lhs_repeat
                }
        return analysis
    
    def get_cycles_bp(self, pattern, batch_size, bitwidth=16):
        bw_mod = 16 // bitwidth
        inc, inf, k_size, outc, outf = pattern
        # bs, outc, outf | outc, inc, k_size, k_size
        lhs = (batch_size * outf, outc)
        rhs = (outc, inc*k_size)
        loading_factor = 1
        #if rhs[1] > lhs[0]:
        #    rhs, lhs = (lhs[1], lhs[0]), (rhs[1], rhs[0])
        #    N, run_depth, io_factor = self.lhs_buffer(lhs, rhs, loading_factor, bitwidth)
        #    weight_transpose = True
        #else:
        N, run_depth, io_factor = self.lhs_buffer(lhs, rhs, loading_factor)
        weight_transpose = False
        n_lhs_row_full = math.floor(lhs[0] / run_depth)
        n_rhs_row = math.ceil(rhs[0] / (self.hw_config['R']*self.hw_config['Nw']))
        n_rhs_col = math.ceil(rhs[1] / self.hw_config['C'])
        
        lhs_residue = lhs[0] - n_lhs_row_full * run_depth
        mac_cycles = io_factor * (1/bw_mod) * (run_depth * n_lhs_row_full / self.hw_config['Nc'] + math.ceil(lhs_residue/self.hw_config['Nc'])) * n_rhs_row * n_rhs_col
        w_load_cycles = rhs[0] * rhs[1] * (1/bw_mod) / self.hw_config['io']
        p_store_cycles = lhs[0] * rhs[1] * (1/bw_mod) / self.hw_config['io']
        # with efficient col2im on-chip
        
        
        #total_cycles = mac_cycles + w_load_cycles + p_store_cycles
        total_cycles = mac_cycles + w_load_cycles
        
        lhs_repeat = max(math.ceil(rhs[0] * rhs[1] / (self.hw_config['rhs_buf'] * loading_factor)), 1)
        lhs_da =  lhs[0] * lhs[1] * lhs_repeat
        rhs_da = rhs[0] * rhs[1]
        out_da = lhs[0] * rhs[1]
        dram_accesses = lhs_da + rhs_da + out_da
        
        analysis = {
                'layer_time' : total_cycles / (2*1e8*batch_size),
                'layer_cycles' : total_cycles,
                'mac_cycles' : mac_cycles,
                'w_load_cycles' : w_load_cycles,
                'p_store_cycles' : p_store_cycles,
                'io_factor' : io_factor,
                'transposed' : weight_transpose,
                'dram_access':  dram_accesses,
                'lhs_da' : lhs_da,
                'rhs_da' : rhs_da,
                'out_da' : out_da,
                'lhs_repeat' : lhs_repeat
                }
        return analysis
    
    def get_cycles_wg(self, pattern, batch_size, bitwidth=16):
        # Not exactly accurate, but an approximation
        bw_mod = 16 // bitwidth
        inc, inf, k_size, outc, outf = pattern
        lhs = (inc*k_size, outf * batch_size)
        rhs = (outf * batch_size, outc)
        loading_factor = 1

        run_length = min(math.floor(self.hw_config['lhs_buf'] / lhs[0]), math.floor(self.hw_config['rhs_buf'] / rhs[1]))
        req_speed = self.hw_config['Nw'] * self.hw_config['Nc'] * self.hw_config['R']
        #io_factor = max(req_speed / self.hw_config['io'], 1)
        io_factor = 1
        #loading_factor = k_size * outf / inf
        #N, run_depth, io_factor = self.lhs_buffer(lhs, rhs, loading_factor)
        
        mac_cycles = io_factor * (1/(bw_mod*bw_mod)) * (lhs[0] * lhs[1] * rhs[1] / (self.hw_config['Nc'] * self.hw_config['Nw'] * self.hw_config['R'] * self.hw_config['C']))
        rhs_repeat = math.ceil(lhs[0] / self.hw_config['Dp'])
        w_load_cycles = rhs[0] * rhs[1] * (1/bw_mod) / self.hw_config['io'] 
        p_store_cycles = lhs[0] * rhs[1] * (1/bw_mod) / self.hw_config['io']
        
        total_cycles = mac_cycles + w_load_cycles + p_store_cycles
        #total_cycles = mac_cycles + w_load_cycles
        
        #lhs_repeat = max(rhs[0] * rhs[1] / (self.hw_config['rhs_buf'] * loading_factor), 1)
        lhs_repeat = math.ceil(rhs[1] / self.hw_config['C'])
        lhs_da = inc * inf * batch_size * lhs_repeat
        #lhs_da = lhs[0] * lhs[1] * lhs_repeat
        rhs_da = rhs[0] * rhs[1] #* rhs_repeat
        out_da = lhs[0] * rhs[1] 
        dram_accesses = lhs_da + rhs_da + out_da
        
        analysis = {
                'layer_time' : total_cycles / (2*1e8*batch_size),
                'layer_cycles' : total_cycles,
                'mac_cycles' : mac_cycles,
                'w_load_cycles' : w_load_cycles,
                'p_store_cycles' : p_store_cycles,
                'io_factor' : io_factor,
                'dram_access':  dram_accesses,
                'lhs_da' : lhs_da,
                'rhs_da' : rhs_da,
                'out_da' : out_da,
                'rhs_repeat' : rhs_repeat
                }
        return analysis
        
        #%%
        
        

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
# Total Cache for Conv-Activations
for pattern in patterns:
    dram_footprint += (256*pattern[0] * pattern[1] + pattern[0]*pattern[2]*pattern[3])
    total_mac += total_MACs(*pattern)
# Total Cache for Conv-indexes in pooling
    
# Total Cache for BNorm-Inputs
    

    # This is only considering for CONV-layers.
print('Total Memory Footprint : %.3fMB'%(dram_footprint/(1024 * 1024)))
print('Total MACs : %.3fGMAC'%(total_mac / (1024*1024*1024)))
#%%
# Testing against various HW configurations
# Analysis for forward phase
#weight_buffer_size = [3*8*2048, 3*8*]
R = 12
C = 12
Nc = 4 
Nw = 3 
Dw = 512 
Dp = 128
B = 2
freq = 200*1e6 # 200 MHz
words_per_cycle = 16 
weight_buffer_size = 2 * C * Nw * R * Dw # Number of words inside 
mux_per_PE = Nc * 3 * 16 / B
print('Partial Sums Buffer Size : %.3fKB'%(C*6*Nc*Dp/1024.0))
print('Weight Buffer Size : %.3fKB'%(C*2*Nw*R*Dw/1024.0))
print('GOPS (8b-8b) : %.3fGOPS'%(2*4*C*R*Nc*Nw*freq/(B*1e9)))
print('GOPS (16b-16b) : %.3fGOPS'%(2*C*R*Nc*Nw*freq/(B*1e9)))

MAC_per_cycle = 4*C*R*Nc*Nw / B
#%%
# Testing against various batch sizes

for j,batch_size in enumerate([1,4,16,64,256]):
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
        #da = min(da_d, da_w, da_p)
        da = da_w
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
    #print('Total DRAM Access for Balanced:\t\t%i'%(sum(balancedmax)))
    print('Total Latency for inference : %i(cycles) | %.3fms'%(sum(latencies), sum(latencies) * 1e3 / freq) )
    print('DRAM Access per image(forward) :%.3f MB'%(sum(balancedmax)/(1024*1024*batch_size)))
    print('DRAM FootPrint for Training: %.3fMB'%(dram_footprint/(1024*1024)))
    #print('Latency on Titan-X : %.5fs'%(gpu_times_vgg[j]))
    print()
    #print('# of MAC for forward pass : %.3fGOPs'%(sum(n_MAC)/(1024*1024*1024*batch_size)))
print('# of MAC for forward pass : %.3fGOPs'%(sum(n_MAC)/(1024*1024*1024*batch_size)))
#%%
for j,batch_size in enumerate([1,4,16,64,256]):
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
    for i, pattern in enumerate(patterns[1:]):
        dram_footprint += (2*batch_size*pattern[0] * pattern[1] + 2*pattern[0]*pattern[2]*pattern[3])
        da_d = int(datamax_DRAMACCESS_b(*pattern, Nc=Nc, Dp=Dp, batch_size=batch_size))
        datamax.append(da_d)
        da_w = int(weightmax_DRAMACCESS_b(*pattern, Nc=Nc, Dp=Dp, weight_buffer_size=weight_buffer_size, batch_size=batch_size))
        weightmax.append(da_w)
        da_p = int(psplit_DRAMACCESS_b(*pattern, Nc=Nc, Dp=Dp, weight_buffer_size=weight_buffer_size, batch_size=batch_size))
        psplitmax.append(da_p)
        da = min(da_d, da_w, da_p)
        da = da_w
        balancedmax.append(da)
        numb_mac = batch_size * total_MACs_b(*pattern)
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
    #print('Total DRAM Access for Balanced:\t\t%i'%(sum(balancedmax)))
    print('Total Latency for inference : %i(cycles) | %.3fms'%(sum(latencies), sum(latencies) * 1e3 / freq) )
    print('DRAM Access per image(forward) :%.3f MB'%(sum(balancedmax)/(1024*1024*batch_size)))
    print('DRAM FootPrint for Training: %.3fMB'%(dram_footprint/(1024*1024)))
    #print('Latency on Titan-X : %.5fs'%(gpu_times_vgg[j]))
    print()
    #print('# of MAC for forward pass : %.3fGOPs'%(sum(n_MAC)/(1024*1024*1024*batch_size)))
print('# of MAC for backward pass : %.3fGOPs'%(sum(n_MAC)/(1024*1024*1024*batch_size)))
#%%
for j,batch_size in enumerate([1,4,16,64,256]):
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
        da_d = int(datamax_DRAMACCESS_w(*pattern, Nc=Nc, Dp=Dp, batch_size=batch_size))
        datamax.append(da_d)
        da_w = int(weightmax_DRAMACCESS_w(*pattern, Nc=Nc, Dp=Dp, weight_buffer_size=weight_buffer_size, batch_size=batch_size))
        weightmax.append(da_w)
        da_p = int(psplit_DRAMACCESS_w(*pattern, Nc=Nc, Dp=Dp, weight_buffer_size=weight_buffer_size, batch_size=batch_size))
        psplitmax.append(da_p)
        da = min(da_d, da_w, da_p)
        da = da_w
        balancedmax.append(da)
        numb_mac = batch_size * total_MACs_w(*pattern)
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
    #print('Total DRAM Access for Balanced:\t\t%i'%(sum(balancedmax)))
    print('Total Latency for inference : %i(cycles) | %.3fms'%(sum(latencies), sum(latencies) * 1e3 / freq) )
    print('DRAM Access per image(forward) :%.3f MB'%(sum(balancedmax)/(1024*1024*batch_size)))
    print('DRAM FootPrint for Training: %.3fMB'%(dram_footprint/(1024*1024)))
    #print('Latency on Titan-X : %.5fs'%(gpu_times_vgg[j]))
    print()
    #print('# of MAC for forward pass : %.3fGOPs'%(sum(n_MAC)/(1024*1024*1024*batch_size)))
print('# of MAC for backward pass : %.3fGOPs'%(sum(n_MAC)/(1024*1024*1024*batch_size)))