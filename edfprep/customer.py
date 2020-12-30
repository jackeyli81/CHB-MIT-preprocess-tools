'''
custermize your own channel pick function

this lib also provide summary-txt generator functions, as long as your edf file is normative
'''

import os, glob, datetime, scipy.io, stft, re
import numpy as np, pandas as pd

from mne.io import RawArray, read_raw_edf


def CHBMIT_Channels(sub = 'chb01' ):
    '''
    this function designs for CHB-MIT channel selection. Not fully Implemented.
    '''
    assert type(sub) is str, Exception('invalid input')
    chs = []
    # , exclude_chs = [], []

    if sub in ['chb13','chb16']:        #   17
        chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3',
                    u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4',
                    u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8',
                    u'FZ-CZ', u'CZ-PZ']
    elif sub in ['chb04']:              #   20
        chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3',
                    u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4',
                    u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'P8-O2',
                    u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9', u'FT10-T8']
    elif sub in ['chb09']:              #   21
        chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3',
                    u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4',
                    u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'P8-O2',
                    u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9', u'FT9-FT10', u'FT10-T8']
    else:                               #   22
        chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3',
                    u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4',
                    u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8',
                    u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9',
                    u'FT9-FT10', u'FT10-T8']
    # if sub in ['chb04','chb09']:
    exclude_chs = [u'T8-P8']

    chs = list( set(chs).difference(set(exclude_chs) ) )
    chs.sort()
    return chs, exclude_chs



def EEGnormative(patient_dir = './chb01'):
    pass

'''
TODO. ylt: please use the following codes to inplement this func

import os, glob, datetime, scipy.io, stft, re
import numpy as np, pandas as pd
from mne.io import RawArray, read_raw_edf

raw_dir = '/mnt/data/share_data/CHB-MIT/physionet.org/files/chbmit/1.0.0'
sub_dir = os.path.join( raw_dir, 'chb24' )



edf_list = glob.glob( os.path.join(sub_dir, '*.edf') )
edf_list.sort()
exclude_chs = [u'T8-P8']
include_chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3',
                    u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4',
                    u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8',
                    u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9',
                    u'FT9-FT10', u'FT10-T8']

for ele in edf_list[:1]:
    rawEEG = read_raw_edf(ele, exclude = exclude_chs, verbose = 0, preload=True)
    rawEEG.pick_channels(include_chs)
    print(type(rawEEG.info) )
    print( rawEEG.info['meas_date'] )
#     rawEEG.plot_sensors()

'''




'''
        ch_header   = 'number,name'
        edf_header  = 'edfs,log_st,log_sp,seiz'
        seiz_header = 'edfs,seiz_number,seiz_st,seiz_sp'

'''













