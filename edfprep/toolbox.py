
### this file works as the basic toolbox


import os, re, time, stft
import pickle as pkl
from datetime import datetime
from mne.io import RawArray, read_raw_edf
import numpy as np, pandas as pd


### Part 1
#   basic tools

def log(cont = 'Hello', dir_high = './'):
    '''
    self-made Logs. default name: MLog.txt
    
    to be substituted by logger in the future. 
    '''
    with open(os.path.join(dir_high, 'MLog.txt'),'a+') as f:
        string = str(datetime.now()) + ': ' + cont + '\r\n'
        f.write(string)

def logger(fil = '~/liyang.txt', cont = 'hello', verbose = True):
    '''
    self-made Logs. u need to assign the Log path and contents.
    
    verbose controls whether display them while writing into disk
    '''
    if os.path.exists(os.path.dirname(fil)):
        if verbose:
            print( cont )
        with open( fil, 'a' ) as fils:
            fils.write( cont + '\n' )

def strcv(x, digits = 2):
    '''
    when u need digits = 3, it turns '1' into '001', '12' into '012'
    
    default digits = 2.
    '''
    assert type(x) is int, Exception('incorrect integer')
    if x < 10:
        return '0' + str(x)
    else:
        return str(x)


def timestump(prefix = ''):
    '''
    one naming manner with timestumps
    '''
    return ( '{}' + ( (time.asctime()).replace(':', '-')).replace(' ', '_') ).format(prefix)

def check_dir(x):
    '''
    if x is not one existed dir, create it (but your father dir shall exist)

    for example, create /ax/bx/ before create /ax/bx/cx/
    '''
    if not os.path.exists(x):
        os.mkdir(x)
        return False
    else:
        return True

def pkl_load(fil = '1'):
    '''
    pickle load everything!
    '''
    with open(fil, 'rb') as filec:
        res = pkl.load(filec)
    return res

def pkl_save(fil = '1', sub = [1,2,3]):
    '''
    pickle save anything!
    '''
    with open(fil, 'wb') as filec:
        pkl.dump(sub, filec)

def DaytimeDelta(a, b):
    '''
    calc time interval length by seconds, assume input is of STR format %H:%M:%S
    '''
    a, b = a.split(':'), b.split(':')
    delta = 0
    for j in range(3):
        delta += ( 60**(2 - j) ) * ( int(a[j]) - int(b[j]) )
    return delta

