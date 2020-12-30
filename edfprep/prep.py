'''
preprocessing utils. implemented projects:
    CHB-MIT     √
    EPILESIAE
    kaggle2014
    FB
'''

import os, glob, datetime, scipy.io, stft, re
import numpy as np, pandas as pd

from mne.io import RawArray, read_raw_edf
# from mne.channels import read_montage
# from mne import create_info, concatenate_raws, pick_types
# from mne.filter import notch_filter

from edfprep.toolbox import check_dir, pkl_load, pkl_save
from edfprep.toolbox import logger, strcv, DaytimeDelta
from edfprep.customer import CHBMIT_Channels




def EEG_Channel_pick(filename, exclude_chs, include_chs):
    '''
    read raw and pick certain channels
    
    
    suggest future re-define, since this only returns numpy data
    '''
    rawEEG = read_raw_edf(filename,
                          exclude = exclude_chs,
                          verbose = 0, preload=True)

    rawEEG.pick_channels(ch_names = include_chs, ordered = True)
    return rawEEG.to_data_frame().to_numpy()

class CHBMITLogCsv(object):
    '''
    this class is designed to deal with ONLY ONE SUBJECT at one time.

    designed for CHBMIT dataset.
    later we shall modify this script into general version for any dataset
    
    please note that, para::location is your stft path
    '''

    def __init__(self, raw_dir, patient,
                file_pattern = '{}/{}-summary.txt',
                settings     = None):
        ### for CHB-MIT, patient shall be like chb.., . be of digits 0-9
        self.summary_file       = file_pattern.format( patient, patient )
        self.summary_file       = os.path.join( raw_dir, self.summary_file )

        sub_folder      = os.path.join( settings.stft_path, patient )
        check_dir(sub_folder)
        sub_csvfolder   = settings.stft_sub_csv_template.format(patient)
        check_dir(sub_csvfolder)
        self.location   = sub_csvfolder
        self.genLogCsv()

    def genLogCsv(self, basic   = 'basic.txt',
                channel_template= 'channel{}.csv',
                edfsrec_template= 'edfsrec{}.csv',
                seizrec_template= 'seizrec{}.csv'):
        '''
        be sure your location is one valid path
        '''
        location = self.location
        check_dir( location )

        basic_txt               = os.path.join( location, basic )
        channel_csv_template    = os.path.join( location, channel_template )
        edfsrec_csv_template    = os.path.join( location, edfsrec_template )
        seizrec_csv_template    = os.path.join( location, seizrec_template )

        ch_header   = 'number,name'
        edf_header  = 'edfs,log_st,log_sp,seiz'
        seiz_header = 'edfs,seiz_number,seiz_st,seiz_sp'

        dicts = self.SummaryParser(logfile = self.summary_file )

        info = 'Data Sampling Rate Hz:\n{}'.format(dicts[0]['fs_rate'])
        logger(fil = basic_txt, cont = info)

        N = len(dicts)
        for cou in range(N):
            CH  = dicts[cou]['channels']
            EDF = dicts[cou]['edf']
            Nch = len(CH)

            logger( fil = channel_csv_template.format(cou), cont = ch_header    )
            logger( fil = edfsrec_csv_template.format(cou), cont = edf_header   )
            logger( fil = seizrec_csv_template.format(cou), cont = seiz_header  )

            for chan in ['Channel {}'.format(ele+1) for ele in range(Nch)]:
                info = '{},{}'.format(chan, CH[chan])
                logger( fil = channel_csv_template.format(cou), cont = info )
            
            edf_list = list(EDF.keys())
            edf_list.sort()
            for edfs in edf_list:
                pairs = EDF[ edfs ]
                info = '{},{},{},{}'.format( edfs, pairs['log_start'],
                                                pairs['log_stop'], pairs['seiz_num'] )
                logger(fil = edfsrec_csv_template.format(cou), cont = info)
                for cou3 in range(pairs['seiz_num'] ):
                    info = '{},{},{},{}'.format( edfs, cou3+1,
                                                pairs[cou3][0], pairs[cou3][1] )
                    logger(fil = seizrec_csv_template.format(cou), cont = info)


    def SummaryParser(self, logfile = './chb01-summary.txt'):
        '''
        this func read and resolve information from summary txt file
        the template for summary txt file can see CHB-MIT's summary file.


        initially designed for CHBMIT dataset
        '''

        #   this func only deals with file named like ./chb..-summary.txt
        res = []
        fs_rate = None
        with open(logfile, 'r') as f:
            line = f.readline()

            ##  first, find sampling rate
            while line:
                tmp = re.findall(r' \d{3} Hz', line)
                if tmp != []:
                    try:
                        fs_rate = int( re.findall(r'\d{3}', tmp[0])[0])
                    except:
                        raise Exception('critical info: sampling rate, invalid value:{}'.format(tmp) )
                    line = f.readline()
                    break
                line = f.readline()
            if fs_rate is None:
                raise Exception('critical info: sampling rate, missing')

            ##  second, parse all pairs defined as (channels, edf-file):
            #   some sub has more than one sampling-channel designs
            while line:
                res.append( {
                    'fs_rate': fs_rate,
                    'channels': {},
                    'edf':{}
                })
                pairs = res[-1]

                while line:
                    CH  = pairs['channels']
                    EDF = pairs['edf']

                    while line:
                        if re.findall(r'Channels in EDF Files:', line) != []:
                            break
                        if re.findall(r'Channels changed:', line) != []:
                            break
                        line = f.readline()
                    while line:
                        if re.findall(r'File Name:', line) != []:
                            break
                        if re.findall(r'Channel \d{1,2}:', line) != []:
                            tmp = line.strip().split(':')
                            tmp[1] = tmp[1].strip()
                            CH[ tmp[0] ] = tmp[1]
                        line = f.readline()

                    ##  third, record whether seizure exists and
                    #   if exists, return Seizure start-time and end-time

                    while line:
                        if re.findall(r'File Name:', line) != []:
                            fileeg = line.strip().split(':')[-1].strip(' ')
                            EDF[ fileeg ] = {
                                'log_st'    : '',
                                'log_sp'    : '',
                                'seiz_num'  : 0
                            }
                            cur_file = EDF[ fileeg ]

                            line = f.readline()
                            line = re.findall(r'\d+:\d+:\d+', line.strip())
                            cur_file['log_st'] = line[0]

                            line = f.readline()
                            line = re.findall(r'\d+:\d+:\d+', line.strip())
                            cur_file['log_sp'] = line[0]

                            line = f.readline()
                            line = re.findall(r'File: \d+', line.strip())
                            try:
                                seiz_num = int(line[0].split(': ')[-1].strip())
                            except:
                                raise Exception(line)
                            else:
                                cur_file['seiz_num'] = seiz_num

                            line = f.readline()
                            if cur_file['seiz_num'] > 0:
                                for j in range(cur_file['seiz_num']):
                                    seiz_start  = re.findall(r'\d+ second', line.strip() )
                                    seiz_start  = int( seiz_start[0].split(' ')[0] )
                                    line        = f.readline()
                                    seiz_stop   = re.findall(r'\d+ second', line.strip() )
                                    seiz_stop   = int( seiz_stop[0].split(' ')[0] )
                                    line        = f.readline()

                                    cur_file[j] = [seiz_start, seiz_stop]
                        if re.findall(r'Channels changed:', line) != []:
                            break
                    if re.findall(r'Channels changed:', line) != []:
                        break
        return res











def load_signals_CHBMIT(settings, window_len = 30, log_file = None, fs_rate = 256):
    '''
    window: x minutes long during each time's prediction
    '''
    sample_freq = fs_rate
    seconds     = 60
    
    seiz_num    = settings['seiz_num']
    edfs_len    = settings['edfs_len']
    edfs        = settings['EDF']
    sub         = settings['sub']
    seiz_rec     = settings['seiz_rec']
    sph         = settings['sph']
    sop         = settings['sop']
    raw_path    = settings['raw_path']
    window_len  = settings['window_len' ]

    raw_dir         = os.path.join( raw_path, sub)
    include_chs, exclude_chs = CHBMIT_Channels(sub)
    time_unit       = window_len * seconds
    
    avg_trial_len   = sum( edfs_len ) / len(edfs_len)
    
    last_state      = None
    prev_sp         = - avg_trial_len
    for j, filename in enumerate(edfs):
        cur_file = os.path.join(raw_dir, filename)
        
        labelictal = None
        if seiz_num[j] > 0:
            labelictal = 1

            window_num  = seiz_num[j]
            st_schedule = list(seiz_rec[2][ seiz_rec[0]==filename ])
            sp_schedule = list(seiz_rec[3][ seiz_rec[0]==filename ])
        else:
            labelictal = 0

            window_num = (edfs_len[j] // time_unit ) - 1
            window_num = max(1, window_num)
            if j < len(edfs) - 1 and seiz_num[j+1] > 0:
                window_num -= 1
            
            st_schedule = [ time_unit * (l+1) for l in range(window_num) ]
            sp_schedule = [ time_unit * (l+1) for l in range(window_num) ]
        
        
        #   find its prev file, because the serial number growth keeps in accordance with time flows
        if filename == 'chb02_16+.edf':
            prevfile    = 'chb02_16.edf'
        elif filename == 'chb02_17.edf':
            prevfile    = 'chb02_16+.edf'
        else:
            seq         = int( filename.split('_')[-1][0:-4])
            prevfile    = '_'.join( [sub, strcv(seq-1) + '.edf'] )
        
        if j>0 and prev_sp > 0:
            prev_sp     -= edfs_len[j-1]
        prev_path   = os.path.join(raw_dir, prevfile)
        
        info = ''
        logger( fil = log_file, cont = info )
        info = '\t reading raw edf: sub {}, {}th trial file: {}, window-num: {}'.format( sub, j, filename, window_num)
        logger( fil = log_file, cont = info )
        info = '\t show edfs_len: {}'.format( edfs_len[j] )
        logger( fil = log_file, cont = info )
        info = '\t show st_schedule: {}, sp_schedule: {}'.format( st_schedule, sp_schedule )
        logger( fil = log_file, cont = info )
        info = '\t show ictallabel: {}'.format( labelictal )
        logger( fil = log_file, cont = info )
        info = '\t show prev_sp: {}'.format( prev_sp )
        logger( fil = log_file, cont = info )

        for k in range( window_num ):
            info = '\t\t *progress on No.{}/{} window'.format( k+1, window_num )
            logger( fil = log_file, cont = info )
            
            st      = st_schedule[k]
            sp      = sp_schedule[k]
            if labelictal == 1:
                sp  = max(sp, st_schedule[k] + sop * seconds)

            # FIND PREV FILE: take care of some special filenames
            tmp_st = st - time_unit - sph * seconds
            
            info = '\t\t show tmp_st: {}'.format( tmp_st )
            logger( fil = log_file, cont = info )
            
            if tmp_st >= prev_sp and labelictal == 0:
                if tmp_st >= 0:
                    cur_data = EEG_Channel_pick( cur_file, exclude_chs, include_chs)
                    data = cur_data[tmp_st*sample_freq : st*sample_freq]
                    logger( fil = log_file, cont = '\t\t\t send {}'.format(labelictal) )
                    yield data, labelictal
                else:
                    if os.path.exists( prev_path ) and prev_sp < 0:
                        cur_data    = EEG_Channel_pick( cur_file, exclude_chs, include_chs)
                        prevtmp     = EEG_Channel_pick(prev_path, exclude_chs, include_chs)
                        data = np.concatenate((prevtmp[tmp_st*sample_freq:], cur_data[:st*sample_freq]), axis = 0)
                        logger( fil = log_file, cont = '\t\t\t send {}'.format(labelictal) )
                        yield data, labelictal
            if labelictal == 1:
                if last_state == 1:
                    tmp_st = max( tmp_st, prev_sp )
                logger( fil = log_file, cont = '\t\t sending {}'.format(labelictal) )
                logger( fil = log_file, cont = '\t\t\t show tmp_st {}'.format(tmp_st) )
                logger( fil = log_file, cont = '\t\t\t show prev_sp {}'.format(prev_sp) )
                logger( fil = log_file, cont = '\t\t\t show prev_file {}'.format(prev_path) )
                
                if tmp_st >= 0:
                    cur_data = EEG_Channel_pick( cur_file, exclude_chs, include_chs)
                    data = cur_data[tmp_st*sample_freq : st*sample_freq]
                    logger( fil = log_file, cont = '\t\t\t send {}'.format(labelictal) )
                    yield data, labelictal
                else:
                    if os.path.exists( prev_path ):
                        cur_data    = EEG_Channel_pick( cur_file, exclude_chs, include_chs)
                        prevtmp     = EEG_Channel_pick(prev_path, exclude_chs, include_chs)
                        data = np.concatenate((prevtmp[tmp_st*sample_freq:], cur_data[:st*sample_freq]), axis = 0)
                        logger( fil = log_file, cont = '\t\t\t send {}'.format(labelictal) )
                        yield data, labelictal
            prev_sp = sp
        last_state = labelictal


class LoadEEGSignals():
    '''
    TODO, read, prepro and save, those 3 funcs shall be of private methods
    to avoid aliasing and effects of power line, 1-128 Hz are selected
    and 60-3 to 60+3, 120-3 to 120+3 are deleted to elinimate power line Hz
    so 128 - 7 - 7 = 114 and
    later this num will reduce to 114 - 2 = 112, 127 and 128 are discarded
    '''
    def __init__(self, settings = {1:1}):
        self.settings   = settings
        self.counter    = 0
        self.log        = settings['MLog_file']
        self.ictal      = 0
        self.inter      = 0
        self.dataset    = settings['dataset']
    
    def __read_raw_signal(self):
        if self.dataset == 'CHBMIT':
            self.target_freq = 256
            return self.load_signals_CHBMIT(settings = self.settings, log_file = self.log)
        else:
            raise NotImplementedError

    def load_signals_CHBMIT(self, settings, window_len = 30, log_file = None, fs_rate = 256):
        '''
        window: x minutes long during each time's prediction
        '''
        sample_freq = fs_rate
        seconds     = 60

        sub         = settings['sub']
        sph         = settings['sph']
        sop         = settings['sop']
        raw_path    = settings['raw_path']
        window_len  = settings['window_len' ]
        
        edfs_rec    = settings['EDF']
        seiz_rec    = settings['seiz_rec']
        

        raw_dir     = os.path.join( raw_path, sub)
        include_chs, exclude_chs = CHBMIT_Channels(sub)
        time_unit   = window_len * seconds

        seiz_dict   = {}
        for ele in range(len(seiz_rec)):
            keys = seiz_rec.iloc[ele]['edfs']
            if keys not in seiz_dict:
                seiz_dict[ keys ] = { 'seiz_st':[], 'seiz_sp':[] }
            seiz_dict[keys]['seiz_st'].append( seiz_rec.iloc[ele]['seiz_st'] )
            seiz_dict[keys]['seiz_sp'].append( seiz_rec.iloc[ele]['seiz_sp'] )
        seiz_num    = list(edfs['seiz'])
        edfs_len    = [ DaytimeDelta( edfs_rec['log_sp'][j], EDF['log_st'][j] ) for j in range( len(edfs_rec)) ]
        
        last_state  = None
        prev_sp     = - sum( edfs_len ) / len(edfs_len)
        local_sp    = - sum( edfs_len ) / len(edfs_len)
        
        
        
        for j, filename in enumerate(edfs):
            
            labelictal = None
            if seiz_num[j] > 0:
                labelictal = 1

                window_num  = seiz_num[j]
                st_schedule = seiz_dict[filename]['seiz_st']
                sp_schedule = seiz_dict[filename]['seiz_st']
            else:
                labelictal = 0

                window_num  = (edfs_len[j] // time_unit )
                window_num  = max(1, window_num)
                nextfile    = self.find_next_file(cur_file = filename)
                if j < len(edfs) - 1 and os.path.exists(nextfile):
                    if seiz_dict.get(nextfile, {'seiz_st':[1e6]})['seiz_st'][0] < sph * seconds + window_len:
                        window_num -= 1

                st_schedule = [ time_unit * (l+1) for l in range(window_num) ]
                sp_schedule = [ time_unit * (l+1) for l in range(window_num) ]


            # info = ''
            # logger( fil = log_file, cont = info )
            # info = '\t reading raw edf: sub {}, {}th trial file: {}, window-num: {}'.format( sub, j, filename, window_num)
            # logger( fil = log_file, cont = info )
            # info = '\t show edfs_len: {}'.format( edfs_len[j] )
            # logger( fil = log_file, cont = info )
            # info = '\t show st_schedule: {}, sp_schedule: {}'.format( st_schedule, sp_schedule )
            # logger( fil = log_file, cont = info )
            # info = '\t show ictallabel: {}'.format( labelictal )
            # logger( fil = log_file, cont = info )
            # info = '\t show prev_sp: {}'.format( prev_sp )
            # logger( fil = log_file, cont = info )

            prevfile = self.find_prev_file(cur_file = filename)
            if j>0 and prev_sp > 0:
                prev_sp     -= edfs_len[j-1]
            prev_path   = os.path.join(raw_dir, prevfile)


            cur_file = os.path.join(raw_dir, filename)
            for k in range( window_num ):
                # info = '\t\t *progress on No.{}/{} window'.format( k+1, window_num )
                # logger( fil = log_file, cont = info )

                st   = st_schedule[k]
                sp   = sp_schedule[k]
                if labelictal == 1:
                    sp = max(sp, st_schedule[k] + sop * seconds)
                tmp_st = st - time_unit - sph * seconds

                # info = '\t\t show tmp_st: {}'.format( tmp_st )
                # logger( fil = log_file, cont = info )

                if tmp_st >= prev_sp and labelictal == 0:
                    if tmp_st >= 0:
                        cur_data = EEG_Channel_pick( cur_file, exclude_chs, include_chs)
                        data = cur_data[tmp_st*sample_freq : st*sample_freq]
                        logger( fil = log_file, cont = '\t\t\t send {}'.format(labelictal) )
                        yield data, labelictal
                    else:
                        if os.path.exists( prev_path ) and prev_sp < 0:
                            cur_data    = EEG_Channel_pick( cur_file, exclude_chs, include_chs)
                            prevtmp     = EEG_Channel_pick(prev_path, exclude_chs, include_chs)
                            data = np.concatenate((prevtmp[tmp_st*sample_freq:], cur_data[:st*sample_freq]), axis = 0)
                            logger( fil = log_file, cont = '\t\t\t send {}'.format(labelictal) )
                            yield data, labelictal
                if labelictal == 1:
                    if last_state == 1:
                        tmp_st = max( tmp_st, prev_sp )
                        #   这条逻辑是时间尽可能取得长，对于正样本
                        #   这个是否合理？我不认为这是对的，早期EEG可能是负样本信号
                        #   13972P7, Roger Kun. 20:46
                    # logger( fil = log_file, cont = '\t\t sending {}'.format(labelictal) )
                    # logger( fil = log_file, cont = '\t\t\t show tmp_st {}'.format(tmp_st) )
                    # logger( fil = log_file, cont = '\t\t\t show prev_sp {}'.format(prev_sp) )
                    # logger( fil = log_file, cont = '\t\t\t show prev_file {}'.format(prev_path) )

                    if tmp_st >= 0:
                        cur_data = EEG_Channel_pick( cur_file, exclude_chs, include_chs)
                        data = cur_data[tmp_st*sample_freq : st*sample_freq]
                        logger( fil = log_file, cont = '\t\t\t send {}'.format(labelictal) )
                        yield data, labelictal
                    else:
                        if os.path.exists( prev_path ):
                            cur_data    = EEG_Channel_pick( cur_file, exclude_chs, include_chs)
                            prevtmp     = EEG_Channel_pick(prev_path, exclude_chs, include_chs)
                            data = np.concatenate((prevtmp[tmp_st*sample_freq:], cur_data[:st*sample_freq]), axis = 0)
                            logger( fil = log_file, cont = '\t\t\t send {}'.format(labelictal) )
                            yield data, labelictal
                prev_sp = sp
            last_state = labelictal


    def find_prev_file(self, cur_file = ''):
        prevfile = 'felt,ylt'
        
        if self.dataset == 'CHBMIT':
            #   find its prev file, because the serial number growth keeps in accordance with time flows
            if cur_file == 'chb02_16+.edf':
                prevfile    = 'chb02_16.edf'
            elif cur_file == 'chb02_17.edf':
                prevfile    = 'chb02_16+.edf'
            else:
                seq         = int( cur_file.split('_')[-1][0:-4])
                prevfile    = '_'.join( [sub, strcv(seq-1) + '.edf'] )

        return prevfile


    def find_next_file(self, cur_file = ''):
        nextfile = 'felt,ylt'
        
        if self.dataset == 'CHBMIT':
            #   find its prev file, because the serial number growth keeps in accordance with time flows
            if cur_file == 'chb02_16+.edf':
                nextfile    = 'chb02_17.edf'
            elif cur_file == 'chb02_16.edf':
                nextfile    = 'chb02_16+.edf'
            else:
                seq         = int( cur_file.split('_')[-1][0:-4])
                nextfile    = '_'.join( [sub, strcv(seq+1) + '.edf'] )

        return nextfile



    def __preprocess(self, data_raw):

        targetFrequency = self.target_freq  # re-sample to target frequency
        slice_time      = 30                #   in seconds
        stft_path       = self.settings['stft_path']
        sub             = self.settings['sub']
        block_name      = os.path.join( stft_path, sub, '_'.join(['{}']*6) )
        #   we will split into slices with length 30 seconds
        collects        = []
        ictal_ovl_pt    = 0.5
        ictal_ovl_len   = int( targetFrequency * slice_time * ictal_ovl_pt )
        window_len      = int( targetFrequency * slice_time )

        for data_X, y_value in data_raw:
            if y_value == 1 or ( y_value == 0 and self.inter<20 ):
                self.counter+= 1
                ictal       = y_value == 1
                interictal  = y_value == 0
                assert ictal or interictal, NotImplementedError
                self.inter  += 1 - y_value
                X_temp = []
                totalSample = data_X.shape[0]//window_len
                info = '\t\t total sample groups : {}'.format(totalSample)
                logger( fil = self.log, cont = info )
                info = '\t\t edf shape : {}'.format(data_X.shape)
                logger( fil = self.log, cont = info )
                info = '\t\t len data : {}'.format( len(data_X) )
                logger( fil = self.log, cont = info )
                for i in range(totalSample):
                    #   window_len = 30*256
                    stft_data = stft.spectrogram(
                        data = data_X[i*window_len:(i+1)*window_len,:],
                        framelength = targetFrequency,
                        centered = False,
                        halved   = True
                        )
                    stft_data = np.transpose(stft_data, (2,1,0) )
                    logger(fil = self.log, cont = '\t\t\t shape stft: {}'.format(stft_data.shape) )
                    logger(fil = self.log, cont = '\t\t\t stft head: {}'.format(stft_data[:2, :2, :2]) )
                    if self.settings['dataset'] == 'CHBMIT':
                        stft_data = np.concatenate(
                            (stft_data[:,:,1:57], stft_data[:,:,64:117], stft_data[:,:,124:127]),
                            axis=-1)
                        #   127 = 112 - (57 - 1 ) - (117 - 64 ) + 124
                        stft_data = stft_data[np.newaxis, :,:,:]
                        # stft_data = stft_data[np.newaxis, :,:56,:112]
                    else:
                        raise NotImplementedError
                    # stft_data = np.abs(stft_data) + 1e-6
                    # stft_data = np.log10(stft_data)
                    # stft_data[np.where(stft_data <= 0)] = 0
                    #   whether this step makes sense? negative shall be kept
                    X_temp.append(stft_data)


                #overdsampling are deleted
                if X_temp != []:
                    X_temp = np.concatenate(X_temp, axis=0)
                    # block_name = os.path.join( stft_path, sub, '_'.join(['{}']*6) )
                    filname = block_name.format( self.counter, *X_temp.shape, y_value )
                    pkl_save( fil = filname, sub = X_temp )
                    collects.append( filname )
        
        return collects







    def apply(self):
        data    = self.__read_raw_signal()
        collects= self.__preprocess(data)
        return self.counter, collects
























########################################################################
####    ****    Deprecated or To Be Implemented ****    ####
########################################################################

'''
# class LoadEEGSignals():
#     def __init__(self, target, label, sph = 5, sop = 2):
#         assert label in ['ictal', 'interictal'], NotImplementedError
        
#         self.patient = target
#         self.label = label
#         self.sph = sph
#         self.sop = sop


#         self.global_proj = np.array( [0.0]*114 )
#         # self.significant_channels = None

#     ### TODO, read, prepro and save, those 3 funcs shall be of private methods
#     def read_raw_signal(self):
#         if self.settings['dataset'] == 'CHBMIT':
#             self.samp_freq = 256
#             self.freq = 256
#             self.global_proj = np.array( [0.0]*114 )
#             #   to avoid aliasing and effects of power line, 1-128 Hz are selected
#             #   and 60-3 to 60+3, 120-3 to 120+3 are deleted to elinimate power line Hz
#             #   so 128 - 7 - 7 = 114
#             # from utils.CHBMIT_channels import channels
#             # try:
#             #     self.significant_channels = channels[self.target]
#             # except:
#             #     pass
#             return load_signals_CHBMIT(self.settings['datadir'],
#                                         self.patient,
#                                         self.label,
#                                         self.sph)
#         else:
#             raise NotImplementedError
#         # elif self.settings['dataset'] == 'FB':
#         #     self.samp_freq = 256
#         #     self.freq = 256
#         #     self.global_proj = np.array([0.0]*114)
#         #     return load_signals_FB(self.settings['datadir'], self.target, self.type, self.sph)
#         # elif self.settings['dataset'] == 'Kaggle2014Pred':
#         #     if self.type == 'ictal':
#         #         data_type = 'preictal'
#         #     else:
#         #         data_type = self.type
#         #     from utils.Kaggle2014Pred_channels import channels
#         #     try:
#         #         self.significant_channels = channels[self.target]
#         #     except:
#         #         pass
#         #     print (self.target,self.significant_channels)
#         #     return load_signals_Kaggle2014Pred(self.settings['datadir'], self.target, data_type)
#         # elif self.settings['dataset'] == 'EpilepsiaSurf':
#         #     self.samp_freq = 256
#         #     self.freq = 256
#         #     self.global_proj = np.array([0.0] * 128)
#         #     return  load_signals_EpilepsiaSurf(self.settings['datadir'], self.target, self.type, self.sph)

#     def preprocess(self, data_):
#         ictal = self.label == 'ictal'
#         interictal = self.label == 'interictal'
#         assert ictal or interictal, NotImplementedError

#         targetFrequency = self.freq  # re-sample to target frequency
#         window_time = 30
        
#         df_sampling = pd.read_csv(
#             'sampling_%s.csv' % self.settings['dataset'],
#             header=0,index_col=None)
#         trg = int(self.target)
#         print (df_sampling)
#         print (df_sampling[df_sampling.Subject==trg].ictal_ovl.values)
#         ictal_ovl_pt = \
#             df_sampling[df_sampling.Subject==trg].ictal_ovl.values[0]
#         ictal_ovl_len = int(targetFrequency*ictal_ovl_pt*window_time)

#         def process_raw_data(mat_data):            
#             print ('Loading data')
#             X = []
#             y = []
#             #scale_ = scale_coef[target]
#             for data in mat_data:
#                 if self.settings['dataset'] == 'FB':
#                     data = data.transpose()
#                 if self.significant_channels is not None:
#                     print ('Reducing number of channels')
#                     data = data[:,self.significant_channels]
#                 if ictal:
#                     y_value=1
#                 else:
#                     y_value=0

#                 X_temp = []
#                 y_temp = []
    
#                 totalSample = int(data.shape[0]/targetFrequency/window_time) + 1
#                 window_len = int(targetFrequency*window_time)
#                 for i in range(totalSample):
#                     if (i+1)*window_len <= data.shape[0]:
#                         s = data[i*window_len:(i+1)*window_len,:]

#                         stft_data = stft.spectrogram(s,framelength=targetFrequency,centered=False)
#                         stft_data = np.transpose(stft_data,(2,1,0))
#                         stft_data = np.abs(stft_data)+1e-6

#                         if self.settings['dataset'] == 'FB':
#                             stft_data = np.concatenate((stft_data[:,:,1:47],
#                                                         stft_data[:,:,54:97],
#                                                         stft_data[:,:,104:]),
#                                                        axis=-1)
#                         elif self.settings['dataset'] == 'CHBMIT':
#                             stft_data = np.concatenate((stft_data[:,:,1:57],
#                                                         stft_data[:,:,64:117],
#                                                         stft_data[:,:,124:]),
#                                                        axis=-1)
#                         elif self.settings['dataset'] == 'EpilepsiaSurf':
#                             stft_data = stft_data[:,:,1:]
#                         stft_data = np.log10(stft_data)
#                         indices = np.where(stft_data <= 0)
#                         stft_data[indices] = 0                      

#                         if self.settings['dataset'] in ['FB', 'CHBMIT']:
#                             stft_data = stft_data[:,:56,:112]
#                         elif self.settings['dataset'] == 'EpilepsiaSurf':
#                             stft_data = stft_data[:,:56,:]
#                         stft_data = stft_data.reshape(-1, stft_data.shape[0],
#                                                       stft_data.shape[1],
#                                                       stft_data.shape[2])


#                         X_temp.append(stft_data)
#                         y_temp.append(y_value)

#                 #overdsampling
#                 if ictal:
#                     i = 1
#                     print ('ictal_ovl_len =', ictal_ovl_len)
#                     while (window_len + (i + 1)*ictal_ovl_len <= data.shape[0]):
#                         s = data[i*ictal_ovl_len:i*ictal_ovl_len + window_len, :]

#                         stft_data = stft.spectrogram(s, framelength=targetFrequency,centered=False)
#                         stft_data = np.transpose(stft_data, (2, 1, 0))
#                         stft_data = np.abs(stft_data)+1e-6

#                         if self.settings['dataset'] == 'FB':
#                             stft_data = np.concatenate((stft_data[:,:,1:47],
#                                                         stft_data[:,:,54:97],
#                                                         stft_data[:,:,104:]),
#                                                        axis=-1)
#                         elif self.settings['dataset'] == 'CHBMIT':
#                             stft_data = np.concatenate((stft_data[:,:,1:57],
#                                                         stft_data[:,:,64:117],
#                                                         stft_data[:,:,124:]),
#                                                        axis=-1)
#                         elif self.settings['dataset'] == 'EpilepsiaSurf':
#                             stft_data = stft_data[:, :, 1:]
#                         stft_data = np.log10(stft_data)
#                         indices = np.where(stft_data <= 0)
#                         stft_data[indices] = 0

#                         proj = np.sum(stft_data,axis=(0,1),keepdims=False)
#                         self.global_proj += proj/1000.0

#                         #stft_data = np.multiply(stft_data,1.0/scale_)

#                         if self.settings['dataset'] in ['FB', 'CHBMIT']:
#                             stft_data = stft_data[:,:56,:112]
#                         elif self.settings['dataset'] == 'EpilepsiaSurf':
#                             stft_data = stft_data[:,:56,:]
#                         stft_data = stft_data.reshape(-1, stft_data.shape[0],
#                                                       stft_data.shape[1],
#                                                       stft_data.shape[2])

#                         X_temp.append(stft_data)
#                         # differentiate between non-overlapped and overlapped
#                         # samples. Testing only uses non-overlapped ones.
#                         y_temp.append(2)
#                         i += 1

#                 if len(X_temp)>0:
#                     X_temp = np.concatenate(X_temp, axis=0)
#                     y_temp = np.array(y_temp)
#                     X.append(X_temp)
#                     y.append(y_temp)

#             if ictal or interictal:
#                 #y = np.array(y)
#                 try:
#                     print ('X', len(X), X[0].shape, 'y', len(y), y[0].shape)
#                 except:
#                     print ('!!!!!!!!!!!!DEBUG!!!!!!!!!!!!!:', X)
#                 return X, y
#             else:
#                 print ('X', X.shape)
#                 return X

#         data = process_raw_data(data_)

#         return data

#     def save_STFT_to_files(self, X, over_spl):
#         pre = None
#         # oversampling for GAN training
#         ovl_pct = 0.1
#         # oversampling for GAN training
#         if isinstance(X, list):
#             index=0
#             ovl_len = int(ovl_pct*X[0].shape[-2]) # oversampling for GAN training
#             ovl_num = int(np.floor(1.0/ovl_pct) - 1) # oversampling for GAN training
#             for x in X:
#                 for i in range(x.shape[0]):
#                     fn = '%s_%s_%d_%d.npy' % (self.target,self.type,index,i)
#                     if self.settings['dataset'] in ['FB','CHBMIT']:
#                         x_ = x[i,:,:56,:112]
#                     elif self.settings['dataset'] == 'Kaggle2014Pred':
#                         if 'Dog' in self.target:
#                             x_ = x[i,:,:56,:96]
#                         elif 'Patient' in self.target:
#                             x_ = x[i,:,:112,:96]
#                     elif self.settings['dataset'] == 'EpilepsiaSurf':
#                         x_ = x[i,:,:,:]
#                     np.save(os.path.join(dir,fn),x_)
#                     # Generate overlapping samples for GAN
#                     if over_spl:
#                         if i>0:
#                             for j in range(1, ovl_num+1):
#                                 fn = '%s_ovl_%s_%d_%d_%d.npy' % (self.target,self.type,index,i,j)
#                                 x_2 = np.concatenate((pre[:,:j*ovl_len,:], x_[:,j*ovl_len:,:]),axis=1)
#                                 assert x_2.shape == x_.shape
#                                 np.save(os.path.join(dir,fn),x_2)
#                         pre = x_
#                 index += 1
#         else:
#             ovl_len = int(ovl_pct*X.shape[-2]) # oversampling for GAN training
#             ovl_num = np.floor(1.0/ovl_pct) - 1 # oversampling for GAN training
#             for i in range(X.shape[0]):
#                 fn = '%s_%s_0_%d.npy' % (self.patient, self.type, i)
#                 if self.settings['dataset'] in ['FB','CHBMIT']:
#                     x_ = X[i,:,:56,:112]
#                 elif self.settings['dataset'] == 'Kaggle2014Pred':
#                     if 'Dog' in self.target:
#                         x_ = X[i,:,:56,:96]
#                     elif 'Patient' in self.target:
#                         x_ = X[i,:,:112,:96]
#                 np.save(os.path.join(dir,fn), x_)
#                 # Generate overlapping samples for GAN
#                 if over_spl:
#                     if i>0:
#                         for j in range(1, ovl_num+1):
#                             fn = '%s_ovl_%s_%d_%d_%d.npy' % (self.target,self.type,index,i,j)
#                             x_2 = np.concatenate((pre[:,:j*ovl_len,:], x_[:,j*ovl_len:,:]),axis=-1)
#                             assert x_2.shape == x_.shape
#                             np.save(os.path.join(dir,fn),x_2)
#                     pre = x_
#         print('Finished saving STFT to %s' % dir)
#         return None


#     def apply(self, csv_dir, npy_dir):
#         data = self.read_raw_signal()

#         if self.settings['dataset']=='Kaggle2014Pred':
#             X, y = self.preprocess_Kaggle(data)
#         else:
#             X, y = self.preprocess(data)
#         pkl_save( filename, [X, y])


#         if save_STFT:
#             return self.save_STFT_to_files(X, over_spl)
#         else:
#             return X, y
'''


########################################################################
####    ****    Deprecated or To Be Implemented ****    ####
########################################################################
'''
def load_signals_EpilepsiaSurf(raw_dir='', target='1', data_type='preictal', sph=5):
    #########################################################################
    def load_raw_data(filename):
        fn = filename + '.data'
        hd = filename + '.head'

        h = pd.read_csv(hd, header=None, index_col=None, sep='=')
        start_ts = h[h[0] == 'start_ts'][1].values[0]
        num_samples = int(h[h[0] == 'num_samples'][1])
        sample_freq = int(h[h[0] == 'sample_freq'][1])
        conversion_factor = float(h[h[0] == 'conversion_factor'][1])
        num_channels = int(h[h[0] == 'num_channels'][1])
        elec_names = h[h[0] == 'elec_names'][1].values[0]
        elec_names = elec_names[1:-1]
        elec_names = elec_names.split(',')
        duration_in_sec = int(h[h[0] == 'duration_in_sec'][1])

        # print ('start_ts', start_ts)
        # print ('num_samples', num_samples)
        # print ('sample_freq', sample_freq)
        # print ('conversion_factor', conversion_factor)
        # print ('num_channels', num_channels)
        # print ('elec_names', elec_names)
        # print ('duration_in_sec', duration_in_sec)

        m = np.fromfile(fn, '<i2')
        m = m * conversion_factor
        m = m.reshape(-1, num_channels)
        assert m.shape[0] == num_samples

        ch_fn = './utils/include_chs.txt'
        with open(ch_fn, 'r') as f:
            include_chs = f.read()
            include_chs = include_chs.split(',')
        ch_ind = np.array([elec_names.index(ch) for ch in include_chs])
        m_s = m[:, ch_ind]
        assert m_s.shape[1] == len(include_chs)
        #print (m.shape)
        return m_s, include_chs, int(sample_freq)
    #########################################################################

    #########################################################################
    # Load all filenames per patient
    all_fn = './utils/epilepsia_recording_blocks.csv'
    all_pd = pd.read_csv(all_fn, header=0, index_col=None)
    pat_pd = all_pd[all_pd['pat']==int(target)]
    #print (pat_pd)
    pat_fd = pat_pd['folder'].values
    pat_fd = list(set(list(pat_fd)))
    assert len(pat_fd)==1
    #print (pat_fd[0])
    pat_adm = os.path.join(raw_dir,pat_fd[0])
    pat_adm = glob.glob(pat_adm + '/adm_*')
    assert len(pat_adm)==1
    #print (pat_adm[0])
    pat_fns = list(pat_pd['filename'].values)
    #########################################################################

    #########################################################################
    # Load seizure info
    all_sz_fn = './utils/epilepsia_seizure_master.csv'
    all_sz_pd = pd.read_csv(all_sz_fn, header=0, index_col=None)
    pat_sz_pd = all_sz_pd[all_sz_pd['pat']==int(target)]
    pat_sz_pd = pat_sz_pd[pat_sz_pd['leading_sz']==1]
    print (pat_sz_pd)


    #########################################################################
    ii=0
    fmt = "%d/%m/%Y %H:%M:%S"

    # exi
    count_interictal = 0
    for i_fn in range(len(pat_fns)):
        pat_fn = pat_fns[i_fn]
        #print (pat_fn)
        rec_fd = pat_fn.split('_')[0]
        rec_fd = 'rec_' + rec_fd
        #print (rec_fd)
        fn = pat_fn.split('.')[0]
        fn = os.path.join(pat_adm[0],rec_fd,fn)
        # print (fn)
        this_fn_pd = pat_pd[pat_pd['filename']==pat_fn]
        gap = list(this_fn_pd['gap'].values)
        assert len(gap)==1
        gap = gap[0]
        # print (gap)

        begin_rec = list(this_fn_pd['begin'].values)
        assert len(begin_rec)==1
        begin_rec = datetime.datetime.strptime(begin_rec[0], fmt)
        # print (begin_rec)

        # m, elec_names, sample_freq = load_raw_data(filename=fn)
        # print (m.shape, sample_freq, elec_names)

        # with open(elec_file, 'a') as f:
        #     f.write('%s, %s \n' %(target,','.join(elec_names)))


        if data_type=='interictal':
            # check if current recording is at least 4 hour away from sz
            flag_sz = False
            dist_to_sz = 0
            ind = 0
            while (dist_to_sz < 4*3600*256) and (ind <= i_fn):
                full_fn = pat_fns[i_fn-ind]
                fn_ = full_fn.split('.')[0]
                fn_pd_ = pat_pd[pat_pd['filename']==full_fn]
                if ind > 0:
                    dist_to_sz = dist_to_sz + int(fn_pd_['samples']) + int(fn_pd_['gap'])*256
                ind += 1
                pd_ = pat_sz_pd[pat_sz_pd['filename']==fn_]
                #print ('!DEBUG:', i_c, pat_fns[i_c], pd_)
                if pd_.shape[0]>0:
                    flag_sz = True
                    break
                print ('DEBUG: 1', dist_to_sz,ind)
            dist_to_sz = 0
            ind = 1
            while (dist_to_sz < 4*3600*256) and (ind <= (len(pat_fns)-i_fn-1)):
                full_fn = pat_fns[i_fn+ind]
                fn_ = full_fn.split('.')[0]
                fn_pd_ = pat_pd[pat_pd['filename']==full_fn]
                dist_to_sz = dist_to_sz + int(fn_pd_['samples']) + int(fn_pd_['gap'])*256
                ind += 1
                pd_ = pat_sz_pd[pat_sz_pd['filename']==fn_]
                #print ('!DEBUG:', i_c, pat_fns[i_c], pd_)
                if pd_.shape[0]>0:
                    flag_sz = True
                    break
                print ('DEBUG: 2', dist_to_sz,ind)

            if not flag_sz:
                m, elec_names, sample_freq = load_raw_data(filename=fn)
                count_interictal += 1
                print (data_type, count_interictal, m.shape, sample_freq, elec_names)
                yield m

            # for i_c in range(max(0,i_fn-4), min(len(pat_fns),i_fn+4)):
            #     fn_ = pat_fns[i_c].split('.')[0]
            #     pd_ = pat_sz_pd[pat_sz_pd['filename']==fn_]
            #     #print ('!DEBUG:', i_c, pat_fns[i_c], pd_)
            #     if pd_.shape[0]>0:
            #         # not > 4 hour away from sz
            #         break
            #     else:
            #         yield m

        elif data_type=='ictal': # actually preictal

            pat_onset_pd = pat_sz_pd[pat_sz_pd['filename']==os.path.basename(fn)]
            onset = list(pat_onset_pd['onset'].values)
            # print (os.path.basename(fn))
            # print ('!!!ONSET', len(onset), onset)
            if len(onset)>0:
                m, elec_names, sample_freq = load_raw_data(filename=fn)
                print (data_type, m.shape, sample_freq, elec_names)

                window_unit = 30 * 60 * sample_freq  # window_unit = 30 min
                window_unit = sph * 60 * sample_freq

                dt = datetime.datetime.strptime(onset[0], fmt)
                print (begin_rec, dt)
                time_to_sz = dt - begin_rec
                time_to_sz = int(np.floor(time_to_sz.total_seconds())) * sample_freq
                print (time_to_sz)
                if time_to_sz >= window_unit + window_unit:
                    #yield data here
                    st = time_to_sz - window_unit - window_unit
                    sp = time_to_sz - window_unit
                    data = m[st:sp]
                    print ('!DATA shape', data.shape)

                else: # concatenate preictal signals from previous recording
                    if time_to_sz > window_unit:
                        n_spls_fr_pre = window_unit + window_unit - time_to_sz

                        pat_fn_pre = pat_fns[i_fn-1]
                        rec_fd_pre = pat_fn_pre.split('_')[0]
                        rec_fd_pre = 'rec_' + rec_fd_pre

                        fn_pre = pat_fn_pre.split('.')[0]
                        fn_pre = os.path.join(pat_adm[0], rec_fd_pre, fn_pre)
                        print (fn_pre)

                        m_pre, _, _ = load_raw_data(filename=fn_pre)

                        data = np.concatenate((m_pre[-n_spls_fr_pre:], m[0:time_to_sz - window_unit]), axis=0)
                        print ('!DATA shape with pre', data.shape)
                    else: # all preictal data extracted from previous recording
                        pat_fn_pre = pat_fns[i_fn-1]
                        rec_fd_pre = pat_fn_pre.split('_')[0]
                        rec_fd_pre = 'rec_' + rec_fd_pre

                        fn_pre = pat_fn_pre.split('.')[0]
                        fn_pre = os.path.join(pat_adm[0], rec_fd_pre, fn_pre)
                        print (fn_pre)

                        m_pre, _, _ = load_raw_data(filename=fn_pre)

                        data = m[m_pre.shape[0]+time_to_sz - window_unit - window_unit:m_pre.shape[0]+time_to_sz - window_unit]
                        print ('!DATA shape with pre', data.shape)

                yield data



    #return None

def load_signals_Kaggle2014Pred(raw_dir, target, data_type):
    print ('load_signals_Kaggle2014Pred for Patient', target)

    dir = os.path.join(raw_dir, target)
    done = False
    i = 0
    while not done:
        i += 1
        if i < 10:
            nstr = '000%d' %i
        elif i < 100:
            nstr = '00%d' %i
        elif i < 1000:
            nstr = '0%d' %i
        else:
            nstr = '%d' %i

        filename = '%s/%s_%s_segment_%s.mat' % (dir, target, data_type, nstr)
        if os.path.exists(filename):
            data = scipy.io.loadmat(filename)
            # discard preictal segments from 66 to 35 min prior to seizure
            if data_type == 'preictal':
                for skey in data.keys():
                    if "_segment_" in skey.lower():
                        mykey = skey
                sequence = data[mykey][0][0][4][0][0]
                if (sequence <= 3):
                    print ('Skipping %s....' %filename)
                    continue
            yield(data)
        else:
            if i == 1:
                raise Exception("file %s not found" % filename)
            done = True

def load_signals_FB(raw_dir, target, data_type, sph):
    print ('load_signals_FB for Patient', target)

    def strcv(i):
        if i < 10:
            return '000' + str(i)
        elif i < 100:
            return '00' + str(i)
        elif i < 1000:
            return '0' + str(i)
        elif i < 10000:
            return str(i) 

    if int(target) < 10:
        strtrg = '00' + str(target)
    elif int(target) < 100:
        strtrg = '0' + str(target)

    if data_type == 'ictal':

        window_unit = 30*60*256
        target_ = 'pat%sIktal' % strtrg
        dir = os.path.join(raw_dir, target_)
        df_sz = pd.read_csv(
            os.path.join(raw_dir,'seizure.csv'),index_col=None,header=0)
        df_sz = df_sz[df_sz.patient==int(target)]
        df_sz.reset_index(inplace=True,drop=True)

        print (df_sz)
        print ('Patient %s has %d seizures' % (target,df_sz.shape[0]))
        for i in range(df_sz.shape[0]):
            data = []
            filename = df_sz.iloc[i]['filename']
            st = df_sz.iloc[i]['start'] - sph*60*256
            print ('Seizure %s starts at %d' % (filename, st))
            for ch in range(1,7):
                filename2 = '%s/%s_%d.asc' % (dir, filename, ch)
                if os.path.exists(filename2):
                    tmp = np.loadtxt(filename2)
                    seq = int(filename[-4:])
                    prevfile = '%s/%s%s_%d.asc' % (dir, filename[:-4], strcv(seq - 1), ch)

                    if st - window_unit >= 0:
                        tmp = tmp[st - window_unit:st]
                    else:
                        prevtmp = np.loadtxt(prevfile)
                        if os.path.exists(prevfile):
                            if st > 0:
                                tmp = np.concatenate((prevtmp[st - window_unit:], tmp[:st]))
                            else:
                                tmp = prevtmp[st - window_unit:st]
                        else:
                            if st > 0:
                                tmp = tmp[:st]
                            else:
                                raise Exception("file %s does not contain useful info" % filename)

                    tmp = tmp.reshape(1, tmp.shape[0])
                    data.append(tmp)

                else:
                    raise Exception("file %s not found" % filename)
            if len(data) > 0:
                concat = np.concatenate(data)
                print (concat.shape)
                yield (concat)

    elif data_type == 'interictal':
        target_ = 'pat%sInteriktal' % strtrg
        dir = os.path.join(raw_dir, target_)
        text_files = [f for f in os.listdir(dir) if f.endswith('.asc')]
        prefixes = [text[:8] for text in text_files]
        prefixes = set(prefixes)
        prefixes = sorted(prefixes)

        totalfiles = len(text_files)
        print (prefixes, totalfiles)

        done = False
        count = 0

        for prefix in prefixes:
            i = 0
            while not done:

                i += 1

                stri = strcv(i)
                data = []
                for ch in range(1, 7):
                    filename = '%s/%s_%s_%d.asc' % (dir, prefix, stri, ch)

                    if os.path.exists(filename):
                        try:                           
                            tmp = np.loadtxt(filename)
                            tmp = tmp.reshape(1, tmp.shape[0])
                            data.append(tmp)
                            count += 1
                        except:
                            print ('OOOPS, this file can not be loaded', filename)
                    elif count >= totalfiles:
                        done = True
                    elif count < totalfiles:
                        break
                    else:
                        raise Exception("file %s not found" % filename)

                if i > 99999:
                    break

                if len(data) > 0:
                    yield (np.concatenate(data))	

'''