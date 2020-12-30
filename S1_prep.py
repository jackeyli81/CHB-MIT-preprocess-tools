import os, time, glob

from edfprep.toolbox import check_dir, pkl_load, pkl_save
from edfprep.toolbox import DaytimeDelta, logger, timestump

from edfprep.prep import LoadEEGSignals, CHBMITLogCsv
from config import LogEEG
import pandas as pd



def main(dataset = 'CHBMIT',
        sph = 5, sop = 2,
        stftstump = ''):
    '''
    to assign target subjects,
    u should modify codes to assign subjects to be processed
    '''
    
    assert dataset in ['kaggle2014', 'EPILEPSIAE', 'CHBMIT', 'FB'], NotImplementedError
    subjects        = []
    stft_configu    = None

    stftstump = timestump(prefix = stftstump)
    
    if dataset == 'CHBMIT':
        subjects += [ 'chb0{}'.format(ele) for ele in range(1, 9+1)]
        subjects += [ 'chb1{}'.format(ele) for ele in [0,1,3] ]
        subjects += [ 'chb2{}'.format(ele) for ele in [0,1,2,3] ]
        #   1-11, 13, 20-23 together 16 people
        subjects = sorted(list( set(subjects) ))

        stft_configu = LogEEG(dataset = dataset,
                         stft_stump = stftstump,
                         subjects = subjects)
        
        ### the configuration file is pickled with filename 'Log{}'
        log_save = os.path.join( stft_configu.stft_path, 'Log{}'.format(dataset) )
        pkl_save(fil = log_save, sub = stft_configu)
    else:
        raise NotImplementedError


    for sub in subjects:

        CHBMITLogCsv(
            raw_dir         = stft_configu.raw_path,
            file_pattern    = stft_configu.summary_template,
            patient         = sub,
            settings        = stft_configu
            )



    ### TODO: complete stft transform
    for sub in subjects:

        sub_folder      = os.path.join( stft_configu.stft_path, sub )
        sub_csvfolder   = stft_configu.stft_sub_csv_template.format(sub)

        csv_list        = glob.glob( os.path.join(sub_csvfolder, '*.csv') )
        N               = len(csv_list) // 3
        #   channel: not null;  edfsrec: not null;  seizrec: null exists
        
        edfsrec_template= os.path.join(sub_csvfolder, 'edfsrec{}.csv')
        seizrec_template= os.path.join(sub_csvfolder, 'seizrec{}.csv')
        edfsrec_file    = edfsrec_template.format(0)
        seizrec_file    = seizrec_template.format(0)
        
        EDF = pd.read_csv(edfsrec_file, delimiter = ',', header = None)
        try:
            SEIZ = pd.read_csv(seizrec_file, delimiter = ',', header = None)
        except:
            SEIZ = None
        for j in range(1, N):
            # CH = pd.concat( [CH, pd.read_csv(channel.format(j), delimiter = ',', header = None)],
                            # ignore_index = True )
            EDF = pd.concat( [EDF, pd.read_csv(edfsrec_template.format(j), delimiter = ',', header = None)],
                            ignore_index = True )
            try:
                tmp = pd.read_csv(seizrec_template.format(j), delimiter = ',', header = None)
            except:
                continue
            else:
                SEIZ = pd.concat( [SEIZ, tmp], ignore_index = True )



        EDF.to_csv( os.path.join(sub_folder, 'worklog_EDF.csv') )
        if SEIZ is not None:
            SEIZ.to_csv( os.path.join(sub_folder, 'worklog_SEIZ.csv') )
        else:
            pd.DataFrame([]).to_csv( os.path.join(sub_folder, 'worklog_SEIZ.csv') )
            # with open( os.path.join(sub_folder, 'worklog_SEIZ.csv'), 'w' ) as fil131:
            #     fil131.write('No seizures')
        # CH = list(CH[1])
        seiz_num        = list(EDF[3])      #   this indicates who are ictal and who are interictal
        edfs_len    = [ DaytimeDelta( EDF[2][j], EDF[1][j] ) for j in range( len(seiz_num)) ]
        EDF             = list(EDF[0])      #   edfs_len and EDF are linked to each other
        log_file        = os.path.join( stft_configu.stft_path, sub, 'Mlog.txt' )
        config_file     = os.path.join( stft_configu.stft_path, sub, 'Clog' )
        
        
        stft_configu.stft_settings.update(
            {
            'seiz_num'  : seiz_num,
            'edfs_len'  : edfs_len,
            'EDF'       : EDF,
            'sub'       : sub,
            'raw_path'  : stft_configu.raw_path,
            'stft_path' : stft_configu.stft_path,
            'seizrec'   : SEIZ,
            'sph'       : sph,
            'sop'       : sop,
            'MLog_file' : log_file
            }
        )

        info = 'start on processing subject: {}'.format(sub)
        logger(fil = log_file, cont = info)

        log         = LoadEEGSignals( settings = stft_configu.stft_settings )
        _, collects = log.apply()

        name_list   = os.path.join( stft_configu.stft_path, 'pklog'+ '_' + sub )
        pkl_save( fil = name_list, sub = collects )
        
        info = 'complete on processing subject: {}'.format(sub)
        logger(fil = log_file, cont = info)
        pkl_save( fil = config_file, sub = stft_configu )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default = 'CHBMIT',
                        help = "FB, CHBMIT, kaggle2014 or EpilepsiaSurf")
    parser.add_argument("--sig", type = str, default = None,
                        help = "signature/biomark")

    parser.add_argument("--sph", type = int, default = 5,
                        help = "0, 5, etc")
    parser.add_argument("--sop", type = int, default = 2,
                        help = "seizure occurence period")
    parser.add_argument("--logonly", default = True, action = 'store_false',
                        help = "signature/biomark")
    parser.add_argument("--new", default = True, action = 'store_false',
                        help = "whether build one new log dir")
    parser.add_argument("--online", default = False, action = 'store_true',
                        help = "whether choose online running")
    args = parser.parse_args()

    if args.online:
        raise NotImplementedError
    else:
        main( dataset = args.dataset,
            stftstump  = args.sig,
            sph       = args.sph,
            sop       = args.sop)







