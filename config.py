from edfprep.toolbox import check_dir, pkl_load, pkl_save
import os
import numpy as np



class LogEEG(object):
    '''
    this is subject level
    '''

    __raw_path_dict = {
        'CHBMIT'    :   '/mnt/data/share_data/CHB-MIT/physionet.org/files/chbmit/1.0.0',
        'EPILEPSIA' :   None,
        'FB'        :   None,
        'kaggle2014':   None
        }
    __summary_template = {
        'CHBMIT':'{}/{}-summary.txt'
        }
    __settings_dict = {
        'CHBMIT':{
            'dataset'       : 'CHBMIT',
            'channels'      : None,
            'seiz_num'      : None,
            'edfs_len'  : None,
            'EDF'           : None,
            'raw_path_sub'  : None,
            'sub'           : None,
            'seizrec'       : None,
            'sph'           : 5,
            'sop'           : 2,
            'window_len'    : 35
        }
    }
    
    stft_template           = './{}'        #   sigstump
    stft_sub_csv_template   = '{}/CsvLog'   #   csv log, recording labels and time length

    def __init__(self, dataset = '', stft_stump = '', subjects = [], root_dir = './'):
        
        assert check_dir(root_dir), Exception( 'invalid root dir: {}'.format(root_dir) )
        assert dataset in self.__raw_path_dict.keys(), Exception( 'invalid dataset: {}'.format(dataset) )

        self.subjects               = subjects
        self.raw_path               = self.__raw_path_dict[dataset]
        self.summary_template       = self.__summary_template[dataset]
        self.stft_settings          = self.__settings_dict[dataset]
        self.stft_path              = os.path.join( root_dir,\
            self.stft_template.format( '_'.join([dataset, stft_stump]) )
        )
        assert not check_dir(self.stft_path), Exception( 'coincided stft store path: {}'.format(self.stft_path) )
        self.stft_sub_csv_template  = os.path.join(self.stft_path, self.stft_sub_csv_template)




























'''
class ModelConfig(object):
    npy_pattern = '{}/{}'
    ckpt_pattern = '{}/checkpoints'
    run_para = {
        "learning_rate" : 0.001,
        "beta1"         : 0.5,
        'max_epoch'     : 10,
        
        'reload_ckpt_sig': 'ckpt-Mon_Nov__2_22-21-21_2020',
        'reloading'     : True,
        # 'reloading'     : False,
        
        'stft_files'    : [1,2,3],
        'len_epoch'     : 40,
        'train_rate'    : 0.7,
        'train'         : None,
        'test'          : None,
        'valid'         : None,
        'L1_outsize'    : 100,
        'L2_outsize'    : 2
        #       max and len could only exist one work
    }

    model_para = {
        'crop'          : False,
        'batch_size'    : 64,
        'sample_num'    : 64,
        'input_height'  : 56,
        'input_width'   : 112,
        'output_height' : 56,
        'output_width'  : 112,
        'y_dim'        : 2,
        'z_dim'        : 100,
        'gf_dim'        : 16,
        'df_dim'        : 16,
        'gfc_dim'       : 1024,
        'dfc_dim'       : 1024,
        'c_dim'         : 22,
        "learning_rate" : 0.0001,
        "beta1"         : 0.5,
        'L1_outsize'    : 100,
        'L2_outsize'    : 2
    }

    def __init__(self, model = 'dcgan',
                    stft_dir = '',
                    subjects = [1,2,3]):
        self.run_para['model'] = model
        self.run_para['stft_dir'] = stft_dir
        self.run_para['subjects'] = subjects
        self.update()

    def update(self, dicts = None):
        if dicts is None:
            self.run_para[ 'ckpt_dir' ] = self.ckpt_pattern.format(
                self.run_para['stft_dir']
            )
            check_dir( self.run_para[ 'ckpt_dir' ] )



        # raise NotImplementedError

    def sub_stft(self, sub = 'chb01'):
        return self.npy_pattern.format(sub)
'''





# run_records = pkl_load('./run_records')




### History records

####    the prototype template for argparser:
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--model", default = 'stft',
                    # help = "stft, dcgan, cvgan, gcn")
# parser.add_argument("--dataset", default = 'CHBMIT',
                    # help = "FB, CHBMIT, kaggle2014 or EpilepsiaSurf")
# parser.add_argument("--sph", type = int, default = 5,
                    # help = "0, 5, etc")
# parser.add_argument("--sop", type = int, default = 2,
                    # help = "seizure occurence period")
# parser.add_argument("--sig", type = str, default = None,
                    # help = "signature/biomark")
# parser.add_argument("--logonly", default = True, action = 'store_false',
                    # help = "signature/biomark")
# parser.add_argument("--new", default = True, action = 'store_false',
                    # help = "whether build one new log dir")
# parser.add_argument("--online", default = False, action = 'store_true',
                    # help = "whether choose online running")
# args = parser.parse_args()


