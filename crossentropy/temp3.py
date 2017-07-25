import numpy as np
import h5py
scorefile="batch5_hiddensize256_train6988_hard_initialized_t1/scores_epoch15.h5"
scorefile="batch5_hiddensize64_train6988_soft_initialized_t2/scores_epoch15.h5"
#scorefile="/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/batch10_epoch110_hiddensize128/outfile100.h5"
with h5py.File(scorefile,'r') as hf:
	data1 = hf.get('score')
	#data1=hf.get('data')
	attenmaps = np.array(data1)
