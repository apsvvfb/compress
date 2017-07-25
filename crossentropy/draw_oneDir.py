import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot,savefig

import matplotlib.pyplot as plt
import glob
import sys
import numpy as np
import os

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def readfile_t(inputfile):
        f=open(inputfile,'r')
        lines=f.readlines()
        f.close()

        batchsize=5
        lenf=len(lines)/batchsize

        trainloss=np.zeros(100)
        losses=np.zeros(lenf)
        e=0
        i=0
        epo_now=1
        epo_tmp_num=0
        ifepo=0
        for ix, line in enumerate(lines):
                data=line.strip().split(',')[0].split(':')[0]
                if data == "Loss": #isfloat(data):
                        #losses[i]=float(data)
			losses[i] = float(line.strip().split(' ')[1])
			#print("loss:%f" %losses[i])
                        i=i+1
                        if ifepo==1:
                                trainloss[e]=sum(losses)/(i)
                                e=e+1
				#print("e:%d,i:%d" %(e,i))
                                i=0
                                ifepo=0
                        epo_tmp_num=0
                elif data == "epoch":
                        epoch=int(line.strip().split(',')[0].split(':')[1])
			#print("epoch:%d" %epoch)
                        if epo_tmp_num == 0 and epo_now != epoch:
                                ifepo=1
                                epo_now=epoch
                        epo_tmp_num=epo_tmp_num+1

        return trainloss[0:e-1],len(trainloss[0:e-1])

def readfile_e(inputpath,x_e):
    #files = glob.glob('%s/ap_*.csv' %inputpath)
    avg=np.zeros(len(x_e))
    for i,ei in enumerate(x_e):
	file='%s/ap_%d.csv' %(inputpath,ei)
        f = open("%s" %file, 'r')
        f_lines = f.readlines()
        f.close()
        res=np.zeros(20)
        for ix, line in enumerate(f_lines):
                 res[ix]=float(line)
        avg[i]=np.average(res)
    return avg

def draw(dir):
	inpath=dir
	inputfile_t=glob.glob(r"%s/logTrain_*" %(inpath))[0] 
	name_t=inputfile_t.split("/")[1]
	print(name_t)
        imgname='%s.png' %(name_t)
        plttitle='%s' %(name_t)

        loss_e= readfile_e(inpath,x_e)
        lens=len(x_e)

        loss_t,len_t= readfile_t(inputfile_t)
        lens=len_t

        x_tmp=range(lens)
        x_t=[i+1 for i in x_tmp]

        plt.figure()
        fig,ax1=plt.subplots()
        ax2=ax1.twinx()

        y_loss1=loss_t #[0:lens]
        p1, =ax1.plot(x_t,y_loss1,'m*-', label="train_loss",linewidth=2)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('train_loss')

        y_loss1=loss_e #[0:lens]
        p2, =ax2.plot(x_e,y_loss1,'bo-', label="test_prec",linewidth=2)
        ax2.set_ylabel('testdata_precistion')
        plt.legend(loc='upper right')
        plt.title(plttitle, fontsize = 16)

        ax1.yaxis.label.set_color(p1.get_color())
        ax2.yaxis.label.set_color(p2.get_color())

        tkw = dict(size=5, width=1.5)
        ax1.tick_params(axis='y', colors=p1.get_color(), **tkw)
        ax1.set_ylim(ymin=0)
        ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
        ax2.set_ylim(ymin=0,ymax=0.5)
        ax1.tick_params(axis='x', **tkw)

        savefig(imgname)

if __name__ == "__main__":
        x_e=[(i+1)*5 for i in range(20)]
#	x_e=[(i+1)*5 for i in range(19)]
	dir=sys.argv[1]
	draw(dir)
