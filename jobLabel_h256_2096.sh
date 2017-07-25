#!/bin/bash
cd ${PBS_O_WORKDIR}
hiddenSize=256
modelpath=/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/batch5_epoch5_hiddensize256_cw1/
batchSize=5
epochs=75
outpath=batch5_hiddensize256to64_2rd
if [ ! -r $outpath ];then mkdir $outpath; fi
#typenum  1: train6988 2:test 3:train2096
typenum=3

mainfile=1_generateLabels_2rdLayer.lua
logfile=logLabel_h${hiddenSize}_typenum${typenum}
th ${mainfile} ${hiddenSize} ${modelpath} ${batchSize} ${epochs} ${outpath} $typenum > $logfile
cp ${mainfile} jobLabel_h256_2096.sh  ${outpath}
