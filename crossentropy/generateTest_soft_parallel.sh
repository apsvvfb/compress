#/bin/bash
function generate() {
	#Number of hidden units of lstm
	subpath="MED14"
	TRAIN_SAMPLE_NUM=$1
	HIDDEN_NUM=$2
	TEMPRATURE=$5

	#Test annotations file path
	TEST_ANNOTATION_PATH=/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm/Test_MED14
	#How many sample in the file
	TEST_SAMPLE_NUM=23953

	#Dimension of the input feature
	FEAT_DIM=1024
	#Maximum number of unrolling steps of Lstm. 
	#If the sequence length is longer than 'SEQ_LENGTH_MAX', Lstm only unrolls for the first 'SEQ_LENGTH_MAX' steps
	SEQ_LENGTH_MAX=2000
	#Output class number
	TARGET_CLASS_NUM=21

	#After every how many epochs the model should be saved
	MODEL_SAVING_STEP=5
	#Total trained epoch num
	EPOCH_NUM=100
	#Batch size
	BATCH_SIZE=5
	#Learning rate
	LEARNING_RATE=$6
	#Learning rate decay
	LEARNING_RATE_DECAY=1e-4
	#Weight decay
	WEIGHT_DECAY=0.005
	#Momentum
	MOMENTUM=0.9
	#model initialization #6988

	mainpath=batch${BATCH_SIZE}_hiddensize${HIDDEN_NUM}_train${TRAIN_SAMPLE_NUM}
	TYPE1=hard
	TYPE2=""
	TYPE3=""
	lrtmp=`echo $LEARNING_RATE | cut -d'.' -f2`
	HARD_TARGET=$4
	RandInit=$3
	if [ $HARD_TARGET -eq 0 ];then 
		TYPE1=soft
	fi
	if [ $lrtmp -ne 5 ];then TYPE3=_lr${LEARNING_RATE};fi
	if [ $RandInit -eq 1 ];then TYPE2=_initialized; fi
	mainpath=${mainpath}_${TYPE1}${TYPE2}_t${TEMPRATURE}${TYPE3}
	
	#if [ ! -r $mainpath ];then mkdir $mainpath; fi

	#The directory where the trained models are saved
	MODEL_SAVING_DIR=${mainpath}
	#The directory where the test results are saved	
	OUTPUT_PATH=${subpath}/${mainpath}
	if [ ! -r ${OUTPUT_PATH} ];then mkdir ${OUTPUT_PATH};fi

	#test shell
	epochn=5
	submit="./submitcomm"
	echo "" > ${submit}
	while [ $epochn -le $EPOCH_NUM ]; do
		testfile="jobTest_h${HIDDEN_NUM}_${TRAIN_SAMPLE_NUM}_${TYPE1}${TYPE2}_t${TEMPRATURE}${TYPE3}_MED14_${epochn}.sh"
		touch $testfile && chmod +x $testfile
		logtest=logTest_h${HIDDEN_NUM}_${TRAIN_SAMPLE_NUM}_${TYPE1}${TYPE2}_t${TEMPRATURE}${TYPE3}_MED14_${epochn}
		echo "#!/bin/bash" > $testfile
	        echo "cd \${PBS_O_WORKDIR}" >> $testfile
        	echo "th 2_test.lua -MODEL_SAVING_DIR ${MODEL_SAVING_DIR} -START_EPOCH ${epochn} -EPOCH_NUM ${epochn} -OUTPUT_PATH ${OUTPUT_PATH} -HIDDEN_NUM ${HIDDEN_NUM} -TEST_ANNOTATION_PATH ${TEST_ANNOTATION_PATH} -TEST_SAMPLE_NUM ${TEST_SAMPLE_NUM} > $logtest" >> $testfile
		if [ $epochn -eq $EPOCH_NUM ];then echo "cp 2_test.lua ${testfile} ${OUTPUT_PATH}" >> $testfile; fi
		echo "t2sub -l walltime=2:${lrtmp}:${epochn} -et 1 -m abe -M apsvvfb@gmail.com -W group_list=t2g-crest-deep-sh -q S -l select=1:ncpus=8:gpus=1:mem=30gb ./${testfile}" >> ${submit}
		epochn=$[epochn+MODEL_SAVING_STEP]
	done
	echo "" >> ${submit}
}

ids=("6988")
ts=("1" "2")
#ts=("1")
#model initialization  
#0: randomly 1:initialized by other model
randinit=1

#hard targets or soft targets
#1: hard targets, 0: soft targets
#hardtars=("0" "1")
hardtars=("0")

#learning rate
lrs=("0.01" "0.005")
lrs=("0.01")
for hardtar in ${hardtars[@]};do
        hiddens=("64")
        if [ $hardtar -eq 1 ]; then hiddens=("64" "256"); fi
        for id in ${ids[@]};do
                for hidden in ${hiddens[@]};do
                        for t in ${ts[@]};do
                                for lr in ${lrs[@]};do
                                        generate $id $hidden $randinit $hardtar $t $lr
                                done
                        done
                done
        done
done
