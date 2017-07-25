#/bin/bash
function generate() {
	#Number of hidden units of lstm
	traindata="TRAINDATA_AS_TESTDATA"
	if [ ! -r $traindata ];then mkdir $traindata; fi
	HIDDEN_NUM=$2
	TEMPRATURE=$5

	#How many sample in train file
	TRAIN_SAMPLE_NUM=$1
	#Txt annotations file path
	TRAIN_ANNOTATION_PATH=/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/Train-${TRAIN_SAMPLE_NUM}
	if [ $TRAIN_SAMPLE_NUM -eq 6988 ]; then
		TRAIN_ANNOTATION_PATH="/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/Train-${TRAIN_SAMPLE_NUM}-shuffle"
	fi

	#Test annotations file path
	#TEST_ANNOTATION_PATH=/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/Test
	TEST_ANNOTATION_PATH=/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/Train-6988-shuffle
	#How many sample in the file
	#TEST_SAMPLE_NUM=12632
	TEST_SAMPLE_NUM=6988

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
	RandInit=$3
	INIT_MODEL_PATH="/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/batch5_epoch5_hiddensize${HIDDEN_NUM}_cw1/model_100ex_batch5_unit${HIDDEN_NUM}_epoch5"

	#GPU ID
	GPU_ID=0

	#path of label file
	HARD_TARGET=$4
	if [ $HARD_TARGET -eq 1 ] ; then TEMPRATURE=1; fi
	LABELFILE="/work0/t2g-shinoda2011/15M54105/compress/batch5_hiddensize256to64_2rd/Train${TRAIN_SAMPLE_NUM}_score_epoch70_t${TEMPRATURE}.h5"

	mainpath=batch${BATCH_SIZE}_hiddensize${HIDDEN_NUM}_train${TRAIN_SAMPLE_NUM}
	TYPE1=hard
	TYPE2=""
	TYPE3=""
	lrtmp=`echo $LEARNING_RATE | cut -d'.' -f 2`
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
	OUTPUT_PATH=${traindata}/${mainpath}
	if [ ! -r ${OUTPUT_PATH} ];then mkdir ${OUTPUT_PATH};fi

	#test shell
	testfile="jobTest_h${HIDDEN_NUM}_${TRAIN_SAMPLE_NUM}_${TYPE1}${TYPE2}_t${TEMPRATURE}${TYPE3}_traindata.sh"
	touch $testfile && chmod +x $testfile
	logtest=logTest_h${HIDDEN_NUM}_${TRAIN_SAMPLE_NUM}_${TYPE1}${TYPE2}_t${TEMPRATURE}${TYPE3}_traindata
	echo "#!/bin/bash" > $testfile
        echo "cd \${PBS_O_WORKDIR}" >> $testfile
        echo "th 2_test.lua -MODEL_SAVING_DIR ${MODEL_SAVING_DIR} -EPOCH_NUM ${EPOCH_NUM} -OUTPUT_PATH ${OUTPUT_PATH} -HIDDEN_NUM ${HIDDEN_NUM} -TEST_ANNOTATION_PATH ${TEST_ANNOTATION_PATH} -TEST_SAMPLE_NUM ${TEST_SAMPLE_NUM} > $logtest" >> $testfile
	echo "cp 2_test.lua ${testfile} ${OUTPUT_PATH}" >> $testfile

}

#ids=("2096" "6988")
#hiddens=("32" "64" "128" "256" "512")

ids=("6988")
#ts=("1" "2" "4" "6" "8" "10" "16")
ts=("1" "2" "4" "8")
ts=("1")
#model initialization  
#0: randomly 1:initialized by other model
randinit=1

#hard targets or soft targets
#1: hard targets, 0: soft targets
hardtars=("0" "1")
hardtars=("1")

#learning rate
lrs=("0.01" "0.005")
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
