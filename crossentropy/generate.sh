#/bin/bash
function generate() {
	#Number of hidden units of lstm
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
	TEST_ANNOTATION_PATH=/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/Test
	#How many sample in the file
	TEST_SAMPLE_NUM=12632

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
	LEARNING_RATE=0.005
	#LEARNING_RATE=0.01
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
	if [ $HARD_TARGET -eq 0 ];then TYPE1=soft; fi
	if [ $RandInit -eq 1 ];then TYPE2=_initialized; fi
	mainpath=${mainpath}_${TYPE1}${TYPE2}_t${TEMPRATURE}
	
	if [ ! -r $mainpath ];then mkdir $mainpath; fi

	#The directory where the trained models are saved
	MODEL_SAVING_DIR=${mainpath}
	#The directory where the test results are saved	
	OUTPUT_PATH=${MODEL_SAVING_DIR}
	if [ ! -r ${MODEL_SAVING_DIR} ];then mkdir ${MODEL_SAVING_DIR};fi

	#train shell
        trainfile="jobTrain_h${HIDDEN_NUM}_${TRAIN_SAMPLE_NUM}_${TYPE1}${TYPE2}_t${TEMPRATURE}.sh"
        touch $trainfile && chmod +x $trainfile
	logtrain=logTrain_h${HIDDEN_NUM}_${TRAIN_SAMPLE_NUM}_${TYPE1}${TYPE2}_t${TEMPRATURE}
        echo "#!/bin/bash" > $trainfile
        echo "cd \${PBS_O_WORKDIR}" >> $trainfile
        echo "th 1_train_memory2.lua -LEARNING_RATE $LEARNING_RATE -TRAIN_ANNOTATION_PATH "${TRAIN_ANNOTATION_PATH}" -TRAIN_SAMPLE_NUM ${TRAIN_SAMPLE_NUM} -HIDDEN_NUM ${HIDDEN_NUM} -MODEL_SAVING_DIR ${MODEL_SAVING_DIR} -EPOCH_NUM ${EPOCH_NUM} -INIT_MODEL_PATH ${INIT_MODEL_PATH} -RandInit ${RandInit} -HARD_TARGET ${HARD_TARGET} -LABELFILE ${LABELFILE} > ${logtrain}" >> $trainfile
        echo "cp 1_train_memory2.lua ${trainfile} ${logtrain} ${MODEL_SAVING_DIR}" >> $trainfile

	#test shell
	testfile="jobTest_h${HIDDEN_NUM}_${TRAIN_SAMPLE_NUM}_${TYPE1}${TYPE2}_t${TEMPRATURE}.sh"
	touch $testfile && chmod +x $testfile
	logtest=logTest_h${HIDDEN_NUM}_${TRAIN_SAMPLE_NUM}_${TYPE1}${TYPE2}_t${TEMPRATURE}
	echo "#!/bin/bash" > $testfile
        echo "cd \${PBS_O_WORKDIR}" >> $testfile
        echo "th 2_test.lua -MODEL_SAVING_DIR ${MODEL_SAVING_DIR} -EPOCH_NUM ${EPOCH_NUM} -OUTPUT_PATH ${OUTPUT_PATH} -HIDDEN_NUM ${HIDDEN_NUM} > $logtest" >> $testfile
	echo "cp 2_test.lua ${testfile} ${OUTPUT_PATH}" >> $testfile

}

#ids=("2096" "6988")
#hiddens=("32" "64" "128" "256" "512")

ids=("6988")
#ts=("1" "2" "4" "6" "8" "10" "16")
#ts=("1" "2" "4" "8" "16")
ts=("1")

#model initialization  
#0: randomly 1:initialized by other model
randinit=1

#hard targets or soft targets
#1: hard targets, 0: soft targets
#hardtars=("0" "1")
hardtars=("0")

for hardtar in ${hardtars[@]};do
	hiddens=("64")
	if [ $hardtar -eq 1 ]; then hiddens=("64" "256"); fi
	for id in ${ids[@]};do
		for hidden in ${hiddens[@]};do
			for t in ${ts[@]};do
			        generate $id $hidden $randinit $hardtar $t
			done
		done
	done
done
