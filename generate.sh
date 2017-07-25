#/bin/bash

function generate() {
	CRI=$2
	#Number of hidden units of lstm
	HIDDEN_NUM=64
	mainpath=batch5_hiddensize256to${HIDDEN_NUM}_initialized
	if [ ! -r $mainpath ];then mkdir $mainpath; fi

	#How many sample in train file
	TRAIN_SAMPLE_NUM=$1
	#Txt annotations file path
	TRAIN_ANNOTATION_PATH=/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/Train-${TRAIN_SAMPLE_NUM}
	if [ $TRAIN_SAMPLE_NUM -eq 6988 ]; then
		TRAIN_ANNOTATION_PATH=/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/Train-${TRAIN_SAMPLE_NUM}-shuffle
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

	#which criterion(1-3) 1:MSECriterion 2.DistKLDivCriterion
	if [ $CRI -eq 1 ];then
        	TYPE=MSECriterion
	elif [ $CRI -eq 2 ];then
        	TYPE=DistKLDivCriterion
	fi

	#The directory where the trained models are saved
	MODEL_SAVING_DIR=${mainpath}/${TYPE}_${TRAIN_SAMPLE_NUM}
	#The directory where the test results are saved	
	OUTPUT_PATH=${MODEL_SAVING_DIR}
	#After every how many epochs the model should be saved
	MODEL_SAVING_STEP=5
	#Total trained epoch num
	EPOCH_NUM=75
	#Batch size
	BATCH_SIZE=5
	#Learning rate
	LEARNING_RATE=0.005
	#Learning rate decay
	LEARNING_RATE_DECAY=1e-4
	#Weight decay
	WEIGHT_DECAY=0.005
	#Momentum
	MOMENTUM=0.9
	#model initialization
	model="/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/batch5_epoch75_hiddensize${HIDDEN_NUM}/model_100ex_batch5_unit${HIDDEN_NUM}_epoch5"
	
	#GPU ID
	GPU_ID=0

	#path of label file
	LABELFILE="/work0/t2g-shinoda2011/15M54105/compress/batch5_hiddensize256to64/Train${TRAIN_SAMPLE_NUM}_score_epoch70.h5"

	if [ ! -r ${MODEL_SAVING_DIR} ];then mkdir ${MODEL_SAVING_DIR};fi

	#train shell
        trainfile="jobTrain_h256to${HIDDEN_NUM}_${TRAIN_SAMPLE_NUM}_${TYPE}.sh"
        touch $trainfile && chmod +x $trainfile
	logtrain=logTrain_h${HIDDEN_NUM}_${TYPE}_${TRAIN_SAMPLE_NUM}
        echo "#!/bin/bash" > $trainfile
        echo "cd \${PBS_O_WORKDIR}" >> $trainfile
        echo "th 2_train.lua ${TRAIN_ANNOTATION_PATH} ${TRAIN_SAMPLE_NUM} ${FEAT_DIM} ${SEQ_LENGTH_MAX} ${TARGET_CLASS_NUM} ${HIDDEN_NUM} ${MODEL_SAVING_DIR} ${MODEL_SAVING_STEP} ${EPOCH_NUM} ${BATCH_SIZE} ${LEARNING_RATE} ${LEARNING_RATE_DECAY} ${WEIGHT_DECAY} ${MOMENTUM} ${CRI} ${GPU_ID} ${LABELFILE} ${model} > ${logtrain}" >> $trainfile
        echo "cp 2_train.lua ${outfile} ${logtrain} ${MODEL_SAVING_DIR}" >> $trainfile

	#test shell
	outfile="jobTest_h256to${HIDDEN_NUM}_${TRAIN_SAMPLE_NUM}_${TYPE}.sh"
	touch $outfile && chmod +x $outfile
	logtest=logTest_h${HIDDEN_NUM}_${TYPE}_${TRAIN_SAMPLE_NUM}
	echo "#!/bin/bash" > $outfile
        echo "cd \${PBS_O_WORKDIR}" >> $outfile
        echo "th 3_test.lua ${TEST_ANNOTATION_PATH} ${TEST_SAMPLE_NUM} ${FEAT_DIM} ${SEQ_LENGTH_MAX} ${TARGET_CLASS_NUM} ${MODEL_SAVING_DIR} ${EPOCH_NUM} ${BATCH_SIZE} ${OUTPUT_PATH} ${GPU_ID} ${HIDDEN_NUM} > $logtest" >> $outfile
	echo "cp 3_test.lua ${outfile} ${OUTPUT_PATH}" >> $outfile

}

ids=("2096" "6988")
#cris=("MSECriterion" "DistKLDivCriterion")
cris=("1" "2")
for id in ${ids[@]};do
	for cri in ${cris[@]};do
	        generate $id $cri
	done
done
