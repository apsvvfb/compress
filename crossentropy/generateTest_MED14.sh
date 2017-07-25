#/bin/bash
function generate() {
	#Number of hidden units of lstm
	HIDDEN_NUM=$1

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
	EPOCH_NUM=75
	#Batch size
	BATCH_SIZE=5
	#GPU ID
	GPU_ID=0

	#The directory where the trained models are saved
	#MODEL_SAVING_DIR=/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/batch5_epoch5_hiddensize${HIDDEN_NUM}_cw1
	MODEL_SAVING_DIR=/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/batch5_epoch75_hiddensize${HIDDEN_NUM}
	#The directory where the test results are saved
	OUTPUT_PATH_TMP=`basename $MODEL_SAVING_DIR`
	OUTPUT_PATH=MED14/${OUTPUT_PATH_TMP}
	if [ ! -r ${OUTPUT_PATH} ];then mkdir ${OUTPUT_PATH};fi

	#test shell
	testfile="jobTest_h${HIDDEN_NUM}_MED14.sh"
	touch $testfile && chmod +x $testfile
	logtest=logTest_h${HIDDEN_NUM}_MED14
	echo "#!/bin/bash" > $testfile
        echo "cd \${PBS_O_WORKDIR}" >> $testfile
        echo "th 2_test_ori.lua -MODEL_SAVING_DIR ${MODEL_SAVING_DIR} -EPOCH_NUM ${EPOCH_NUM} -OUTPUT_PATH ${OUTPUT_PATH} -HIDDEN_NUM ${HIDDEN_NUM} -TEST_ANNOTATION_PATH ${TEST_ANNOTATION_PATH} -TEST_SAMPLE_NUM ${TEST_SAMPLE_NUM} > $logtest" >> $testfile
	echo "cp 2_test_ori.lua ${testfile} ${OUTPUT_PATH}" >> $testfile

}

hiddens=("16" "32" "64" "128" "256" "512" "1024")

for hidden in ${hiddens[@]};do
        generate $hidden
done
