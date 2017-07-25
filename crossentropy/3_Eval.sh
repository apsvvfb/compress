#!/bin/bash
if [ -r ap.csv ] ;then
rm ap.csv	
fi
dir=$1
testtype="med14test"
#testtype="kindred"
path1=/work1/t2g-shinoda2011/15M54105/trecvid/EventAgent/database_titech/
path2=/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm
temperature=""
if [ $testtype == "kindred" ];then
        CLIP_FILE="${path2}/Test_clip"
        TEST_EVENTDB="${path1}/Kindred14-Test_20140428_EventDB.csv"
        TEST_REF="${path1}/Kindred14-Test_20140428_Ref.csv"
	num=12632
elif [ $testtype == "med14test" ];then
        CLIP_FILE="${path2}/Test_clip_MED14"
        TEST_EVENTDB="${path2}/MED14-Test_20140513_EventDB.csv"
        TEST_REF="${path2}/MED14-Test_20140513_Ref.csv"
	num=23953
elif [ $testtype == "traindata" ];then
	temperature=`echo $dir | cut -d"_" -f8`
	temperature=${temperature:1:1}
	TEST_EVENTDB="${path1}/Kindred14-Test_20140428_EventDB.csv"
	TEST_REF="${path2}/Train_6988_Ref.csv"
	CLIP_FILE="${path2}/Train_clip"
        num=6988
fi
APB=/work1/t2g-shinoda2011/15M54105/trecvid/EventAgent/depend/ap.sh

starte=100
ende=5
for outfile in ` ls $dir/scores_epoch*.h5 `
do
	#echo $outfile
	id=`echo ${outfile##*/} | cut -d'.' -f1 `
	id=${id:12}
	#echo $id
	if [ $id -gt $ende ]; then ende=$id; fi
	if [ $id -lt $starte ]; then starte=$id; fi
done
echo $dir $num $starte $ende $temperature
#th score_post_processing.lua $dir $num $starte $ende $temperature
INPUT=testResList 
for outfile in ` ls $dir/outfile*.h5 `
do
	echo $outfile
	python hdf5ToList.py $outfile $CLIP_FILE
	${APB} ${INPUT} ${TEST_EVENTDB} ${TEST_REF}
	id=`echo ${outfile##*/} | cut -d'.' -f1 `
	id=${id:7}
	echo $id
	sed 's/\"//g' ap.csv | cut -d',' -f2 > ${dir}/ap_${id}.csv
	rm ap.csv
done
echo "start_epoch: $starte, end_epoch: $ende"
python combineRes.py  $dir $starte $ende
