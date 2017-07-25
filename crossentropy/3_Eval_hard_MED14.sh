#!/bin/bash
if [ -r ap.csv ] ;then
rm ap.csv	
fi
dir=$1
#testtype="kindred"
testtype="med14test"
path1=/work1/t2g-shinoda2011/15M54105/trecvid/EventAgent/database_titech/
path2=/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm
if [ $testtype == "kindred" ];then
	testclip="${path2}/Test_clip"
	TEST_EVENTDB="${path1}/Kindred14-Test_20140428_EventDB.csv"
	TEST_REF="${path1}/Kindred14-Test_20140428_Ref.csv"
elif [ $testtype == "med14test" ];then
	testclip="${path2}/Test_clip_MED14"
	TEST_EVENTDB="${path2}/MED14-Test_20140513_EventDB.csv"
	TEST_REF="${path2}/MED14-Test_20140513_Ref.csv"
fi
APB=/work1/t2g-shinoda2011/15M54105/trecvid/EventAgent/depend/ap.sh
INPUT=testResList 
starte=200
ende=5
for outfile in ` ls $dir/outfile*.h5 `
do
        echo $outfile
        python /work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm4/hdf5ToList.py $outfile $testclip $INPUT
        ${APB} ${INPUT} ${TEST_EVENTDB} ${TEST_REF}
        id=`echo ${outfile##*/} | cut -d'.' -f1 `
        id=${id:7}
        echo $id
        if [ $id -gt $ende ]; then ende=$id; fi
        if [ $id -lt $starte ]; then starte=$id; fi
        sed 's/\"//g' ap.csv | cut -d',' -f2 > ${dir}/ap_${id}.csv
        rm ap.csv
done
echo "start_epoch: $starte, end_epoch: $ende"
python combineRes.py  $dir $starte $ende

