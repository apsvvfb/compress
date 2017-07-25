#!/bin/bash
if [ -r ap.csv ] ;then
rm ap.csv	
fi
dir=$1
path1=/work1/t2g-shinoda2011/15M54105/trecvid/EventAgent/database_titech/
TEST_EVENTDB=${path1}"Kindred14-Test_20140428_EventDB.csv"
TEST_REF=${path1}"Kindred14-Test_20140428_Ref.csv"
APB=/work1/t2g-shinoda2011/15M54105/trecvid/EventAgent/depend/ap.sh
INPUT=testResList 
for outfile in ` ls $dir/outfile*.h5 `
do
	echo $outfile
	python /work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm2/hdf5ToList.py $outfile
	${APB} ${INPUT} ${TEST_EVENTDB} ${TEST_REF}
	id=`echo ${outfile##*/} | cut -d'.' -f1 `
	id=${id:7}
	echo $id
	sed 's/\"//g' ap.csv | cut -d',' -f2 > ${dir}/ap_${id}.csv
	rm ap.csv
done
