import sys 
import numpy as np
dir=sys.argv[1]
starte=int(sys.argv[2])
ende=int(sys.argv[3])
outfile='%s/ap_all.csv' %dir
#nums=[(i+1)*5 for i in range(20)]
rangenum=(ende-starte)/5+1
nums=[i*5+starte for i in range(rangenum)]
pres=np.zeros((20,len(nums)))
for j,num in enumerate(nums):
	file='%s/ap_%d.csv' %(dir,num)
	f=open(file,'r')
	lines=f.readlines()
	f.close()
	for ix, line in enumerate(lines):
		pres[ix,j]=line
avg=np.average(pres, axis=0)
with open(outfile, 'w') as writer:
	writer.truncate()
	numsa=np.asarray(nums)
	feat = numsa.reshape(1,-1)
        np.savetxt(writer, feat, delimiter='\t', fmt='%.8g')

	for i in range(20):
	     feat = pres[i].reshape(1,-1)
	     np.savetxt(writer, feat, delimiter='\t', fmt='%.8g')

	feat = avg.reshape(1,-1)
        np.savetxt(writer, feat, delimiter='\t',fmt='%.8g')
