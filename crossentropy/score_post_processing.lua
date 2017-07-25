require 'nn'

require 'hdf5'
require 'os'
require 'rnn'
local filepath=arg[1]
local testNum=tonumber(arg[2])
local startepochs=tonumber(arg[3])
local epochs=tonumber(arg[4])
local seqLengthMax=2000
local numTargetClasses=21
local model = nn.MaskZero(nn.SoftMax(),1) 
if testNum == 6988 then
	temperature=tonumber(arg[5])
	print(temperature)
	model = nn.Sequential()
                :add(nn.MaskZero(nn.MulConstant(1/temperature),1))
                :add(nn.MaskZero(nn.SoftMax(),1))
end
for epoch = startepochs, epochs, 5 do
local scorefile=string.format("%s/scores_epoch%d.h5",filepath,epoch)
print(scorefile)
local scoreF = hdf5.open(scorefile, 'r')
local scoredata = scoreF:read('score'):all() --testNum  x seqLengthMax x numTargetClasses
scoreF:close()
print(#scoredata)
local finalscore = torch.zeros(testNum,numTargetClasses)
for i = 1,testNum do
	local seq_len = scoredata[i]:sum(2):ne(0):double():sum()
	local score_i = model:forward(scoredata[i]) --seqLengthMax x numTargetClasses
	local score_sum = score_i:sum(1)	-- 1 x numTargetClasses	
	finalscore[i] = score_sum / seq_len 
end
local outfile = hdf5.open(string.format('%s/outfile%d.h5', filepath,epoch), 'w')
outfile:write('data', finalscore) 
outfile:close()
end
