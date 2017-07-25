--[[
copy from /work/na/programs/CREST-MED/lstm/Lstm$ gedit lstm_test.lua
--]]

--sequence x batch x featdim
--right-aligned
require 'cutorch'
require 'cunn'
require 'rnn'
require 'optim'
require 'hdf5'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Test a LSTM network')
cmd:text()
cmd:option('-TEST_ANNOTATION_PATH', '/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/Test','Test annotations file path')
cmd:option('-TEST_SAMPLE_NUM', 12632, 'How many samples in test file')
cmd:option('-FEAT_DIM', 1024, 'Dimension of the input feature')
cmd:option('-SEQ_LENGTH_MAX', 2000, 'Maximum number of unrolling steps of Lstm.If the sequence length is longer than SEQ_LENGTH_MAX, Lstm only unrolls for the first SEQ_LENGTH_MAX steps')
cmd:option('-TARGET_CLASS_NUM', 21, 'Output class number')
cmd:option('-HIDDEN_NUM', 256, 'HIDDEN_NUM for lstm unit')
cmd:option('-MODEL_SAVING_STEP', 5, 'After every how many epochs the model should be saved')
cmd:option('-EPOCH_NUM', 100, 'Total trained epoch num')
cmd:option('-BATCH_SIZE', 5, 'Batch size')
cmd:option('-MODEL_SAVING_DIR', 'batch5_hiddensize256_train988_tmp', 'The directory where the trained models are saved')
cmd:option('-OUTPUT_PATH','batch5_hiddensize256_train988_tmp', 'The directory where the test results are saved')
cmd:option('-START_EPOCH', 5, '')

local opt = cmd:parse(arg)
print(opt)
local testAnnotationPath = opt.TEST_ANNOTATION_PATH
local testNum = opt.TEST_SAMPLE_NUM

local featdim = opt.FEAT_DIM
local seqLengthMax = opt.SEQ_LENGTH_MAX
local numTargetClasses = opt.TARGET_CLASS_NUM

local modelPath = opt.MODEL_SAVING_DIR
local epoch = opt.EPOCH_NUM
local batchSize = opt.BATCH_SIZE
local modelstep = opt.MODEL_SAVING_STEP

local outputPath = opt.OUTPUT_PATH
local hiddensize = opt.HIDDEN_NUM

local startepoch = opt.START_EPOCH
cutorch.setDevice(1)

local timer = nil
local model = nil
timer = torch.Timer()
print("Loading model...")
for epochnum = startepoch,epoch,modelstep do
modelname = string.format('%s/model_100ex_batch5_unit%s_epoch%d', modelPath, hiddensize, epochnum)
model = torch.load(modelname)
--local m = model.modules
--m[1].module.modules[3]=nn.Identity()
print(model)
print("Model Loaded...")

model:cuda()

local bactchseqLenMax = 0
local bactchseqLens = torch.Tensor(batchSize):fill(0)
local feattemp = torch.Tensor(batchSize, seqLengthMax, featdim):fill(0)
local labelbatch = torch.Tensor(batchSize):fill(0)
local targets = {}
local linenum = 0

local finalRes = torch.Tensor(testNum, numTargetClasses):fill(0)
--local scores=torch.Tensor(testNum,seqLengthMax,numTargetClasses):fill(0)
for line in io.lines(testAnnotationPath) do
	linenum = linenum + 1
	print(linenum)
	local i = linenum % batchSize
	if i == 0 then 
		i = batchSize 
	end
	local featpath = line:split(' ')[1]
	local labeli = line:split(' ')[2]
	labelbatch[i] = tonumber(labeli)

	--dirs
	local myFile = hdf5.open(featpath, 'r')
	local data = myFile:read('feature'):all()
	bactchseqLens[i] = data:size(1)
	if bactchseqLens[i] > seqLengthMax then
		data = data[{{1,seqLengthMax}}]
		bactchseqLens[i] = seqLengthMax
	end
	feattemp[i][{{1,bactchseqLens[i]}, {}}] = data

	if (i == batchSize) then
		--for last batch
		bactchseqLenMax = torch.max(bactchseqLens)
		local input = {}  
		local seqPadding = torch.Tensor(batchSize):fill(bactchseqLenMax) - bactchseqLens
		----right-aligned, padding zero in the left
		for seq = 1, bactchseqLenMax do
			local forOneTimeStep = torch.Tensor(batchSize,featdim):fill(0)
			local labeltemp = torch.Tensor(batchSize):fill(0)
			forOneTimeStep = forOneTimeStep:cuda()
			labeltemp = labeltemp:cuda()
			for batchi = 1, batchSize do
				if seqPadding[batchi] < seq then
					forOneTimeStep[batchi] = feattemp[batchi][seq-seqPadding[batchi]]
					labeltemp[batchi] = labelbatch[batchi]
				end
			end
			table.insert(input,forOneTimeStep)
			table.insert(targets, labeltemp)
		end
		----RNN
		local output = model:forward(input)
		for batchj = 1,batchSize do
			for seqj = seqPadding[batchj]+1, bactchseqLenMax do
				finalRes[linenum-batchSize+batchj] = finalRes[linenum-batchSize+batchj] + output[seqj][batchj]:exp():double()
			end
			finalRes[linenum-batchSize+batchj] = finalRes[linenum-batchSize+batchj] / bactchseqLens[batchj]
		end
		--[[	
		local start1=linenum-batchSize+1
		local score_tmp = torch.zeros(seqLengthMax, batchSize, numTargetClasses)
            	score_tmp = score_tmp:cuda()
	        for seqj= 1, bactchseqLenMax do
        	        score_tmp[seqj] = m[1].sharedClones[seqj].modules[2].output
	        end
        	scores[{{start1,start1+batchSize-1},{},{} }]=score_tmp:transpose(1,2):double()
		--]]
		--for new batch
		bactchseqLenMax = 0
		bactchseqLens = torch.Tensor(batchSize):fill(0)
		feattemp = torch.Tensor(batchSize,seqLengthMax,featdim):fill(0)
		labelbatch = torch.Tensor(batchSize):fill(0)
		targets = {}
	end
end
io.input():close()
--rest
local restnum = testNum % batchSize
bactchseqLenMax = torch.max(bactchseqLens)
local input = {}  
local seqPadding = torch.Tensor(batchSize):fill(bactchseqLenMax) - bactchseqLens
for seq = 1, bactchseqLenMax do
	local forOneTimeStep = torch.Tensor(restnum,featdim):fill(0)
	local labeltemp = torch.Tensor(restnum):fill(0)
	forOneTimeStep = forOneTimeStep:cuda()
	labeltemp = labeltemp:cuda()
	for batchi = 1, restnum do
		if seqPadding[batchi] < seq then
			forOneTimeStep[batchi] = feattemp[batchi][seq-seqPadding[batchi]]
			labeltemp[batchi] = labelbatch[batchi]
		end
	end
	table.insert(input,forOneTimeStep)
	table.insert(targets, labeltemp)
end
local output = model:forward(input)
for batchj = 1,restnum do
	for seqj = seqPadding[batchj] + 1,bactchseqLenMax do
		finalRes[testNum - restnum + batchj] = finalRes[testNum - restnum+batchj] + output[seqj][batchj]:exp():double()
	end
	finalRes[testNum - restnum + batchj] = finalRes[testNum - restnum + batchj] / bactchseqLens[batchj]
end
local outfile = hdf5.open(string.format('%s/outfile%d.h5', outputPath,epochnum), 'w')
outfile:write('data', finalRes)	
outfile:close()
--[[
local score_tmp = torch.zeros(seqLengthMax, restnum, numTargetClasses)
score_tmp=score_tmp:cuda()
for seqj= 1, bactchseqLenMax do
	score_tmp[seqj] = m[1].sharedClones[seqj].modules[2].output
end
scores[{{testNum-restnum+1,testNum},{},{} }]=score_tmp:transpose(1,2):double()
local scorefile=string.format("%s/scores_epoch%d.h5",outputPath,epochnum)
local myFile = hdf5.open(scorefile, 'w')
myFile:write('score', scores)
myFile:close()
--]]
end
print('Time elapsed: ' .. timer:time().real .. ' seconds')
--end

