--copy from /work0/t2g-shinoda2011/15M54105/compress/2_train.lua
--sequence x batch x featdim
--right-aligned
require 'cutorch'
require 'cunn'
require 'rnn'
require 'optim'
require 'hdf5'
require 'os'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a LSTM network')
cmd:text()
cmd:option('-TRAIN_ANNOTATION_PATH', '/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/Train-6988-shuffle','Txt annotations file path')
cmd:option('-TRAIN_SAMPLE_NUM', 6988, 'How many samples in train file')
cmd:option('-FEAT_DIM', 1024, 'Dimension of the input feature')
cmd:option('-SEQ_LENGTH_MAX', 2000, 'Maximum number of unrolling steps of Lstm.If the sequence length is longer than SEQ_LENGTH_MAX, Lstm only unrolls for the first SEQ_LENGTH_MAX steps')
cmd:option('-TARGET_CLASS_NUM', 21, 'Output class number')
cmd:option('-HIDDEN_NUM', 256, 'HIDDEN_NUM for lstm unit')
cmd:option('-MODEL_SAVING_STEP', 5, 'After every how many epochs the model should be saved')
cmd:option('-EPOCH_NUM', 100, 'Total trained epoch num')
cmd:option('-BATCH_SIZE', 5, 'Batch size')

cmd:option('-LEARNING_RATE', 0.005, 'lr')
cmd:option('-LEARNING_RATE_DECAY', 1e-4, 'Learning rate decay')
cmd:option('-WEIGHT_DECAY', 0.005, 'Weight decay')
cmd:option('-MOMENTUM', 0.9, 'Momentum')

cmd:option('-RandInit', 1, 'model will be randomly initialed: 1 or 0')
cmd:option('-INIT_MODEL_PATH', '', 'the path of the initial model')
cmd:option('-MODEL_SAVING_DIR', 'batch5_hiddensize256_train988_tmp', 'The directory where the trained models are saved')
cmd:option('-HARD_TARGET', 1, 'soft target(0) or hard target(1)')
cmd:option('-LABELFILE','','The softlabel file: /work0/t2g-shinoda2011/15M54105/compress/batch5_hiddensize256to64/Train6988_score_epoch70.h5')

cmd:option('-TEMPRATURE',1, 'Temperature')

local opt = cmd:parse(arg)
print(opt)
local trainAnnotationPath = opt.TRAIN_ANNOTATION_PATH
local trainNum = opt.TRAIN_SAMPLE_NUM
local featdim = opt.FEAT_DIM
local seqLengthMax = opt.SEQ_LENGTH_MAX
local numTargetClasses = opt.TARGET_CLASS_NUM
local hiddenSize = opt.HIDDEN_NUM

local modelSavingDir = opt.MODEL_SAVING_DIR
local modelSavingStep = opt.MODEL_SAVING_STEP
local epoch = opt.EPOCH_NUM
local batchSize = opt.BATCH_SIZE
local lr = opt.LEARNING_RATE
local lrd = opt.LEARNING_RATE_DECAY
local wd = opt.WEIGHT_DECAY
local momentum = opt.MOMENTUM
local hardtarget = opt.HARD_TARGET
local scoreF, scoredata
local labelfile = opt.LABELFILE
local temperature = opt.TEMPRATURE
if hardtarget == 0 then
	scoreF = hdf5.open(labelfile, 'r')
	scoredata = scoreF:read('score'):all()  --trainNum x seqLengthMax x numTargetClasses = 6988 x 2000 x 21
	scoreF:close()
end

local RandInit = opt.RandInit
local gpuId = 0
local model = nn.Sequencer(
        nn.Sequential()
                :add(nn.FastLSTM(featdim, hiddenSize):maskZero(1))
                :add(nn.MaskZero(nn.Linear(hiddenSize, numTargetClasses),1))
		:add(nn.MaskZero(nn.MulConstant(1/temperature),1))
                :add(nn.MaskZero(nn.LogSoftMax(),1))
)
model:cuda()
-- get weights and loss wrt weights from the model
local params, grads = model:getParameters()
if RandInit ~= 0 then
	local init_model = torch.load(opt.INIT_MODEL_PATH)
	local init_params, init_grads = init_model:getParameters()
	for i = 1, (#params)[1] do
		params[i] = init_params[i]
		grads[i] = init_grads[i]
	end
	init_model, init_params, init_grads = nil, nil, nil
	params, grads = model:getParameters()
end
print('Model created')

local timer = torch.Timer()
local weight=torch.Tensor(21):fill(1)
weight[21]=0.02
--[[local c1 = nn.MyCrossEntropyCriterion(weight)
c1.maskZero = true
c1.sizeAverage = true
local criterion = nn.SequencerCriterion(c1)--]]
--[[local criterion_hard = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(weight),1))
criterion_hard:cuda()--]]
local criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.MyCrossEntropyCriterionGPU(weight),1))
criterion:cuda()

local bactchseqLenMax = 0
local bactchseqLens = torch.Tensor(batchSize):fill(0)
local feattemp = torch.Tensor(batchSize, seqLengthMax, featdim):fill(0)
--local scorelabels = torch.Tensor(batchSize, seqLengthMax, numTargetClasses):fill(0)
local scorelabels = torch.Tensor(batchSize):fill(0)
local labelbatch = torch.Tensor(batchSize):fill(0)

local sgd_params = {
	learningRate = lr,
	learningRateDecay = lrd,
	weightDecay = wd,
	momentum = momentum
}

local allnum = 0
for epi = 1, epoch do
	local linenum = 0
        --print("epoch:"..epi)
        --print("memory:"..collectgarbage("count").."KB")
	for line in io.lines(trainAnnotationPath) do
		linenum = linenum + 1
		allnum = trainNum * (epi - 1) + linenum
		print("epoch:"..epi..", linenum:"..linenum)
		local i = allnum % batchSize
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
		myFile, data = nil, nil
		if hardtarget == 0 then
			--scorelabels[i] = scoredata[linenum]
			scorelabels[i] = linenum
		end
		if (i == batchSize) then
			--for last batch
			bactchseqLenMax = torch.max(bactchseqLens)
			local input = {} 
			local targets = {}
			--local targets_hard = {}
			local seqPadding = torch.Tensor(batchSize):fill(bactchseqLenMax)-bactchseqLens

			----right-aligned, padding zero in the left
			local forOneTimeStep, labeltemp
			for seq = 1, bactchseqLenMax do
				forOneTimeStep = torch.Tensor(batchSize,featdim):fill(0)
				--labeltemp_hard = torch.Tensor(batchSize):fill(0)
				labeltemp = torch.Tensor(batchSize, numTargetClasses):fill(0)
				forOneTimeStep = forOneTimeStep:cuda()
				--labeltemp_hard = labeltemp_hard:cuda()
				for batchi = 1, batchSize do
					if seqPadding[batchi] < seq then
						forOneTimeStep[batchi]=feattemp[batchi][seq-seqPadding[batchi]]
						--labeltemp_hard[batchi]=labelbatch[batchi]
						labeltemp[batchi][labelbatch[batchi]] = 1
						if hardtarget == 0 then
							--labeltemp[batchi]=scorelabels[batchi][seq-seqPadding[batchi]]
							labeltemp[batchi]=scoredata[scorelabels[batchi]][seq-seqPadding[batchi]]
						end
					end
				end
				table.insert(targets, labeltemp)
				--table.insert(targets_hard, labeltemp_hard)
				table.insert(input,forOneTimeStep)
			end
			forOneTimeStep, labeltemp = nil, nil
			----RNN	   	    
			local feval = function(x_new)
				-- copy the weight if are changed
				if params ~= x_new then
					params:copy(x_new)
				end
				-- reset gradients (gradients are always accumulated, to accommodate
				-- batch methods)
				grads:zero()

				-- evaluate the loss function and its derivative with respect to x, given a mini batch
				local output = model:forward(input)
				local loss_x = criterion:forward(output, targets)
				local gradOutputs = criterion:backward(output, targets)
				model:backward(input, gradOutputs)

				output, gradOutputs = nil, nil
				return loss_x, grads

				--[[
				local loss_x_hard = criterion_hard:forward(output, targets_hard)
				local gradOutputs_hard = criterion_hard:backward(output, targets_hard)
				model:backward(input, gradOutputs_hard)
				return loss_x_hard, grads
				--]]
			end
			local _, fs = optim.sgd(feval, params, sgd_params)
			print("Loss: " .. fs[1])
			bactchseqLenMax = 0
			bactchseqLens, feattemp, labelbatch, scorelabels, input, targets = nil, nil, nil, nil, nil, nil
			bactchseqLens = torch.Tensor(batchSize):fill(0)
			feattemp = torch.Tensor(batchSize, seqLengthMax, featdim):fill(0)
			labelbatch = torch.Tensor(batchSize):fill(0)	    
			--scorelabels = torch.Tensor(batchSize, seqLengthMax, numTargetClasses):fill(0)
			scorelabels = torch.Tensor(batchSize)
			--print("memory:"..collectgarbage("count").."KB")	
		end
		--collectgarbage()
	end
	io.input():close()
	--if (epi > modelSavingStep and epi % modelSavingStep == 0) then
	if (epi % modelSavingStep == 0) then
		print("saving model-epoch:".. epi)
		print('Time elapsed: ' .. timer:time().real .. ' seconds')
		--collectgarbage("collect")
		model:clearState()
		torch.save(string.format("%s/model_100ex_batch%d_unit%d_epoch%d", modelSavingDir, batchSize, hiddenSize, epi), model)
	end
end
--model:clearState()
--torch.save("rnnmodel_100ex-all", model)
print("finish training!")
print('Time elapsed: ' .. timer:time().real .. ' seconds')
