--sequence x batch x featdim
--right-aligned

--[[
copy from
/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm4/batch5_epoch100_hiddensize128_cw1/2_test_train+test.lua
--]]

require 'cutorch'
require 'cunn'
require 'rnn'
require 'optim'
require 'hdf5'

--1:train
Train6988="/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/Train-6988-shuffle"
trainNum6988=6988
--2:test
Test="/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/Test"
testNum=12632
--3:train-2096
Train2096="/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm2/Train-2096"
trainNum2096=2096

hiddenSize = tonumber(arg[1]) --256
inpath=arg[2]
batchSize = tonumber(arg[3])
epochs=tonumber(arg[4])
outpath=arg[5]
typenum=tonumber(arg[6])

featdim = 1024
seqLengthMax = 2000
numTargetClasses=21

inputfiles={Train6988,Test,Train2096}
intypes={"Train6988","Test","Train2096"}
inputnums={trainNum6988,testNum,trainNum2096}

inputfile=inputfiles[typenum]
inputnum=inputnums[typenum]
intype=intypes[typenum]

function TableToTensor(table)
  local tensorSize = table[1]:size()
  local tensorSizeTable = {-1}
  for i=1,tensorSize:size(1) do
    tensorSizeTable[i+1] = tensorSize[i]
  end
  merge=nn.Sequential()
    :add(nn.JoinTable(1))
    :add(nn.View(unpack(tensorSizeTable)))

  return merge:forward(table)
end

--for mei=15,epochs,5 do
for mei=70,70,5 do
  model=nil
  model = torch.load(string.format("%s/model_100ex_batch%d_unit%d_epoch%d",inpath,batchSize,hiddenSize,mei))
  criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))
  model:cuda()
  criterion:cuda()
  
  --[[for id_f=1,2,1 do
    inputfile=inputfiles[id_f]
    inputnum=inputnums[id_f]
    intype=intypes[id_f]
  --]]
    scores=torch.Tensor(inputnum,seqLengthMax,numTargetClasses):fill(0)
  
    bactchseqLenMax=0
    bactchseqLens=torch.Tensor(batchSize):fill(0)
    feattemp=torch.Tensor(batchSize,seqLengthMax,featdim):fill(0)
    labelbatch=torch.Tensor(batchSize):fill(0)
    targets = {}
    linenum=0
    finalRes=torch.Tensor(inputnum,numTargetClasses):fill(0)

    for line in io.lines(inputfile) do
        linenum=linenum+1
	print(linenum)
	i=linenum % batchSize
	if i==0 then 
	    i=batchSize 
	end
	featpath=line:split(' ')[1]
	labeli=line:split(' ')[2]
	labelbatch[i]=tonumber(labeli)

	--dirs
	local myFile = hdf5.open(featpath, 'r')
	local data = myFile:read('feature'):all()
	bactchseqLens[i]=data:size(1)
        if bactchseqLens[i]>seqLengthMax then
                data=data[{{1,seqLengthMax}}]
                bactchseqLens[i]=seqLengthMax
        end
	feattemp[i][{{1,bactchseqLens[i]}, {}}]=data

	if (i == batchSize) then
	    --for last batch
	    bactchseqLenMax=torch.max(bactchseqLens)
	    input={}  
	    local seqPadding=torch.Tensor(batchSize):fill(bactchseqLenMax)-bactchseqLens
	    ----right-aligned, padding zero in the left
	    for seq=1,bactchseqLenMax do
	        forOneTimeStep=torch.Tensor(batchSize,featdim):fill(0)
		labeltemp=torch.Tensor(batchSize):fill(0)
		forOneTimeStep=forOneTimeStep:cuda()
		labeltemp=labeltemp:cuda()
		for batchi=1,batchSize do
		    if seqPadding[batchi] < seq then
			forOneTimeStep[batchi]=feattemp[batchi][seq-seqPadding[batchi]]
			labeltemp[batchi]=labelbatch[batchi]
		    end
		end
		table.insert(input,forOneTimeStep)
		table.insert(targets, labeltemp)
	    end
	    ----RNN
	    local output = model:forward(input)

	    output_tensor=TableToTensor(output)
            local output_trans = output_tensor:transpose(1,2)
            local start1=linenum-batchSize+1
            scores[{{start1,start1+batchSize-1},{1,bactchseqLenMax},{} }]=output_trans
            output_trans=nil

	    --[[
	    for batchj = 1,batchSize do
		for seqj=seqPadding[batchj]+1,bactchseqLenMax do
			finalRes[linenum-batchSize+batchj]= finalRes[linenum-batchSize+batchj] + output[seqj][batchj]:exp():double()
		end
		finalRes[linenum-batchSize+batchj]=finalRes[linenum-batchSize+batchj]/bactchseqLens[batchj]
	    end
	    --]]

	    --for new batch
	    bactchseqLenMax=0
	    bactchseqLens=torch.Tensor(batchSize):fill(0)
	    feattemp=torch.Tensor(batchSize,seqLengthMax,featdim):fill(0)
	    labelbatch=torch.Tensor(batchSize):fill(0)
	    targets = {}
        end
    end
    io.input():close()
    --rest
    restnum=inputnum%batchSize
    bactchseqLenMax=torch.max(bactchseqLens)
    input={}  
    local seqPadding=torch.Tensor(batchSize):fill(bactchseqLenMax)-bactchseqLens
    for seq=1,bactchseqLenMax do
      forOneTimeStep=torch.Tensor(restnum,featdim):fill(0)
      labeltemp=torch.Tensor(restnum):fill(0)
      forOneTimeStep=forOneTimeStep:cuda()
      labeltemp=labeltemp:cuda()
      for batchi=1,restnum do
        if seqPadding[batchi] < seq then
 	    forOneTimeStep[batchi]=feattemp[batchi][seq-seqPadding[batchi]]
	    labeltemp[batchi]=labelbatch[batchi]
        end
      end
      table.insert(input,forOneTimeStep)
      table.insert(targets, labeltemp)
    end
    output = model:forward(input)

    output_tensor=TableToTensor(output)
    local output_trans = output_tensor:transpose(1,2)
    scores[{{inputnum-restnum+1,inputnum},{1,bactchseqLenMax},{} }]=output_trans
    output_trans=nil

    --[[
    for batchj = 1,restnum do
	for seqj=seqPadding[batchj]+1,bactchseqLenMax do
		finalRes[inputnum-restnum+batchj]= finalRes[inputum-restnum+batchj] + output[seqj][batchj]:exp():double()
	end
	finalRes[inputnum-restnum+batchj]=finalRes[inputnum-restnum+batchj]/bactchseqLens[batchj]
    end
    outfile = hdf5.open(string.format("%s/%s_outfile%d.h5",outpath,intype,mei),'w')
    outfile:write('data', finalRes)	
    outfile:close()
    --]]

    print("start saving score file!")
    print(#scores)
    local scorefile=string.format("%s/%s_score_epoch%d.h5",outpath,intype,mei)
    local myFile = hdf5.open(scorefile, 'w')
    myFile:write('score', scores)
    print("finish saving score file!")
    myFile:close()	
--  end
end
