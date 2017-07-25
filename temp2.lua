require 'cutorch'
require 'cunn'
require 'rnn'
require 'os'
--[[
net = nn.Sequencer(
   nn.Sequential()
      :add(nn.MaskZero(nn.FastLSTM(10,6),1))
      :add(nn.MaskZero(nn.Linear(6,4),1))
      :add(nn.MaskZero(nn.LogSoftMax(),1))
)
parameters, gradParameters = net:getParameters()
lightModel = net:clone('weight','bias','running_mean','running_std')
torch.save('model.t7',lightModel)
--]]

net=torch.load("model.t7")
net:cuda()
local m = net.modules
model = (nn.MaskZero(nn.LogSoftMax(),1)):cuda()
--[[
tensor1 = torch.zeros(5,10)
tensor1[3]=torch.Tensor{3,4,5,6,7,8,23,2,12,90}
tensor2 = torch.ones(5,10)
tensor2[{{1,2},{}}]=torch.Tensor{ {1,3,4,5,6,0,3,2,56,2}, {5,3,2,5,7,3,45,78,235,10}}
tensor2[4]=torch.ones(1,10):fill(3.2)
tensor2[5]=torch.zeros(1,10)
input = {tensor1,tensor2}
--]]
--net=torch.load("/work1/t2g-shinoda2011/15M54105/trecvid/torch-lstm3/batch5_epoch5_hiddensize256_cw1/model_100ex_batch5_unit256_epoch70")
--[[
array = {}
tensor1  = torch.zeros(5,10)
tensor1[3]=torch.rand(1,10)
tensor2 = torch.rand(5,10)
tensor3 = torch.rand(5,10)
tensor4 = torch.rand(5,10)
tensor1=tensor1:cuda()
tensor2=tensor2:cuda()
tensor3=tensor3:cuda()
tensor4=tensor4:cuda()
table.insert(array, tensor1)
table.insert(array, tensor2)
table.insert(array, tensor3)
table.insert(array, tensor4)
file = torch.DiskFile('input.asc', 'w')
file:writeObject(array)
file:close()
os.exit()
--]]
--[[
file = torch.DiskFile('input.asc', 'r')
input = file:readObject()
--]]
seq=7
input={}
for i=1,seq do
	table.insert(input,torch.rand(5,10):cuda())
end
output = net:forward(input)
print(m)
--[[
for seqj = 1, #input do
	print(seqj)
	res = m[1].sharedClones[seqj].modules[2].output
	out1=output[seqj]
	out2=model:forward(res)
	print(out1-out2)
end
--]]
seq=9
input={}
for i=1,seq do
        table.insert(input,torch.rand(5,10):cuda())
end
output = net:forward(input)
print(m)
os.exit()
for seqj = 1, #input do
        print(seqj)
        res = m[1].sharedClones[seqj].modules[2].output
        out1=output[seqj]
        out2=model:forward(res)
        print(out1-out2)
end
