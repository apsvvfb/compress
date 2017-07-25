require 'cutorch'
require 'cunn'
require 'rnn'
require 'optim'
require 'hdf5'
require 'os'
local model = nn.LogSoftMax()
local criterion = nn.ClassNLLCriterion()

local input=torch.Tensor{{1,2,3,4},{3,5,1,1},{7,2,3,4}}
local targets= torch.Tensor{3,2,1}
local output=model:forward(input)
local loss_x = criterion:forward(output, targets)
local gradOutputs = criterion:backward(output, targets)
print("input")
print(input)
print("label")
print(targets)
print("output")
print(output)
print("loss")
print(loss_x)
print("gradient")
print(gradOutputs)
model:backward(input, gradOutputs)


