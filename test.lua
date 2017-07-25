require 'cutorch'
require 'cunn'
require 'rnn'
require 'os'

file = torch.DiskFile('foo.asc', 'r')
input = file:readObject()
print(input)
model=nn.LogSoftMax()
model:cuda()
out=model:forward(input)
print(out)
