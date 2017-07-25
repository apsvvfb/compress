
require 'nn'
require 'os'
require 'rnn'
--[[
scorefile="batch5_hiddensize256_train6988_hard_initialized_t1/scores_epoch5.h5"
local scoreF = hdf5.open(scorefile, 'r')
local scoredata = scoreF:read('score'):all() --testNum  x seqLengthMax x numTargetClasses
scoreF:close()
--]]
a=nn.Sequencer(nn.FastLSTM(3, 4):maskZero(1))
b=nn.Sequencer(nn.MaskZero(nn.FastLSTM(3,4),1))
input={}
seq1=torch.rand(5,3)
seq1[1]=torch.zeros(1,3)
seq1[2]=torch.zeros(1,3)
seq2=torch.rand(5,3)
seq2[2]=torch.zeros(1,3)
seq3=torch.rand(5,3)
table.insert(input,seq1)
table.insert(input,seq2)
table.insert(input,seq3)
print(a,b)
out1=a:forward(input)
out2=b:forward(input)
back1=a:backward(input)
back2=b:backward(input)
for i = 1,3 do
print(out1[i],out2[i])
end
