require 'nn'
require 'rnn'
require 'os'
require 'cunn'
featdim=10
hiddenSize=5
temperature=2
numTargetClasses=21
batch=3
seq=4
model = nn.Sequencer(
        nn.Sequential()
                :add(nn.FastLSTM(featdim, hiddenSize):maskZero(1))
                :add(nn.MaskZero(nn.Linear(hiddenSize, numTargetClasses),1))
                :add(nn.MaskZero(nn.MulConstant(1/temperature),1))
                :add(nn.MaskZero(nn.LogSoftMax(),1))
)
print(model)
input={}
for i=1,seq do
	table.insert(input,torch.rand(batch,featdim))
end
out1=model:forward(input)
local m = model.modules
m[1].modules[1].modules[3]=nn.Identity()
--m[1].recurrentModule.modules[3]=nn.Sequencer(nn.Identity())
print(model)
out2=model:forward(input)
for i=1,seq do
	print(out1[i]*2-out2[i])
end

