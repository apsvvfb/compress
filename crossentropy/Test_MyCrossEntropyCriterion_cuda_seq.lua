require 'nn'
require 'os'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'MyCrossEntropyCriterion'

function CudaTabel2DoublTabel (table1)
local table2={}
for i = 1, #table1 do
        table2[i]=table1[i]:double()
end
return table2
end
function DoublTabel2CudaTabel (table1)
local table2={}
for i = 1, #table1 do
        table2[i]=table1[i]:cuda()
end
return table2
end

for maskzero = 0,1 do
for data_include_zero = 0,1 do
--maskzero = 1
--data_include_0 = 0

sizeAverage = true 
--sizeAverage = false

--weight=torch.ones(3)
weight=torch.Tensor{0.1,0.2,0.4}

--label="soft"
label="hard"

print("=======================================================")
print(string.format("use maskzero:%d. data includes all-zero:%d.", maskzero, data_include_zero))
print(string.format("sizeAverage:%s. label_type:%s", tostring(sizeAverage),label))
print("=======================================================")
if data_include_zero == 1 then
        y_1 = torch.Tensor{{1,2,3},{0,0,0},{0,0,0}}
        y_h1 = torch.Tensor{3,0,0}
        y_d1 = torch.Tensor{{0,0,1},{0,0,0},{0,0,0}}
        y_h1 = torch.Tensor{3,1,1}
        y_d1 = torch.Tensor{{0,0,1},{1,0,0},{1,0,0}}
	y_s1 = torch.Tensor{{0.1,0.2,0.7},{0,0,0},{0,0,0}}

        y_2 = torch.Tensor{{23,17,3},{2,41,7},{0,0,0}}
        y_h2 = torch.Tensor{1,2,0}
        y_d2 = torch.Tensor{{1,0,0},{0,1,0},{0,0,0}}
        y_h2 = torch.Tensor{1,2,3}
        y_d2 = torch.Tensor{{1,0,0},{0,1,0},{0,0,1}}
	y_s2 = torch.Tensor{{0.8,0.15,0.05},{0.02,0.9,0.18},{0,0,0}}

        y_3 = torch.Tensor{{11,32,1},{1,45,34},{12,2,6}}
        y_h3 = torch.Tensor{2,2,1}
        y_d3 = torch.Tensor{{0,1,0},{0,1,0},{1,0,0}}
	y_s3 = torch.Tensor{{0.25,0.7,0.05},{0.02,0.9,0.18},{0.6,0.1,0.3}}

else
        y_1 = torch.Tensor{{1,2,3},{14,12,3},{33,2,7}}
        y_h1 = torch.Tensor{3,1,1}
        y_d1 = torch.Tensor{{0,0,1},{1,0,0},{1,0,0}}
        y_s1 = torch.Tensor{{0.1,0.2,0.7},{0.6,0.2,0.2},{0.44,0.2,0.36}}

        y_2 = torch.Tensor{{23,17,3},{2,41,7},{8,23,51}}
        y_h2 = torch.Tensor{1,2,3}
        y_d2 = torch.Tensor{{1,0,0},{0,1,0},{0,0,1}}
        y_s2 = torch.Tensor{{0.8,0.15,0.05},{0.02,0.9,0.18},{0.2,0.3,0.5}}

        y_3 = torch.Tensor{{11,32,1},{1,45,34},{12,2,6}}
        y_h3 = torch.Tensor{2,2,1}
        y_d3 = torch.Tensor{{0,1,0},{0,1,0},{1,0,0}}
        y_s3 = torch.Tensor{{0.25,0.7,0.05},{0.02,0.9,0.18},{0.6,0.1,0.3}}
end
y_ = {}
y_d = {}
y_s = {}
y_h = {}
model=nn.MaskZero(nn.LogSoftMax(),1)
table.insert(y_,model:forward(y_1))
table.insert(y_d,y_d1)
table.insert(y_s,y_s1)
table.insert(y_h,y_h1)
table.insert(y_,model:forward(y_2))
table.insert(y_d,y_d2)
table.insert(y_s,y_s2)
table.insert(y_h,y_h2)
table.insert(y_,model:forward(y_3))
table.insert(y_d,y_d3)
table.insert(y_s,y_s3)
table.insert(y_h,y_h3)
seq=3
if maxzero == 1 then
	c1_single = nn.MyCrossEntropyCriterion(weight)
	c1_single.maskZero = true 
	c1_single.sizeAverage = sizeAverage

	c2_t = nn.ClassNLLCriterion(weight)
	c2_t.sizeAverage = sizeAverage
        c2_single = nn.MaskZeroCriterion(c2_t,1)

	c3_t = nn.MyCrossEntropyCriterionGPU(weight)
	c3_t.sizeAverage = sizeAverage
	c3_single = nn.MaskZeroCriterion(c3_t,1)
else
	c1_single = nn.MyCrossEntropyCriterion(weight)
	c1_single.sizeAverage = sizeAverage

        c2_single = nn.ClassNLLCriterion(weight)
        c2_single.sizeAverage = sizeAverage

	c3_single = nn.MyCrossEntropyCriterionGPU(weight)
	c3_single.sizeAverage = sizeAverage
end
c1=nn.SequencerCriterion(c1_single)
c2=nn.SequencerCriterion(c2_single)
c3=nn.SequencerCriterion(c3_single)

y_l=y_s
if label=="hard" then
	y_l=y_d
end
--cpu
print("cpu")
o1 = c1:forward(y_,y_l)
o2 = c2:forward(y_,y_h)
o3 = c3:forward(y_,y_l)
print(o1-o3)
if label=="hard" then
	print(o2-o1,o2-o3)
end
o1b = c1:backward(y_,y_l)
o2b = c2:backward(y_,y_h)
o3b = c3:backward(y_,y_l)
for i=1,seq do
	print(o1b[i]-o3b[i])
	if label=="hard" then
		print(o2b[i]-o1b[i],o2b[i]-o3b[i])
	end
end
--gpu
print("gpu")
c2_gpu=c2:clone()
c3_gpu=c3:clone()
c2_gpu:cuda()
c3_gpu:cuda()
o2_gpu = c2_gpu:forward(DoublTabel2CudaTabel(y_),DoublTabel2CudaTabel(y_h))
o3_gpu = c3_gpu:forward(DoublTabel2CudaTabel(y_),DoublTabel2CudaTabel(y_l))
print(o1-o3_gpu)
if label=="hard" then
	print(o2_gpu-o3_gpu)
end
o2b_gpu = c2_gpu:backward(DoublTabel2CudaTabel(y_),DoublTabel2CudaTabel(y_h))
o3b_gpu = c3_gpu:backward(DoublTabel2CudaTabel(y_),DoublTabel2CudaTabel(y_l))
for i=1,seq do
	print(o1b[i]-o3b_gpu[i]:double())
	if label=="hard" then
		print(o2b_gpu[i]-o3b_gpu[i])
	end
end
end
end
