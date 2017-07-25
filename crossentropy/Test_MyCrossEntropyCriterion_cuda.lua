require 'nn'
require 'os'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'MyCrossEntropyCriterion'
for maskzero = 0,1 do
for data_include_zero = 0,1 do
--maskzero = 1
--data_include_0 = 0

sizeAverage = true 
--sizeAverage = false

--weight=torch.ones(3)
weight=torch.Tensor{0.1,0.2,0.4}

label="soft"
--label="hard"

print("=======================================================")
print(string.format("use maskzero:%d. data includes all-zero:%d.", maskzero, data_include_zero))
print(string.format("sizeAverage:%s. label_type:%s", tostring(sizeAverage),label))
print("=======================================================")


if data_include_zero == 1 then
	y_ = torch.Tensor{{-1.20397280433,-2.30258509299,-0.51082562376},{-2.30258509299,-0.22314355131,-2.30258509299},{0,0,0}} --{log0.3,log0.1,log0.6},{log0.1,log0.8,log0.1}
	y_ = torch.Tensor{{1,2,3},{3,5,1},{0,0,0}}
	y_h = torch.Tensor{3,2,1}
	y_d = torch.Tensor{{0,0,1},{0,1,0},{1,0,0}} --y_s[3] can be any tensor
	y_s = torch.Tensor{{0.1,0.3,0.6},{0.2,7,0.1},{0,0,0}}
else
	y_ = torch.Tensor{{-1.20397280433,-2.30258509299,-0.51082562376},{-2.30258509299,-0.22314355131,-2.30258509299}} --{log0.3,log0.1,log0.6},{log0.1,log0.8,log0.1}
	y_ = torch.Tensor{{1,2,3},{3,5,1}}
	y_h = torch.Tensor{3,2}
	y_d = torch.Tensor{{0,0,1},{0,1,0}}
	y_s = torch.Tensor{{0.1,0.3,0.6},{0.2,0.7,0.1}}
end
if maxzero == 1 then
	c1 = nn.MyCrossEntropyCriterion(weight)
	c1.maskZero = true 
	c1.sizeAverage = sizeAverage

	c2_t = nn.ClassNLLCriterion(weight)
	c2_t.sizeAverage = sizeAverage
        c2 = nn.MaskZeroCriterion(c2_t,1)

	c3_t = nn.MyCrossEntropyCriterionGPU(weight)
	c3_t.sizeAverage = sizeAverage
	c3 = nn.MaskZeroCriterion(c3_t,1)
else
	c1 = nn.MyCrossEntropyCriterion(weight)
	c1.sizeAverage = sizeAverage

        c2 = nn.ClassNLLCriterion(weight)
        c2.sizeAverage = sizeAverage

	c3 = nn.MyCrossEntropyCriterionGPU(weight)
	c3.sizeAverage = sizeAverage
end
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
print(o1b-o3b)
if label=="hard" then
	print(o2b-o1b,o2b-o3b)
end
--gpu
print("gpu")
c2_gpu=c2:clone()
c3_gpu=c3:clone()
c2_gpu:cuda()
c3_gpu:cuda()
o2_gpu = c2_gpu:forward(y_:cuda(),y_h:cuda())
o3_gpu = c3_gpu:forward(y_:cuda(),y_l:cuda())
print(o1-o3_gpu)
if label=="hard" then
	print(o2_gpu-o3_gpu)
end
o2b_gpu = c2_gpu:backward(y_:cuda(),y_h:cuda())
o3b_gpu = c3_gpu:backward(y_:cuda(),y_l:cuda())
print(o1b-o3b_gpu:double())
if label=="hard" then
	print(o2b_gpu-o3b_gpu)
end
end
end
