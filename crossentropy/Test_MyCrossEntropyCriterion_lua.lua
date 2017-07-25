require 'nn'
require 'os'
require 'rnn'
require 'MyCrossEntropyCriterion'
--for maxzero = 0,1 do
--for data = 0,1 do
maxzero = 0
data =  1
sizeAverage = true 
--weight=torch.ones(3)
weight=torch.Tensor{0.1,0.2,0.4}

if data == 0 then
	y_ = torch.Tensor{{-1.20397280433,-2.30258509299,-0.51082562376},{-2.30258509299,-0.22314355131,-2.30258509299},{0,0,0}} --{log0.3,log0.1,log0.6},{log0.1,log0.8,log0.1}
	y_ = torch.Tensor{{1,2,3},{3,5,1},{0,0,0}}
	y_d = torch.Tensor{3,2,1}  --y_d[3] can be any number(1<=y_d[3]<=3)
	y_s = torch.Tensor{{0,0,1},{0,1,0},{1,0,0}} --y_s[3] can be any tensor
	--y_s = torch.Tensor{{0.1,0.2,0.7},{0.09,0.8,0.11},{0.5,0.2,0.3}}
else
	y_ = torch.Tensor{{-1.20397280433,-2.30258509299,-0.51082562376},{-2.30258509299,-0.22314355131,-2.30258509299}} --{log0.3,log0.1,log0.6},{log0.1,log0.8,log0.1}
	y_ = torch.Tensor{{1,2,3},{3,5,1}}
	y_d = torch.Tensor{3,2}
	--y_s = torch.Tensor{{0,0,1},{0,1,0}}
	y_s = torch.Tensor{{0.1,0.2,0.7},{0.2,0.6,0.2}}
end
print("data")
print(y_)
print("hard_targ, soft_targ")
print(y_d,y_s)
if maxzero == 1 then
	c1 = nn.MyCrossEntropyCriterion(weight)
	c1.maskZero = true 
	c1.sizeAverage = sizeAverage
	c3_t = nn.ClassNLLCriterion(weight)
	c3_t.sizeAverage = sizeAverage
	c3 = nn.MaskZeroCriterion(c3_t,1)
else
	c1 = nn.MyCrossEntropyCriterion(weight)
	c1.sizeAverage = sizeAverage
	c3 = nn.ClassNLLCriterion(weight)
	c3.sizeAverage = sizeAverage
end

o1 = c1:forward(y_,y_s)
o3 = c3:forward(y_,y_d)
print(o1)
--print(o1-o3)
o1b = c1:backward(y_,y_s)
o3b = c3:backward(y_,y_d)
print(o1b)
--print(o1b-o3b)
--end
--end
