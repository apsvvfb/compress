require 'nn'
require 'os'
require 'rnn'
require 'MyCrossEntropyCriterion'
for maxzero = 0, 1 do
for data = 0, 1 do 
--maxzero = 1
--data = 0
print(maxzero,data)
sizeAverage = true 
sizeAverage = false

--weight=torch.ones(3)
weight=torch.Tensor{0.1,0.2,0.3}
weight=torch.Tensor{1,0.2,1}

y_ = {}
y_d = {}
y_s = {}

model=nn.MaskZero(nn.LogSoftMax(),1)
if data == 0 then
	y_1 = torch.Tensor{{1,2,3},{0,0,0},{0,0,0}}
	y_d1 = torch.Tensor{3,0,0}  
	y_s1 = torch.Tensor{{0,0,1},{0,0,0},{0,0,0}} 
	y_d1 = torch.Tensor{3,1,1}
        y_s1 = torch.Tensor{{0,0,1},{1,0,0},{1,0,0}}

        y_2 = torch.Tensor{{23,17,3},{2,41,7},{0,0,0}}
        y_d2 = torch.Tensor{1,2,0}  
        y_s2 = torch.Tensor{{1,0,0},{0,1,0},{0,0,0}} 
	y_d2 = torch.Tensor{1,2,3}  
        y_s2 = torch.Tensor{{1,0,0},{0,1,0},{0,0,1}}

        y_3 = torch.Tensor{{11,32,1},{1,45,34},{12,2,6}}
        y_d3 = torch.Tensor{2,2,1}  
        y_s3 = torch.Tensor{{0,1,0},{0,1,0},{1,0,0}}
else
        y_1 = torch.Tensor{{1,2,3},{14,12,3},{33,2,7}}
        y_d1 = torch.Tensor{3,1,1}
        y_s1 = torch.Tensor{{0,0,1},{1,0,0},{1,0,0}}

        y_2 = torch.Tensor{{23,17,3},{2,41,7},{8,23,51}}
        y_d2 = torch.Tensor{1,2,3}
        y_s2 = torch.Tensor{{1,0,0},{0,1,0},{0,0,1}}

        y_3 = torch.Tensor{{11,32,1},{1,45,34},{12,2,6}}
        y_d3 = torch.Tensor{2,2,1}
        y_s3 = torch.Tensor{{0,1,0},{0,1,0},{1,0,0}}
end
table.insert(y_,model:forward(y_1))
table.insert(y_d,y_d1)
table.insert(y_s,y_s1)
table.insert(y_,model:forward(y_2))
table.insert(y_d,y_d2)
table.insert(y_s,y_s2)
table.insert(y_,model:forward(y_3))
table.insert(y_d,y_d3)
table.insert(y_s,y_s3)

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
c1_seq=nn.SequencerCriterion(c1)
c3_seq=nn.SequencerCriterion(c3)

o1 = c1_seq:forward(y_,y_s)
o3 = c3_seq:forward(y_,y_d)
print(o1-o3)
o1b = c1_seq:backward(y_,y_s)
o3b = c3_seq:backward(y_,y_d)
for i=1,3 do
	print(string.format("seq:%d", i))
	print("mine")
	print(o1b[i])
	print("ori")
	print(o3b[i])
	print(string.format("%f",o1b[i][1][1]))
	os.exit()
	print(o1b[i]-o3b[i])
end
end
end
