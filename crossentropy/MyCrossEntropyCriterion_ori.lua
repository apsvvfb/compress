require 'nn'
require 'os'
local MyCrossEntropyCriterion, parent = torch.class('nn.MyCrossEntropyCriterion', 'nn.Criterion')

function MyCrossEntropyCriterion:__init(weights, sizeAverage, maskZero)
	parent.__init(self)
	self.sizeAverage = true
	if sizeAverage ~= nil then
		self.sizeAverage = sizeAverage
	end
	self.maskZero = false
	if maskZero ~= nil then
		self.maskZero = maskZero
	end

	if weights then
		assert(weights:dim() == 1, "weights input should be 1-D Tensor")
		self.weights = weights
	end

	self.output_tensor = torch.zeros(1)
	self.total_weight_tensor = torch.ones(1)
	self.target = torch.zeros(1):long()
end

function MyCrossEntropyCriterion:updateOutput(input, target)
	local temp
	if input:dim() == 2 then --batchsize > 1
		self.out = torch.zeros(1,input:size()[2])
		self.batchsize = input:size()[1]
		for i = 1, self.batchsize do
			temp = target[i]:clone()
			temp:cmul(input[i])
			self.out:add(temp)		
		end
	elseif input:dim() == 1 then --batchsize=1
		self.batchsize = 1
		temp = target:clone()
                temp:cmul(input)
                self.out = temp
	end

	if (self.weights) then
		self.out:cmul(self.weights)
	end
	
	if self.sizeAverage then
		local divnum = target:clone()
		local mask = torch.ones(input:size())
		if self.maskZero then
			local rows = torch.sum(input, 2)
			local rowid = rows:ne(0) --:nonzero()[{ {},{1} }]
			--print(rows,rowid)
			rowid = torch.repeatTensor(rowid,1,input:size()[2])
			mask = torch.DoubleTensor()
			mask:resize(rowid:size()):copy(rowid)
		end
		--print(mask)
		if (self.weights) then
			weight_rep = torch.repeatTensor(self.weights, self.batchsize, 1)	
			divnum:cmul(weight_rep)
		end
		divnum:cmul(mask)
		self.divnum = divnum:sum()
		self.out:div(self.divnum)
	end

	self.output = -torch.sum(self.out)
	
	return self.output
end
function MyCrossEntropyCriterion:updateGradInput(input, target)
	--self.gradInput = -torch.cdiv(target, input)
	self.gradInput = -target
	if (self.weights) then
		weight_rep = torch.repeatTensor(self.weights, self.batchsize, 1)
		self.gradInput:cmul(weight_rep)
	end
	if self.sizeAverage then
                self.gradInput:div(self.divnum)
        end
	return self.gradInput
end

return MyCrossEntropyCriterion
