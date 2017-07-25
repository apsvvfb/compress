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
end

function MyCrossEntropyCriterion:updateOutput(input, target)
	self.mask = torch.ones(input:size())
	if self.maskZero then
		local rows = torch.sum(input, 2)
		local rowid = rows:ne(0) --:nonzero()[{ {},{1} }]
		--print(rows,rowid)
		rowid = torch.repeatTensor(rowid,1,input:size()[2])
		self.mask = torch.DoubleTensor()
		self.mask:resize(rowid:size()):copy(rowid)
		target:cmul(self.mask)
		if torch.sum(self.mask) == 0 then
			print("torch.sum(self.mask) == 0 !")
			os.exit()
		end
	end
	local temp
	if input:dim() == 2 then --batchsize > 1
		self.out = torch.zeros(1,input:size()[2])
		self.batchsize = input:size()[1]
		for i = 1, self.batchsize do
			temp = target[i]:clone()
			temp:cmul(input[i])
			if (self.weights) then
				temp:cmul(self.weights)
			end
			self.out:add(temp)		
		end
	elseif input:dim() == 1 then --batchsize=1
		self.batchsize = 1
		temp = target:clone()
                temp:cmul(input)
                self.out = temp
	end
	if self.sizeAverage then
		local divnum = target:clone()
		if (self.weights) then
			weight_rep = torch.repeatTensor(self.weights, self.batchsize, 1)	
			divnum:cmul(weight_rep)
		end
		divnum:cmul(self.mask)
		self.divnum = divnum:sum()
		if self.divnum == 0 then
			print("self.divnum==0!")
			os.exit()
		else
			self.out:div(self.divnum)
		end
	end

	self.output = -torch.sum(self.out)
	
	return self.output
end
function MyCrossEntropyCriterion:updateGradInput(input, target)
	--self.gradInput = -torch.cdiv(target, input)
	self.gradInput = -target
	if self.maskZero then
		self.gradInput:cmul(self.mask)
	end
	if (self.weights) then
		weight_rep = torch.repeatTensor(self.weights, self.batchsize, 1)
		self.gradInput:cmul(weight_rep)
	end
	if self.sizeAverage then
                self.gradInput:div(self.divnum)
        end
	--print("MyCriterion_backward memory: " .. collectgarbage("count") .. "KB")
	return self.gradInput
end

return MyCrossEntropyCriterion
