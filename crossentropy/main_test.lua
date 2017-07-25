require 'rnn'
require 'os'
require 'nn'
require 'optim'
require 'MyCrossEntropyCriterion'

printout=1
debug=0
rnn= nn.Sequential()
      :add(nn.LSTM(3,10))
      :add(nn.Linear(10,2))
      :add(nn.LogSoftMax())
params, gradParams = rnn:getParameters()
--inputs = {torch.randn(2,3), torch.randn(2,3), torch.randn(2,3)}
criterion = nn.MyCrossEntropyCriterion(weight)
inputs = {torch.Tensor{{1,5,2},{2,5,2}},torch.Tensor{{1,5,2},{2,5,2}},torch.Tensor{{1,5,2},{2,5,2}}}
targ =torch.Tensor{{2,3},{1,5}}
targets = {targ, targ, targ}
nStep=3
------------------------------------------ single step
-------------------------optim
sgd_params = {
   learningRate = 0.01,
}
outputs, loss = {}, 0
for step=1,3 do
   outputs[step] = rnn:forward(inputs[step])
print(outputs)
print(targets)
   loss = loss + criterion:forward(outputs[step], targets[step])
end
gradOutputs, gradInputs = {}, {}
rnn:zeroGradParameters()
for step=3,1,-1 do
  gradOutputs[step] = criterion:backward(outputs[step], targets[step])
  gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
end
feval = function(params_new)
        -- copy the weight if are changed
        if params ~= params_new then
                params:copy(params_new)
        end
        -- select a training batch
        --local inputs, targets = nextBatch()

        -- reset gradients (gradients are always accumulated, to accommodate
        -- batch methods)
        --dl_dx:zero()

        -- evaluate the loss function and its derivative with respect to x, given a mini batch

        --local outputs = seq:forward(inputs)
        --local loss_x = criterion:forward(outputs, targets)
        --local gradOutputs = criterion:backward(outputs, targets)
        --local gradInputs = seq:backward(inputs, gradOutputs)
        return loss, gradParams
end
_, fs = optim.sgd(feval,params, sgd_params)
rnn:forget()
rnn:zeroGradParameters()

