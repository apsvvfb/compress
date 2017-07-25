require 'rnn'
require 'nn'
model = nn.LSTM(10, 20)
m=model.modules[1]
print(m)
