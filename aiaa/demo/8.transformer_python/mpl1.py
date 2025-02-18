import torch
import torch.nn as nn
import torch.optim as optim
import random



class mpl(nn.Module):
	def __init__(self):
		super(mpl, self).__init__()
		self.fc1 = nn.Linear(2, 10)
		self.fc1.dtype = torch.float32
		self.fc2 = nn.Linear(10, 10)
		self.fc2.dtype = torch.float32
		self.fc3 = nn.Linear(10, 2)
		self.fc3.dtype = torch.float32
		self.optimizer = optim.Adam(self.parameters(), lr=0.001)
		self.criterion = torch.nn.MSELoss()
		self.dropout = nn.Dropout(0.2)

	def forward(self, input):
		output = self.fc1(input)
		output = self.dropout(output)
		output = self.fc2(output)
		output = self.dropout(output)
		output = self.fc3(output)
		return output

	def train(self, input, result):
		self.optimizer.zero_grad()
		output = self.forward(input)
		loss = self.criterion(output, result)
		loss.backward()
		self.optimizer.step()
		return loss

	def eval(self, input):
		output = self.forward(input)
		return output



t = mpl()

for i in range(5000):
	x = random.randint(0, 100)
	y = random.randint(0, 100)
	result = torch.tensor([x + y, x * y], dtype=torch.float32)

	t.train(torch.tensor([x, y], dtype=torch.float32), result)


print(t.eval(torch.tensor([30,500], dtype=torch.float32)))
