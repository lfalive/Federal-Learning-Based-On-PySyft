import argparse
import sys
from datetime import datetime

import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from syft.frameworks.torch.fl import utils
from syft.workers.websocket_client import WebsocketClientWorker
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 用于记录控制台输出结果的Logger类，将控制台内容到指定文本文档中
class Logger(object):
	def __init__(self, filename='./log/default.log', stream=sys.stdout):
		self.terminal = stream
		self.log = open(filename, 'a')

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		pass


time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')  # 将时间格式化为指定格式字符串
sys.stdout = Logger(filename="./log/" + time_str + ".log")


# 自定义带时间戳的print
def mylogger(str):
	logger_time = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
	print(logger_time, end=" - ")
	print(str)


def define_and_get_arguments():
	parser = argparse.ArgumentParser(description="Run federated learning using websocket client workers.")
	parser.add_argument("--batch_size", type=int, default=64, help="batch size of the training")
	parser.add_argument("--test_batch_size", type=int, default=1000, help="batch size used for the test data")
	parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train")
	parser.add_argument("--federate_after_n_batches", type=int, default=50,
						help="number of training steps performed on each remote worker " "before averaging")
	parser.add_argument("--log_interval", type=int, default=25, help="interval of the console printing log")
	parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
	parser.add_argument("--cuda", action="store_true", help="use cuda")
	parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
	parser.add_argument("--verbose", "-v", action="store_true",
						help="if set, websocket client workers will " "be started in verbose mode")
	return parser.parse_args(args=sys.argv[1:])


args = define_and_get_arguments()
use_cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
mylogger(args)

hook = sy.TorchHook(torch)

kwargs_websocket = {"hook": hook, "verbose": args.verbose}
alice = WebsocketClientWorker(id="alice", port=8777, host='localhost', **kwargs_websocket)
bob = WebsocketClientWorker(id="bob", port=8778, host='localhost', **kwargs_websocket)
charlie = WebsocketClientWorker(id="charlie", port=8779, host='localhost', **kwargs_websocket)

workers = [alice, bob, charlie]
for worker in workers:
	mylogger(worker)

federated_train_loader = sy.FederatedDataLoader(
	datasets.MNIST(
		"./data",
		train=True,
		download=True,
		transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
	).federate(tuple(workers)),
	batch_size=args.batch_size,
	shuffle=True,
	iter_per_worker=True
)

test_loader = DataLoader(
	datasets.MNIST(
		"./data",
		train=False,
		transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
	),
	batch_size=args.test_batch_size,
	shuffle=True
)


class AlexNet(nn.Module):
	def __init__(self):
		super(AlexNet, self).__init__()

		# 由于MNIST为28x28， 而最初AlexNet的输入图片是227x227的。所以网络层数和参数需要调节
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # AlexCONV1(3,96, k=11,s=4,p=0)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # AlexPool1(k=3, s=2)
		self.relu1 = nn.ReLU()

		# self.conv2 = nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # AlexCONV2(96, 256,k=5,s=1,p=2)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # AlexPool2(k=3,s=2)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # AlexCONV3(256,384,k=3,s=1,p=1)
		# self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # AlexCONV4(384, 384, k=3,s=1,p=1)
		self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # AlexCONV5(384, 256, k=3, s=1,p=1)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # AlexPool3(k=3,s=2)
		self.relu3 = nn.ReLU()

		self.fc6 = nn.Linear(256 * 3 * 3, 1024)  # AlexFC6(256*6*6, 4096)
		self.fc7 = nn.Linear(1024, 512)  # AlexFC6(4096,4096)
		self.fc8 = nn.Linear(512, 10)  # AlexFC6(4096,1000)

	def forward(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.relu1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = self.relu2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.pool3(x)
		x = self.relu3(x)
		x = x.view(-1, 256 * 3 * 3)  # Alex: x = x.view(-1, 256*6*6)
		x = self.fc6(x)
		x = f.relu(x)
		x = self.fc7(x)
		x = f.relu(x)
		x = self.fc8(x)
		return f.log_softmax(x, dim=1)


model = AlexNet().to(device)
print(model)


def get_next_batches(fdataloader: sy.FederatedDataLoader, nr_batches: int):
	"""retrieve next nr_batches of the federated data loader and group the batches by worker

    Args:
        fdataloader (sy.FederatedDataLoader): federated data loader
        over which the function will iterate
        nr_batches (int): number of batches (per worker) to retrieve

    Returns:
        Dict[syft.workers.BaseWorker, List[batches]]

    """
	batches = {}
	for worker_id in fdataloader.workers:
		worker = fdataloader.federated_dataset.datasets[worker_id].location
		batches[worker] = []
	try:
		for i in range(nr_batches):
			next_batches = next(fdataloader)
			for worker in next_batches:
				batches[worker].append(next_batches[worker])
	except StopIteration:
		pass
	return batches


def train_on_batches(worker, batches, model_in, device, lr):
	"""Train the model on the worker on the provided batches

    Args:
        worker(syft.workers.BaseWorker): worker on which the
        training will be executed
        batches: batches of data of this worker
        model_in: machine learning model, training will be done on a copy
        device (torch.device): where to run the training
        lr: learning rate of the training steps

    Returns:
        model, loss: obtained model and loss after training

    """
	model = model_in.copy()
	optimizer = optim.SGD(model.parameters(), lr=lr)
	# optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

	model.train()
	model.send(worker)
	loss_local = False

	for batch_idx, (data, target) in enumerate(batches):
		loss_local = False
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = f.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			loss = loss.get()  # <-- NEW: get the loss back
			loss_local = True
			mylogger("Train Worker {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
				worker.id,
				batch_idx,
				len(batches),
				100.0 * batch_idx / len(batches),
				loss.item(),
			)
			)

	if not loss_local:
		loss = loss.get()  # <-- NEW: get the loss back
	model.get()  # <-- NEW: get the model back
	return model, loss


def train(model, device, federated_train_loader, lr, federate_after_n_batches, abort_after_one=False):
	model.train()

	nr_batches = federate_after_n_batches

	models = {}
	loss_values = {}

	iter(federated_train_loader)  # initialize iterators
	batches = get_next_batches(federated_train_loader, nr_batches)
	counter = 0

	while True:
		mylogger(f"Starting training round, batches [{counter}, {counter + nr_batches}]")
		data_for_all_workers = True
		for worker in batches:
			curr_batches = batches[worker]
			if curr_batches:
				models[worker], loss_values[worker] = train_on_batches(
					worker, curr_batches, model, device, lr
				)
			else:
				data_for_all_workers = False
		counter += nr_batches
		if not data_for_all_workers:
			mylogger("At least one worker ran out of data, stopping.")
			break

		model = utils.federated_avg(models)
		batches = get_next_batches(federated_train_loader, nr_batches)
		if abort_after_one:
			break
	return model


def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += f.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
			pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	accuracy = 100.0 * correct / len(test_loader.dataset)
	mylogger(
		"Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
			test_loss, correct, len(test_loader.dataset), accuracy
		)
	)


for epoch in range(1, args.epochs + 1):
	mylogger("Starting epoch {}/{}".format(epoch, args.epochs))
	model = train(model, device, federated_train_loader, args.lr, args.federate_after_n_batches, abort_after_one=False)
	test(model, device, test_loader)
