args = []
abort_after_one = False

import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
import torch
from torchvision import datasets, transforms

import run_websocket_client as rwc

args = rwc.define_and_get_arguments(args=args)
use_cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
print(args)


hook = sy.TorchHook(torch)

kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": args.verbose}
alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
bob = WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)
charlie = WebsocketClientWorker(id="charlie", port=8779, **kwargs_websocket)

workers = [alice, bob, charlie]
print(workers)


federated_train_loader = sy.FederatedDataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ).federate(tuple(workers)),
    batch_size=args.batch_size,
    shuffle=True,
    iter_per_worker=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.test_batch_size,
    shuffle=True
)


model = rwc.Net().to(device)
print(model)


import logging
import sys
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s")
handler.setFormatter(formatter)
logger.handlers = [handler]

for epoch in range(1, args.epochs + 1):
    print("Starting epoch {}/{}".format(epoch, args.epochs))
    model = rwc.train(model, device, federated_train_loader, args.lr, args.federate_after_n_batches,
                      abort_after_one=abort_after_one)
    rwc.test(model, device, test_loader)
