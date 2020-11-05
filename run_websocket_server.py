import argparse
import logging
import os
from multiprocessing import Process

import syft as sy
import torch
from syft.workers.websocket_server import WebsocketServerWorker


def start_proc(participant, kwargs):  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    def target():
        server = participant(**kwargs)
        server.start()

    p = Process(target=target)
    p.start()
    return p


if __name__ == "__main__":

    # Logging setup
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d, p:%(process)d) - %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("run_websocket_server")
    logger.setLevel(level=logging.DEBUG)

    # Parse args
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="port number of the websocket server worker, e.g. --port 8777",
    )
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    parser.add_argument(
        "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help=(
            "if set, websocket server worker will load "
            "the test dataset instead of the training dataset"
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""if set, websocket server worker will be started in verbose mode""",
    )
    parser.add_argument(
        "--notebook",
        type=str,
        default="normal",
        help=(
            "can run websocket server for websockets examples of mnist/mnist-parallel or "
            "pen_testing/steal_data_over_sockets. Type 'mnist' for starting server "
            "for websockets-example-MNIST, `mnist-parallel` for websockets-example-MNIST-parallel "
            "and 'steal_data' for pen_tesing stealing data over sockets"
        ),
    )
    parser.add_argument("--pytest_testing", action="store_true", help="""Used for pytest testing""")
    args = parser.parse_args()

    # Hook and start server
    hook = sy.TorchHook(torch)

    # server = start_proc(WebsocketServerWorker, kwargs)
    kwargs = {
        "id": args.id,
        "host": args.host,
        "port": args.port,
        "hook": hook,
        "verbose": args.verbose,
    }
    if os.name != "nt":
        server = start_proc(WebsocketServerWorker, kwargs)
    else:
        server = WebsocketServerWorker(**kwargs)
        server.start()
