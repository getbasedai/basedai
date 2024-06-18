# The MIT License (MIT)
# Copyright © 2023 Based Labs

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from munch import Munch, munchify

defaults: Munch = munchify(
    {
        "netuid": 1,
        "basednode": {"network": "prometheus", "chain_endpoint": None, "_mock": False},
        "pow_register": {
            "num_processes": None,
            "update_interval": 50000,
            "output_in_place": True,
            "verbose": False,
            "cuda": {"dev_id": [0], "use_cuda": False, "tpb": 256},
        },
        "brainport": {
            "port": 8091,
            "ip": "[::]",
            "external_port": None,
            "external_ip": None,
            "max_workers": 10,
            "maximum_concurrent_rpcs": 400,
        },
        "priority": {"max_workers": 5, "maxsize": 10},
        "prometheus": {"port": 7091, "level": "INFO"},
        "wallet": {
            "name": "default",
            "computekey": "default",
            "path": "~/.basedai/wallets/",
        },
        "dataset": {
            "batch_size": 10,
            "block_size": 20,
            "num_workers": 0,
            "dataset_names": "default",
            "data_dir": "~/.basedai/data/",
            "save_dataset": False,
            "max_datasets": 3,
            "num_batches": 100,
        },
        "logging": {
            "debug": False,
            "trace": False,
            "record_log": False,
            "logging_dir": "~/.basedai/miners",
        },
    }
)

from .brainstore import BrainStoreListCommand
from .core import (CoreGetWeightsCommand, CoreList, CoreMemorizeCommand,
                   CoreSetDecreaseCommand, CoreSetIncreaseCommand,
                   CoreSetWeightsCommand)
from .delegates import (DelegateStakeCommand, DelegateUnstakeCommand,
                        ListDelegatesCommand, NominateCommand,
                        PortfolioCommand)
from .gigabrain import (GigaBrainsCommand, GigaBrainsMemorizeCommand,
                        GigaBrainsResignCommand, ProposalsCommand,
                        ShowVotesCommand, VoteCommand)
from .inspect import InspectCommand
from .list import ListCommand
from .memorize import (MemorizeCommand, PowMemorizeCommand, RunFaucetCommand,
                       SwapComputekeyCommand)
from .network import (BrainGetParametersCommand, BrainListCommand,
                      BrainParametersCommand, BrainSetParametersCommand,
                      LinkBrainCommand)
from .overview import OverviewCommand
from .stake import StakeCommand, StakeShow
from .stem import StemCommand
from .transfer import TransferCommand
from .unstake import UnStakeCommand
from .wallets import (GetWalletHistoryCommand, NewComputekeyCommand,
                      NewPersonalkeyCommand, RegenComputekeyCommand,
                      RegenPersonalkeyCommand, RegenPersonalkeypubCommand,
                      WalletBalanceCommand, WalletCreateCommand)

# from .identity import GetIdentityCommand, SetIdentityCommand

__all__ = [
    "defaults",
    "StakeCommand",
    "StakeShow",
    "UnStakeCommand",
    "OverviewCommand",
    "PowMemorizeCommand",
    "MemorizeCommand",
    "RunFaucetCommand",
    "SwapComputekeyCommand",
    "NominateCommand",
    "ListDelegatesCommand",
    "DelegateStakeCommand",
    "DelegateUnstakeCommand",
    "PortfolioCommand",
    "NewPersonalkeyCommand",
    "NewComputekeyCommand",
    "RegenPersonalkeyCommand",
    "RegenPersonalkeypubCommand",
    "RegenComputekeyCommand",
    "WalletCreateCommand",
    "WalletBalanceCommand",
    "GetWalletHistoryCommand",
    "TransferCommand",
    "InspectCommand",
    "StemCommand",
    "ListCommand",
    "GigaBrainsCommand",
    "ProposalsCommand",
    "ShowVotesCommand",
    "GigaBrainsMemorizeCommand",
    "GigaBrainsResignCommand",
    "VoteCommand",
    "LinkBrainCommand",
    "BrainListCommand",
    "BrainParametersCommand",
    "BrainSetParametersCommand",
    "BrainGetParametersCommand",
    "CoreMemorizeCommand",
    "CoreList",
    "CoreSetWeightsCommand",
    "CoreGetWeightsCommand",
    "CoreSetIncreaseCommand",
    "CoreSetDecreaseCommand",
    "BrainStoreListCommand",
]
