# The MIT License (MIT)
# Copyright © 2024 Sean Wellington

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from rich.console import Console
from rich.traceback import install

# Install and apply nest asyncio to allow the async functions
# to run in a .ipynb
import nest_asyncio

nest_asyncio.apply()

# BasedAI code and protocol version.
__version__ = "1.0.5"

version_split = __version__.split(".")
__version_as_int__ = (
    (100 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)
__new_signature_version__ = 360

# Rich console.
__console__ = Console()
__use_console__ = True

# Remove overdue locals in debug training.
install(show_locals=False)


def turn_console_off():
    global __use_console__
    global __console__
    from io import StringIO

    __use_console__ = False
    __console__ = Console(file=StringIO(), stderr=False)


def turn_console_on():
    global __use_console__
    global __console__
    __use_console__ = True
    __console__ = Console()


turn_console_off()


# Logging helpers.
def trace(on: bool = True):
    logging.set_trace(on)


def debug(on: bool = True):
    logging.set_debug(on)


# Substrate chain block time (seconds).
__blocktime__ = 12

# Pip address for versioning
__pipaddress__ = "https://pypi.org/pypi/basedai/json"

# Raw github url for delegates registry file
__delegates_details_url__: str = "https://raw.githubusercontent.com/delegates.json"

# Substrate ss58_format
__ss58_format__ = 42

# Wallet ss58 address length
__ss58_address_length__ = 48

__networks__ = ["local", "prometheus", "test", "archive"]

__prometheus_entrypoint__ = "wss://prometheus.basedaibridge.com:443"

__prometheus_test_entrypoint__ = "wss://test.prometheus.basedaibridge.com:443/"

__archive_entrypoint__ = "wss://archive.basedaibridge.com:443/"

# Needs to use wss://
__bellagene_entrypoint__ = "wss://prometheus.basedaibridge.com:443"

__local_entrypoint__ = "ws://127.0.0.1:9944"

__basedai_symbol__: str = chr(0x1D539)

__betaai_symbol__: str = chr(0x03B2)

## Must all be polkadotjs explorer urls
__network_explorer_map__ = {
    "mainnet": {
        "local": "https://polkadot.js.org/apps/?rpc=wss%3A%2F%2Fprometheus.basedaibridge.com%3A443#/explorer",
        "endpoint": "https://polkadot.js.org/apps/?rpc=wss%3A%2F%2Fprometheus.basedaibridge.com%3A443#/explorer",
        "prometheus": "https://polkadot.js.org/apps/?rpc=wss%3A%2F%2Fprometheus.basedaibridge.com%3A443#/explorer",
    },
    "basedaiscan": {
        "local": "https://x.basedaiscan.com",
        "endpoint": "https://x.basedaiscan.com",
        "prometheus": "https://x.basedaiscan.com",
    },
}

# --- Type Registry ---
__type_registry__ = {
    "types": {
        "Balance": "u64",  # Need to override default u128
    },
    "runtime_api": {
        "NeuronInfoRuntimeApi": {
            "methods": {
                "get_neuron_lite": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                        {
                            "name": "uid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                },
                "get_neurons_lite": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                },
            }
        },
        "StakeInfoRuntimeApi": {
            "methods": {
                "get_stake_info_for_personalkey": {
                    "params": [
                        {
                            "name": "personalkey_account_vec",
                            "type": "Vec<u8>",
                        },
                    ],
                    "type": "Vec<u8>",
                },
                "get_stake_info_for_personalkeys": {
                    "params": [
                        {
                            "name": "personalkey_account_vecs",
                            "type": "Vec<Vec<u8>>",
                        },
                    ],
                    "type": "Vec<u8>",
                },
            },
        },
        "ValidatorIPRuntimeApi": {
            "methods": {
                "get_associated_validator_ip_info_for_subnet": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                },
            },
        },
        "SubnetInfoRuntimeApi": {
            "methods": {
                "get_subnet_hyperparams": {
                    "params": [
                        {
                            "name": "netuid",
                            "type": "u16",
                        },
                    ],
                    "type": "Vec<u8>",
                }
            }
        },
        "SubnetRegistrationRuntimeApi": {
            "methods": {"get_network_registration_cost": {"params": [], "type": "u64"}}
        },
    },
}

from .errors import *

from substrateinterface import Keypair as Keypair
from .config import *
from .keyfile import *
from .wallet import *

from .utils import *
from .utils.balance import Balance as Balance
from .chain_data import *
from .basednode import basednode as basednode
from .cli import cli as cli, COMMANDS as ALL_COMMANDS
from .btlogging import logging as logging
from .stem import stem as stem
from .threadpool import PriorityThreadPoolExecutor as PriorityThreadPoolExecutor

from .brainresponder import *
from .stream import *
from .tensor import *
from .brainport import brainport as brainport
from .brainrequester import brainrequester as brainrequester


configs = [
    brainport.config(),
    basednode.config(),
    PriorityThreadPoolExecutor.config(),
    wallet.config(),
    logging.config(),
]
defaults = config.merge_all(configs)
