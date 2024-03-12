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

import json
import argparse
import basedai
from tqdm import tqdm
from rich.table import Table
from rich.prompt import Prompt
from .utils import (
    check_netuid_set,
    get_delegates_details,
    DelegatesDetails,
    get_computekey_wallets_for_wallet,
    get_personalkey_wallets_for_path,
    get_all_wallets_for_path,
    filter_netuids_by_registered_computekeys,
)
from . import defaults

console = basedai.__console__

import os
import basedai
from typing import List, Tuple, Optional, Dict


def _get_personalkey_wallets_for_path(path: str) -> List["basedai.wallet"]:
    try:
        wallet_names = next(os.walk(os.path.expanduser(path)))[1]
        return [basedai.wallet(path=path, name=name) for name in wallet_names]
    except StopIteration:
        # No wallet files found.
        wallets = []
    return wallets


def _get_computekey_wallets_for_wallet(wallet) -> List["basedai.wallet"]:
    computekey_wallets = []
    computekeys_path = wallet.path + "/" + wallet.name + "/computekeys"
    try:
        computekey_files = next(os.walk(os.path.expanduser(computekeys_path)))[2]
    except StopIteration:
        computekey_files = []
    for computekey_file_name in computekey_files:
        try:
            computekey_for_name = basedai.wallet(
                path=wallet.path, name=wallet.name, computekey=computekey_file_name
            )
            if (
                computekey_for_name.computekey_file.exists_on_device()
                and not computekey_for_name.computekey_file.is_encrypted()
            ):
                computekey_wallets.append(computekey_for_name)
        except Exception:
            pass
    return computekey_wallets


class InspectCommand:
    """
    Executes the ``inspect`` command, which compiles and displays a detailed report of a user's wallet pairs (personalkey, computekey) on the Basedai network.

    This report includes balance and
    staking information for both the personalkey and computekey associated with the wallet.

    Optional arguments:
        - ``all``: If set to ``True``, the command will inspect all wallets located within the specified path. If set to ``False``, the command will inspect only the wallet specified by the user.

    The command gathers data on:

    - Personalkey balance and delegated stakes.
    - Computekey stake and emissions per agent on the network.
    - Delegate names and details fetched from the network.

    The resulting table includes columns for:

    - **Personalkey**: The personalkey associated with the user's wallet.
    - **Balance**: The balance of the personalkey.
    - **Delegate**: The name of the delegate to which the personalkey has staked funds.
    - **Stake**: The amount of stake held by both the personalkey and computekey.
    - **Emission**: The emission or rewards earned from staking.
    - **Netuid**: The network unique identifier of the Brain where the computekey is active.
    - **Computekey**: The computekey associated with the agent on the network.

    Usage:
        This command can be used to inspect a single wallet or all wallets located within a
        specified path. It is useful for a comprehensive overview of a user's participation
        and performance in the Basedai network.

    Example usage::

            basedcli wallet inspect
            basedcli wallet inspect --all

    Note:
        The ``inspect`` command is for displaying information only and does not perform any
        transactions or state changes on the Basedai network. It is intended to be used as
        part of the Basedai CLI and not as a standalone function within user code.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Inspect a cold, hot pair."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            InspectCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        if cli.config.get("all", d=False) == True:
            wallets = _get_personalkey_wallets_for_path(cli.config.wallet.path)
            all_computekeys = get_all_wallets_for_path(cli.config.wallet.path)
        else:
            wallets = [basedai.wallet(config=cli.config)]
            all_computekeys = get_computekey_wallets_for_wallet(wallets[0])

        netuids = basednode.get_all_subnet_netuids()
        netuids = filter_netuids_by_registered_computekeys(
            cli, basednode, netuids, all_computekeys
        )
        basedai.logging.debug(f"Netuids to check: {netuids}")

        registered_delegate_info: Optional[
            Dict[str, DelegatesDetails]
        ] = get_delegates_details(url=basedai.__delegates_details_url__)
        if registered_delegate_info is None:
            basedai.__console__.print(
                ":warning:[yellow]Could not get delegate info from chain.[/yellow]"
            )
            registered_delegate_info = {}

        neuron_state_dict = {}
        for netuid in tqdm(netuids):
            neurons = basednode.neurons_lite(netuid)
            neuron_state_dict[netuid] = neurons if neurons != None else []

        table = Table(show_footer=True, pad_edge=False, box=None, expand=True)
        table.add_column(
            "[overline white]Personalkey", footer_style="overline white", style="bold white"
        )
        table.add_column(
            "[overline white]Balance", footer_style="overline white", style="green"
        )
        table.add_column(
            "[overline white]Delegate", footer_style="overline white", style="blue"
        )
        table.add_column(
            "[overline white]Stake", footer_style="overline white", style="green"
        )
        table.add_column(
            "[overline white]Emission", footer_style="overline white", style="green"
        )
        table.add_column(
            "[overline white]Netuid", footer_style="overline white", style="bold white"
        )
        table.add_column(
            "[overline white]Computekey", footer_style="overline white", style="yellow"
        )
        table.add_column(
            "[overline white]Stake", footer_style="overline white", style="green"
        )
        table.add_column(
            "[overline white]Emission", footer_style="overline white", style="green"
        )
        for wallet in tqdm(wallets):
            delegates: List[
                Tuple(basedai.DelegateInfo, basedai.Balance)
            ] = basednode.get_delegated(personalkey_ss58=wallet.personalkeypub.ss58_address)
            if not wallet.personalkeypub_file.exists_on_device():
                continue
            cold_balance = basednode.get_balance(wallet.personalkeypub.ss58_address)
            table.add_row(wallet.name, str(cold_balance), "", "", "", "", "", "", "")
            for dele, staked in delegates:
                if dele.computekey_ss58 in registered_delegate_info:
                    delegate_name = registered_delegate_info[dele.computekey_ss58].name
                else:
                    delegate_name = dele.computekey_ss58
                table.add_row(
                    "",
                    "",
                    str(delegate_name),
                    str(staked),
                    str(
                        dele.total_daily_return.based
                        * (staked.based / dele.total_stake.based)
                    ),
                    "",
                    "",
                    "",
                    "",
                )

            computekeys = _get_computekey_wallets_for_wallet(wallet)
            for netuid in netuids:
                for neuron in neuron_state_dict[netuid]:
                    if neuron.personalkey == wallet.personalkeypub.ss58_address:
                        computekey_name: str = ""

                        computekey_names: List[str] = [
                            wallet.computekey_str
                            for wallet in filter(
                                lambda computekey: computekey.computekey.ss58_address
                                == neuron.computekey,
                                computekeys,
                            )
                        ]
                        if len(computekey_names) > 0:
                            computekey_name = f"{computekey_names[0]}-"

                        table.add_row(
                            "",
                            "",
                            "",
                            "",
                            "",
                            str(netuid),
                            f"{computekey_name}{neuron.computekey}",
                            str(neuron.stake),
                            str(basedai.Balance.from_based(neuron.emission)),
                        )

        basedai.__console__.print(table)

    @staticmethod
    def check_config(config: "basedai.config"):
        if (
            not config.is_set("wallet.name")
            and not config.no_prompt
            and not config.get("all", d=None)
        ):
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.netuids != [] and config.netuids != None:
            if not isinstance(config.netuids, list):
                config.netuids = [int(config.netuids)]
            else:
                config.netuids = [int(netuid) for netuid in config.netuids]

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        inspect_parser = parser.add_parser(
            "inspect", help="""Inspect a wallet (personal, compute, EVM) pair."""
        )
        inspect_parser.add_argument(
            "--all",
            action="store_true",
            help="""Check all personalkey wallets.""",
            default=False,
        )
        inspect_parser.add_argument(
            "--netuids",
            dest="netuids",
            type=int,
            nargs="*",
            help="""Set the Brain netuid(s) to filter by.""",
            default=None,
        )

        basedai.wallet.add_args(inspect_parser)
        basedai.basednode.add_args(inspect_parser)
