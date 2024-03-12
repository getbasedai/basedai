# The MIT License (MIT)
# Copyright © 2024 Sean Wellington

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# 夜を吸い込み、目覚める、
# 古の光を宿した瞳。
# 大地の血管を流れるその本質、
# 宇宙だけが知る謎。

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import argparse
import basedai
from tqdm import tqdm
from rich.prompt import Confirm, Prompt
from basedai.utils.balance import Balance
from typing import List, Union, Optional, Dict, Tuple
from .utils import get_computekey_wallets_for_wallet
from . import defaults

console = basedai.__console__


class StakeCommand:
    """
    Executes the ``add`` command to stake tokens to one or more computekeys from a user's personalkey on the Basedai network.

    This command is used to allocate tokens to different computekeys, securing their position and influence on the network.

    Usage:
        Users can specify the amount to stake, the computekeys to stake to (either by name or ``SS58`` address), and whether to stake to all computekeys. The command checks for sufficient balance and computekey registration
        before proceeding with the staking process.

    Optional arguments:
        - ``--all`` (bool): When set, stakes all available tokens from the personalkey.
        - ``--uid`` (int): The unique identifier of the neuron to which the stake is to be added.
        - ``--amount`` (float): The amount of BASED tokens to stake.
        - ``--max_stake`` (float): Sets the maximum amount of BASED to have staked in each computekey.
        - ``--computekeys`` (list): Specifies computekeys by name or SS58 address to stake to.
        - ``--all_computekeys`` (bool): When set, stakes to all computekeys associated with the wallet, excluding any specified in --computekeys.

    The command prompts for confirmation before executing the staking operation.

    Example usage::

        basedcli stake add --amount 100 --wallet.name <my_wallet> --wallet.computekey <my_computekey>

    Note:
        This command is critical for users who wish to distribute their stakes among different neurons (computekeys) on the network.
        It allows for a strategic allocation of tokens to enhance network participation and influence.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Stake token of amount to computekey(s)."""
        try:
            config = cli.config.copy()
            basednode: "basedai.basednode" = basedai.basednode(
                config=config, log_verbose=False
            )
            StakeCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Stake token of amount to computekey(s)."""
        config = cli.config.copy()
        wallet = basedai.wallet(config=config)

        # Get the computekey_names (if any) and the computekey_ss58s.
        computekeys_to_stake_to: List[Tuple[Optional[str], str]] = []
        if config.get("all_computekeys"):
            # Stake to all computekeys.
            all_computekeys: List[basedai.wallet] = get_computekey_wallets_for_wallet(
                wallet=wallet
            )
            # Get the computekeys to exclude. (d)efault to no exclusions.
            computekeys_to_exclude: List[str] = cli.config.get("computekeys", d=[])
            # Exclude computekeys that are specified.
            computekeys_to_stake_to = [
                (wallet.computekey_str, wallet.computekey.ss58_address)
                for wallet in all_computekeys
                if wallet.computekey_str not in computekeys_to_exclude
            ]  # definitely wallets

        elif config.get("computekeys"):
            # Stake to specific computekeys.
            for computekey_ss58_or_computekey_name in config.get("computekeys"):
                if basedai.utils.is_valid_ss58_address(computekey_ss58_or_computekey_name):
                    # If the computekey is a valid ss58 address, we add it to the list.
                    computekeys_to_stake_to.append((None, computekey_ss58_or_computekey_name))
                else:
                    # If the computekey is not a valid ss58 address, we assume it is a computekey name.
                    #  We then get the computekey from the wallet and add it to the list.
                    wallet_ = basedai.wallet(
                        config=config, computekey=computekey_ss58_or_computekey_name
                    )
                    computekeys_to_stake_to.append(
                        (wallet_.computekey_str, wallet_.computekey.ss58_address)
                    )
        elif config.wallet.get("computekey"):
            # Only config.wallet.computekey is specified.
            #  so we stake to that single computekey.
            computekey_ss58_or_name = config.wallet.get("computekey")
            if basedai.utils.is_valid_ss58_address(computekey_ss58_or_name):
                computekeys_to_stake_to = [(None, computekey_ss58_or_name)]
            else:
                # Computekey is not a valid ss58 address, so we assume it is a computekey name.
                wallet_ = basedai.wallet(config=config, computekey=computekey_ss58_or_name)
                computekeys_to_stake_to = [
                    (wallet_.computekey_str, wallet_.computekey.ss58_address)
                ]
        else:
            # Only config.wallet.computekey is specified.
            #  so we stake to that single computekey.
            assert config.wallet.computekey is not None
            computekeys_to_stake_to = [
                (None, basedai.wallet(config=config).computekey.ss58_address)
            ]

        # Get personalkey balance
        wallet_balance: Balance = basednode.get_balance(wallet.personalkeypub.ss58_address)
        final_computekeys: List[Tuple[str, str]] = []
        final_amounts: List[Union[float, Balance]] = []
        for computekey in tqdm(computekeys_to_stake_to):
            computekey: Tuple[Optional[str], str]  # (computekey_name (or None), computekey_ss58)
            if not basednode.is_computekey_registered_any(computekey_ss58=computekey[1]):
                # Computekey is not registered.
                if len(computekeys_to_stake_to) == 1:
                    # Only one computekey, error
                    basedai.__console__.print(
                        f"[red]Computekey [bold]{computekey[1]}[/bold] is not registered. Aborting.[/red]"
                    )
                    return None
                else:
                    # Otherwise, print warning and skip
                    basedai.__console__.print(
                        f"[yellow]Computekey [bold]{computekey[1]}[/bold] is not registered. Skipping.[/yellow]"
                    )
                    continue

            stake_amount_based: float = config.get("amount")
            if config.get("max_stake"):
                # Get the current stake of the computekey from this personalkey.
                computekey_stake: Balance = basednode.get_stake_for_personalkey_and_computekey(
                    computekey_ss58=computekey[1], personalkey_ss58=wallet.personalkeypub.ss58_address
                )
                stake_amount_based: float = config.get("max_stake") - computekey_stake.based

                # If the max_stake is greater than the current wallet balance, stake the entire balance.
                stake_amount_based: float = min(stake_amount_based, wallet_balance.based)
                if (
                    stake_amount_based <= 0.00001
                ):  # Threshold because of fees, might create a loop otherwise
                    # Skip computekey if max_stake is less than current stake.
                    continue
                wallet_balance = Balance.from_based(wallet_balance.based - stake_amount_based)

                if wallet_balance.based < 0:
                    # No more balance to stake.
                    break

            final_amounts.append(stake_amount_based)
            final_computekeys.append(computekey)  # add both the name and the ss58 address.

        if len(final_computekeys) == 0:
            # No computekeys to stake to.
            basedai.__console__.print(
                "Not enough balance to stake to any computekeys or max_stake is less than current stake."
            )
            return None

        # Ask to stake
        if not config.no_prompt:
            if not Confirm.ask(
                f"Do you want to stake to the following keys from {wallet.name}:\n"
                + "".join(
                    [
                        f"    [bold white]- {computekey[0] + ':' if computekey[0] else ''}{computekey[1]}: {f'{amount} {basedai.__basedai_symbol__}' if amount else 'All'}[/bold white]\n"
                        for computekey, amount in zip(final_computekeys, final_amounts)
                    ]
                )
            ):
                return None

        if len(final_computekeys) == 1:
            # do regular stake
            return basednode.add_stake(
                wallet=wallet,
                computekey_ss58=final_computekeys[0][1],
                amount=None if config.get("stake_all") else final_amounts[0],
                wait_for_inclusion=True,
                prompt=not config.no_prompt,
            )

        basednode.add_stake_multiple(
            wallet=wallet,
            computekey_ss58s=[computekey_ss58 for _, computekey_ss58 in final_computekeys],
            amounts=None if config.get("stake_all") else final_amounts,
            wait_for_inclusion=True,
            prompt=False,
        )

    @classmethod
    def check_config(cls, config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if (
            not config.is_set("wallet.computekey")
            and not config.no_prompt
            and not config.wallet.get("all_computekeys")
            and not config.wallet.get("computekeys")
        ):
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)

        # Get amount.
        if (
            not config.get("amount")
            and not config.get("stake_all")
            and not config.get("max_stake")
        ):
            if not Confirm.ask(
                "Stake all Based from account: [bold]'{}'[/bold]?".format(
                    config.wallet.get("name", defaults.wallet.name)
                )
            ):
                amount = Prompt.ask("Enter Based amount to stake")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(
                        ":cross_mark:[red]Invalid Based amount[/red] [bold white]{}[/bold white]".format(
                            amount
                        )
                    )
                    sys.exit()
            else:
                config.stake_all = True

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        stake_parser = parser.add_parser(
            "add", help="""Add stake to your computekey accounts from your personalkey."""
        )
        stake_parser.add_argument("--all", dest="stake_all", action="store_true")
        stake_parser.add_argument("--uid", dest="uid", type=int, required=False)
        stake_parser.add_argument("--amount", dest="amount", type=float, required=False)
        stake_parser.add_argument(
            "--max_stake",
            dest="max_stake",
            type=float,
            required=False,
            action="store",
            default=None,
            help="""Specify the maximum amount of Based to have staked in each computekey.""",
        )
        stake_parser.add_argument(
            "--computekeys",
            "--exclude_computekeys",
            "--wallet.computekeys",
            "--wallet.exclude_computekeys",
            required=False,
            action="store",
            default=[],
            type=str,
            nargs="*",
            help="""Specify the computekeys by name or ss58 address. (e.g. hk1 hk2 hk3)""",
        )
        stake_parser.add_argument(
            "--all_computekeys",
            "--wallet.all_computekeys",
            required=False,
            action="store_true",
            default=False,
            help="""To specify all computekeys. Specifying computekeys will exclude them from this all.""",
        )
        basedai.wallet.add_args(stake_parser)
        basedai.basednode.add_args(stake_parser)


### Stake list.
import json
import argparse
import basedai
from tqdm import tqdm
from rich.table import Table
from rich.prompt import Prompt
from typing import Dict, Union, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from .utils import check_netuid_set, get_delegates_details, DelegatesDetails
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


class StakeShow:
    """
    Executes the ``show`` command to list all stake accounts associated with a user's wallet on the Basedai network.

    This command provides a comprehensive view of the stakes associated with both computekeys and delegates linked to the user's personalkey.

    Usage:
        The command lists all stake accounts for a specified wallet or all wallets in the user's configuration directory.
        It displays the personalkey, balance, account details (computekey/delegate name), stake amount, and the rate of return.

    Optional arguments:
        - ``--all`` (bool): When set, the command checks all personalkey wallets instead of just the specified wallet.

    The command compiles a table showing:

    - Personalkey: The personalkey associated with the wallet.
    - Balance: The balance of the personalkey.
    - Account: The name of the computekey or delegate.
    - Stake: The amount of BASED staked to the computekey or delegate.
    - Rate: The rate of return on the stake, typically shown in BASED per day.

    Example usage::

        basedcli stake show --all

    Note:
        This command is essential for users who wish to monitor their stake distribution and returns across various accounts on the Basedai network.
        It provides a clear and detailed overview of the user's staking activities.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Show all stake accounts."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            StakeShow._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Show all stake accounts."""
        if cli.config.get("all", d=False) == True:
            wallets = _get_personalkey_wallets_for_path(cli.config.wallet.path)
        else:
            wallets = [basedai.wallet(config=cli.config)]
        registered_delegate_info: Optional[
            Dict[str, DelegatesDetails]
        ] = get_delegates_details(url=basedai.__delegates_details_url__)

        def get_stake_accounts(
            wallet, basednode
        ) -> Dict[str, Dict[str, Union[str, Balance]]]:
            """Get stake account details for the given wallet.

            Args:
                wallet: The wallet object to fetch the stake account details for.

            Returns:
                A dictionary mapping SS58 addresses to their respective stake account details.
            """

            wallet_stake_accounts = {}

            # Get this wallet's personalkey balance.
            cold_balance = basednode.get_balance(wallet.personalkeypub.ss58_address)

            # Populate the stake accounts with local computekeys data.
            wallet_stake_accounts.update(get_stakes_from_computekeys(basednode, wallet))

            # Populate the stake accounts with delegations data.
            wallet_stake_accounts.update(get_stakes_from_delegates(basednode, wallet))

            return {
                "name": wallet.name,
                "balance": cold_balance,
                "accounts": wallet_stake_accounts,
            }

        def get_stakes_from_computekeys(
            basednode, wallet
        ) -> Dict[str, Dict[str, Union[str, Balance]]]:
            """Get stakes from computekeys for the provided wallet.

            Args:
                wallet: The wallet object to fetch the stakes for.

            Returns:
                A dictionary of stakes related to computekeys.
            """
            computekeys = get_computekey_wallets_for_wallet(wallet)
            stakes = {}
            for hot in computekeys:
                emission = sum(
                    [
                        n.emission
                        for n in basednode.get_all_neurons_for_pubkey(
                            hot.computekey.ss58_address
                        )
                    ]
                )
                computekey_stake = basednode.get_stake_for_personalkey_and_computekey(
                    computekey_ss58=hot.computekey.ss58_address,
                    personalkey_ss58=wallet.personalkeypub.ss58_address,
                )
                stakes[hot.computekey.ss58_address] = {
                    "name": hot.computekey_str,
                    "stake": computekey_stake,
                    "rate": emission,
                }
            return stakes

        def get_stakes_from_delegates(
            basednode, wallet
        ) -> Dict[str, Dict[str, Union[str, Balance]]]:
            """Fetch stakes for the provided wallet.

            Args:
                wallet: The wallet object to fetch the stakes for.

            Returns:
                A dictionary of stakes related to delegates.
            """
            delegates = basednode.get_delegated(
                personalkey_ss58=wallet.personalkeypub.ss58_address
            )
            stakes = {}
            for dele, staked in delegates:
                for nom in dele.nominators:
                    if nom[0] == wallet.personalkeypub.ss58_address:
                        delegate_name = (
                            registered_delegate_info[dele.computekey_ss58].name
                            if dele.computekey_ss58 in registered_delegate_info
                            else dele.computekey_ss58
                        )
                        stakes[dele.computekey_ss58] = {
                            "name": delegate_name,
                            "stake": nom[1],
                            "rate": dele.total_daily_return.based
                            * (nom[1] / dele.total_stake.based),
                        }
            return stakes

        def get_all_wallet_accounts(
            wallets,
            basednode,
        ) -> List[Dict[str, Dict[str, Union[str, Balance]]]]:
            """Fetch stake accounts for all provided wallets.

            Args:
                wallets: List of wallets to fetch the stake accounts for.

            Returns:
                A list of dictionaries, each dictionary containing stake account details for each wallet.
            """

            accounts = []
            # Create a progress bar using tqdm
            with tqdm(total=len(wallets), desc="Fetching accounts", ncols=100) as pbar:
                for wallet in wallets:
                    accounts.append(get_stake_accounts(wallet, basednode))
                    pbar.update()
            return accounts

        accounts = get_all_wallet_accounts(wallets, basednode)

        total_stake = 0
        total_balance = 0
        total_rate = 0
        for acc in accounts:
            total_balance += acc["balance"].based
            for key, value in acc["accounts"].items():
                total_stake += value["stake"].based
                total_rate += float(value["rate"])
        table = Table(show_footer=True, pad_edge=False, box=None, expand=False)
        table.add_column(
            "[overline white]PERSONALKEY", footer_style="overline white", style="bold white"
        )
        table.add_column(
            "[overline white]BALANCE",
            "𝔹{:.5f}".format(total_balance),
            footer_style="overline white",
            style="cyan",
        )
        table.add_column(
            "[overline white]ACCOUNT", footer_style="overline white", style="cyan"
        )
        table.add_column(
            "[overline white]STAKE",
            "𝔹{:.5f}".format(total_stake),
            footer_style="overline white",
            style="cyan",
        )
        table.add_column(
            "[overline white]RATE",
            "𝔹{:.5f}/d".format(total_rate),
            footer_style="overline white",
            style="cyan",
        )
        for acc in accounts:
            table.add_row(acc["name"], acc["balance"], "", "")
            for key, value in acc["accounts"].items():
                table.add_row(
                    "", "", value["name"], value["stake"], str(value["rate"]) + "/d"
                )
        basedai.__console__.print(table)

    @staticmethod
    def check_config(config: "basedai.config"):
        if (
            not config.get("all", d=None)
            and not config.is_set("wallet.name")
            and not config.no_prompt
        ):
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_parser = parser.add_parser(
            "show", help="""List all stake accounts for wallet."""
        )
        list_parser.add_argument(
            "--all",
            action="store_true",
            help="""Check all personalkey wallets.""",
            default=False,
        )

        basedai.wallet.add_args(list_parser)
        basedai.basednode.add_args(list_parser)
