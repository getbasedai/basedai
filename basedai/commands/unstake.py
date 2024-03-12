# The MIT License (MIT)
# Copyright © 2024 Saul Finney

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

import sys
import basedai
from tqdm import tqdm
from rich.prompt import Confirm, Prompt
from basedai.utils.balance import Balance
from typing import List, Union, Optional, Tuple
from .utils import get_computekey_wallets_for_wallet
from . import defaults

console = basedai.__console__


class UnStakeCommand:
    """
    Executes the ``remove`` command to unstake BASED tokens from one or more computekeys and transfer them back to the user's personalkey on the Basedai network.

    This command is used to withdraw tokens previously staked to different computekeys.

    Usage:
        Users can specify the amount to unstake, the computekeys to unstake from (either by name or ``SS58`` address), and whether to unstake from all computekeys. The command checks for sufficient stake and prompts for confirmation before proceeding with the unstaking process.

    Optional arguments:
        - ``--all`` (bool): When set, unstakes all staked tokens from the specified computekeys.
        - ``--amount`` (float): The amount of BASED tokens to unstake.
        - --computekey_ss58address (str): The SS58 address of the computekey to unstake from.
        - ``--max_stake`` (float): Sets the maximum amount of BASED to remain staked in each computekey.
        - ``--computekeys`` (list): Specifies computekeys by name or SS58 address to unstake from.
        - ``--all_computekeys`` (bool): When set, unstakes from all computekeys associated with the wallet, excluding any specified in --computekeys.

    The command prompts for confirmation before executing the unstaking operation.

    Example usage::

        basedcli stake remove --amount 100 --computekeys hk1,hk2

    Note:
        This command is important for users who wish to reallocate their stakes or withdraw them from the network.
        It allows for flexible management of token stakes across different neurons (computekeys) on the network.
    """

    @classmethod
    def check_config(cls, config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if (
            not config.get("computekey_ss58address", d=None)
            and not config.is_set("wallet.computekey")
            and not config.no_prompt
            and not config.get("all_computekeys")
            and not config.get("computekeys")
        ):
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)

        # Get amount.
        if (
            not config.get("computekey_ss58address")
            and not config.get("amount")
            and not config.get("unstake_all")
            and not config.get("max_stake")
        ):
            computekeys: str = ""
            if config.get("all_computekeys"):
                computekeys = "all computekeys"
            elif config.get("computekeys"):
                computekeys = str(config.computekeys).replace("[", "").replace("]", "")
            else:
                computekeys = str(config.wallet.computekey)
            if not Confirm.ask(
                "Unstake all Based from: [bold]'{}'[/bold]?".format(computekeys)
            ):
                amount = Prompt.ask("Enter Based amount to unstake")
                config.unstake_all = False
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(
                        ":cross_mark:[red] Invalid Based amount[/red] [bold white]{}[/bold white]".format(
                            amount
                        )
                    )
                    sys.exit()
            else:
                config.unstake_all = True

    @staticmethod
    def add_args(command_parser):
        unstake_parser = command_parser.add_parser(
            "remove",
            help="""Remove stake from the specified computekey into the personalkey balance.""",
        )
        unstake_parser.add_argument(
            "--all", dest="unstake_all", action="store_true", default=False
        )
        unstake_parser.add_argument(
            "--amount", dest="amount", type=float, required=False
        )
        unstake_parser.add_argument(
            "--computekey_ss58address", dest="computekey_ss58address", type=str, required=False
        )
        unstake_parser.add_argument(
            "--max_stake",
            dest="max_stake",
            type=float,
            required=False,
            action="store",
            default=None,
            help="""Specify the maximum amount of Based to have staked in each computekey.""",
        )
        unstake_parser.add_argument(
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
        unstake_parser.add_argument(
            "--all_computekeys",
            "--wallet.all_computekeys",
            required=False,
            action="store_true",
            default=False,
            help="""To specify all computekeys. Specifying computekeys will exclude them from this all.""",
        )
        basedai.wallet.add_args(unstake_parser)
        basedai.basednode.add_args(unstake_parser)

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Unstake token of amount from computekey(s)."""
        try:
            config = cli.config.copy()
            basednode: "basedai.basednode" = basedai.basednode(
                config=config, log_verbose=False
            )
            UnStakeCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Unstake token of amount from computekey(s)."""
        config = cli.config.copy()
        wallet = basedai.wallet(config=config)

        # Get the computekey_names (if any) and the computekey_ss58s.
        computekeys_to_unstake_from: List[Tuple[Optional[str], str]] = []
        if cli.config.get("computekey_ss58address"):
            # Stake to specific computekey.
            computekeys_to_unstake_from = [(None, cli.config.get("computekey_ss58address"))]
        elif cli.config.get("all_computekeys"):
            # Stake to all computekeys.
            all_computekeys: List[basedai.wallet] = get_computekey_wallets_for_wallet(
                wallet=wallet
            )
            # Get the computekeys to exclude. (d)efault to no exclusions.
            computekeys_to_exclude: List[str] = cli.config.get("computekeys", d=[])
            # Exclude computekeys that are specified.
            computekeys_to_unstake_from = [
                (wallet.computekey_str, wallet.computekey.ss58_address)
                for wallet in all_computekeys
                if wallet.computekey_str not in computekeys_to_exclude
            ]  # definitely wallets

        elif cli.config.get("computekeys"):
            # Stake to specific computekeys.
            for computekey_ss58_or_computekey_name in cli.config.get("computekeys"):
                if basedai.utils.is_valid_ss58_address(computekey_ss58_or_computekey_name):
                    # If the computekey is a valid ss58 address, we add it to the list.
                    computekeys_to_unstake_from.append((None, computekey_ss58_or_computekey_name))
                else:
                    # If the computekey is not a valid ss58 address, we assume it is a computekey name.
                    #  We then get the computekey from the wallet and add it to the list.
                    wallet_ = basedai.wallet(
                        config=cli.config, computekey=computekey_ss58_or_computekey_name
                    )
                    computekeys_to_unstake_from.append(
                        (wallet_.computekey_str, wallet_.computekey.ss58_address)
                    )
        elif cli.config.wallet.get("computekey"):
            # Only cli.config.wallet.computekey is specified.
            #  so we stake to that single computekey.
            computekey_ss58_or_name = cli.config.wallet.get("computekey")
            if basedai.utils.is_valid_ss58_address(computekey_ss58_or_name):
                computekeys_to_unstake_from = [(None, computekey_ss58_or_name)]
            else:
                # Computekey is not a valid ss58 address, so we assume it is a computekey name.
                wallet_ = basedai.wallet(
                    config=cli.config, computekey=computekey_ss58_or_name
                )
                computekeys_to_unstake_from = [
                    (wallet_.computekey_str, wallet_.computekey.ss58_address)
                ]
        else:
            # Only cli.config.wallet.computekey is specified.
            #  so we stake to that single computekey.
            assert cli.config.wallet.computekey is not None
            computekeys_to_unstake_from = [
                (None, basedai.wallet(config=cli.config).computekey.ss58_address)
            ]

        final_computekeys: List[Tuple[str, str]] = []
        final_amounts: List[Union[float, Balance]] = []
        for computekey in tqdm(computekeys_to_unstake_from):
            computekey: Tuple[Optional[str], str]  # (computekey_name (or None), computekey_ss58)
            unstake_amount_based: float = cli.config.get(
                "amount"
            )  # The amount specified to unstake.
            computekey_stake: Balance = basednode.get_stake_for_personalkey_and_computekey(
                computekey_ss58=computekey[1], personalkey_ss58=wallet.personalkeypub.ss58_address
            )
            if unstake_amount_based == None:
                unstake_amount_based = computekey_stake.based
            if cli.config.get("max_stake"):
                # Get the current stake of the computekey from this personalkey.
                unstake_amount_based: float = computekey_stake.based - cli.config.get(
                    "max_stake"
                )
                cli.config.amount = unstake_amount_based
                if unstake_amount_based < 0:
                    # Skip if max_stake is greater than current stake.
                    continue
            else:
                if unstake_amount_based is not None:
                    # There is a specified amount to unstake.
                    if unstake_amount_based > computekey_stake.based:
                        # Skip if the specified amount is greater than the current stake.
                        continue

            final_amounts.append(unstake_amount_based)
            final_computekeys.append(computekey)  # add both the name and the ss58 address.

        if len(final_computekeys) == 0:
            # No computekeys to unstake from.
            basedai.__console__.print(
                "Not enough stake to unstake from any computekeys or max_stake is more than current stake."
            )
            return None

        # Ask to unstake
        if not cli.config.no_prompt:
            if not Confirm.ask(
                f"Do you want to unstake from the following keys to {wallet.name}:\n"
                + "".join(
                    [
                        f"    [bold white]- {computekey[0] + ':' if computekey[0] else ''}{computekey[1]}: {f'{amount} {basedai.__basedai_symbol__}' if amount else 'All'}[/bold white]\n"
                        for computekey, amount in zip(final_computekeys, final_amounts)
                    ]
                )
            ):
                return None

        if len(final_computekeys) == 1:
            # do regular unstake
            return basednode.unstake(
                wallet=wallet,
                computekey_ss58=final_computekeys[0][1],
                amount=None if cli.config.get("unstake_all") else final_amounts[0],
                wait_for_inclusion=True,
                prompt=not cli.config.no_prompt,
            )

        basednode.unstake_multiple(
            wallet=wallet,
            computekey_ss58s=[computekey_ss58 for _, computekey_ss58 in final_computekeys],
            amounts=None if cli.config.get("unstake_all") else final_amounts,
            wait_for_inclusion=True,
            prompt=False,
        )
