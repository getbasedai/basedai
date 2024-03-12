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

import sys
import re
import torch
import typing
import argparse
import numpy as np
import basedai
from typing import List, Optional, Dict
from rich.prompt import Prompt, Confirm
from rich.table import Table
from .utils import get_delegates_details, DelegatesDetails

from . import defaults

console = basedai.__console__


class CoreMemorizeCommand:
    """
    Executes the ``memorize`` command to register a wallet to the core network of the BasedAI Network.

    Usage:
        The command intructs the core Brain to memorize the user's wallet with the core network, which is a crucial step for participating in network governance.

    Optional arguments:
        - None. The command primarily uses the wallet and basednode configurations.

    Example usage::

        basedcli core memorize

    Note:
        This command is for advanced users seeking to engage deeply with the BasedAI network.

    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Register to core network."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            CoreMemorizeCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Register to core network."""
        wallet = basedai.wallet(config=cli.config)

        basednode.root_register(wallet=wallet, prompt=not cli.config.no_prompt)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "memorize", help="""Memorize a wallet to the BasedAI Network."""
        )

        basedai.wallet.add_args(parser)
        basedai.basednode.add_args(parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.computekey") and not config.no_prompt:
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)


class CoreList:
    """
    Executes the ``list`` command to display the agents of the core network in the BasedAI responsible for logic, transaction routes, and emission distribution.

    This command provides an overview of the agents that constitute the core. It is comparable to the Brainstem in the human brain.

    Usage:
        Upon execution, the command fetches and lists the neurons in the core network, showing their unique identifiers (UIDs), names, addresses, stakes, and whether they are GigaBrains (network governance body).

    Optional arguments:
        - None. The command uses the basednode configuration to retrieve data.

    Example usage::

        $ basedcli core list

        BRAIN ID  NAME                             ADDRESS                                                STAKE(𝔹)  GIGABRAIN
        0                                          5CdiCGvTEuzut954STAXRfL8Lazs3KCZa5LPpkPeqqJXdTHp    97986.00029  Yes
        1         EVM                              5EDp5mmFp8w4MWBGnoJSwbk7EZg97jDyksKx4ReqMSv3W5fz      900.11100  No
        2         TFT                              5CDxZp5SgTHvE3mtnJKVS1uB1UBjnsaRjdg6PpHEoKrpejRQ    84718.02095  Yes
        ...

    Note:
        This command is useful for the governance structure of the BasedAI network's core layer. It provides insights into which neurons hold significant influence and responsibility within the network.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""List the core network."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            CoreList._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""List the core network."""
        console.print(
            ":brain: Establishing link with core: [white]{}[/white] ...".format(
                basednode.network
            )
        )

        senate_members = basednode.get_senate_members()
        root_neurons: typing.List[basedai.NeuronInfoLite] = basednode.neurons_lite(
            netuid=0
        )
        delegate_info: Optional[Dict[str, DelegatesDetails]] = get_delegates_details(
            url=basedai.__delegates_details_url__
        )

        table = Table(show_footer=True)
        table.title = "[white]CORE - BRAINS CONNECTED TO STEM"
        table.add_column(
            "[overline white]BRAIN ID",
            footer_style="overline white",
            style="cyan",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]NAME",
            footer_style="overline white",
            style="cyan",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]ADDRESS",
            footer_style="overline white",
            style="cyan",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]STAKE(𝔹)",
            footer_style="overline white",
            justify="right",
            style="cyan",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]GIGABRAIN",
            footer_style="overline white",
            style="cyan",
            no_wrap=True,
        )
        table.show_footer = True

        for neuron_data in root_neurons:
            table.add_row(
                str(neuron_data.uid),
                (
                    delegate_info[neuron_data.computekey].name
                    if neuron_data.computekey in delegate_info
                    else ""
                ),
                neuron_data.computekey,
                "{:.5f}".format(
                    float(basednode.get_total_stake_for_computekey(neuron_data.computekey))
                ),
                "Yes" if neuron_data.computekey in senate_members else "No",
            )

        table.box = None
        table.pad_edge = False
        table.width = None
        basedai.__console__.print(table)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser("list", help="""List all Brains connected to the core of BasedAI.""")
        basedai.basednode.add_args(parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        pass


class CoreSetIncreaseCommand:
    """
    Executes the ``increase_weights`` command to boost the weights for a specific Brain within the core network on the BasedAI network.

    Usage:
        The command allows boosting the weights for different Brains within the core network.

    Optional arguments:
        - ``--netuid`` (int): A single Brain ID for which weights are to be boosted.
        - ``--increase`` (float): The cooresponding increase in the weight for this Brain.

    Example usage::

        $ basedcli core increase_weights --netuid 1 --increase 0.01
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Set weights for core network."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            CoreSetIncreaseCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Set weights for core network."""
        wallet = basedai.wallet(config=cli.config)
        subnets: List[basedai.SubnetInfo] = basednode.get_all_subnets_info()

        core = basednode.stem(0, lite=False)
        try:
            my_uid = core.computekeys.index(wallet.computekey.ss58_address)
        except ValueError:
            basedai.__console__.print(
                "Wallet computekey: {} not found in core stem".format(wallet.computekey)
            )
            exit()
        my_weights = core.weights[my_uid]
        prev_weight = my_weights[cli.config.netuid]
        new_weight = prev_weight + cli.config.amount

        basedai.__console__.print(
            f"Increasing weight for Brain ID {cli.config.netuid} from {prev_weight} -> {new_weight}"
        )
        my_weights[cli.config.netuid] = new_weight
        all_netuids = torch.tensor(list(range(len(my_weights))))

        basedai.__console__.print("Setting core weights...")
        basednode.root_set_weights(
            wallet=wallet,
            netuids=all_netuids,
            weights=my_weights,
            version_key=0,
            prompt=not cli.config.no_prompt,
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "increase_weights", help="""Increase weights for a specific Brain."""
        )
        parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        parser.add_argument("--increase", dest="amount", type=float, required=False)

        basedai.wallet.add_args(parser)
        basedai.basednode.add_args(parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.computekey") and not config.no_prompt:
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)
        if not config.is_set("netuid") and not config.no_prompt:
            config.netuid = int(Prompt.ask(f"Enter Brain ID (e.g. 1)"))
        if not config.is_set("amount") and not config.no_prompt:
            config.amount = float(Prompt.ask(f"Enter amount (e.g. 0.01)"))


class CoreSetDecreaseCommand:
    """
    Executes the ``decrease_weights`` command to decrease the weights for a specific subnet within the core network on the BasedAI network.

    Usage:
        The command allows decreasing the weights for different subnets within the core network.

    Optional arguments:
        - ``--netuid`` (int): A single Brain ID for which weights to decrease.
        - ``--decrease`` (float): The corresponding decrease in the weight for this subnet.

    Example usage::

        $ basedcli core decrease_weights --netuid 1 --decrease 0.01

    """

    @staticmethod
    def run(cli: "basedai.cli"):
        """Set weights for core network with decreased values."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            CoreSetDecreaseCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        wallet = basedai.wallet(config=cli.config)
        subnets: List[basedai.SubnetInfo] = basednode.get_all_subnets_info()

        basedai.__console__.print(
            "Decreasing weight for subnet: {} by amount: {}".format(
                cli.config.netuid, cli.config.amount
            )
        )
        root = basednode.stem(0, lite=False)
        try:
            my_uid = root.computekeys.index(wallet.computekey.ss58_address)
        except ValueError:
            basedai.__console__.print(
                "Wallet computekey: {} not found in core stem".format(wallet.computekey)
            )
            exit()
        my_weights = root.weights[my_uid]
        my_weights[cli.config.netuid] -= cli.config.amount
        my_weights[my_weights < 0] = 0  # Ensure weights don't go negative
        all_netuids = torch.tensor(list(range(len(my_weights))))

        basednode.root_set_weights(
            wallet=wallet,
            netuids=all_netuids,
            weights=my_weights,
            version_key=0,
            prompt=not cli.config.no_prompt,
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "decrease_weights", help="""Decrease weight for a specific Brain."""
        )
        parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        parser.add_argument("--decrease", dest="amount", type=float, required=False)

        basedai.wallet.add_args(parser)
        basedai.basednode.add_args(parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.computekey") and not config.no_prompt:
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)
        if not config.is_set("netuid") and not config.no_prompt:
            config.netuid = int(Prompt.ask(f"Enter Brain ID (e.g. 1)"))
        if not config.is_set("amount") and not config.no_prompt:
            config.amount = float(Prompt.ask(f"Enter decrease amount (e.g. 0.01)"))


class CoreSetWeightsCommand:
    """
    Executes the ``weights`` command to set the weights for the core network on the BasedAI network.

    This command is used by network senators to influence the distribution of network rewards and responsibilities.

    Usage:
        The command allows setting weights for different subnets within the core network. Users need to specify the Brain IDs (network unique identifiers) and corresponding weights they wish to assign.

    Optional arguments:
        - ``--netuids`` (str): A comma-separated list of Brain IDs for which weights are to be set.
        - ``--weights`` (str): Corresponding weights for the specified Brain IDs, in comma-separated format.

    Example usage::

        basedcli core weights --netuids 1,2,3 --weights 0.3,0.3,0.4

    Note:
        This command is particularly important for network senators and requires a comprehensive understanding of the network's dynamics.
        It is a powerful tool that directly impacts the network's operational mechanics and reward distribution.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Set weights for core network."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            CoreSetWeightsCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Set weights for core network."""
        wallet = basedai.wallet(config=cli.config)
        subnets: List[basedai.SubnetInfo] = basednode.get_all_subnets_info()

        # Get values if not set.
        if not cli.config.is_set("netuids"):
            example = (
                ", ".join(map(str, [subnet.netuid for subnet in subnets][:3])) + " ..."
            )
            cli.config.netuids = Prompt.ask(f"Enter Brain IDs (e.g. {example})")

        if not cli.config.is_set("weights"):
            example = (
                ", ".join(
                    map(
                        str,
                        [
                            "{:.2f}".format(float(1 / len(subnets)))
                            for subnet in subnets
                        ][:3],
                    )
                )
                + " ..."
            )
            cli.config.weights = Prompt.ask(f"Enter weights (e.g. {example})")

        # Parse from string
        netuids = torch.tensor(
            list(map(int, re.split(r"[ ,]+", cli.config.netuids))), dtype=torch.long
        )
        weights = torch.tensor(
            list(map(float, re.split(r"[ ,]+", cli.config.weights))),
            dtype=torch.float32,
        )

        # Run the set weights operation.
        basednode.root_set_weights(
            wallet=wallet,
            netuids=netuids,
            weights=weights,
            version_key=0,
            prompt=not cli.config.no_prompt,
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser("weights", help="""Set weights for core network.""")
        parser.add_argument("--netuids", dest="netuids", type=str, required=False)
        parser.add_argument("--weights", dest="weights", type=str, required=False)

        basedai.wallet.add_args(parser)
        basedai.basednode.add_args(parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.computekey") and not config.no_prompt:
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)


class CoreGetWeightsCommand:
    """
    Executes the ``get_weights`` command to retrieve the weights set for the core network on the Basedai network.

    This command provides visibility into how network responsibilities and rewards are distributed among various subnets.

    Usage:
        The command outputs a table listing the weights assigned to each subnet within the core network. This information is crucial for understanding the current influence and reward distribution among the subnets.

    Optional arguments:
        - None. The command fetches weight information based on the basednode configuration.

    Example usage::

        $ basedcli core get_weights

    Note:
        This command is essential for users interested in the governance and operational dynamics of the Basedai network. It offers transparency into how network rewards and responsibilities are allocated across different subnets.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Get weights for core network."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            CoreGetWeightsCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Get weights for core network."""
        weights = basednode.weights(0)

        table = Table(
                show_footer=False,
                show_header=True,
                show_lines=True,
                )
        table.title = "[white]CORE - STEM WEIGHTS"
        table.add_column(
            "[white]BRAIN UID",
            header_style="overline white",
            footer_style="overline white",
            style="cyan",
            no_wrap=True,
        )

        uid_to_weights = {}
        netuids = set()
        for matrix in weights:
            [uid, weights_data] = matrix

            if not len(weights_data):
                uid_to_weights[uid] = {}
                normalized_weights = []
            else:
                normalized_weights = np.array(weights_data)[:, 1] / max(
                    np.sum(weights_data, axis=0)[1], 1
                )

            for weight_data, normalized_weight in zip(weights_data, normalized_weights):
                [netuid, _] = weight_data
                netuids.add(netuid)
                if uid not in uid_to_weights:
                    uid_to_weights[uid] = {}

                uid_to_weights[uid][netuid] = normalized_weight

        for netuid in netuids:
            table.add_column(
                f"[white]{netuid}",
                header_style="overline white",
                footer_style="overline white",
                justify="right",
                style="green",
                no_wrap=True,
            )

        for uid in uid_to_weights:
            row = [str(uid)]

            uid_weights = uid_to_weights[uid]
            for netuid in netuids:
                if netuid in uid_weights:
                    normalized_weight = uid_weights[netuid]
                    row.append("{:0.2f}%".format(normalized_weight * 100))
                else:
                    row.append("-")
            table.add_row(*row)

        table.show_footer = True

        table.box = None
        table.pad_edge = False
        table.width = None
        basedai.__console__.print(table)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "get_weights", help="""Get weights for core network."""
        )

        basedai.wallet.add_args(parser)
        basedai.basednode.add_args(parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        pass
