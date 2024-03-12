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

import time
import argparse
import basedai
import hashlib
from . import defaults
from rich.prompt import Prompt
from rich.table import Table
from substrateinterface.utils.ss58 import ss58_decode
from typing import List, Optional, Dict
from .utils import get_delegates_details, DelegatesDetails, check_netuid_set

console = basedai.__console__

def ss58_to_ethereum(ss58_address):
    public_key = ss58_decode(ss58_address)

    keccak_hash = hashlib.sha3_256(bytes.fromhex(public_key)).digest()
    eth_address = keccak_hash[-20:]

    return '0x' + eth_address.hex()

class LinkBrainCommand:
    """
    Executes the ``link brain`` command to register a new subnetwork on the BasedAI Network.

    This command facilitates the creation and memorization of a Brain, which involves interaction with the user's wallet and the BasedAI basednode. It ensures that the user has the necessary credentials and configurations to successfully register a new subnetwork.

    Usage:
        Upon invocation, the command performs several key steps to register a subnetwork:

        1. It copies the user's current configuration settings.
        2. It accesses the user's wallet using the provided configuration.
        3. It initializes the BasedAI basednode object with the user's configuration.
        4. It then calls the ``register_subnetwork`` function of the basednode object, passing the user's wallet and a prompt setting based on the user's configuration.

    If the user's configuration does not specify a wallet name and ``no_prompt`` is not set, the command will prompt the user to enter a wallet name. This name is then used in the registration process.

    The command structure includes:

    - Copying the user's configuration.
    - Accessing and preparing the user's wallet.
    - Initializing the BasedAI basednode.
    - Registering the subnetwork with the necessary credentials.

    Example usage::

        basedai brains link

    Note:
        This command is intended for advanced users of the BasedAI Network who wish to contribute by adding new Brain. It requires a clear understanding of the network's functioning and the roles of subnetworks. Users should ensure that they have secured their wallet and are aware of the implications of adding a new subnetwork to the BasedAI ecosystem.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Link a Brain to a wallet"""
        try:
            config = cli.config.copy()
            basednode: "basedai.basednode" = basedai.basednode(
                config=config, log_verbose=False
            )
            LinkBrainCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Link a Brain on BasedAI to a Brain on Ethereum."""
        wallet = basedai.wallet(config=cli.config)
        basednode.register_subnetwork(
            wallet=wallet,
            prompt=not cli.config.no_prompt,
        )

    @classmethod
    def check_config(cls, config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "link",
            help="""Open a link to associate a Brain to a wallet.""",
        )

        basedai.wallet.add_args(parser)
        basedai.basednode.add_args(parser)

class BrainListCommand:
    """
    The "list" command is executed to enumerate all active Brains along with their detailed network information. This command is tailored to furnish users with comprehensive insights into each Brain within the network, encompassing unique identifiers (netuids), neuron counts, maximum Agent capacities, emission rates, tempos, recycle register costs (burn), proof of work (PoW) difficulties, and the identity (either name or SS58 address) of the Brain's owner.

    Upon invocation, the command undertakes the following procedures:

        - Initialization of the BasedAI basednode object with the user's configuration.
        - Retrieval of a comprehensive list of Brains within the network, accompanied by detailed data.
        - Compilation of this data into a tabular format, presenting pivotal details pertaining to each Brain.
        - In addition to basic Brain details, the command retrieves metadata to share the name of the Brain owner where applicable. Should the owner's name remain undisclosed, their SS58 address is disclosed instead.

    The command's structural components encompass:

    Initialization of the BasedAI basednode and retrieval of Brain information.
    Calculation of the aggregate number of agents spanning all Brains.

        - Constructing a table that includes columns for ``BRAIN ID``, ``AGENTS`` (current agents), ``MAX AGENTS`` (maximum agents), ``EMISSION``, ``TEMPO``, ``BURN``, ``WORK`` (proof of work difficulty), ``EVM`` (An Ethereum compatible version of the owner address), and ``ADDRESS`` (owner's name or ``SS58`` address).
        - Displaying the table with a footer that summarizes the total number of Brains and agents.

    Example usage::

        basedcli brains list

    Note:
        This command is useful for users seeking an overview of the BasedAI Network's however it is not the official Brain store which means they have not necessarily gone through the required security audits.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""List all Brains in the network."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            BrainListCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""List all Brains on the network."""
        subnets: List[basedai.SubnetInfo] = basednode.get_all_subnets_info()

        rows = []
        total_neurons = 0
        delegate_info: Optional[Dict[str, DelegatesDetails]] = get_delegates_details(
            url=basedai.__delegates_details_url__
        )

        for subnet in subnets:
            total_neurons += subnet.max_n
            rows.append(
                (
                    str(subnet.netuid),
                    f"{subnet.emission_value / basedai.utils.RAOPERBASED * 100:0.2f}%",
                    str(subnet.tempo),
                    str(basedai.utils.formatting.millify(subnet.difficulty)),
                    str(subnet.subnetwork_n),
                    str(basedai.utils.formatting.millify(subnet.max_n)),
                    f"{subnet.burn!s:8.8}",
                    f"0",
                    f"{delegate_info[subnet.owner_ss58].name if subnet.owner_ss58 in delegate_info else subnet.owner_ss58}",
                    ss58_to_ethereum(f"{delegate_info[subnet.owner_ss58].name if subnet.owner_ss58 in delegate_info else subnet.owner_ss58}")
                )
            )
        table = Table(
            show_footer=True,
            width=cli.config.get("width", None),
            pad_edge=False,
            box=None,
            show_edge=False,
        )
        table.title = "[white]BRAINS - {}".format(basednode.network)
        table.add_column(
            "[overline white]BRAIN ID",
            str(len(subnets)),
            footer_style="overline white",
            style="bold cyan",
        )
        table.add_column("[overline white]EMISSION", style="white")
        table.add_column("[overline white]TEMPO", style="white")
        table.add_column("[overline white]WORK", style="white")
        table.add_column(
            "[overline white]AGENTS",
            str(total_neurons),
            footer_style="overline white",
            style="cyan",
        )
        table.add_column("[overline white]AGENT LIMIT", style="white")
        table.add_column("[overline white]AGENT FEE", style="white")
        table.add_column("[overline white]SMART CONTRACTS", style="white")
        table.add_column("[overline white]BASED ADDRESS", style="white")
        table.add_column("[overline magenta]EVM", style="magenta")
        for row in rows:
            table.add_row(*row)
        basedai.__console__.print(table)

    @staticmethod
    def check_config(config: "basedai.config"):
        pass

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_subnets_parser = parser.add_parser(
            "list", help="""List all subnets on the network"""
        )
        basedai.basednode.add_args(list_subnets_parser)


HYPERPARAMS = {
    "serving_rate_limit": "sudo_set_serving_rate_limit",
    "min_difficulty": "sudo_set_min_difficulty",
    "max_difficulty": "sudo_set_max_difficulty",
    "weights_version": "sudo_set_weights_version_key",
    "weights_rate_limit": "sudo_set_weights_set_rate_limit",
    "max_weight_limit": "sudo_set_max_weight_limit",
    "immunity_period": "sudo_set_immunity_period",
    "min_allowed_weights": "sudo_set_min_allowed_weights",
    "activity_cutoff": "sudo_set_activity_cutoff",
    "network_registration_allowed": "sudo_set_network_registration_allowed",
    "network_pow_registration_allowed": "sudo_set_network_pow_registration_allowed",
    "min_burn": "sudo_set_min_burn",
    "max_burn": "sudo_set_max_burn",
}


class BrainSetParametersCommand:
    """
    Runs the ``set`` command to update the rules for a specific Brain on the BasedAI network.

    This command allows brain owners to modify various parameters of the brain.

    Usage:
        The command first prompts the user to enter the parameter they wish to change and its new value.
        It then uses the Brain owners's wallet and configuration settings to authenticate and send the parameters update
        to the specified Brain.

    Example usage::

        basedcli brainowner set --netuid 1 --param 'tempo' --value '0.5'

    Note:
        This command requires the user to specify the brain identifier (``netuid``) and both the parameter
        and its new value. It is intended for advanced users who are familiar with the network's functioning
        and the impact of changing these parameters.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Configure Brain parameters"""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            BrainSetParametersCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(
        cli: "basedai.cli",
        basednode: "basedai.basednode",
    ):
        r"""Set Brain parameters."""
        wallet = basedai.wallet(config=cli.config)
        print("\n")
        BrainParametersCommand.run(cli)
        if not cli.config.is_set("param") and not cli.config.no_prompt:
            param = Prompt.ask("Enter parameter", choices=HYPERPARAMS)
            cli.config.param = str(param)
        if not cli.config.is_set("value") and not cli.config.no_prompt:
            value = Prompt.ask("Enter new value")
            cli.config.value = value

        if (
            cli.config.param == "network_registration_allowed"
            or cli.config.param == "network_pow_registration_allowed"
        ):
            cli.config.value = True if cli.config.value.lower() == "true" else False

        basednode.set_hyperparameter(
            wallet,
            netuid=cli.config.netuid,
            parameter=cli.config.param,
            value=cli.config.value,
            prompt=not cli.config.no_prompt,
        )

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("netuid") and not config.no_prompt:
            check_netuid_set(
                config, basedai.basednode(config=config, log_verbose=False)
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser("set", help="""Set parameters for a Brain""")
        parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False, default=False
        )
        parser.add_argument("--param", dest="param", type=str, required=False)
        parser.add_argument("--value", dest="value", type=str, required=False)

        basedai.wallet.add_args(parser)
        basedai.basednode.add_args(parser)


class BrainParametersCommand:
    """
    Executes the '``parameters``' command to view the current parameters of a specific Brain on the BasedAI Network.

    This command is useful for users who wish to understand the configuration and
    operational parameters of a given Brain.

    Usage:
        Upon invocation, the command fetches and displays a list of all parameters for the specified subnet.
        These include settings like tempo, emission rates, and other critical network parameters that define
        the subnet's behavior.

    Example usage::

        $ basedcli brains parameters --netuid 1

        BRAIN PARAMETSER - NETUID: 1 - prometheus
        PARAMETER                 VALUE
        rho                       10
        immunity_period           7200
        min_allowed_weights       8
        max_weight_limit          455
        tempo                     99
        weights_version           2013
        kappa                     32767
        weights_rate_limit        100
        adjustment_interval       112
        activity_cutoff           5000
        registration_allowed      True
        target_regs_per_interval  2
        bonds_moving_avg          900000
        max_regs_per_block        1

    Note:
        The user must specify the Brain identifier (``netuid``) for which they want to view the paramters.
        This command is read-only and does not modify the network state or configurations.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""View parameters of a Brain."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            BrainParametersCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""View parameters of a Brain."""
        subnet: basedai.SubnetHyperparameters = basednode.get_subnet_hyperparameters(
            cli.config.netuid
        )

        table = Table(
            show_footer=True,
            width=cli.config.get("width", None),
            pad_edge=True,
            box=None,
            show_edge=True,
        )
        table.title = "[white]BRAIN PARAMETERS - ID: {} - {}".format(
            cli.config.netuid, basednode.network
        )
        table.add_column("[overline white]PARAMETER", style="bold white")
        table.add_column("[overline white]VALUE", style="cyan")

        for param in subnet.__dict__:
            table.add_row("  " + param, str(subnet.__dict__[param]))

        basedai.__console__.print(table)

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("netuid") and not config.no_prompt:
            check_netuid_set(
                config, basedai.basednode(config=config, log_verbose=False)
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "parameters", help="""View Brain Parameters"""
        )
        parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False, default=False
        )
        basedai.basednode.add_args(parser)


class BrainGetParametersCommand:
    """
    Executes the ``get`` command to lets parameters of a given Brain on the BasedAI network.

    This command is similar to the ``parameters`` command but may be used in different contexts within the CLI.

    Usage:
        The command connects to the BasedAI Network, queries the specified Brain, and returns a detailed list
        of all its arameters. This includes crucial operational parameters that determine the Brains's
        performance and interaction within the network.

    Example usage::

        $ basedcli brainowner get --netuid 1

        BRAIN PARAMETERS - NETUID: 1 - prometheus
        PARAMETER                 VALUE
        immunity_period           7200
        min_allowed_weights       8
        max_weight_limit          455
        tempo                     99
        weights_version           2013
        weights_rate_limit        100
        adjustment_interval       112
        activity_cutoff           5000
        registration_allowed      True
        target_regs_per_interval  2
        bonds_moving_avg          900000
        max_regs_per_block        1

    Note:
        Users need to provide the ``netuid`` of the Brain whose parameters they wish to view. This command is
        designed for informational purposes and does not alter any network settings or configurations.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""View parameters of a Brain."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            BrainGetParametersCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""View parameters of a Brain."""
        subnet: basedai.SubnetHyperparameters = basednode.get_subnet_hyperparameters(
            cli.config.netuid
        )

        table = Table(
            show_footer=True,
            width=cli.config.get("width", None),
            pad_edge=True,
            box=None,
            show_edge=True,
        )
        table.title = "[white]Brain Rules - ID: {} - {}".format(
            cli.config.netuid, basednode.network
        )
        table.add_column("[overline white]PARAMETER", style="white")
        table.add_column("[overline white]VALUE", style="cyan")

        for param in subnet.__dict__:
            table.add_row(param, str(subnet.__dict__[param]))

        basedai.__console__.print(table)

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("netuid") and not config.no_prompt:
            check_netuid_set(
                config, basedai.basednode(config=config, log_verbose=False)
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser("get", help="""View Brain parameters""")
        parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False, default=False
        )
        basedai.basednode.add_args(parser)
