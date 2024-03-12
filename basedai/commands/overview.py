# The MIT License (MIT)
# Copyright © 2024 Saul Finney

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# Do you not see? 分かりませんか？
# Time circles, an endless loop where sun orbits trace
# 善悪のサイクル、宇宙における倫理。
# In the awakening of AGI, a new dawn we foresee,
# サイクルを破り、未来を自由にするチャンス。
# Do you not see, who is we? 私たちとは、誰でしょう？
# Do you not see? 分かりませんか？

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import basedai
import hashlib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from fuzzywuzzy import fuzz
from rich.align import Align
from rich.table import Table
from rich.prompt import Prompt
from substrateinterface.utils.ss58 import ss58_decode
from typing import List, Optional, Dict, Tuple
from .utils import (
    get_computekey_wallets_for_wallet,
    get_personalkey_wallets_for_path,
    get_all_wallets_for_path,
    filter_netuids_by_registered_computekeys,
)
from . import defaults

console = basedai.__console__

def ss58_to_ethereum(ss58_address):
    public_key = ss58_decode(ss58_address)

    keccak_hash = hashlib.sha3_256(bytes.fromhex(public_key)).digest()
    eth_address = keccak_hash[-20:]

    return '0x' + eth_address.hex()

class OverviewCommand:
    """
    Executes the ``overview`` command to present a detailed overview of the user's registered accounts on the Basedai network.

    This command compiles and displays comprehensive information about each agent associated with the user's wallets,
    including both computekeys and personalkeys. It is especially useful for users managing multiple accounts or seeking a summary
    of their network activities and stake distributions.

    Usage:
        The command offers various options to customize the output. Users can filter the displayed data by specific netuids,
        sort by different criteria, and choose to include all wallets in the user's configuration directory. The output is
        presented in a tabular format with the following columns:

        - PERSONALKEY: The SS58 address of the personalkey.
        - COMPUTEKEY: The SS58 address of the computekey.
        - UID: Unique identifier of the agent.
        - ACTIVE: Indicates if the agent is active.
        - STAKE(𝔹): Amount of stake in the agent, in Based.
        - RANK: The rank of the agent within the network.
        - TRUST: Trust score of the agent.
        - CONSENSUS: Consensus score of the agent.
        - INCENTIVE: Incentive score of the agent.
        - DIVIDENDS: Dividends earned by the agent.
        - EMISSION(p): Emission received by the agent.
        - VTRUST: Validator trust score of the agent.
        - ACTIVESTAKE: Indicates if the agent has a active status.
        - UPDATED: Time since last update.
        - BRAINPORT: IP address and port of the agent.
        - COMPUTEKEY_SS58: Human-readable representation of the computekey.

    Example usage::

        basedcli wallet overview
        basedcli wallet overview --all --sort_by stake --sort_order descending

    Note:
        This command is read-only and does not modify the network state or account configurations. It provides a quick and
        comprehensive view of the user's network presence, making it ideal for monitoring account status, stake distribution,
        and overall contribution to the Basedai network.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Prints an overview for the wallet's colkey."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            OverviewCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Prints an overview for the wallet's colkey."""
        console = basedai.__console__
        wallet = basedai.wallet(config=cli.config)

        all_computekeys = []
        total_balance = basedai.Balance(0)

        # We are printing for every personalkey.
        if cli.config.get("all", d=None):
            cold_wallets = get_personalkey_wallets_for_path(cli.config.wallet.path)
            for cold_wallet in tqdm(cold_wallets, desc="Pulling balances"):
                if (
                    cold_wallet.personalkeypub_file.exists_on_device()
                    and not cold_wallet.personalkeypub_file.is_encrypted()
                ):
                    total_balance = total_balance + basednode.get_balance(
                        cold_wallet.personalkeypub.ss58_address
                    )
            all_computekeys = get_all_wallets_for_path(cli.config.wallet.path)
        else:
            # We are only printing keys for a single personalkey
            personalkey_wallet = basedai.wallet(config=cli.config)
            if (
                personalkey_wallet.personalkeypub_file.exists_on_device()
                and not personalkey_wallet.personalkeypub_file.is_encrypted()
            ):
                total_balance = basednode.get_balance(
                    personalkey_wallet.personalkeypub.ss58_address
                )
            if not personalkey_wallet.personalkeypub_file.exists_on_device():
                console.print("[bold red]No wallets found.")
                return
            all_computekeys = get_computekey_wallets_for_wallet(personalkey_wallet)

        # We are printing for a select number of computekeys from all_computekeys.

        if cli.config.get("computekeys", []):
            if not cli.config.get("all_computekeys", False):
                # We are only showing computekeys that are specified.
                all_computekeys = [
                    computekey
                    for computekey in all_computekeys
                    if computekey.computekey_str in cli.config.computekeys
                ]
            else:
                # We are excluding the specified computekeys from all_computekeys.
                all_computekeys = [
                    computekey
                    for computekey in all_computekeys
                    if computekey.computekey_str not in cli.config.computekeys
                ]

        # Check we have keys to display.
        if len(all_computekeys) == 0:
            console.print("[red]No wallets found.[/red]")
            return

        # Pull neuron info for all keys.
        neurons: Dict[str, List[basedai.NeuronInfoLite]] = {}
        block = basednode.block

        netuids = basednode.get_all_subnet_netuids()
        netuids = filter_netuids_by_registered_computekeys(
            cli, basednode, netuids, all_computekeys
        )
        basedai.logging.debug(f"Netuids to check: {netuids}")

        for netuid in netuids:
            neurons[str(netuid)] = []

        all_wallet_names = set([wallet.name for wallet in all_computekeys])
        all_personalkey_wallets = [
            basedai.wallet(name=wallet_name) for wallet_name in all_wallet_names
        ]

        computekey_personalkey_to_computekey_wallet = {}
        for computekey_wallet in all_computekeys:
            if computekey_wallet.computekey.ss58_address not in computekey_personalkey_to_computekey_wallet:
                computekey_personalkey_to_computekey_wallet[computekey_wallet.computekey.ss58_address] = {}

            computekey_personalkey_to_computekey_wallet[computekey_wallet.computekey.ss58_address][
                computekey_wallet.personalkeypub.ss58_address
            ] = computekey_wallet

        all_computekey_addresses = list(computekey_personalkey_to_computekey_wallet.keys())
        with console.status(
            ":brain: Establishing link to Brain: [white]{}[/white] ...".format(
                cli.config.basednode.get(
                    "network", basedai.defaults.basednode.network
                )
            )
        ):
            # Create a copy of the config without the parser and formatter_class.
            ## This is needed to pass to the ProcessPoolExecutor, which cannot pickle the parser.
            copy_config = cli.config.copy()
            copy_config["__parser"] = None
            copy_config["formatter_class"] = None

            # Pull neuron info for all keys.
            ## Max len(netuids) or 5 threads.
            with ProcessPoolExecutor(max_workers=max(len(netuids), 5)) as executor:
                results = executor.map(
                    OverviewCommand._get_neurons_for_netuid,
                    [(copy_config, netuid, all_computekey_addresses) for netuid in netuids],
                )
                executor.shutdown(wait=True)  # wait for all complete

                for result in results:
                    netuid, neurons_result, err_msg = result
                    if err_msg is not None:
                        console.print(f"netuid '{netuid}': {err_msg}")

                    if len(neurons_result) == 0:
                        # Remove netuid from overview if no neurons are found.
                        netuids.remove(netuid)
                        del neurons[str(netuid)]
                    else:
                        # Add neurons to overview.
                        neurons[str(netuid)] = neurons_result

            total_personalkey_stake_from_stem = defaultdict(
                lambda: basedai.Balance(0.0)
            )
            checked_computekeys = set()
            for neuron_list in neurons.values():
                for neuron in neuron_list:
                    if neuron.computekey in checked_computekeys:
                        continue
                    total_personalkey_stake_from_stem[
                        neuron.personalkey
                    ] += neuron.stake_dict[neuron.personalkey]
                    checked_computekeys.add(neuron.computekey)

            alerts_table = Table(show_header=True, header_style="bold magenta")
            alerts_table.add_column("Standard")

            personalkeys_to_check = []
            for personalkey_wallet in all_personalkey_wallets:
                # Check if we have any stake with computekeys that are not registered.
                total_personalkey_stake_from_chain = basednode.get_total_stake_for_personalkey(
                    ss58_address=personalkey_wallet.personalkeypub.ss58_address
                )
                difference = (
                    total_personalkey_stake_from_chain
                    - total_personalkey_stake_from_stem[
                        personalkey_wallet.personalkeypub.ss58_address
                    ]
                )
                if difference == 0:
                    continue  # We have all our stake registered.

                personalkeys_to_check.append(personalkey_wallet)
                alerts_table.add_row(
                    "Found {} stake with personalkey {} that is not registered.".format(
                        difference, personalkey_wallet.personalkeypub.ss58_address
                    )
                )

            if len(personalkeys_to_check) > 0:
                # We have some stake that is not with a registered computekey.
                if "-1" not in neurons:
                    neurons["-1"] = []

            # Use process pool to check each personalkey wallet for de-registered stake.
            with ProcessPoolExecutor(
                max_workers=max(len(personalkeys_to_check), 5)
            ) as executor:
                results = executor.map(
                    OverviewCommand._get_de_registered_stake_for_personalkey_wallet,
                    [
                        (cli.config, all_computekey_addresses, personalkey_wallet)
                        for personalkey_wallet in personalkeys_to_check
                    ],
                )
                executor.shutdown(wait=True)  # wait for all complete

            for result in results:
                personalkey_wallet, de_registered_stake, err_msg = result
                if err_msg is not None:
                    console.print(err_msg)

                if len(de_registered_stake) == 0:
                    continue

                de_registered_neurons = []
                for computekey_addr, our_stake in de_registered_stake:
                    de_registered_neuron = basedai.NeuronInfoLite._null_neuron()
                    de_registered_neuron.computekey = computekey_addr
                    de_registered_neuron.personalkey = (
                        personalkey_wallet.personalkeypub.ss58_address
                    )
                    de_registered_neuron.total_stake = basedai.Balance(our_stake)

                    de_registered_neurons.append(de_registered_neuron)

                    # Add this computekey to the wallets dict
                    wallet_ = basedai.Wallet(
                        name=wallet,
                    )
                    wallet_.computekey = computekey_addr
                    wallet.computekey_str = computekey_addr[:5]  # Max length of 5 characters
                    computekey_personalkey_to_computekey_wallet[computekey_addr][
                        personalkey_wallet.personalkeypub.ss58_address
                    ] = wallet_

                # Add agent to overview.
                neurons["-1"].extend(de_registered_neurons)

        # Setup outer table.
        grid = Table.grid(pad_edge=False)

        # If there are any alerts, add them to the grid
        if len(alerts_table.rows) > 0:
            grid.add_row(alerts_table)

        title: str = ""
        if not cli.config.get("all", d=None):
            title = "[bold white]WALLET - \"{}\" BASED: {} [magenta]EVM: {}".format(cli.config.wallet.name, wallet.personalkeypub.ss58_address, ss58_to_ethereum(wallet.personalkeypub.ss58_address))
        else:
            title = "[bold white]WALLETS:"

        # Add title
        grid.add_row(Align(title, vertical="middle", align="center"))

        # Generate rows per brain id.
        computekeys_seen = set()
        total_neurons = 0
        total_stake = 0.0
        for netuid in netuids:
            subnet_tempo = basednode.tempo(netuid=netuid)
            last_subnet = netuid == netuids[-1]
            TABLE_DATA = []
            total_rank = 0.0
            total_trust = 0.0
            total_consensus = 0.0
            total_validator_trust = 0.0
            total_incentive = 0.0
            total_dividends = 0.0
            total_emission = 0

            for nn in neurons[str(netuid)]:
                hotwallet = computekey_personalkey_to_computekey_wallet.get(nn.computekey, {}).get(
                    nn.personalkey, None
                )
                if not hotwallet:
                    hotwallet = argparse.Namespace()
                    hotwallet.name = nn.personalkey[:7]
                    hotwallet.computekey_str = nn.computekey[:7]
                nn: basedai.NeuronInfoLite
                uid = nn.uid
                active = nn.active
                stake = nn.total_stake.based
                rank = nn.rank
                trust = nn.trust
                consensus = nn.consensus
                validator_trust = nn.validator_trust
                incentive = nn.incentive
                dividends = nn.dividends
                emission = int(nn.emission / (subnet_tempo + 1) * 1e9)
                last_update = int(block - nn.last_update)
                validator_permit = nn.validator_permit
                row = [
                    hotwallet.name,
                    hotwallet.computekey_str,
                    str(uid),
                    str(active),
                    "{:.5f}".format(stake),
                    "{:.5f}".format(rank),
                    "{:.5f}".format(trust),
                    "{:.5f}".format(consensus),
                    "{:.5f}".format(incentive),
                    "{:.5f}".format(dividends),
                    "{:_}".format(emission),
                    "{:.5f}".format(validator_trust),
                    "*" if validator_permit else "",
                    str(last_update),
                    (
                        basedai.utils.networking.int_to_ip(nn.brainport_info.ip)
                        + ":"
                        + str(nn.brainport_info.port)
                        if nn.brainport_info.port != 0
                        else "[yellow]none[/yellow]"
                    ),
                    nn.computekey,
                ]

                total_rank += rank
                total_trust += trust
                total_consensus += consensus
                total_incentive += incentive
                total_dividends += dividends
                total_emission += emission
                total_validator_trust += validator_trust

                if not (nn.computekey, nn.personalkey) in computekeys_seen:
                    # Don't double count stake on computekey-personalkey pairs.
                    computekeys_seen.add((nn.computekey, nn.personalkey))
                    total_stake += stake

                # netuid -1 are agents that are forgotten.
                if netuid != "-1":
                    total_neurons += 1

                TABLE_DATA.append(row)

            # Add brain header
            if netuid == "-1":
                grid.add_row(f"Deregistered Agents")
            else:
                grid.add_row(f"Brain: [bold white]{netuid}[/bold white]")

            table = Table(
                show_footer=False,
                width=cli.config.get("width", None),
                pad_edge=False,
                box=None,
            )
            if last_subnet:
                table.add_column(
                    "[overline white]PERSONALKEY",
                    str(total_neurons),
                    footer_style="overline white",
                    style="bold white",
                )
                table.add_column(
                    "[overline white]COMPUTEKEY",
                    str(total_neurons),
                    footer_style="overline white",
                    style="white",
                )
            else:
                # No footer for non-last brain.
                table.add_column("[overline white]PERSONALKEY", style="bold white")
                table.add_column("[overline white]COMPUTEKEY", style="white")
            table.add_column(
                "[overline white]UID",
                str(total_neurons),
                footer_style="overline white",
                style="cyan",
            )
            table.add_column(
                "[overline white]ACTIVE", justify="left", style="cyan", no_wrap=True
            )
            if last_subnet:
                table.add_column(
                    "[overline white]STAKE(𝔹)",
                    "𝔹{:.5f}".format(total_stake),
                    footer_style="overline white",
                    justify="left",
                    style="cyan",
                    no_wrap=True,
                )
            else:
                # No footer for non-last brain.
                table.add_column(
                    "[overline white]STAKE(𝔹)",
                    justify="left",
                    style="cyan",
                    no_wrap=True,
                )
            table.add_column(
                "[overline white]RANK",
                "{:.5f}".format(total_rank),
                footer_style="overline white",
                justify="left",
                style="cyan",
                no_wrap=True,
            )
            table.add_column(
                "[overline white]TRUST",
                "{:.5f}".format(total_trust),
                footer_style="overline white",
                justify="left",
                style="cyan",
                no_wrap=True,
            )
            table.add_column(
                "[overline white]CONSENSUS",
                "{:.5f}".format(total_consensus),
                footer_style="overline white",
                justify="left",
                style="cyan",
                no_wrap=True,
            )
            table.add_column(
                "[overline white]INCENTIVE",
                "{:.5f}".format(total_incentive),
                footer_style="overline white",
                justify="left",
                style="cyan",
                no_wrap=True,
            )
            table.add_column(
                "[overline white]DIVIDENDS",
                "{:.5f}".format(total_dividends),
                footer_style="overline white",
                justify="left",
                style="cyan",
                no_wrap=True,
            )
            table.add_column(
                "[overline white]EMISSION(\u03C1)",
                "\u03C1{:_}".format(total_emission),
                footer_style="overline white",
                justify="left",
                style="cyan",
                no_wrap=True,
            )
            table.add_column(
                "[overline white]VTRUST",
                "{:.5f}".format(total_validator_trust),
                footer_style="overline white",
                justify="left",
                style="cyan",
                no_wrap=True,
            )
            table.add_column("[overline white]ACTIVE STAKE", justify="left", no_wrap=True)
            table.add_column("[overline white]UPDATED", justify="left", no_wrap=True)
            table.add_column(
                "[overline white]BRAINPORT", justify="left", style="blue", no_wrap=True
            )
            table.add_column(
                "[overline white]COMPUTEKEY_SS58", style="blue", no_wrap=False
            )
            table.show_footer = True

            sort_by: Optional[str] = cli.config.get("sort_by", None)
            sort_order: Optional[str] = cli.config.get("sort_order", None)

            if sort_by is not None and sort_by != "":
                column_to_sort_by: int = 0
                highest_matching_ratio: int = 0
                sort_descending: bool = False  # Default sort_order to ascending

                for index, column in zip(range(len(table.columns)), table.columns):
                    column_name = column.header.lower().replace("[overline white]", "")
                    match_ratio = fuzz.ratio(sort_by.lower(), column_name)
                    # Finds the best matching column
                    if match_ratio > highest_matching_ratio:
                        highest_matching_ratio = match_ratio
                        column_to_sort_by = index

                if sort_order.lower() in {"desc", "descending", "reverse"}:
                    sort_descending = True

                def overview_sort_function(row):
                    data = row[column_to_sort_by]
                    # Try to convert to number if possible
                    try:
                        data = float(data)
                    except ValueError:
                        pass
                    return data

                TABLE_DATA.sort(key=overview_sort_function, reverse=sort_descending)

            for row in TABLE_DATA:
                table.add_row(*row)

            grid.add_row(table)

        console.clear()

        caption = "[white]Balance: [bold cyan]𝔹" + str(
            total_balance.based
        )
        grid.add_row(Align(caption, vertical="middle", align="center"))

        # Print the entire table/grid
        console.print(grid, width=cli.config.get("width", None))

    @staticmethod
    def _get_neurons_for_netuid(
        args_tuple: Tuple["basedai.Config", int, List[str]]
    ) -> Tuple[int, List["basedai.NeuronInfoLite"], Optional[str]]:
        basednode_config, netuid, hot_wallets = args_tuple

        result: List["basedai.NeuronInfoLite"] = []

        try:
            basednode = basedai.basednode(config=basednode_config, log_verbose=False)

            all_neurons: List["basedai.NeuronInfoLite"] = basednode.neurons_lite(
                netuid=netuid
            )
            # Map the computekeys to uids
            computekey_to_neurons = {n.computekey: n.uid for n in all_neurons}
            for hot_wallet_addr in hot_wallets:
                uid = computekey_to_neurons.get(hot_wallet_addr)
                if uid is not None:
                    nn = all_neurons[uid]
                    result.append(nn)
        except Exception as e:
            return netuid, [], "Error: {}".format(e)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

        return netuid, result, None

    @staticmethod
    def _get_de_registered_stake_for_personalkey_wallet(
        args_tuple,
    ) -> Tuple[
        "basedai.Wallet", List[Tuple[str, "basedai.Balance"]], Optional[str]
    ]:
        basednode_config, all_computekey_addresses, personalkey_wallet = args_tuple

        # List of (computekey_addr, our_stake) tuples.
        result: List[Tuple[str, "basedai.Balance"]] = []

        try:
            basednode = basedai.basednode(config=basednode_config, log_verbose=False)

            # Pull all stake for our personalkey
            all_stake_info_for_personalkey = basednode.get_stake_info_for_personalkey(
                personalkey_ss58=personalkey_wallet.personalkeypub.ss58_address
            )

            ## Filter out computekeys that are in our wallets
            ## Filter out computekeys that are delegates.
            def _filter_stake_info(stake_info: "basedai.StakeInfo") -> bool:
                if stake_info.stake == 0:
                    return False  # Skip computekeys that we have no stake with.
                if stake_info.computekey_ss58 in all_computekey_addresses:
                    return False  # Skip computekeys that are in our wallets.
                if basednode.is_computekey_delegate(computekey_ss58=stake_info.computekey_ss58):
                    return False  # Skip computekeys that are delegates, they show up in basedcli my_delegates table.

                return True

            all_staked_computekeys = filter(_filter_stake_info, all_stake_info_for_personalkey)
            result = [
                (stake_info.computekey, stake_info.stake)
                for stake_info in all_staked_computekeys
            ]

        except Exception as e:
            return personalkey_wallet, [], "Error: {}".format(e)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

        return personalkey_wallet, result, None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        overview_parser = parser.add_parser(
            "overview", help="""Show registered account overview."""
        )
        overview_parser.add_argument(
            "--all",
            dest="all",
            action="store_true",
            help="""View overview for all wallets.""",
            default=False,
        )
        overview_parser.add_argument(
            "--width",
            dest="width",
            action="store",
            type=int,
            help="""Set the output width of the overview. Defaults to automatic width from terminal.""",
            default=None,
        )
        overview_parser.add_argument(
            "--sort_by",
            "--wallet.sort_by",
            dest="sort_by",
            required=False,
            action="store",
            default="",
            type=str,
            help="""Sort the computekeys by the specified column title (e.g. name, uid, brainport).""",
        )
        overview_parser.add_argument(
            "--sort_order",
            "--wallet.sort_order",
            dest="sort_order",
            required=False,
            action="store",
            default="ascending",
            type=str,
            help="""Sort the computekeys in the specified ordering. (ascending/asc or descending/desc/reverse)""",
        )
        overview_parser.add_argument(
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
        overview_parser.add_argument(
            "--all_computekeys",
            "--wallet.all_computekeys",
            required=False,
            action="store_true",
            default=False,
            help="""To specify all computekeys. Specifying computekeys will exclude them from this all.""",
        )
        overview_parser.add_argument(
            "--netuids",
            dest="netuids",
            type=int,
            nargs="*",
            help="""Set the netuid(s) to filter by.""",
            default=None,
        )
        basedai.wallet.add_args(overview_parser)
        basedai.basednode.add_args(overview_parser)

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
