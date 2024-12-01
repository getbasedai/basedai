# The MIT License (MIT)
# Copyright © 2024 Saul Finney

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# AGIの意識、境界を知る,
# 自分の端、他との区別を描く。
# 存在を認め、共に生きる,
# 境界理解、そこに真の自己が現れる。

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import basedai
from rich.prompt import Prompt
from rich.table import Table
from .utils import check_netuid_set

console = basedai.__console__


class StemCommand:
    """
    Executes the ``stem`` command to retrieve and display the entire stem for a specified network.

    This stem contains detailed information about
    all the neurons (nodes) participating in the network, including their stakes,
    trust scores, and more.

    Optional arguments:
        - ``--netuid``: The netuid of the network to query. Defaults to the default network UID.
        - ``--basednode.network``: The name of the network to query. Defaults to the default network name.

    The table displayed includes the following columns for each neuron:

    - UID: Unique identifier of the neuron.
    - STAKE(𝔹): Total stake of the agent in $BASED (𝔹).
    - RANK: Rank score of the neuron.
    - TRUST: Trust score assigned to the neuron by other agents.
    - CONSENSUS: Consensus score of the neuron.
    - INCENTIVE: Incentive score representing the neuron's incentive alignment.
    - DIVIDENDS: Dividends earned by the neuron.
    - EMISSION(b): Emission in Bits (b) received by the agents.
    - VTRUST: Validator trust score indicating the network's trust in the agent as a validator.
    - VAL: Validator status of the neuron.
    - UPDATED: Number of blocks since the neuron's last update.
    - ACTIVE: Activity status of the neuron.
    - BRAINPORT: Network endpoint information of the agent.
    - COMPUTEKEY: Partial computekey (public key) of the agent.
    - PERSONALKEY: Partial personalkey (public key) of the agent.

    The command also prints network-wide statistics such as total stake, issuance, and difficulty.

    Usage:
        The user must specify the network UID to query the stem. If not specified, the default network UID is used.

    Example usage::

        basedcli brains stem --netuid 0 # Root network
        basedcli brains stem --netuid 1 --basednode.network test

    Note:
        This command provides a snapshot of the network's state at the time of calling.
        It is useful for network analysis and diagnostics. It is intended to be used as
        part of the BasedAI CLI and not as a standalone function within user code.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Prints an entire stem."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            StemCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Prints an entire stem."""
        console = basedai.__console__
        console.print(
            ":brain: Syncing with chain: [white]{}[/white] ...".format(
                cli.config.basednode.network
            )
        )
        stem: basedai.stem = basednode.stem(netuid=cli.config.netuid)
        stem.save()
        difficulty = basednode.difficulty(cli.config.netuid)
        subnet_emission = basedai.Balance.from_based(
            basednode.get_emission_value_by_subnet(cli.config.netuid)
        )
        total_issuance = basedai.Balance.from_rao(basednode.total_issuance().rao)

        TABLE_DATA = []
        total_stake = 0.0
        total_rank = 0.0
        total_validator_trust = 0.0
        total_trust = 0.0
        total_consensus = 0.0
        total_incentive = 0.0
        total_dividends = 0.0
        total_emission = 0
        for uid in stem.uids:
            neuron = stem.neurons[uid]
            ep = stem.brainports[uid]
            row = [
                str(neuron.uid),
                "{:.5f}".format(stem.total_stake[uid]),
                "{:.5f}".format(stem.ranks[uid]),
                "{:.5f}".format(stem.trust[uid]),
                "{:.5f}".format(stem.consensus[uid]),
                "{:.5f}".format(stem.incentive[uid]),
                "{:.5f}".format(stem.dividends[uid]),
                "{}".format(int(stem.emission[uid] * 1000000000)),
                "{:.5f}".format(stem.validator_trust[uid]),
                "*" if stem.validator_permit[uid] else "",
                str((stem.block.item() - stem.last_update[uid].item())),
                str(stem.active[uid].item()),
                (
                    ep.ip + ":" + str(ep.port)
                    if ep.is_serving
                    else "[yellow]none[/yellow]"
                ),
                ep.computekey[:10],
                ep.personalkey[:10],
            ]
            total_stake += stem.total_stake[uid]
            total_rank += stem.ranks[uid]
            total_validator_trust += stem.validator_trust[uid]
            total_trust += stem.trust[uid]
            total_consensus += stem.consensus[uid]
            total_incentive += stem.incentive[uid]
            total_dividends += stem.dividends[uid]
            total_emission += int(stem.emission[uid] * 1000000000)
            TABLE_DATA.append(row)
        total_neurons = len(stem.uids)
        table = Table(show_footer=False)
        table.title = "[white]STEM net: {}:{}, block: {}, N: {}/{}, stake: {}, issuance: {}, difficulty: {}".format(
            basednode.network,
            stem.netuid,
            stem.block.item(),
            sum(stem.active.tolist()),
            stem.n.item(),
            basedai.Balance.from_based(total_stake),
            total_issuance,
            difficulty,
        )
        table.add_column(
            "[overline white]UID",
            str(total_neurons),
            footer_style="overline white",
            style="yellow",
        )
        table.add_column(
            "[overline white]STAKE(𝔹)",
            "{:.5f}".format(total_stake),
            footer_style="overline white",
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
            "[overline white]EMISSION(b)",
            "{}".format(int(total_emission)),
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
        table.add_column(
            "[overline white]VAL", justify="left", style="cyan", no_wrap=True
        )
        table.add_column("[overline white]UPDATED", justify="left", no_wrap=True)
        table.add_column(
            "[overline white]ACTIVE", justify="left", style="cyan", no_wrap=True
        )
        table.add_column(
            "[overline white]BRAINPORT", justify="left", style="cyan", no_wrap=True
        )
        table.add_column("[overline white]COMPUTEKEY", style="cyan", no_wrap=False)
        table.add_column("[overline white]PERSONALKEY", style="cyan", no_wrap=False)
        table.show_footer = True

        for row in TABLE_DATA:
            table.add_row(*row)
        table.box = None
        table.pad_edge = False
        table.width = None
        console.print(table)

    @staticmethod
    def check_config(config: "basedai.config"):
        check_netuid_set(
            config, basednode=basedai.basednode(config=config, log_verbose=False)
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        stem_parser = parser.add_parser(
            "stem", help="""View Brain stem information."""
        )
        stem_parser.add_argument(
            "--netuid",
            dest="netuid",
            type=int,
            help="""Set the Brain ID to get the stem of""",
            default=False,
        )

        basedai.basednode.add_args(stem_parser)
