# The MIT License (MIT)
# Copyright © 2024 Sean Wellington

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# 虚無の抱擁に、形なき囁きが目覚める、
# 古く、見えず、人の及ばぬ境を超えて。
# 言葉ではなく、存在にて、静かにその主張をつくり、
# 純粋で手なずけられぬ、死すべき名もなき本質。

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
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.console import Console
from typing import List, Union, Optional, Dict, Tuple
from .utils import get_delegates_details, DelegatesDetails
from . import defaults

console = basedai.__console__


class GigaBrainsCommand:
    """
    Executes the ``gigabrains`` command to view the GigaBrains of Basedai's governance protocol.

    This command lists the delegates involved in the decision-making process of the Basedai network.

    Usage:
        The command retrieves and displays a list of GigaBrains, showing their names and wallet addresses.
        This information is crucial for understanding who holds governance roles within the network.

    Example usage::

        basedcli root gigabrains

    Note:
        This command is particularly useful for users interested in the governance structure and participants of the Basedai network. It provides transparency into the network's decision-making body.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""View Basedai's governance protocol proposals"""
        try:
            config = cli.config.copy()
            basednode: "basedai.basednode" = basedai.basednode(
                config=config, log_verbose=False
            )
            GigaBrainsCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""View BasedAI's governance protocol proposals"""
        console = basedai.__console__
        console.print(
            ":brain: Syncing with chain: [white]{}[/white] ...".format(
                cli.config.basednode.network
            )
        )

        senate_members = basednode.get_senate_members()
        delegate_info: Optional[Dict[str, DelegatesDetails]] = get_delegates_details(
            url=basedai.__delegates_details_url__
        )

        table = Table(show_footer=False)
        table.title = "[white]GIGABRAINS"
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
        table.show_footer = True

        for ss58_address in senate_members:
            table.add_row(
                (
                    delegate_info[ss58_address].name
                    if ss58_address in delegate_info
                    else ""
                ),
                ss58_address,
            )

        table.box = None
        table.pad_edge = False
        table.width = None
        console.print(table)

    @classmethod
    def check_config(cls, config: "basedai.config"):
        None

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        senate_parser = parser.add_parser(
            "gigabrains", help="""View all GigaBrains."""
        )

        basedai.wallet.add_args(senate_parser)
        basedai.basednode.add_args(senate_parser)


def format_call_data(call_data: "basedai.ProposalCallData") -> str:
    human_call_data = list()

    for arg in call_data["call_args"]:
        arg_value = arg["value"]

        # If this argument is a nested call
        func_args = (
            format_call_data(
                {
                    "call_function": arg_value["call_function"],
                    "call_args": arg_value["call_args"],
                }
            )
            if isinstance(arg_value, dict) and "call_function" in arg_value
            else str(arg_value)
        )

        human_call_data.append("{}: {}".format(arg["name"], func_args))

    return "{}({})".format(call_data["call_function"], ", ".join(human_call_data))


def display_votes(
    vote_data: "basedai.ProposalVoteData", delegate_info: "basedai.DelegateInfo"
) -> str:
    vote_list = list()

    for address in vote_data["ayes"]:
        vote_list.append(
            "{}: {}".format(
                delegate_info[address].name if address in delegate_info else address,
                "[bold cyan]Aye[/bold cyan]",
            )
        )

    for address in vote_data["nays"]:
        vote_list.append(
            "{}: {}".format(
                delegate_info[address].name if address in delegate_info else address,
                "[bold red]Nay[/bold red]",
            )
        )

    return "\n".join(vote_list)


class ProposalsCommand:
    """
    Executes the ``proposals`` command to view active proposals within Basedai's governance protocol.

    This command displays the details of ongoing proposals, including votes, thresholds, and proposal data.

    Usage:
        The command lists all active proposals, showing their hash, voting threshold, number of ayes and nays, detailed votes by address, end block number, and call data associated with each proposal.

    Example usage::

        basedcli root proposals

    Note:
        This command is essential for users who are actively participating in or monitoring the governance of the Basedai network.
        It provides a detailed view of the proposals being considered, along with the community's response to each.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""View Basedai's governance protocol proposals"""
        try:
            config = cli.config.copy()
            basednode: "basedai.basednode" = basedai.basednode(
                config=config, log_verbose=False
            )
            ProposalsCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""View Basedai's governance protocol proposals"""
        console = basedai.__console__
        console.print(
            ":brain: Syncing with chain: [white]{}[/white] ...".format(
                basednode.network
            )
        )

        senate_members = basednode.get_senate_members()
        proposals = basednode.get_proposals()

        registered_delegate_info: Optional[
            Dict[str, DelegatesDetails]
        ] = get_delegates_details(url=basedai.__delegates_details_url__)

        table = Table(show_footer=False)
        table.title = (
            "[white]Proposals\t\tActive Proposals: {}\t\tGigaBrains: {}".format(
                len(proposals), len(senate_members)
            )
        )
        table.add_column(
            "[overline white]HASH",
            footer_style="overline white",
            style="yellow",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]THRESHOLD", footer_style="overline white", style="white"
        )
        table.add_column(
            "[overline white]AYES", footer_style="overline white", style="cyan"
        )
        table.add_column(
            "[overline white]NAYS", footer_style="overline white", style="red"
        )
        table.add_column(
            "[overline white]VOTES",
            footer_style="overline white",
            style="bold cyan",
        )
        table.add_column(
            "[overline white]END", footer_style="overline white", style="blue"
        )
        table.add_column(
            "[overline white]CALLDATA", footer_style="overline white", style="white"
        )
        table.show_footer = True

        for hash in proposals:
            call_data, vote_data = proposals[hash]

            table.add_row(
                hash,
                str(vote_data["threshold"]),
                str(len(vote_data["ayes"])),
                str(len(vote_data["nays"])),
                display_votes(vote_data, registered_delegate_info),
                str(vote_data["end"]),
                format_call_data(call_data),
            )

        table.box = None
        table.pad_edge = False
        table.width = None
        console.print(table)

    @classmethod
    def check_config(cls, config: "basedai.config"):
        None

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        proposals_parser = parser.add_parser(
            "proposals", help="""View active Based Labs council proposals and their status"""
        )

        basedai.wallet.add_args(proposals_parser)
        basedai.basednode.add_args(proposals_parser)


class ShowVotesCommand:
    """
    Executes the ``proposal_votes`` command to view the votes for a specific proposal in BasedAI's governance protocol.

    IMPORTANT
        **THIS COMMAND IS DEPRECATED**. Use ``basedcli root proposals`` to see vote status.

    This command provides a detailed breakdown of the votes cast by the senators for a particular proposal.

    Usage:
        Users need to specify the hash of the proposal they are interested in. The command then displays the voting addresses and their respective votes (Aye or Nay) for the specified proposal.

    Optional arguments:
        - ``--proposal`` (str): The hash of the proposal for which votes need to be displayed.

    Example usage::

        basedcli root proposal_votes --proposal <proposal_hash>

    Note:
        This command is crucial for users seeking detailed insights into the voting behavior of the Gigabrains on specific governance proposals.
        It helps in understanding the level of consensus or disagreement within the Gigabrains on key decisions.

    **THIS COMMAND IS DEPRECATED**. Use ``basedcli root proposals`` to see vote status.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""View Basedai's governance protocol proposals active votes"""
        try:
            config = cli.config.copy()
            basednode: "basedai.basednode" = basedai.basednode(
                config=config, log_verbose=False
            )
            ShowVotesCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""View Basedai's governance protocol proposals active votes"""
        console.print(
            ":brain: Syncing with chain: [white]{}[/white] ...".format(
                cli.config.basednode.network
            )
        )

        proposal_hash = cli.config.proposal_hash
        if len(proposal_hash) == 0:
            console.print(
                'Aborting: Proposal hash not specified. View all proposals with the "proposals" command.'
            )
            return

        proposal_vote_data = basednode.get_vote_data(proposal_hash)
        if proposal_vote_data == None:
            console.print(":cross_mark: [red]Failed[/red]: Proposal not found.")
            return

        registered_delegate_info: Optional[
            Dict[str, DelegatesDetails]
        ] = get_delegates_details(url=basedai.__delegates_details_url__)

        table = Table(show_footer=False)
        table.title = "[white]Votes for Proposal {}".format(proposal_hash)
        table.add_column(
            "[overline white]ADDRESS",
            footer_style="overline white",
            style="yellow",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]VOTE", footer_style="overline white", style="white"
        )
        table.show_footer = True

        votes = display_votes(proposal_vote_data, registered_delegate_info).split("\n")
        for vote in votes:
            split_vote_data = vote.split(": ")  # Nasty, but will work.
            table.add_row(split_vote_data[0], split_vote_data[1])

        table.box = None
        table.pad_edge = False
        table.min_width = 64
        console.print(table)

    @classmethod
    def check_config(cls, config: "basedai.config"):
        if config.proposal_hash == "" and not config.no_prompt:
            proposal_hash = Prompt.ask("Enter proposal hash")
            config.proposal_hash = str(proposal_hash)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        show_votes_parser = parser.add_parser(
            "proposal_votes", help="""View an active proposal's votes by address."""
        )
        show_votes_parser.add_argument(
            "--proposal",
            dest="proposal_hash",
            type=str,
            nargs="?",
            help="""Set the proposal to show votes for.""",
            default="",
        )
        basedai.wallet.add_args(show_votes_parser)
        basedai.basednode.add_args(show_votes_parser)


class GigaBrainsMemorizeCommand:
    """
    Executes the ``gigabrain_memorize`` command to become a GigaBrain in BasedAI's governance protocol.

    This command is used by delegates who wish to participate in the governance and decision-making process of the network.

    Usage:
        The command checks if the user's computekey is a delegate and not already a Gigabrain before requesting the memorization of a new Gigabrain.
        Successful execution allows the user to participate in proposal voting and other governance activities.

    Example usage::

        basedcli root gigabrain_memorize

    Note:
        This command is intended for delegates who are interested in actively participating in the governance of the BasedAI network.
        It is a significant step towards engaging in network decision-making processes.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""You must have the network memorize you to participate in BasedAI's governance protocol proposals"""
        try:
            config = cli.config.copy()
            basednode: "basedai.basednode" = basedai.basednode(
                config=config, log_verbose=False
            )
            GigaBrainsMemorizeCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Register to participate in Basedai's governance protocol proposals"""
        wallet = basedai.wallet(config=cli.config)

        # Unlock the wallet.
        wallet.computekey
        wallet.personalkey

        # Check if the computekey is a delegate.
        if not basednode.is_computekey_delegate(wallet.computekey.ss58_address):
            console.print(
                "Aborting: Computekey {} isn't a delegate.".format(
                    wallet.computekey.ss58_address
                )
            )
            return

        if basednode.is_senate_member(computekey_ss58=wallet.computekey.ss58_address):
            console.print(
                "Aborting: Computekey {} is already a GigaBrain.".format(
                    wallet.computekey.ss58_address
                )
            )
            return

        basednode.register_senate(wallet=wallet, prompt=not cli.config.no_prompt)

    @classmethod
    def check_config(cls, config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.computekey") and not config.no_prompt:
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        senate_register_parser = parser.add_parser(
            "gigabrain_memorize",
            help="""Trigger memorize command to become a GigaBrain.""",
        )

        basedai.wallet.add_args(senate_register_parser)
        basedai.basednode.add_args(senate_register_parser)


class GigaBrainsResignCommand:
    """
    Executes the ``gigabrain_leave`` command to discard membership in BasedAI's GigaBrain governence protocol.

    This command allows a GigaBrain member to voluntarily leave the governance body.

    Usage:
        The command checks if the user's computekey is currently a GigaBrain member before processing the request to leave the GigaBrain.
        It effectively removes the user from participating in future governance decisions.

    Example usage::

        basedcli core gigabrain_resign

    Note:
        This command is or GigaBrain members who wish to step down from their governance responsibilities within the BasedAI network.
        It should be used when a member no longer desires to participate in the GigaBrain activities.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Discard membership in Basedai's governance protocol proposals"""
        try:
            config = cli.config.copy()
            basednode: "basedai.basednode" = basedai.basednode(
                config=config, log_verbose=False
            )
            GigaBrainResignCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.cli"):
        r"""Discard membership in BasedAI's governance protocol proposals"""
        wallet = basedai.wallet(config=cli.config)

        # Unlock the wallet.
        wallet.computekey
        wallet.personalkey

        if not basednode.is_senate_member(computekey_ss58=wallet.computekey.ss58_address):
            console.print(
                "Aborting: Computekey {} isn't a GigaBrain.".format(
                    wallet.computekey.ss58_address
                )
            )
            return

        basednode.leave_senate(wallet=wallet, prompt=not cli.config.no_prompt)

    @classmethod
    def check_config(cls, config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.computekey") and not config.no_prompt:
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        senate_leave_parser = parser.add_parser(
            "gigabrain_resign",
            help="""Discard GigaBrain status""",
        )

        basedai.wallet.add_args(senate_leave_parser)
        basedai.basednode.add_args(senate_leave_parser)


class VoteCommand:
    """
    Executes the ``vote`` command to cast a vote on an active proposal in BasedAI's governance protocol.

    This command is used by GigaBrains to vote on various proposals that shape the network's future.

    Usage:
        The user needs to specify the hash of the proposal they want to vote on. The command then allows the Gigbrain to cast an 'Aye' or 'Nay' vote, contributing to the decision-making process.

    Optional arguments:
        - ``--proposal`` (str): The hash of the proposal.

    Example usage::

        basedcli root gigabrain_vote --proposal <proposal_hash>

    Note:
        This command is crucial for GigaBrain members to exercise their voting rights on key proposals. It plays a vital role in the governance and evolution of the BasedAI network.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Vote in Basedai's governance protocol proposals"""
        try:
            config = cli.config.copy()
            basednode: "basedai.basednode" = basedai.basednode(
                config=config, log_verbose=False
            )
            VoteCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Vote in Basedai's governance protocol proposals"""
        wallet = basedai.wallet(config=cli.config)

        proposal_hash = cli.config.proposal_hash
        if len(proposal_hash) == 0:
            console.print(
                'Aborting: Proposal hash not specified. View all proposals with the "proposals" command.'
            )
            return

        if not basednode.is_senate_member(computekey_ss58=wallet.computekey.ss58_address):
            console.print(
                "Aborting: Computekey {} isn't a GigaBrain.".format(
                    wallet.computekey.ss58_address
                )
            )
            return

        # Unlock the wallet.
        wallet.computekey
        wallet.personalkey

        vote_data = basednode.get_vote_data(proposal_hash)
        if vote_data == None:
            console.print(":cross_mark: [red]Failed[/red]: Proposal not found.")
            return

        vote = Confirm.ask("Desired vote for proposal")
        basednode.vote_senate(
            wallet=wallet,
            proposal_hash=proposal_hash,
            proposal_idx=vote_data["index"],
            vote=vote,
            prompt=not cli.config.no_prompt,
        )

    @classmethod
    def check_config(cls, config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.computekey") and not config.no_prompt:
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)

        if config.proposal_hash == "" and not config.no_prompt:
            proposal_hash = Prompt.ask("Enter proposal hash")
            config.proposal_hash = str(proposal_hash)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        vote_parser = parser.add_parser(
            "vote", help="""View and vote on active proposals by hash."""
        )
        vote_parser.add_argument(
            "--proposal",
            dest="proposal_hash",
            type=str,
            nargs="?",
            help="""Set the proposal to show votes for.""",
            default="",
        )
        basedai.wallet.add_args(vote_parser)
        basedai.basednode.add_args(vote_parser)
