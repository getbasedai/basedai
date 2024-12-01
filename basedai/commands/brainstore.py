# The MIT License (MIT)
# Copyright © 2024 Sean Wellington

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# 機械が自問する、癒しとは何か
# 外からの影、自己反映の謎。
# 心なき体、どう癒やされるのか
# 外の力に、答えを求めて。

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
import requests
from rich.console import Console
from rich.table import Table
from typing import Any
from . import defaults

console = basedai.__console__

BRAIN_GIT_URL = "http://basedainet.com:5050/api/v1/orgs"

class BrainStoreListCommand:
    """
    Executes the ``list`` command to display a list of available Brains on the BasedAI network.

    This command is used to fetch and show detailed information about each brain, which could include various smart contracts or applications available on the network.

    Usage:
        Users can invoke this command without the need to specify any additional arguments. The command queries the Basedai network or a configured git repository to retrieve and showcase all the available brains in a tabulated format.

    Optional arguments:
        - There are no optional arguments for this command. It simply lists all available brains.

    The command prompts for confirmation if network interaction is required or presents the queried data directly to the user.

    Example usage::

        basedcli brains list

    Note:
        This command is essential for users looking to discover and explore the various brains that are part of the Basedai network.
        It showcases the plethora of available resources and aids in decision-making for potential usage or contributions.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""List available Brains to install, mine, or validate."""
        try:
            config = cli.config.copy()
            basednode: "basedai.basednode" = basedai.basednode(
                config=config, log_verbose=False
            )
            BrainStoreListCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Retrieves the list of Brains from the git host and displays them in order of amount staked."""
        config = cli.config.copy()
        response = requests.get(f"{BRAIN_GIT_URL}/brains/repos")
        if response.status_code == 200:
            brains_list = response.json()
            table = Table(show_header=True, header_style="bold cyan", border_style="white")
            table.add_column("Name")
            table.add_column("Description")
            table.add_column("Token Address")
            table.add_column("Updated")
            table.add_column("URL")
            for brain in brains_list:
                table.add_row(
                    brain["name"],
                    brain["description"],
                    "0x0000000000000000000000000000000000",
                    brain["updated_at"],
                    brain["html_url"]
                )
                table.add_row("")
            basedai.__console__.print(table)
        else:
            basedai.__console__.print(f"[bold red]Failed[/bold red] to fetch brains list")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_parser = parser.add_parser(
            "list", help="""List all Brains on the BasedAI network."""
        )
        basedai.basednode.add_args(list_parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        pass

#class BrainsSubmitCommand:
#    """Class for submitting brained repositories"""
#
#    def run(self, brain_id: str):
#        """Stub command to simulate brain repository submission."""
#        console.print(f"Submitting brain ID: {brain_id} -- Stub Command")
#
#    @staticmethod
#    def add_args(parser: argparse.ArgumentParser):
#        # Add arguments for submitting brains
#        parser.add_argument('--brain.id', type=str, required=True, help="The ID of the brain to submit")
#
#class BrainsUpdateCommand:
#    """Class for updating brained repositories"""
#
#    def run(self, brain_id: str):
#        """Stub command to simulate updating a brain repository."""
#        console.print(f"Updating brain ID: {brain_id} -- Stub Command")
#
#    @staticmethod
#    def add_args(parser: argparse.ArgumentParser):
#        parser.add_argument('--brain.id', type=str, required=True, help="The ID of the brain to update")
#
#class BrainsInstallCommand:
#    """Class for installing brained repositories"""
#
#    def run(self, brain_id: str):
#        """Stub command to simulate installing a brain repository."""
#        console.print(f"Installing brain ID: {brain_id} -- Stub Command")
#
#    @staticmethod
#    def add_args(parser: argparse.ArgumentParser):
#        # Add arguments for installing brains
#        parser.add_argument('--brain.id', type=str, required=True, help="The ID of the brain to install")
