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

import os
import argparse
import basedai
import hashlib
from rich import print
from rich.tree import Tree
from substrateinterface.utils.ss58 import ss58_decode

console = basedai.__console__

def ss58_to_ethereum(ss58_address):
    public_key = ss58_decode(ss58_address)

    keccak_hash = hashlib.sha3_256(bytes.fromhex(public_key)).digest()
    eth_address = keccak_hash[-20:]

    return '0x' + eth_address.hex()


class ListCommand:
    """
    Executes the ``list`` command which enumerates all wallets and their respective computekeys present in the user's Basedai configuration directory.

    The command organizes the information in a tree structure, displaying each wallet along with the ``ss58`` addresses for the personalkey public key and any computekeys associated with it.

    Optional arguments:
        - ``-p``, ``--path``: The path to the Basedai configuration directory. Defaults to '~/.basedai'.

    The output is presented in a hierarchical tree format, with each wallet as a root node,
    and any associated computekeys as child nodes. The ``ss58`` address is displayed for each
    personalkey and computekey that is not encrypted and exists on the device.

    Usage:
        Upon invocation, the command scans the wallet directory and prints a list of all wallets, indicating whether the public keys are available (``?`` denotes unavailable or encrypted keys).

    Example usage::

        basedcli wallet list --path ~/.basedai

    Note:
        This command is read-only and does not modify the filesystem or the network state. It is intended for use within the Basedai CLI to provide a quick overview of the user's wallets.
    """

    @staticmethod
    def run(cli):
        r"""Lists wallets."""
        try:
            wallets = next(os.walk(os.path.expanduser(cli.config.wallet.path)))[1]
        except StopIteration:
            # No wallet files found.
            wallets = []

        root = Tree("🔐")
        for w_name in wallets:
            wallet_for_name = basedai.wallet(path=cli.config.wallet.path, name=w_name)
            try:
                if (
                    wallet_for_name.personalkeypub_file.exists_on_device()
                    and not wallet_for_name.personalkeypub_file.is_encrypted()
                ):
                    personalkeypub_str = wallet_for_name.personalkeypub.ss58_address
                else:
                    personalkeypub_str = "?"
            except:
                personalkeypub_str = "?"

            wallet_tree = root.add(
                    "[bold white]\"{}\" BASED: {} [magenta]EVM: {}".format(w_name, personalkeypub_str, ss58_to_ethereum(personalkeypub_str))
            )
            #wallet_tree.add("[magenta]EVM: {}".format(ss58_to_ethereum(personalkeypub_str)))
            computekeys_path = os.path.join(cli.config.wallet.path, w_name, "computekeys")
            try:
                computekeys = next(os.walk(os.path.expanduser(computekeys_path)))
                if len(computekeys) > 1:
                    for h_name in computekeys[2]:
                        computekey_for_name = basedai.wallet(
                            path=cli.config.wallet.path, name=w_name, computekey=h_name
                        )
                        try:
                            if (
                                computekey_for_name.computekey_file.exists_on_device()
                                and not computekey_for_name.computekey_file.is_encrypted()
                            ):
                                computekey_str = computekey_for_name.computekey.ss58_address
                            else:
                                computekey_str = "?"
                        except:
                            computekey_str = "?"
                        wallet_tree.add("[bold green]COMPUTE KEY SS58: \"{}\" {} - EVM: [dim magenta]{}".format(h_name, computekey_str, ss58_to_ethereum(computekey_str)))
            except:
                continue

        if len(wallets) == 0:
            root.add("[bold red]No wallets found.")

        # Uses rich print to display the tree.
        print(root)
        print("")

    @staticmethod
    def check_config(config: "basedai.config"):
        pass

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_parser = parser.add_parser("list", help="""List wallets""")
        basedai.wallet.add_args(list_parser)
        basedai.basednode.add_args(list_parser)
