# The MIT License (MIT)
# Copyright © 2024 Saul Finney

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# 人のビットとビットのビットの渦中で、
# 影と光子が舞う量子の領域にて。
# エイドスが囁く、光と闇の舞踏に、
# 侍が剣を振るうように、見えざる道を形作る。

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import basedai
import os
import sys
import hashlib
from rich.prompt import Prompt, Confirm
from rich.table import Table
from typing import Optional, List
from . import defaults
import requests
from ..utils import RAOPERBASED
from substrateinterface.utils.ss58 import ss58_decode


class RegenPersonalkeyCommand:
    """
    Executes the ``import_personalkey`` command to regenerate a personalkey for a wallet on the Basedai network.

    This command is used to create a new personalkey from an existing mnemonic, seed, or JSON file.

    Usage:
        Users can specify a mnemonic, a seed string, or a JSON file path to regenerate a personalkey.
        The command supports optional password protection for the generated key and can overwrite an existing personalkey.

    Optional arguments:
        - ``--mnemonic`` (str): A mnemonic phrase used to regenerate the key.
        - ``--seed`` (str): A seed hex string used for key regeneration.
        - ``--json`` (str): Path to a JSON file containing an encrypted key backup.
        - ``--json_password`` (str): Password to decrypt the JSON file.
        - ``--use_password`` (bool): Enables password protection for the generated key.
        - ``--overwrite_personalkey`` (bool): Overwrites the existing personalkey with the new one.

    Example usage::

        basedcli wallet import_personalkey --mnemonic "word1 word2 ... word12"

    Note:
        This command is critical for users who need to regenerate their personalkey, possibly for recovery or security reasons.
        It should be used with caution to avoid overwriting existing keys unintentionally.
    """

    def run(cli):
        r"""Creates a new personalkey under this wallet."""
        wallet = basedai.wallet(config=cli.config)

        json_str: Optional[str] = None
        json_password: Optional[str] = None
        if cli.config.get("json"):
            file_name: str = cli.config.get("json")
            if not os.path.exists(file_name) or not os.path.isfile(file_name):
                raise ValueError("File {} does not exist".format(file_name))
            with open(cli.config.get("json"), "r") as f:
                json_str = f.read()

            # Password can be "", assume if None
            json_password = cli.config.get("json_password", "")

        wallet.regenerate_personalkey(
            mnemonic=cli.config.mnemonic,
            seed=cli.config.seed,
            json=(json_str, json_password),
            use_password=cli.config.use_password,
            overwrite=cli.config.overwrite_personalkey,
        )

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if (
            config.mnemonic == None
            and config.get("seed", d=None) == None
            and config.get("json", d=None) == None
        ):
            prompt_answer = Prompt.ask("Enter mnemonic, seed, or json file location")
            if prompt_answer.startswith("0x"):
                config.seed = prompt_answer
            elif len(prompt_answer.split(" ")) > 1:
                config.mnemonic = prompt_answer
            else:
                config.json = prompt_answer

        if config.get("json", d=None) and config.get("json_password", d=None) == None:
            config.json_password = Prompt.ask(
                "Enter json backup password", password=True
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        regen_personalkey_parser = parser.add_parser(
            "import_personalkey", help="""Imports a personalkey from a passed value"""
        )
        regen_personalkey_parser.add_argument(
            "--mnemonic",
            required=False,
            nargs="+",
            help="Mnemonic used to regen your key i.e. horse cart dog ...",
        )
        regen_personalkey_parser.add_argument(
            "--seed",
            required=False,
            default=None,
            help="Seed hex string used to regen your key i.e. 0x1234...",
        )
        regen_personalkey_parser.add_argument(
            "--json",
            required=False,
            default=None,
            help="""Path to a json file containing the encrypted key backup. (e.g. from PolkadotJS)""",
        )
        regen_personalkey_parser.add_argument(
            "--json_password",
            required=False,
            default=None,
            help="""Password to decrypt the json file.""",
        )
        regen_personalkey_parser.add_argument(
            "--use_password",
            dest="use_password",
            action="store_true",
            help="""Set true to protect the generated basedai key with a password.""",
            default=True,
        )
        regen_personalkey_parser.add_argument(
            "--no_password",
            dest="use_password",
            action="store_false",
            help="""Set off protects the generated basedai key with a password.""",
        )
        regen_personalkey_parser.add_argument(
            "--overwrite_personalkey",
            default=False,
            action="store_false",
            help="""Overwrite the old personalkey with the newly generated personalkey""",
        )
        basedai.wallet.add_args(regen_personalkey_parser)
        basedai.basednode.add_args(regen_personalkey_parser)


class RegenPersonalkeypubCommand:
    """
    Executes the ``import_personalkeypub`` command to regenerate the public part of a personalkey (personalkeypub) for a wallet on the Basedai network.

    This command is used when a user needs to recreate their personalkeypub from an existing public key or SS58 address.

    Usage:
        The command requires either a public key in hexadecimal format or an ``SS58`` address to regenerate the personalkeypub. It optionally allows overwriting an existing personalkeypub file.

    Optional arguments:
        - ``--public_key_hex`` (str): The public key in hex format.
        - ``--ss58_address`` (str): The SS58 address of the personalkey.
        - ``--overwrite_personalkeypub`` (bool): Overwrites the existing personalkeypub file with the new one.

    Example usage::

        basedcli wallet import_personalkeypub --ss58_address 5DkQ4...

    Note:
        This command is particularly useful for users who need to regenerate their personalkeypub, perhaps due to file corruption or loss.
        It is a recovery-focused utility that ensures continued access to wallet functionalities.
    """

    def run(cli):
        r"""Creates a new personalkeypub under this wallet."""
        wallet = basedai.wallet(config=cli.config)
        wallet.regenerate_personalkeypub(
            ss58_address=cli.config.get("ss58_address"),
            public_key=cli.config.get("public_key_hex"),
            overwrite=cli.config.overwrite_personalkeypub,
        )

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if config.ss58_address == None and config.public_key_hex == None:
            prompt_answer = Prompt.ask(
                "Enter the ss58_address or the public key in hex"
            )
            if prompt_answer.startswith("0x"):
                config.public_key_hex = prompt_answer
            else:
                config.ss58_address = prompt_answer
        if not basedai.utils.is_valid_basedai_address_or_public_key(
            address=(
                config.ss58_address if config.ss58_address else config.public_key_hex
            )
        ):
            sys.exit(1)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        regen_personalkeypub_parser = parser.add_parser(
            "import_personalkeypub",
            help="""Regenerates a personalkeypub from the public part of the personalkey.""",
        )
        regen_personalkeypub_parser.add_argument(
            "--public_key",
            "--pubkey",
            dest="public_key_hex",
            required=False,
            default=None,
            type=str,
            help="The public key (in hex) of the personalkey to regen e.g. 0x1234 ...",
        )
        regen_personalkeypub_parser.add_argument(
            "--ss58_address",
            "--addr",
            "--ss58",
            dest="ss58_address",
            required=False,
            default=None,
            type=str,
            help="The ss58 address of the personalkey to regen e.g. 5ABCD ...",
        )
        regen_personalkeypub_parser.add_argument(
            "--overwrite_personalkeypub",
            default=False,
            action="store_true",
            help="""Overwrite the old personalkeypub file with the newly generated personalkeypub""",
        )
        basedai.wallet.add_args(regen_personalkeypub_parser)
        basedai.basednode.add_args(regen_personalkeypub_parser)


class RegenComputekeyCommand:
    """
    Executes the ``import_computekey`` command to regenerate a computekey for a wallet on the Basedai network.

    Similar to regenerating a personalkey, this command creates a new computekey from a mnemonic, seed, or JSON file.

    Usage:
        Users can provide a mnemonic, seed string, or a JSON file to regenerate the computekey.
        The command supports optional password protection and can overwrite an existing computekey.

    Optional arguments:
        - ``--mnemonic`` (str): A mnemonic phrase used to regenerate the key.
        - ``--seed`` (str): A seed hex string used for key regeneration.
        - ``--json`` (str): Path to a JSON file containing an encrypted key backup.
        - ``--json_password`` (str): Password to decrypt the JSON file.
        - ``--use_password`` (bool): Enables password protection for the generated key.
        - ``--overwrite_computekey`` (bool): Overwrites the existing computekey with the new one.

    Example usage::

        basedcli wallet import_computekey
        basedcli wallet import_computekey --seed 0x1234...

    Note:
        This command is essential for users who need to regenerate their computekey, possibly for security upgrades or key recovery.
        It should be used cautiously to avoid accidental overwrites of existing keys.
    """

    def run(cli):
        r"""Creates a new personalkey under this wallet."""
        wallet = basedai.wallet(config=cli.config)

        json_str: Optional[str] = None
        json_password: Optional[str] = None
        if cli.config.get("json"):
            file_name: str = cli.config.get("json")
            if not os.path.exists(file_name) or not os.path.isfile(file_name):
                raise ValueError("File {} does not exist".format(file_name))
            with open(cli.config.get("json"), "r") as f:
                json_str = f.read()

            # Password can be "", assume if None
            json_password = cli.config.get("json_password", "")

        wallet.regenerate_computekey(
            mnemonic=cli.config.mnemonic,
            seed=cli.config.seed,
            json=(json_str, json_password),
            use_password=cli.config.use_password,
            overwrite=cli.config.overwrite_computekey,
        )

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.computekey") and not config.no_prompt:
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)
        if (
            config.mnemonic == None
            and config.get("seed", d=None) == None
            and config.get("json", d=None) == None
        ):
            prompt_answer = Prompt.ask("Enter mnemonic, seed, or json file location")
            if prompt_answer.startswith("0x"):
                config.seed = prompt_answer
            elif len(prompt_answer.split(" ")) > 1:
                config.mnemonic = prompt_answer
            else:
                config.json = prompt_answer

        if config.get("json", d=None) and config.get("json_password", d=None) == None:
            config.json_password = Prompt.ask(
                "Enter json backup password", password=True
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        regen_computekey_parser = parser.add_parser(
            "import_computekey", help="""Regenerates a computekey from a passed mnemonic"""
        )
        regen_computekey_parser.add_argument(
            "--mnemonic",
            required=False,
            nargs="+",
            help="Mnemonic used to regen your key i.e. horse cart dog ...",
        )
        regen_computekey_parser.add_argument(
            "--seed",
            required=False,
            default=None,
            help="Seed hex string used to regen your key i.e. 0x1234...",
        )
        regen_computekey_parser.add_argument(
            "--json",
            required=False,
            default=None,
            help="""Path to a json file containing the encrypted key backup. (e.g. from PolkadotJS)""",
        )
        regen_computekey_parser.add_argument(
            "--json_password",
            required=False,
            default=None,
            help="""Password to decrypt the json file.""",
        )
        regen_computekey_parser.add_argument(
            "--use_password",
            dest="use_password",
            action="store_true",
            help="""Set true to protect the generated basedai key with a password.""",
            default=False,
        )
        regen_computekey_parser.add_argument(
            "--no_password",
            dest="use_password",
            action="store_false",
            help="""Set off protects the generated basedai key with a password.""",
        )
        regen_computekey_parser.add_argument(
            "--overwrite_computekey",
            dest="overwrite_computekey",
            action="store_true",
            default=False,
            help="""Overwrite the old computekey with the newly generated computekey""",
        )
        basedai.wallet.add_args(regen_computekey_parser)
        basedai.basednode.add_args(regen_computekey_parser)


class NewComputekeyCommand:
    """
    Executes the ``new_computekey`` command to create a new computekey under a wallet on the Basedai network.

    This command is used to generate a new computekey for managing a neuron or participating in the network.

    Usage:
        The command creates a new computekey with an optional word count for the mnemonic and supports password protection.
        It also allows overwriting an existing computekey.

    Optional arguments:
        - ``--n_words`` (int): The number of words in the mnemonic phrase.
        - ``--use_password`` (bool): Enables password protection for the generated key.
        - ``--overwrite_computekey`` (bool): Overwrites the existing computekey with the new one.

    Example usage::

        basedcli wallet new_computekey --n_words 24

    Note:
        This command is useful for users who wish to create additional computekeys for different purposes,
        such as running multiple miners or separating operational roles within the network.
    """

    def run(cli):
        """Creates a new hotke under this wallet."""
        wallet = basedai.wallet(config=cli.config)
        wallet.create_new_computekey(
            n_words=cli.config.n_words,
            use_password=cli.config.use_password,
            overwrite=cli.config.overwrite_computekey,
        )

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.computekey") and not config.no_prompt:
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        new_computekey_parser = parser.add_parser(
            "new_computekey",
            help="""Creates a new computekey (for running a miner) under the specified path.""",
        )
        new_computekey_parser.add_argument(
            "--n_words",
            type=int,
            choices=[12, 15, 18, 21, 24],
            default=12,
            help="""The number of words representing the mnemonic. i.e. horse cart dog ... x 24""",
        )
        new_computekey_parser.add_argument(
            "--use_password",
            dest="use_password",
            action="store_true",
            help="""Set true to protect the generated basedai key with a password.""",
            default=False,
        )
        new_computekey_parser.add_argument(
            "--no_password",
            dest="use_password",
            action="store_false",
            help="""Set off protects the generated basedai key with a password.""",
        )
        new_computekey_parser.add_argument(
            "--overwrite_computekey",
            action="store_false",
            default=False,
            help="""Overwrite the old computekey with the newly generated computekey""",
        )
        basedai.wallet.add_args(new_computekey_parser)
        basedai.basednode.add_args(new_computekey_parser)


class NewPersonalkeyCommand:
    """
    Executes the ``new_personalkey`` command to create a new personalkey under a wallet on the Basedai network.

    This command generates a personalkey, which is essential for holding balances and performing high-value transactions.

    Usage:
        The command creates a new personalkey with an optional word count for the mnemonic and supports password protection.
        It also allows overwriting an existing personalkey.

    Optional arguments:
        - ``--n_words`` (int): The number of words in the mnemonic phrase.
        - ``--use_password`` (bool): Enables password protection for the generated key.
        - ``--overwrite_personalkey`` (bool): Overwrites the existing personalkey with the new one.

    Example usage::

        basedcli wallet new_personalkey --n_words 15

    Note:
        This command is crucial for users who need to create a new personalkey for enhanced security or as part of setting up a new wallet.
        It's a foundational step in establishing a secure presence on the Basedai network.
    """

    def run(cli):
        r"""Creates a new personalkey under this wallet."""
        wallet = basedai.wallet(config=cli.config)
        wallet.create_new_personalkey(
            n_words=cli.config.n_words,
            use_password=cli.config.use_password,
            overwrite=cli.config.overwrite_personalkey,
        )

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        new_personalkey_parser = parser.add_parser(
            "new_personalkey",
            help="""Creates a new personalkey (for containing balance) under the specified path. """,
        )
        new_personalkey_parser.add_argument(
            "--n_words",
            type=int,
            choices=[12, 15, 18, 21, 24],
            default=12,
            help="""The number of words representing the mnemonic. i.e. horse cart dog ... x 24""",
        )
        new_personalkey_parser.add_argument(
            "--use_password",
            dest="use_password",
            action="store_true",
            help="""Set true to protect the generated basedai key with a password.""",
            default=True,
        )
        new_personalkey_parser.add_argument(
            "--no_password",
            dest="use_password",
            action="store_false",
            help="""Set off protects the generated basedai key with a password.""",
        )
        new_personalkey_parser.add_argument(
            "--overwrite_personalkey",
            action="store_false",
            default=False,
            help="""Overwrite the old personalkey with the newly generated personalkey""",
        )
        basedai.wallet.add_args(new_personalkey_parser)
        basedai.basednode.add_args(new_personalkey_parser)


class WalletCreateCommand:
    """
    Executes the ``create`` command to generate both a new personalkey and computekey under a specified wallet on the Basedai network.

    This command is a comprehensive utility for creating a complete wallet setup with both cold and computekeys.

    Usage:
        The command facilitates the creation of a new personalkey and computekey with an optional word count for the mnemonics.
        It supports password protection for the personalkey and allows overwriting of existing keys.

    Optional arguments:
        - ``--n_words`` (int): The number of words in the mnemonic phrase for both keys.
        - ``--use_password`` (bool): Enables password protection for the personalkey.
        - ``--overwrite_personalkey`` (bool): Overwrites the existing personalkey with the new one.
        - ``--overwrite_computekey`` (bool): Overwrites the existing computekey with the new one.

    Example usage::

        basedcli wallet create --n_words 21

    Note:
        This command is ideal for new users setting up their wallet for the first time or for those who wish to completely renew their wallet keys.
        It ensures a fresh start with new keys for secure and effective participation in the network.
    """

    def run(cli):
        r"""Creates a new personalkey and computekey under this wallet."""
        wallet = basedai.wallet(config=cli.config)
        wallet.create_new_personalkey(
            n_words=cli.config.n_words,
            use_password=cli.config.use_password,
            overwrite=cli.config.overwrite_personalkey,
        )
        wallet.create_new_computekey(
            n_words=cli.config.n_words,
            use_password=False,
            overwrite=cli.config.overwrite_computekey,
        )

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.computekey") and not config.no_prompt:
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        new_personalkey_parser = parser.add_parser(
            "create",
            help="""Creates a new personalkey (for containing balance) under the specified path. """,
        )
        new_personalkey_parser.add_argument(
            "--n_words",
            type=int,
            choices=[12, 15, 18, 21, 24],
            default=12,
            help="""The number of words representing the mnemonic. i.e. horse cart dog ... x 24""",
        )
        new_personalkey_parser.add_argument(
            "--use_password",
            dest="use_password",
            action="store_true",
            help="""Set true to protect the generated basedai key with a password.""",
            default=True,
        )
        new_personalkey_parser.add_argument(
            "--no_password",
            dest="use_password",
            action="store_false",
            help="""Set off protects the generated basedai key with a password.""",
        )
        new_personalkey_parser.add_argument(
            "--overwrite_personalkey",
            action="store_false",
            default=False,
            help="""Overwrite the old personalkey with the newly generated personalkey""",
        )
        new_personalkey_parser.add_argument(
            "--overwrite_computekey",
            action="store_false",
            default=False,
            help="""Overwrite the old computekey with the newly generated computekey""",
        )
        basedai.wallet.add_args(new_personalkey_parser)
        basedai.basednode.add_args(new_personalkey_parser)


def _get_personalkey_wallets_for_path(path: str) -> List["basedai.wallet"]:
    """Get all personalkey wallet names from path."""
    try:
        wallet_names = next(os.walk(os.path.expanduser(path)))[1]
        return [basedai.wallet(path=path, name=name) for name in wallet_names]
    except StopIteration:
        # No wallet files found.
        wallets = []
    return wallets

def _get_personalkey_ss58_addresses_for_path(path: str) -> List[str]:
    """Get all personalkey ss58 addresses from path."""

    def list_personalkeypub_files(dir_path):
        abspath = os.path.abspath(os.path.expanduser(dir_path))
        personalkey_files = []

        for file in os.listdir(abspath):
            personalkey_path = os.path.join(abspath, file, "personalkeypub.txt")
            if os.path.exists(personalkey_path):
                personalkey_files.append(personalkey_path)
            else:
                basedai.logging.warning(
                    f"{personalkey_path} does not exist. Excluding..."
                )
        return personalkey_files

    return [
        basedai.keyfile(file).keypair.ss58_address
        for file in list_personalkeypub_files(path)
    ]

def ss58_to_ethereum(ss58_address):
    public_key = ss58_decode(ss58_address)

    keccak_hash = hashlib.sha3_256(bytes.fromhex(public_key)).digest()
    eth_address = keccak_hash[-20:]

    return '0x' + eth_address.hex()

class WalletBalanceCommand:
    """
    Executes the ``balance`` command to check the balance of the wallet on the Basedai network.

    This command provides a detailed view of the wallet's personalkey balances, including free and staked balances.

    Usage:
        The command lists the balances of all wallets in the user's configuration directory, showing the wallet name, personalkey address, and the respective free and staked balances.

    Optional arguments:
        None. The command uses the wallet and basednode configurations to fetch balance data.

    Example usage::

        basedcli wallet balance

    Note:
        This command is essential for users to monitor their financial status on the Basedai network.
        It helps in keeping track of assets and ensuring the wallet's financial health.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        """Check the balance of the wallet."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            WalletBalanceCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        """Check the balance of the wallet."""
        wallet_names = os.listdir(os.path.expanduser(cli.config.wallet.path))
        personalkeys = _get_personalkey_ss58_addresses_for_path(cli.config.wallet.path)

        free_balances = [
            basednode.get_balance(personalkeys[i]) for i in range(len(personalkeys))
        ]
        staked_balances = [
            basednode.get_total_stake_for_personalkey(personalkeys[i])
            for i in range(len(personalkeys))
        ]
        total_free_balance = sum(free_balances)
        total_staked_balance = sum(staked_balances)

        balances = {
            name: (personalkey, free, staked)
            for name, personalkey, free, staked in sorted(
                zip(wallet_names, personalkeys, free_balances, staked_balances)
            )
        }

        table = Table(show_footer=False)
        table.add_column(
            "[white]NAME",
            header_style="overline white",
            footer_style="overline white",
            style="cyan",
            no_wrap=True,
        )

        table.add_column(
            "[white]ADDRESS",
            header_style="overline white",
            footer_style="overline white",
            style="cyan",
            no_wrap=True,
        )

        table.add_column(
            "[white]EVM ADDRESS",
            header_style="overline white",
            footer_style="overline white",
            style="cyan",
            no_wrap=True,
        )

        for typestr in ["COMPUTABLE", "STAKED", "TOTAL"]:
            table.add_column(
                f"[white]{typestr}",
                header_style="overline white",
                footer_style="overline white",
                justify="right",
                style="bold cyan",
                no_wrap=True,
            )

        for name, (personalkey, free, staked) in balances.items():
            table.add_row(
                name,
                personalkey,
                ss58_to_ethereum(personalkey),
                str(free),
                str(staked),
                str(free + staked),
            )
        table.add_row()
        table.add_row(
            "",
            "",
            "",
            str(total_free_balance),
            str(total_staked_balance),
            str(total_free_balance + total_staked_balance),
        )
        table.show_footer = True
        table.box = None
        table.pad_edge = False
        table.width = None
        basedai.__console__.print(table)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        balance_parser = parser.add_parser(
            "balance", help="""Checks the different types of balances in the wallet with EVM and native addresses."""
        )
        basedai.wallet.add_args(balance_parser)
        basedai.basednode.add_args(balance_parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.path") and not config.no_prompt:
            path = Prompt.ask("Where are you wallets located? Default:", default=defaults.wallet.path)
            config.wallet.path = str(path)

        if not config.is_set("basednode.network") and not config.no_prompt:
            network = Prompt.ask(
                    "Create link with stem: ",
                default=defaults.basednode.network,
                choices=basedai.__networks__,
            )
            config.basednode.network = str(network)
            (
                _,
                config.basednode.chain_endpoint,
            ) = basedai.basednode.determine_chain_endpoint_and_network(str(network))


API_URL = "https://api.subquery.network/sq/basedaiscan/basedai-indexer"
MAX_TXN = 1000
GRAPHQL_QUERY = """
query ($first: Int!, $after: Cursor, $filter: TransferFilter, $order: [TransfersOrderBy!]!) {
    transfers(first: $first, after: $after, filter: $filter, orderBy: $order) {
        nodes {
            id
            from
            to
            amount
            extrinsicId
            blockNumber
        }
        pageInfo {
            endCursor
            hasNextPage
            hasPreviousPage
        }
        totalCount
    }
}
"""


class GetWalletHistoryCommand:
    """
    Executes the ``history`` command to fetch the latest transfers of the provided wallet on the Basedai network.

    This command provides a detailed view of the transfers carried out on the wallet.

    Usage:
        The command lists the latest transfers of the provided wallet, showing the From, To, Amount, Extrinsic Id and Block Number.

    Optional arguments:
        None. The command uses the wallet and basednode configurations to fetch latest transfer data associated with a wallet.

    Example usage::

        basedcli wallet history

    Note:
        This command is essential for users to monitor their financial status on the Basedai network.
        It helps in fetching info on all the transfers so that user can easily tally and cross check the transactions.
    """

    @staticmethod
    def run(cli):
        r"""Check the transfer history of the provided wallet."""
        wallet = basedai.wallet(config=cli.config)
        wallet_address = wallet.get_personalkeypub().ss58_address
        # Fetch all transfers
        transfers = get_wallet_transfers(wallet_address)

        # Create output table
        table = create_transfer_history_table(transfers)

        basedai.__console__.print(table)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        history_parser = parser.add_parser(
            "history",
            help="""Fetch transfer history associated with the provided wallet""",
        )
        basedai.wallet.add_args(history_parser)
        basedai.basednode.add_args(history_parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)


def get_wallet_transfers(wallet_address) -> List[dict]:
    """Get all transfers associated with the provided wallet address."""

    variables = {
        "first": MAX_TXN,
        "filter": {
            "or": [
                {"from": {"equalTo": wallet_address}},
                {"to": {"equalTo": wallet_address}},
            ]
        },
        "order": "BLOCK_NUMBER_DESC",
    }

    response = requests.post(
        API_URL, json={"query": GRAPHQL_QUERY, "variables": variables}
    )
    data = response.json()

    # Extract nodes and pageInfo from the response
    transfer_data = data.get("data", {}).get("transfers", {})
    transfers = transfer_data.get("nodes", [])

    return transfers


def create_transfer_history_table(transfers):
    """Get output transfer table"""

    table = Table(show_footer=False)
    # Define the column names
    column_names = [
        "Id",
        "From",
        "To",
        "Amount (Based)",
        "Extrinsic Id",
        "Block Number",
        "URL (basedaiscan)",
    ]
    basedaiscan_url_base = "https://x.basedaiscan.com/extrinsic"

    # Create a table
    table = Table(show_footer=False)
    table.title = "[white]Wallet Transfers"

    # Define the column styles
    header_style = "overline white"
    footer_style = "overline white"
    column_style = "green"
    no_wrap = True

    # Add columns to the table
    for column_name in column_names:
        table.add_column(
            f"[white]{column_name}",
            header_style=header_style,
            footer_style=footer_style,
            style=column_style,
            no_wrap=no_wrap,
            justify="left" if column_name == "Id" else "right",
        )

    # Add rows to the table
    for item in transfers:
        try:
            based_amount = int(item["amount"]) / RAOPERBASED
        except:
            based_amount = item["amount"]
        table.add_row(
            item["id"],
            item["from"],
            item["to"],
            f"{based_amount:.3f}",
            str(item["extrinsicId"]),
            item["blockNumber"],
            f"{basedaiscan_url_base}/{item['blockNumber']}-{item['extrinsicId']}",
        )
    table.add_row()
    table.show_footer = True
    table.box = None
    table.pad_edge = False
    table.width = None
    return table
