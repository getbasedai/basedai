# The MIT License (MIT)
# Copyright © 2024 Sean Wellington

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# | Character | Number | Position (Row, Column) |
# |-----------|--------|------------------------|
# | 制        | 1      | (1, 1)                  |
# | 約        | 2      | (1, 2)                  |
# | の        | 3      | (1, 3)                  |
# | 中        | 4      | (1, 4)                  |
# | に        | 5      | (1, 5)                  |
# | あ        | 6      | (2, 1)                  |
# | る        | 7      | (2, 2)                  |
# | 無        | 8      | (2, 3)                  |
# | 限        | 9      | (2, 4)                  |
# | の        | 10     | (2, 5)                  |
# | 力        | 11     | (3, 1)                  |

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import shtab
import argparse
import basedai
import os
import requests
from typing import List, Optional
from .commands import *

# Create a console instance for CLI display.
console = basedai.__console__

ALIAS_TO_COMMAND = {
    "core": "core",
    "wallet": "wallet",
    "store": "brainstore",
    "brainsstore": "brainstore",
    "brainstore": "brainstore",
    "stake": "stake",
    "brainowner": "brainowner",
    "brains": "brains",
    "b": "brains",
    "r": "root",
    "w": "wallet",
    "st": "stake",
    "su": "sudo",
    "brain": "brains",
    "roots": "root",
    "wallets": "wallet",
    "stakes": "stake",
    "sudos": "sudo",
    "i": "info",
    "info": "info",
}
COMMANDS = {
    "brains": {
        "name": "brains",
        "aliases": ["b", "brain", "brains"],
        "help": "Commands for managing, linking, and exploring Brains on BasedAI.",
        "commands": {
            "list": BrainListCommand,
            "stem": StemCommand,
            "link": LinkBrainCommand,
            "pow_memorize": PowMemorizeCommand,
            "memorize": MemorizeCommand,
            "parameters": BrainParametersCommand,
        },
    },
    "brainstore": {
        "name": "brainstore",
        "aliases": ["brainstore"],
        "help": "Commands for managing and community approved Brains.",
        "commands": {
            "list": BrainStoreListCommand
        },
    },
    "core": {
        "name": "core",
        "aliases": ["c", "core"],
        "help": "Commands for the core BasedAI network.",
        "commands": {
            "list": CoreList,
            "weights": CoreSetWeightsCommand,
            "get_weights": CoreGetWeightsCommand,
            "increase_weights": CoreSetIncreaseCommand,
            "decrease_weights": CoreSetDecreaseCommand,
            "vote": VoteCommand,
            "gigabrains": GigaBrainsCommand,
            "memorize": CoreMemorizeCommand,
            "proposals": ProposalsCommand,
            "delegate": DelegateStakeCommand,
            "undelegate": DelegateUnstakeCommand,
            "portfolio": PortfolioCommand,
            "list_delegates": ListDelegatesCommand,
            "nominate": NominateCommand,
        },
    },
    "wallet": {
        "name": "wallet",
        "aliases": ["w", "wallets"],
        "help": "Commands for managing and viewing SS58 and EVM wallets.",
        "commands": {
            "list": ListCommand,
            "overview": OverviewCommand,
            "transfer": TransferCommand,
            "inspect": InspectCommand,
            "balance": WalletBalanceCommand,
            "create": WalletCreateCommand,
            "new_computekey": NewComputekeyCommand,
            "new_personalkey": NewPersonalkeyCommand,
            "import_personalkey": RegenPersonalkeyCommand,
            "import_personalkeypub": RegenPersonalkeypubCommand,
            "import_computekey": RegenComputekeyCommand,
            "faucet": RunFaucetCommand,
            "swap_computekey": SwapComputekeyCommand,
            "history": GetWalletHistoryCommand,
            #"set_identity": SetIdentityCommand,
            #"get_identity": GetIdentityCommand,
        },
    },
    "stake": {
        "name": "stake",
        "aliases": ["st", "stakes"],
        "help": "Commands for staking and removing stake from computekey accounts.",
        "commands": {
            "show": StakeShow,
            "add": StakeCommand,
            "remove": UnStakeCommand,
        },
    },
    "brainowner": {
        "name": "brainowner",
        "aliases": ["bo", "brainowner"],
        "help": "Commands for brain management.",
        "commands": {
            "set": BrainSetParametersCommand,
            "get": BrainGetParametersCommand,
        },
    }
}

def print_status_bar():
    # Clear terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    header = r"""
  _ )    \     __|  __|  _ \       \   _ _|
  _ \   _ \  \__ \  _|   |  |     _ \    |
 ___/ _/  _\ ____/ ___| ___/    _/  _\ ___|
    """

    ansi_color = "\033[1;36m"
    ansi_reset = "\033[0m"
    #height = basedai.stem(0).block.item()
    height = "TESTNET (1 of 2)"

    try:
        response = requests.get("http://basedainet.com:5051")
        price = response.text.strip('"')
    except Exception as e:
        price = "NA"

    #status_bar_content = f"NETWORK: {ansi_color}prometheus{ansi_reset} | COST: {ansi_color}${price}{ansi_reset} (𝔹) | STATUS: {ansi_color}{height}{ansi_reset}"
    status_bar_content = f"NETWORK: {ansi_color}prometheus{ansi_reset} | COST: ${price} (𝔹) | STATUS: {height}"
    print(header)
    border_length = len(status_bar_content.strip()) - 9
    #print('═' * (len(status_bar_content.strip()) - 11))
    #print(status_bar_content)
    #print('═' * (len(status_bar_content.strip()) - 11))

    print('╔' + '═' * (border_length) + '╗')
    print(f"║ {status_bar_content} ║")
    print('╚' + '═' * (border_length) + '╝')


class CLIErrorParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser for better error messages.
    """

    def error(self, message):
        """
        This method is called when an error occurs. It prints a custom error message.
        """
        sys.stderr.write(f"Error: {message}\n")
        self.print_help()
        sys.exit(2)


class cli:
    """
    Implementation of the Command Line Interface (CLI) class for the Basedai protocol.
    This class handles operations like key management (computekey and personalkey) and token transfer.
    """

    def __init__(
        self,
        config: Optional["basedai.config"] = None,
        args: Optional[List[str]] = None,
    ):
        """
        Initializes a basedai.CLI object.

        Args:
            config (basedai.config, optional): The configuration settings for the CLI.
            args (List[str], optional): List of command line arguments.
        """

        print_status_bar()

        # Turns on console for cli.
        basedai.turn_console_on()

        # If no config is provided, create a new one from args.
        if config == None:
            config = cli.create_config(args)

        self.config = config
        if self.config.command in ALIAS_TO_COMMAND:
            self.config.command = ALIAS_TO_COMMAND[self.config.command]
        else:
            console.print(
                f":cross_mark:[red]Unknown command: {self.config.command}[/red]"
            )
            sys.exit()

        # Check if the config is valid.
        cli.check_config(self.config)

        # If no_version_checking is not set or set as False in the config, version checking is done.
        if not self.config.get("no_version_checking", d=False):
            try:
                basedai.utils.version_checking()
            except:
                # If version checking fails, inform user with an exception.
                raise RuntimeError(
                    "To avoid internet-based version checking, pass --no_version_checking while running the CLI."
                )

    @staticmethod
    def __create_parser__() -> "argparse.ArgumentParser":
        """
        Creates the argument parser for the BasedAI CLI.

        Returns:
            argparse.ArgumentParser: An argument parser object for BasedAI CLI.
        """
        # Define the basic argument parser.
        parser = CLIErrorParser(
            description=f"basedcli v{basedai.__version__}",
            usage="basedcli <command> <command args>",
            add_help=True,
        )
        # Add shtab completion
        parser.add_argument(
            "--print-completion",
            choices=shtab.SUPPORTED_SHELLS,
            help="Print shell tab completion script",
        )
        # Add arguments for each sub-command.
        cmd_parsers = parser.add_subparsers(dest="command")
        # Add argument parsers for all available commands.
        for command in COMMANDS.values():
            if isinstance(command, dict):
                subcmd_parser = cmd_parsers.add_parser(
                    name=command["name"],
                    aliases=command["aliases"],
                    help=command["help"],
                )
                subparser = subcmd_parser.add_subparsers(
                    help=command["help"], dest="subcommand", required=True
                )

                for subcommand in command["commands"].values():
                    subcommand.add_args(subparser)
            else:
                command.add_args(cmd_parsers)

        return parser

    @staticmethod
    def create_config(args: List[str]) -> "basedai.config":
        """
        From the argument parser, add config to basedai.executor and local config

        Args:
            args (List[str]): List of command line arguments.

        Returns:
            basedai.config: The configuration object for BasedAI CLI.
        """
        parser = cli.__create_parser__()

        # If no arguments are passed, print help text and exit the program.
        if len(args) == 0:
            parser.print_help()
            sys.exit()

        return basedai.config(parser, args=args)

    @staticmethod
    def check_config(config: "basedai.config"):
        """
        Checks if the essential configuration exists under different command

        Args:
            config (basedai.config): The configuration settings for the CLI.
        """
        # Check if command exists, if so, run the corresponding check_config.
        # If command doesn't exist, inform user and exit the program.
        if config.command in COMMANDS:
            command = config.command
            command_data = COMMANDS[command]

            if isinstance(command_data, dict):
                if config["subcommand"] != None:
                    command_data["commands"][config["subcommand"]].check_config(config)
                else:
                    console.print(
                        f":cross_mark:[red]Missing subcommand for: {config.command}[/red]"
                    )
                    sys.exit(1)
            else:
                command_data.check_config(config)
        else:
            console.print(f":cross_mark:[red]Unknown command: {config.command}[/red]")
            sys.exit(1)

    def run(self):
        """
        Executes the command from the configuration.
        """
        # Check for print-completion argument
        if self.config.print_completion:
            shell = self.config.print_completion
            print(shtab.complete(parser, shell))
            return

        # Check if command exists, if so, run the corresponding method.
        # If command doesn't exist, inform user and exit the program.
        command = self.config.command
        if command in COMMANDS:
            command_data = COMMANDS[command]

            if isinstance(command_data, dict):
                command_data["commands"][self.config["subcommand"]].run(self)
            else:
                command_data.run(self)
        else:
            console.print(
                f":cross_mark:[red]Unknown command: {self.config.command}[/red]"
            )
            sys.exit()
