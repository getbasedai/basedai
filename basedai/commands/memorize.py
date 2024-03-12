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
import argparse
import basedai
from rich.prompt import Prompt, Confirm
from .utils import check_netuid_set, check_for_cuda_reg_config
from copy import deepcopy

from . import defaults

console = basedai.__console__


class MemorizeCommand:
    """
    Executes the ``memorize`` command to have the BasedAI network memorize the address for $BASED.

    This command is used to add a new neuron to a specified Brain within the network, contributing to the decentralization and robustness of Basedai.

    Usage:
        Before registering, the command checks if the specified subnet exists and whether the user's balance is sufficient to cover the registration cost.

        The registration cost is determined by the current recycle amount for the specified subnet. If the balance is insufficient or the subnet does not exist, the command will exit with an appropriate error message.

        If the preconditions are met, and the user confirms the transaction (if ``no_prompt`` is not set), the command proceeds to register the neuron by recycling the required amount of BASED.

    The command structure includes:

    - Verification of subnet existence.
    - Checking the user's balance against the current recycle amount for the subnet.
    - User confirmation prompt for proceeding with registration.
    - Execution of the registration process.

    Columns Displayed in the confirmation prompt:

    - Balance: The current balance of the user's wallet in BASED.
    - Cost to Register: The required amount of BASED needed to register on the specified subnet.

    Example usage::

        basedcli brains register --netuid 1

    Note:
        This command is critical for users who wish to contribute a new neuron to the network. It requires careful consideration of the subnet selection and an understanding of the registration costs. Users should ensure their wallet is sufficiently funded before attempting to register a neuron.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Register agent by recycling some BASED."""
        try:
            config = cli.config.copy()
            basednode: "basedai.basednode" = basedai.basednode(
                config=config, log_verbose=False
            )
            MemorizeCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Register agent with $BASED."""
        wallet = basedai.wallet(config=cli.config)

        # Verify subnet exists
        if not basednode.subnet_exists(netuid=cli.config.netuid):
            basedai.__console__.print(
                f"[red]Brain {cli.config.netuid} does not exist[/red]"
            )
            sys.exit(1)

        # Check current recycle amount
        current_recycle = basednode.recycle(netuid=cli.config.netuid)
        balance = basednode.get_balance(address=wallet.personalkeypub.ss58_address)

        # Check balance is sufficient
        if balance < current_recycle:
            basedai.__console__.print(
                f"[red]Insufficient balance {balance} to register neuron. Current recycle is {current_recycle} BASED[/red]"
            )
            sys.exit(1)

        if not cli.config.no_prompt:
            if (
                Confirm.ask(
                    f"Your balance is: [bold green]{balance}[/bold green]\nThe cost to register by recycle is [bold red]{current_recycle}[/bold red]\nDo you want to continue?",
                    default=False,
                )
                == False
            ):
                sys.exit(1)

        basednode.burned_register(
            wallet=wallet, netuid=cli.config.netuid, prompt=not cli.config.no_prompt
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        register_parser = parser.add_parser(
            "memorize", help="""Have a Brain memorize a wallet."""
        )
        register_parser.add_argument(
            "--netuid",
            type=int,
            help="id for Brain to serve this agent on",
            default=argparse.SUPPRESS,
        )

        basedai.wallet.add_args(register_parser)
        basedai.basednode.add_args(register_parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        if (
            not config.is_set("basednode.network")
            and not config.is_set("basednode.chain_endpoint")
            and not config.no_prompt
        ):
            config.basednode.network = Prompt.ask(
                "Enter basednode network",
                choices=basedai.__networks__,
                default=defaults.basednode.network,
            )
            _, endpoint = basedai.basednode.determine_chain_endpoint_and_network(
                config.basednode.network
            )
            config.basednode.chain_endpoint = endpoint

        check_netuid_set(
            config, basednode=basedai.basednode(config=config, log_verbose=False)
        )

        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.computekey") and not config.no_prompt:
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)


class PowMemorizeCommand:
    """
    Executes the ``pow_memorize`` command to register a neuron on the Basedai network using Proof of Work (PoW).

    This method is an alternative registration process that leverages computational work for securing a neuron's place on the network.

    Usage:
        The command starts by verifying the existence of the specified subnet. If the subnet does not exist, it terminates with an error message.
        On successful verification, the PoW registration process is initiated, which requires solving computational puzzles.

    Optional arguments:
        - ``--netuid`` (int): The netuid for the subnet on which to serve the neuron. Mandatory for specifying the target subnet.
        - ``--pow_memorize.num_processes`` (int): The number of processors to use for PoW registration. Defaults to the system's default setting.
        - ``--pow_memorize.update_interval`` (int): The number of nonces to process before checking for the next block during registration. Affects the frequency of update checks.
        - ``--pow_memorize.no_output_in_place`` (bool): When set, disables the output of registration statistics in place. Useful for cleaner logs.
        - ``--pow_memorize.verbose`` (bool): Enables verbose output of registration statistics for detailed information.
        - ``--pow_memorize.cuda.use_cuda`` (bool): Enables the use of CUDA for GPU-accelerated PoW calculations. Requires a CUDA-compatible GPU.
        - ``--pow_memorize.cuda.no_cuda`` (bool): Disables the use of CUDA, defaulting to CPU-based calculations.
        - ``--pow_memorize.cuda.dev_id`` (int): Specifies the CUDA device ID, useful for systems with multiple CUDA-compatible GPUs.
        - ``--pow_memorize.cuda.tpb`` (int): Sets the number of Threads Per Block for CUDA operations, affecting the GPU calculation dynamics.

    The command also supports additional wallet and basednode arguments, enabling further customization of the registration process.

    Example usage::

        basedcli pow_memorize --netuid 1 --pow_memorize.num_processes 4 --cuda.use_cuda

    Note:
        This command is suited for users with adequate computational resources to participate in PoW registration. It requires a sound understanding
        of the network's operations and PoW mechanics. Users should ensure their systems meet the necessary hardware and software requirements,
        particularly when opting for CUDA-based GPU acceleration.

    This command may be disabled according on the subnet owner's directive. For example, on netuid 1 this is permanently disabled.
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Register agent."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            PowMemorizeCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Register agent."""
        wallet = basedai.wallet(config=cli.config)

        # Verify subnet exists
        if not basednode.subnet_exists(netuid=cli.config.netuid):
            basedai.__console__.print(
                f"[red]Brain {cli.config.netuid} does not exist[/red]"
            )
            sys.exit(1)

        basednode.register(
            wallet=wallet,
            netuid=cli.config.netuid,
            prompt=not cli.config.no_prompt,
            tpb=cli.config.pow_memorize.cuda.get("tpb", None),
            update_interval=cli.config.pow_memorize.get("update_interval", None),
            num_processes=cli.config.pow_memorize.get("num_processes", None),
            cuda=cli.config.pow_memorize.cuda.get(
                "use_cuda", defaults.pow_register.cuda.use_cuda
            ),
            dev_id=cli.config.pow_memorize.cuda.get("dev_id", None),
            output_in_place=cli.config.pow_memorize.get(
                "output_in_place", defaults.pow_register.output_in_place
            ),
            log_verbose=cli.config.pow_memorize.get(
                "verbose", defaults.pow_register.verbose
            ),
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        register_parser = parser.add_parser(
            "pow_memorize", help="""Have a Brain memorize a wallet using PoW."""
        )
        register_parser.add_argument(
            "--netuid",
            type=int,
            help="netuid for subnet to serve this neuron on",
            default=argparse.SUPPRESS,
        )
        register_parser.add_argument(
            "--pow_memorize.num_processes",
            "-n",
            dest="pow_memorize.num_processes",
            help="Number of processors to use for POW registration",
            type=int,
            default=defaults.pow_register.num_processes,
        )
        register_parser.add_argument(
            "--pow_memorize.update_interval",
            "--pow_memorize.cuda.update_interval",
            "--cuda.update_interval",
            "-u",
            help="The number of nonces to process before checking for next block during registration",
            type=int,
            default=defaults.pow_register.update_interval,
        )
        register_parser.add_argument(
            "--pow_memorize.no_output_in_place",
            "--no_output_in_place",
            dest="pow_memorize.output_in_place",
            help="Whether to not ouput the registration statistics in-place. Set flag to disable output in-place.",
            action="store_false",
            required=False,
            default=defaults.pow_register.output_in_place,
        )
        register_parser.add_argument(
            "--pow_memorize.verbose",
            help="Whether to ouput the registration statistics verbosely.",
            action="store_true",
            required=False,
            default=defaults.pow_register.verbose,
        )

        ## Registration args for CUDA registration.
        register_parser.add_argument(
            "--pow_memorize.cuda.use_cuda",
            "--cuda",
            "--cuda.use_cuda",
            dest="pow_memorize.cuda.use_cuda",
            default=defaults.pow_register.cuda.use_cuda,
            help="""Set flag to use CUDA to register.""",
            action="store_true",
            required=False,
        )
        register_parser.add_argument(
            "--pow_memorize.cuda.no_cuda",
            "--no_cuda",
            "--cuda.no_cuda",
            dest="pow_memorize.cuda.use_cuda",
            default=not defaults.pow_register.cuda.use_cuda,
            help="""Set flag to not use CUDA for registration""",
            action="store_false",
            required=False,
        )

        register_parser.add_argument(
            "--pow_memorize.cuda.dev_id",
            "--cuda.dev_id",
            type=int,
            nargs="+",
            default=defaults.pow_register.cuda.dev_id,
            help="""Set the CUDA device id(s). Goes by the order of speed. (i.e. 0 is the fastest).""",
            required=False,
        )
        register_parser.add_argument(
            "--pow_memorize.cuda.tpb",
            "--cuda.tpb",
            type=int,
            default=defaults.pow_register.cuda.tpb,
            help="""Set the number of Threads Per Block for CUDA.""",
            required=False,
        )

        basedai.wallet.add_args(register_parser)
        basedai.basednode.add_args(register_parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        if (
            not config.is_set("basednode.network")
            and not config.is_set("basednode.chain_endpoint")
            and not config.no_prompt
        ):
            config.basednode.network = Prompt.ask(
                "Enter basednode network",
                choices=basedai.__networks__,
                default=defaults.basednode.network,
            )
            _, endpoint = basedai.basednode.determine_chain_endpoint_and_network(
                config.basednode.network
            )
            config.basednode.chain_endpoint = endpoint

        check_netuid_set(
            config, basednode=basedai.basednode(config=config, log_verbose=False)
        )

        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.computekey") and not config.no_prompt:
            computekey = Prompt.ask("Enter computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)

        if not config.no_prompt:
            check_for_cuda_reg_config(config)


class RunFaucetCommand:
    """
    Executes the ``faucet`` command to obtain test BASED tokens by performing Proof of Work (PoW).

    IMPORTANT:
        **THIS COMMAND IS CURRENTLY DISABLED.**

    This command is particularly useful for users who need test tokens for operations on the Basedai testnet.

    Usage:
        The command uses the PoW mechanism to validate the user's effort and rewards them with test BASED tokens. It is typically used in testnet environments where real value transactions are not necessary.

    Optional arguments:
        - ``--faucet.num_processes`` (int): Specifies the number of processors to use for the PoW operation. A higher number of processors may increase the chances of successful computation.
        - ``--faucet.update_interval`` (int): Sets the frequency of nonce processing before checking for the next block, which impacts the PoW operation's responsiveness.
        - ``--faucet.no_output_in_place`` (bool): When set, it disables in-place output of registration statistics for cleaner log visibility.
        - ``--faucet.verbose`` (bool): Enables verbose output for detailed statistical information during the PoW process.
        - ``--faucet.cuda.use_cuda`` (bool): Activates the use of CUDA for GPU acceleration in the PoW process, suitable for CUDA-compatible GPUs.
        - ``--faucet.cuda.no_cuda`` (bool): Disables the use of CUDA, opting for CPU-based calculations.
        - ``--faucet.cuda.dev_id`` (int[]): Allows selection of specific CUDA device IDs for the operation, useful in multi-GPU setups.
        - ``--faucet.cuda.tpb`` (int): Determines the number of Threads Per Block for CUDA operations, affecting GPU calculation efficiency.

    These options provide flexibility in configuring the PoW process according to the user's hardware capabilities and preferences.

    Example usage::

        basedcli wallet faucet --faucet.num_processes 4 --faucet.cuda.use_cuda

    Note:
        This command is meant for use in testnet environments where users can experiment with the network without using real BASED tokens.
        It's important for users to have the necessary hardware setup, especially when opting for CUDA-based GPU calculations.

    **THIS COMMAND IS CURRENTLY DISABLED.**
    """

    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Register agent."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            RunFaucetCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Register agent."""
        wallet = basedai.wallet(config=cli.config)
        basednode.run_faucet(
            wallet=wallet,
            prompt=not cli.config.no_prompt,
            tpb=cli.config.pow_memorize.cuda.get("tpb", None),
            update_interval=cli.config.pow_memorize.get("update_interval", None),
            num_processes=cli.config.pow_memorize.get("num_processes", None),
            cuda=cli.config.pow_memorize.cuda.get(
                "use_cuda", defaults.pow_register.cuda.use_cuda
            ),
            dev_id=cli.config.pow_memorize.cuda.get("dev_id", None),
            output_in_place=cli.config.pow_memorize.get(
                "output_in_place", defaults.pow_register.output_in_place
            ),
            log_verbose=cli.config.pow_memorize.get(
                "verbose", defaults.pow_register.verbose
            ),
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        run_faucet_parser = parser.add_parser(
            "faucet", help="""Receive test Based (𝔹)."""
        )
        run_faucet_parser.add_argument(
            "--faucet.num_processes",
            "-n",
            dest="pow_memorize.num_processes",
            help="Number of processors to use for POW registration",
            type=int,
            default=defaults.pow_register.num_processes,
        )
        run_faucet_parser.add_argument(
            "--faucet.update_interval",
            "--faucet.cuda.update_interval",
            "--cuda.update_interval",
            "-u",
            help="The number of nonces to process before checking for next block during registration",
            type=int,
            default=defaults.pow_register.update_interval,
        )
        run_faucet_parser.add_argument(
            "--faucet.no_output_in_place",
            "--no_output_in_place",
            dest="pow_memorize.output_in_place",
            help="Whether to not ouput the registration statistics in-place. Set flag to disable output in-place.",
            action="store_false",
            required=False,
            default=defaults.pow_register.output_in_place,
        )
        run_faucet_parser.add_argument(
            "--faucet.verbose",
            help="Whether to ouput the registration statistics verbosely.",
            action="store_true",
            required=False,
            default=defaults.pow_register.verbose,
        )

        ## Registration args for CUDA registration.
        run_faucet_parser.add_argument(
            "--faucet.cuda.use_cuda",
            "--cuda",
            "--cuda.use_cuda",
            dest="pow_memorize.cuda.use_cuda",
            default=defaults.pow_register.cuda.use_cuda,
            help="""Set flag to use CUDA to pow_memorize.""",
            action="store_true",
            required=False,
        )
        run_faucet_parser.add_argument(
            "--faucet.cuda.no_cuda",
            "--no_cuda",
            "--cuda.no_cuda",
            dest="pow_memorize.cuda.use_cuda",
            default=not defaults.pow_register.cuda.use_cuda,
            help="""Set flag to not use CUDA for memorization""",
            action="store_false",
            required=False,
        )
        run_faucet_parser.add_argument(
            "--faucet.cuda.dev_id",
            "--cuda.dev_id",
            type=int,
            nargs="+",
            default=defaults.pow_register.cuda.dev_id,
            help="""Set the CUDA device id(s). Goes by the order of speed. (i.e. 0 is the fastest).""",
            required=False,
        )
        run_faucet_parser.add_argument(
            "--faucet.cuda.tpb",
            "--cuda.tpb",
            type=int,
            default=defaults.pow_register.cuda.tpb,
            help="""Set the number of Threads Per Block for CUDA.""",
            required=False,
        )
        basedai.wallet.add_args(run_faucet_parser)
        basedai.basednode.add_args(run_faucet_parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.no_prompt:
            check_for_cuda_reg_config(config)


class SwapComputekeyCommand:
    @staticmethod
    def run(cli: "basedai.cli"):
        r"""Swap your computekey for all registered brainports on the network."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            SwapComputekeyCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    @staticmethod
    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Swap your computekey for all registered brainports on the network."""
        wallet = basedai.wallet(config=cli.config)

        # This creates an unnecessary amount of extra data, but simplifies implementation.
        new_config = deepcopy(cli.config)
        new_config.wallet.computekey = new_config.wallet.computekey_b
        new_wallet = basedai.wallet(config=new_config)

        basednode.swap_computekey(
            wallet=wallet,
            new_wallet=new_wallet,
            wait_for_finalization=False,
            wait_for_inclusion=True,
            prompt=False,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        swap_computekey_parser = parser.add_parser(
            "swap_computekey", help="""Swap your associated computekey."""
        )

        swap_computekey_parser.add_argument(
            "--wallet.computekey_b",
            type=str,
            default=defaults.wallet.computekey,
            help="""Name of the new computekey""",
            required=False,
        )

        basedai.wallet.add_args(swap_computekey_parser)
        basedai.basednode.add_args(swap_computekey_parser)

    @staticmethod
    def check_config(config: "basedai.config"):
        if (
            not config.is_set("basednode.network")
            and not config.is_set("basednode.chain_endpoint")
            and not config.no_prompt
        ):
            config.basednode.network = Prompt.ask(
                "Enter basednode network",
                choices=basedai.__networks__,
                default=defaults.basednode.network,
            )
            _, endpoint = basedai.basednode.determine_chain_endpoint_and_network(
                config.basednode.network
            )
            config.basednode.chain_endpoint = endpoint

        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.computekey") and not config.no_prompt:
            computekey = Prompt.ask("Enter old computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey = str(computekey)

        if not config.is_set("wallet.computekey_b") and not config.no_prompt:
            computekey = Prompt.ask("Enter new computekey name", default=defaults.wallet.computekey)
            config.wallet.computekey_b = str(computekey)
