import argparse
from sys import getsizeof

from rich.prompt import Prompt
from rich.table import Table

import basedai


class SetIdentityCommand:
    """
    Executes the :func:`set_identity` command within the BasedAI network, which allows for the creation or update of a delegate's on-chain identity.

    This identity includes various
    attributes such as display name, legal name, web URL, PGP fingerprint, and contact
    information, among others.

    Optional Arguments:
        - ``display``: The display name for the identity.
        - ``web``: The web URL for the identity.
        - ``riot``: The riot handle for the identity.
        - ``email``: The email address for the identity.
        - ``pgp_fingerprint``: The PGP fingerprint for the identity.
        - ``image``: The image URL for the identity.
        - ``legal``: The formal or legal name for the identity.
        - ``info``: The info for the identity.
        - ``twitter``: The X (twitter) URL for the identity.

    The command prompts the user for the different identity attributes and validates the
    input size for each attribute. It provides an option to update an existing validator
    computekey identity. If the user consents to the transaction cost, the identity is updated
    on the blockchain.

    Each field has a maximum size of 64 bytes. The PGP fingerprint field is an exception
    and has a maximum size of 20 bytes. The user is prompted to enter the PGP fingerprint
    as a hex string, which is then converted to bytes. The user is also prompted to enter
    the personalkey or computekey ``ss58`` address for the identity to be updated. If the user does
    not have a computekey, the personalkey address is used by default.

    If setting a validator identity, the computekey will be used by default. If the user is
    setting an identity for a Brain, the personalkey will be used by default.

    Usage:
        The user should call this command from the command line and follow the interactive
        prompts to enter or update the identity information. The command will display the
        updated identity details in a table format upon successful execution.

    Example usage::

        basedcli wallet set_identity

    Note:
        This command should only be used if the user is willing to incur the 1 BASED transaction
        fee associated with setting an identity on the blockchain. It is a high-level command
        that makes changes to the blockchain state and should not be used programmatically as
        part of other scripts or applications.
    """

    def run(cli: "basedai.cli"):
        r"""Create a new or update existing identity on-chain."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            SetIdentityCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        r"""Create a new or update existing identity on-chain."""
        console = basedai.__console__

        wallet = basedai.wallet(config=cli.config)

        id_dict = {
            "display": cli.config.display,
            "legal": cli.config.legal,
            "web": cli.config.web,
            "pgp_fingerprint": cli.config.pgp_fingerprint,
            "riot": cli.config.riot,
            "email": cli.config.email,
            "image": cli.config.image,
            "twitter": cli.config.twitter,
            "info": cli.config.info,
        }

        for field, string in id_dict.items():
            if getsizeof(string) > 113:  # 64 + 49 overhead bytes for string
                raise ValueError(f"Identity value `{field}` must be <= 64 raw bytes")

        identified = (
            wallet.computekey.ss58_address
            if str(
                Prompt.ask(
                    "Are you updating a validator computekey identity?",
                    default="y",
                    choices=["y", "n"],
                )
            ).lower()
            == "y"
            else None
        )

        if (
            str(
                Prompt.ask(
                    "Cost to permanently update the memory of the Identity is [bold white italic]0.1 Based[/bold white italic], are you sure you wish to continue?",
                    default="n",
                    choices=["y", "n"],
                )
            ).lower()
            == "n"
        ):
            console.print(":cross_mark: Aborted!")
            exit(0)

        wallet.personalkey  # unlock personalkey
        with console.status(":brain: [bold green]Memorizing identity on-chain..."):
            try:
                basednode.update_identity(
                    identified=identified,
                    wallet=wallet,
                    params=id_dict,
                )
            except Exception as e:
                console.print(
                    f"[red]:cross_mark: Identity memorization failed![/red] {e}"
                )
                exit(1)

            console.print(":white_heavy_check_mark: Success!")

        identity = basednode.query_identity(
            identified or wallet.personalkey.ss58_address
        )

        table = Table(title="[bold white italic]Memorized On-Chain Identity")
        table.add_column("Key", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Address", identified or wallet.personalkey.ss58_address)
        for key, value in identity.items():
            table.add_row(key, str(value) if value is not None else "None")

        console.print(table)

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            config.wallet.name = Prompt.ask(
                "Enter wallet name", default=basedai.defaults.wallet.name
            )
        if not config.is_set("wallet.computekey") and not config.no_prompt:
            config.wallet.computekey = Prompt.ask(
                "Enter wallet computekey", default=basedai.defaults.wallet.computekey
            )
        if not config.is_set("basednode.network") and not config.no_prompt:
            config.basednode.network = Prompt.ask(
                "Enter BasedAI network",
                default=basedai.defaults.basednode.network,
                choices=basedai.__networks__,
            )
            (
                _,
                config.basednode.chain_endpoint,
            ) = basedai.basednode.determine_chain_endpoint_and_network(
                config.basednode.network
            )
        if not config.is_set("display") and not config.no_prompt:
            config.display = Prompt.ask("Enter display name", default="")
        if not config.is_set("legal") and not config.no_prompt:
            config.legal = Prompt.ask("Enter legal string", default="")
        if not config.is_set("web") and not config.no_prompt:
            config.web = Prompt.ask("Enter web url", default="")
        if not config.is_set("pgp_fingerprint") and not config.no_prompt:
            config.pgp_fingerprint = Prompt.ask(
                "Enter pgp fingerprint (must be 20 bytes)", default=None
            )
        if not config.is_set("riot") and not config.no_prompt:
            config.riot = Prompt.ask("Enter riot", default="")
        if not config.is_set("email") and not config.no_prompt:
            config.email = Prompt.ask("Enter email address", default="")
        if not config.is_set("image") and not config.no_prompt:
            config.image = Prompt.ask("Enter image url", default="")
        if not config.is_set("twitter") and not config.no_prompt:
            config.twitter = Prompt.ask("Enter twitter url", default="")
        if not config.is_set("info") and not config.no_prompt:
            config.info = Prompt.ask("Enter info", default="")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        new_personalkey_parser = parser.add_parser(
            "set_identity",
            help="""Brain owners can update their identity on-chain for a given cold wallet. The Brain must be already linked using the 'basedcli brains link'.""",
        )
        new_personalkey_parser.add_argument(
            "--display",
            type=str,
            help="""The display name for the identity.""",
        )
        new_personalkey_parser.add_argument(
            "--legal",
            type=str,
            help="""The formal legal name for the identity of the business or proprieter.""",
        )
        new_personalkey_parser.add_argument(
            "--web",
            type=str,
            help="""The web url for the identity.""",
        )
        new_personalkey_parser.add_argument(
            "--riot",
            type=str,
            help="""The riot handle for the identity.""",
        )
        new_personalkey_parser.add_argument(
            "--email",
            type=str,
            help="""The email address for the identity.""",
        )
        new_personalkey_parser.add_argument(
            "--pgp_fingerprint",
            type=str,
            help="""The pgp fingerprint for the identity.""",
        )
        new_personalkey_parser.add_argument(
            "--image",
            type=str,
            help="""The image url for the identity.""",
        )
        new_personalkey_parser.add_argument(
            "--info",
            type=str,
            help="""The info for the identity.""",
        )
        new_personalkey_parser.add_argument(
            "--twitter",
            type=str,
            help="""The twitter url for the identity.""",
        )
        basedai.wallet.add_args(new_personalkey_parser)
        basedai.basednode.add_args(new_personalkey_parser)


class GetIdentityCommand:
    """
    Executes the :func:`get_identity` command, which retrieves and displays the identity details of a user's personalkey or computekey associated with the BasedAI network. This function
    queries the basednode chain for information such as the stake, rank, and trust associated
    with the provided key.

    Optional Arguments:
        - ``key``: The ``ss58`` address of the personalkey or computekey to query. The EVM address is not supported.

    The command performs the following actions:

    - Connects to the basednode network and retrieves the identity information.
    - Displays the information in a structured table format.

    The displayed table includes:

    - **Address**: The ``ss58`` address of the queried key.
    - **Item**: Various attributes of the identity such as stake, rank, and trust.
    - **Value**: The corresponding values of the attributes.

    Usage:
        The user must provide an ``ss58`` address as input to the command. If the address is not
        provided in the configuration, the user is prompted to enter one.

    Example usage::

        basedcli wallet get_identity --key <s58_address>

    Note:
        This function is designed for CLI use and should be executed in a terminal. It is
        primarily used for informational purposes and has no side effects on the network state.
    """

    def run(cli: "basedai.cli"):
        r"""Queries the basednode chain for user identity."""
        try:
            basednode: "basedai.basednode" = basedai.basednode(
                config=cli.config, log_verbose=False
            )
            GetIdentityCommand._run(cli, basednode)
        finally:
            if "basednode" in locals():
                basednode.close()
                basedai.logging.debug("closing basednode connection")

    def _run(cli: "basedai.cli", basednode: "basedai.basednode"):
        console = basedai.__console__

        with console.status(":brain: [bold green]Querying chain identity..."):
            identity = basednode.query_identity(cli.config.key)

        table = Table(title="[bold white italic]On-Chain Identity")
        table.add_column("Item", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Address", cli.config.key)
        for key, value in identity.items():
            table.add_row(key, str(value) if value is not None else "None")

        console.print(table)

    @staticmethod
    def check_config(config: "basedai.config"):
        if not config.is_set("key") and not config.no_prompt:
            config.key = Prompt.ask(
                "Enter personalkey or computekey ss58 address", default=None
            )
            if config.key is None:
                raise ValueError("key must be set")
        if not config.is_set("basednode.network") and not config.no_prompt:
            config.basednode.network = Prompt.ask(
                "Enter basednode network",
                default=basedai.defaults.basednode.network,
                choices=basedai.__networks__,
            )
            (
                _,
                config.basednode.chain_endpoint,
            ) = basedai.basednode.determine_chain_endpoint_and_network(
                config.basednode.network
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        new_personalkey_parser = parser.add_parser(
            "get_identity",
            help="""Creates a new personalkey (for containing balance) under the specified path. """,
        )
        new_personalkey_parser.add_argument(
            "--key",
            type=str,
            default=None,
            help="""The personalkey or computekey ss58 address to query.""",
        )
        basedai.wallet.add_args(new_personalkey_parser)
        basedai.basednode.add_args(new_personalkey_parser)
