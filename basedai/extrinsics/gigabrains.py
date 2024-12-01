# The MIT License (MIT)
# Copyright © 2024 Sean Wellington
#

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

# Imports
import basedai

import time
from rich.prompt import Confirm
from ..errors import *


def register_gigabrains_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
) -> bool:
    r"""Memorizes the wallet to permanent memory for GigaBrain voting.

    Args:
        wallet (basedai.wallet):
            Basedai wallet object.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or included in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    wallet.personalkey  # unlock personalkey
    wallet.computekey  # unlock computekey

    if prompt:
        # Prompt user for confirmation.
        if not Confirm.ask(f"Confirm delegate computekey to permanent memory in the GigaBrains?"):
            return False

    with basedai.__console__.status(":brain: Memorizing GigaBrain..."):
        with basednode.substrate as substrate:
            # create extrinsic call
            call = substrate.compose_call(
                call_module="BasedNode",
                call_function="join_senate",
                call_params={"computekey": wallet.computekey.ss58_address},
            )
            extrinsic = substrate.create_signed_extrinsic(
                call=call, keypair=wallet.personalkey
            )
            response = substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            # process if registration successful
            response.process_events()
            if not response.is_success:
                basedai.__console__.print(
                    ":cross_mark: [red]Failed[/red]: error:{}".format(
                        response.error_message
                    )
                )
                time.sleep(0.5)

            # Successful registration, final check for membership
            else:
                is_registered = wallet.is_senate_member(basednode)

                if is_registered:
                    basedai.__console__.print(
                        ":white_heavy_check_mark: [green]Memorized[/green]"
                    )
                    return True
                else:
                    # neuron not found, try again
                    basedai.__console__.print(
                        ":cross_mark: [red]Unknown error. GigaBrain not found.[/red]"
                    )


def dismiss_gigabrains_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
) -> bool:
    r"""Removes the wallet from chain for GigaBrain voting.

    Args:
        wallet (basedai.wallet):
            Basedai wallet object.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or included in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    wallet.personalkey  # unlock personalkey
    wallet.computekey  # unlock computekey

    if prompt:
        # Prompt user for confirmation.
        if not Confirm.ask(f"Remove delegate computekey from GigaBrains?"):
            return False

    with basedai.__console__.status(":brain: Leaving GigaBrains..."):
        with basednode.substrate as substrate:
            # create extrinsic call
            call = substrate.compose_call(
                call_module="BasedNode",
                call_function="leave_senate",
                call_params={"computekey": wallet.computekey.ss58_address},
            )
            extrinsic = substrate.create_signed_extrinsic(
                call=call, keypair=wallet.personalkey
            )
            response = substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            # process if registration successful
            response.process_events()
            if not response.is_success:
                basedai.__console__.print(
                    ":cross_mark: [red]Failed[/red]: error:{}".format(
                        response.error_message
                    )
                )
                time.sleep(0.5)

            # Successful registration, final check for membership
            else:
                is_registered = wallet.is_senate_member(basednode)

                if not is_registered:
                    basedai.__console__.print(
                        ":white_heavy_check_mark: [green]Left GigaBrains[/green]"
                    )
                    return True
                else:
                    # neuron not found, try again
                    basedai.__console__.print(
                        ":cross_mark: [red]Unknown error. GigaBrain membership still active.[/red]"
                    )


def vote_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    proposal_hash: str,
    proposal_idx: int,
    vote: bool,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
) -> bool:
    r"""Removes the wallet from chain from voting as a GigaBrain.

    Args:
        wallet (basedai.wallet):
            Basedai wallet object.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or included in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    wallet.personalkey  # unlock personalkey
    wallet.computekey  # unlock computekey

    if prompt:
        # Prompt user for confirmation.
        if not Confirm.ask("Cast a vote of {}?".format(vote)):
            return False

    with basedai.__console__.status(":brain: Casting vote.."):
        with basednode.substrate as substrate:
            # create extrinsic call
            call = substrate.compose_call(
                call_module="BasedNode",
                call_function="vote",
                call_params={
                    "computekey": wallet.computekey.ss58_address,
                    "proposal": proposal_hash,
                    "index": proposal_idx,
                    "approve": vote,
                },
            )
            extrinsic = substrate.create_signed_extrinsic(
                call=call, keypair=wallet.personalkey
            )
            response = substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            # process if vote successful
            response.process_events()
            if not response.is_success:
                basedai.__console__.print(
                    ":cross_mark: [red]Failed[/red]: error:{}".format(
                        response.error_message
                    )
                )
                time.sleep(0.5)

            # Successful vote, final check for data
            else:
                vote_data = basednode.get_vote_data(proposal_hash)
                has_voted = (
                    vote_data["ayes"].count(wallet.computekey.ss58_address) > 0
                    or vote_data["nays"].count(wallet.computekey.ss58_address) > 0
                )

                if has_voted:
                    basedai.__console__.print(
                        ":white_heavy_check_mark: [green]Vote cast.[/green]"
                    )
                    return True
                else:
                    # computekey not found in ayes/nays
                    basedai.__console__.print(
                        ":cross_mark: [red]Unknown error. Couldn't find vote.[/red]"
                    )
