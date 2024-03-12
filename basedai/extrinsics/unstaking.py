# The MIT License (MIT)
# Copyright © 2024 Saul Finney
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

import basedai
from rich.prompt import Confirm
from time import sleep
from typing import List, Dict, Union, Optional
from basedai.utils.balance import Balance


def __do_remove_stake_single(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    computekey_ss58: str,
    amount: "basedai.Balance",
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    r"""
    Executes an unstake call to the chain using the wallet and the amount specified.

    Args:
        wallet (basedai.wallet):
            Basedai wallet object.
        computekey_ss58 (str):
            Computekey address to unstake from.
        amount (basedai.Balance):
            Amount to unstake as Basedai balance object.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    Raises:
        basedai.errors.StakeError:
            If the extrinsic fails to be finalized or included in the block.
        basedai.errors.NotRegisteredError:
            If the computekey is not memorized by any subnets.

    """
    # Decrypt keys,
    wallet.personalkey

    success = basednode._do_unstake(
        wallet=wallet,
        computekey_ss58=computekey_ss58,
        amount=amount,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    return success


def unstake_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    computekey_ss58: Optional[str] = None,
    amount: Union[Balance, float] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""Removes stake into the wallet personalkey from the specified computekey ``uid``.

    Args:
        wallet (basedai.wallet):
            Basedai wallet object.
        computekey_ss58 (Optional[str]):
            The ``ss58`` address of the computekey to unstake from. By default, the wallet computekey is used.
        amount (Union[Balance, float]):
            Amount to stake as Basedai balance, or ``float`` interpreted as Based.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    # Decrypt keys,
    wallet.personalkey

    if computekey_ss58 is None:
        computekey_ss58 = wallet.computekey.ss58_address  # Default to wallet's own computekey.

    with basedai.__console__.status(
        ":brain: Syncing with chain: [white]{}[/white] ...".format(
            basednode.network
        )
    ):
        old_balance = basednode.get_balance(wallet.personalkeypub.ss58_address)
        old_stake = basednode.get_stake_for_personalkey_and_computekey(
            personalkey_ss58=wallet.personalkeypub.ss58_address, computekey_ss58=computekey_ss58
        )

    # Convert to basedai.Balance
    if amount == None:
        # Unstake it all.
        unstaking_balance = old_stake
    elif not isinstance(amount, basedai.Balance):
        unstaking_balance = basedai.Balance.from_based(amount)
    else:
        unstaking_balance = amount

    # Check enough to unstake.
    stake_on_uid = old_stake
    if unstaking_balance > stake_on_uid:
        basedai.__console__.print(
            ":cross_mark: [red]Not enough stake[/red]: [green]{}[/green] to unstake: [blue]{}[/blue] from computekey: [white]{}[/white]".format(
                stake_on_uid, unstaking_balance, wallet.computekey_str
            )
        )
        return False

    # Ask before moving on.
    if prompt:
        if not Confirm.ask(
            "Do you want to unstake:\n[bold white]  amount: {}\n  computekey: {}[/bold white ]?".format(
                unstaking_balance, wallet.computekey_str
            )
        ):
            return False

    try:
        with basedai.__console__.status(
            ":brain: Unstaking from chain: [white]{}[/white] ...".format(
                basednode.network
            )
        ):
            staking_response: bool = __do_remove_stake_single(
                basednode=basednode,
                wallet=wallet,
                computekey_ss58=computekey_ss58,
                amount=unstaking_balance,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

        if staking_response == True:  # If we successfully unstaked.
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            basedai.__console__.print(
                ":white_heavy_check_mark: [green]Finalized[/green]"
            )
            with basedai.__console__.status(
                ":brain: Checking Balance on: [white]{}[/white] ...".format(
                    basednode.network
                )
            ):
                new_balance = basednode.get_balance(
                    address=wallet.personalkeypub.ss58_address
                )
                new_stake = basednode.get_stake_for_personalkey_and_computekey(
                    personalkey_ss58=wallet.personalkeypub.ss58_address, computekey_ss58=computekey_ss58
                )  # Get stake on computekey.
                basedai.__console__.print(
                    "Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        old_balance, new_balance
                    )
                )
                basedai.__console__.print(
                    "Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        old_stake, new_stake
                    )
                )
                return True
        else:
            basedai.__console__.print(
                ":cross_mark: [red]Failed[/red]: Error unknown."
            )
            return False

    except basedai.errors.NotRegisteredError as e:
        basedai.__console__.print(
            ":cross_mark: [red]Computekey: {} has not been memorized.[/red]".format(
                wallet.computekey_str
            )
        )
        return False
    except basedai.errors.StakeError as e:
        basedai.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
        return False


def unstake_multiple_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    computekey_ss58s: List[str],
    amounts: List[Union[Balance, float]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""Removes stake from each ``computekey_ss58`` in the list, using each amount, to a common personalkey.

    Args:
        wallet (basedai.wallet):
            The wallet with the personalkey to unstake to.
        computekey_ss58s (List[str]):
            List of computekeys to unstake from.
        amounts (List[Union[Balance, float]]):
            List of amounts to unstake. If ``None``, unstake all.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or included in the block. Flag is ``true`` if any wallet was unstaked. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    if not isinstance(computekey_ss58s, list) or not all(
        isinstance(computekey_ss58, str) for computekey_ss58 in computekey_ss58s
    ):
        raise TypeError("computekey_ss58s must be a list of str")

    if len(computekey_ss58s) == 0:
        return True

    if amounts is not None and len(amounts) != len(computekey_ss58s):
        raise ValueError("amounts must be a list of the same length as computekey_ss58s")

    if amounts is not None and not all(
        isinstance(amount, (Balance, float)) for amount in amounts
    ):
        raise TypeError(
            "amounts must be a [list of basedai.Balance or float] or None"
        )

    if amounts is None:
        amounts = [None] * len(computekey_ss58s)
    else:
        # Convert to Balance
        amounts = [
            basedai.Balance.from_based(amount) if isinstance(amount, float) else amount
            for amount in amounts
        ]

        if sum(amount.based for amount in amounts) == 0:
            # Staking 0 based
            return True

    # Unlock personalkey.
    wallet.personalkey

    old_stakes = []
    with basedai.__console__.status(
        ":brain: Syncing with chain: [white]{}[/white] ...".format(
            basednode.network
        )
    ):
        old_balance = basednode.get_balance(wallet.personalkeypub.ss58_address)

        for computekey_ss58 in computekey_ss58s:
            old_stake = basednode.get_stake_for_personalkey_and_computekey(
                personalkey_ss58=wallet.personalkeypub.ss58_address, computekey_ss58=computekey_ss58
            )  # Get stake on computekey.
            old_stakes.append(old_stake)  # None if not registered.

    successful_unstakes = 0
    for idx, (computekey_ss58, amount, old_stake) in enumerate(
        zip(computekey_ss58s, amounts, old_stakes)
    ):
        # Covert to basedai.Balance
        if amount == None:
            # Unstake it all.
            unstaking_balance = old_stake
        elif not isinstance(amount, basedai.Balance):
            unstaking_balance = basedai.Balance.from_based(amount)
        else:
            unstaking_balance = amount

        # Check enough to unstake.
        stake_on_uid = old_stake
        if unstaking_balance > stake_on_uid:
            basedai.__console__.print(
                ":cross_mark: [red]Not enough stake[/red]: [green]{}[/green] to unstake: [blue]{}[/blue] from computekey: [white]{}[/white]".format(
                    stake_on_uid, unstaking_balance, wallet.computekey_str
                )
            )
            continue

        # Ask before moving on.
        if prompt:
            if not Confirm.ask(
                "Do you want to unstake:\n[bold white]  amount: {}\n  computekey: {}[/bold white ]?".format(
                    unstaking_balance, wallet.computekey_str
                )
            ):
                continue

        try:
            with basedai.__console__.status(
                ":brain: Unstaking from chain: [white]{}[/white] ...".format(
                    basednode.network
                )
            ):
                staking_response: bool = __do_remove_stake_single(
                    basednode=basednode,
                    wallet=wallet,
                    computekey_ss58=computekey_ss58,
                    amount=unstaking_balance,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )

            if staking_response == True:  # If we successfully unstaked.
                # We only wait here if we expect finalization.

                if idx < len(computekey_ss58s) - 1:
                    # Wait for tx rate limit.
                    tx_rate_limit_blocks = basednode.tx_rate_limit()
                    if tx_rate_limit_blocks > 0:
                        basedai.__console__.print(
                            ":hourglass: [yellow]Waiting for tx rate limit: [white]{}[/white] blocks[/yellow]".format(
                                tx_rate_limit_blocks
                            )
                        )
                        sleep(tx_rate_limit_blocks * 12)  # 12 seconds per block

                if not wait_for_finalization and not wait_for_inclusion:
                    successful_unstakes += 1
                    continue

                basedai.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                with basedai.__console__.status(
                    ":brain: Checking Balance on: [white]{}[/white] ...".format(
                        basednode.network
                    )
                ):
                    block = basednode.get_current_block()
                    new_stake = basednode.get_stake_for_personalkey_and_computekey(
                        personalkey_ss58=wallet.personalkeypub.ss58_address,
                        computekey_ss58=computekey_ss58,
                        block=block,
                    )
                    basedai.__console__.print(
                        "Stake ({}): [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                            computekey_ss58, stake_on_uid, new_stake
                        )
                    )
                    successful_unstakes += 1
            else:
                basedai.__console__.print(
                    ":cross_mark: [red]Failed[/red]: Error unknown."
                )
                continue

        except basedai.errors.NotRegisteredError as e:
            basedai.__console__.print(
                ":cross_mark: [red]{} has not been memorized.[/red]".format(computekey_ss58)
            )
            continue
        except basedai.errors.StakeError as e:
            basedai.__console__.print(
                ":cross_mark: [red]Stake Error: {}[/red]".format(e)
            )
            continue

    if successful_unstakes != 0:
        with basedai.__console__.status(
            ":brain: Checking Balance on: ([white]{}[/white] ...".format(
                basednode.network
            )
        ):
            new_balance = basednode.get_balance(wallet.personalkeypub.ss58_address)
        basedai.__console__.print(
            "Balance: [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                old_balance, new_balance
            )
        )
        return True

    return False
