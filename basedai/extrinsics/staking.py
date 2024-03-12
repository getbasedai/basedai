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


def add_stake_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    computekey_ss58: Optional[str] = None,
    amount: Union[Balance, float] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""Adds the specified amount of stake to passed computekey ``uid``.

    Args:
        wallet (basedai.wallet):
            Basedai wallet object.
        computekey_ss58 (Optional[str]):
            The ``ss58`` address of the computekey account to stake to defaults to the wallet's computekey.
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

    Raises:
        basedai.errors.NotRegisteredError:
            If the wallet is not been memorized by the chain.
        basedai.errors.NotDelegateError:
            If the computekey is not a delegate on the chain.
    """
    # Decrypt keys,
    wallet.personalkey

    # Default to wallet's own computekey if the value is not passed.
    if computekey_ss58 is None:
        computekey_ss58 = wallet.computekey.ss58_address

    # Flag to indicate if we are using the wallet's own computekey.
    own_computekey: bool

    with basedai.__console__.status(
        ":brain: Syncing with chain: [white]{}[/white] ...".format(
            basednode.network
        )
    ):
        old_balance = basednode.get_balance(wallet.personalkeypub.ss58_address)
        # Get computekey owner
        computekey_owner = basednode.get_computekey_owner(computekey_ss58)
        own_computekey = wallet.personalkeypub.ss58_address == computekey_owner
        if not own_computekey:
            # This is not the wallet's own computekey so we are delegating.
            if not basednode.is_computekey_delegate(computekey_ss58):
                raise basedai.errors.NotDelegateError(
                    "Computekey: {} is not a delegate.".format(computekey_ss58)
                )

            # Get computekey take
            computekey_take = basednode.get_delegate_take(computekey_ss58)

        # Get current stake
        old_stake = basednode.get_stake_for_personalkey_and_computekey(
            personalkey_ss58=wallet.personalkeypub.ss58_address, computekey_ss58=computekey_ss58
        )

    # Convert to basedai.Balance
    if amount == None:
        # Stake it all.
        staking_balance = basedai.Balance.from_based(old_balance.based)
    elif not isinstance(amount, basedai.Balance):
        staking_balance = basedai.Balance.from_based(amount)
    else:
        staking_balance = amount

    # Remove existential balance to keep key alive.
    if staking_balance > basedai.Balance.from_rao(1000):
        staking_balance = staking_balance - basedai.Balance.from_rao(1000)
    else:
        staking_balance = staking_balance

    # Check enough to stake.
    if staking_balance > old_balance:
        basedai.__console__.print(
            ":cross_mark: [red]Not enough stake[/red]:[bold white]\n  balance:{}\n  amount: {}\n  personalkey: {}[/bold white]".format(
                old_balance, staking_balance, wallet.name
            )
        )
        return False

    # Ask before moving on.
    if prompt:
        if not own_computekey:
            # We are delegating.
            if not Confirm.ask(
                "Do you want to delegate:[bold white]\n  amount: {}\n  to: {}\n  take: {}\n  owner: {}[/bold white]".format(
                    staking_balance, wallet.computekey_str, computekey_take, computekey_owner
                )
            ):
                return False
        else:
            if not Confirm.ask(
                "Do you want to stake:[bold white]\n  amount: {}\n  to: {}[/bold white]".format(
                    staking_balance, wallet.computekey_str
                )
            ):
                return False

    try:
        with basedai.__console__.status(
            ":brain: Staking to: [bold white]{}[/bold white] ...".format(
                basednode.network
            )
        ):
            staking_response: bool = __do_add_stake_single(
                basednode=basednode,
                wallet=wallet,
                computekey_ss58=computekey_ss58,
                amount=staking_balance,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

        if staking_response == True:  # If we successfully staked.
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
                block = basednode.get_current_block()
                new_stake = basednode.get_stake_for_personalkey_and_computekey(
                    personalkey_ss58=wallet.personalkeypub.ss58_address,
                    computekey_ss58=wallet.computekey.ss58_address,
                    block=block,
                )  # Get current stake

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


def add_stake_multiple_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    computekey_ss58s: List[str],
    amounts: List[Union[Balance, float]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""Adds stake to each ``computekey_ss58`` in the list, using each amount, from a common personalkey.

    Args:
        wallet (basedai.wallet):
            Basedai wallet object for the personalkey.
        computekey_ss58s (List[str]):
            List of computekeys to stake to.
        amounts (List[Union[Balance, float]]):
            List of amounts to stake. If ``None``, stake all to the first computekey.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or included in the block. Flag is ``true`` if any wallet was staked. If we did not wait for finalization / inclusion, the response is ``true``.
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

    # Decrypt personalkey.
    wallet.personalkey

    old_stakes = []
    with basedai.__console__.status(
        ":brain: Syncing with chain: [white]{}[/white] ...".format(
            basednode.network
        )
    ):
        old_balance = basednode.get_balance(wallet.personalkeypub.ss58_address)

        # Get the old stakes.
        for computekey_ss58 in computekey_ss58s:
            old_stakes.append(
                basednode.get_stake_for_personalkey_and_computekey(
                    personalkey_ss58=wallet.personalkeypub.ss58_address, computekey_ss58=computekey_ss58
                )
            )

    # Remove existential balance to keep key alive.
    ## Keys must maintain a balance of at least 1000 rao to stay alive.
    total_staking_rao = sum(
        [amount.rao if amount is not None else 0 for amount in amounts]
    )
    if total_staking_rao == 0:
        # Staking all to the first wallet.
        if old_balance.rao > 1000:
            old_balance -= basedai.Balance.from_rao(1000)

    elif total_staking_rao < 1000:
        # Staking less than 1000 rao to the wallets.
        pass
    else:
        # Staking more than 1000 rao to the wallets.
        ## Reduce the amount to stake to each wallet to keep the balance above 1000 rao.
        percent_reduction = 1 - (1000 / total_staking_rao)
        amounts = [
            Balance.from_based(amount.based * percent_reduction) for amount in amounts
        ]

    successful_stakes = 0
    for idx, (computekey_ss58, amount, old_stake) in enumerate(
        zip(computekey_ss58s, amounts, old_stakes)
    ):
        staking_all = False
        # Convert to basedai.Balance
        if amount == None:
            # Stake it all.
            staking_balance = basedai.Balance.from_based(old_balance.based)
            staking_all = True
        else:
            # Amounts are cast to balance earlier in the function
            assert isinstance(amount, basedai.Balance)
            staking_balance = amount

        # Check enough to stake
        if staking_balance > old_balance:
            basedai.__console__.print(
                ":cross_mark: [red]Not enough balance[/red]: [green]{}[/green] to stake: [blue]{}[/blue] from personalkey: [white]{}[/white]".format(
                    old_balance, staking_balance, wallet.name
                )
            )
            continue

        # Ask before moving on.
        if prompt:
            if not Confirm.ask(
                "Do you want to stake:\n[bold white]  amount: {}\n  computekey: {}[/bold white ]?".format(
                    staking_balance, wallet.computekey_str
                )
            ):
                continue

        try:
            staking_response: bool = __do_add_stake_single(
                basednode=basednode,
                wallet=wallet,
                computekey_ss58=computekey_ss58,
                amount=staking_balance,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if staking_response == True:  # If we successfully staked.
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
                    old_balance -= staking_balance
                    successful_stakes += 1
                    if staking_all:
                        # If staked all, no need to continue
                        break

                    continue

                basedai.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )

                block = basednode.get_current_block()
                new_stake = basednode.get_stake_for_personalkey_and_computekey(
                    personalkey_ss58=wallet.personalkeypub.ss58_address,
                    computekey_ss58=computekey_ss58,
                    block=block,
                )
                new_balance = basednode.get_balance(
                    wallet.personalkeypub.ss58_address, block=block
                )
                basedai.__console__.print(
                    "Stake ({}): [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        computekey_ss58, old_stake, new_stake
                    )
                )
                old_balance = new_balance
                successful_stakes += 1
                if staking_all:
                    # If staked all, no need to continue
                    break

            else:
                basedai.__console__.print(
                    ":cross_mark: [red]Failed[/red]: Error unknown."
                )
                continue

        except basedai.errors.NotRegisteredError as e:
            basedai.__console__.print(
                ":cross_mark: [red]Computekey: {} has not been memorized.[/red]".format(
                    computekey_ss58
                )
            )
            continue
        except basedai.errors.StakeError as e:
            basedai.__console__.print(
                ":cross_mark: [red]Stake Error: {}[/red]".format(e)
            )
            continue

    if successful_stakes != 0:
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


def __do_add_stake_single(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    computekey_ss58: str,
    amount: "basedai.Balance",
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    r"""
    Executes a stake call to the chain using the wallet and the amount specified.

    Args:
        wallet (basedai.wallet):
            Basedai wallet object.
        computekey_ss58 (str):
            Computekey to stake to.
        amount (basedai.Balance):
            Amount to stake as Basedai balance object.
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
        basedai.errors.NotDelegateError:
            If the computekey is not a delegate.
        basedai.errors.NotRegisteredError:
            If the computekey has not been memorized by any subnet.

    """
    # Decrypt keys,
    wallet.personalkey

    computekey_owner = basednode.get_computekey_owner(computekey_ss58)
    own_computekey = wallet.personalkeypub.ss58_address == computekey_owner
    if not own_computekey:
        # We are delegating.
        # Verify that the computekey is a delegate.
        if not basednode.is_computekey_delegate(computekey_ss58=computekey_ss58):
            raise basedai.errors.NotDelegateError(
                "Computekey: {} is not a delegate.".format(computekey_ss58)
            )

    success = basednode._do_stake(
        wallet=wallet,
        computekey_ss58=computekey_ss58,
        amount=amount,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    return success
