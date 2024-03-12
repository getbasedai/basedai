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
from ..errors import *
from rich.prompt import Confirm
from typing import List, Dict, Union, Optional
from basedai.utils.balance import Balance
from .staking import __do_add_stake_single

from loguru import logger

logger = logger.opt(colors=True)


def nominate_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    wait_for_finalization: bool = False,
    wait_for_inclusion: bool = True,
) -> bool:
    r"""Becomes a delegate for the computekey.

    Args:
        wallet (basedai.wallet): The wallet to become a delegate for.
    Returns:
        success (bool): ``True`` if the transaction was successful.
    """
    # Unlock the personalkey.
    wallet.personalkey
    wallet.computekey

    # Check if the computekey is already a delegate.
    if basednode.is_computekey_delegate(wallet.computekey.ss58_address):
        logger.error(
            "Computekey {} is already a delegate.".format(wallet.computekey.ss58_address)
        )
        return False

    with basedai.__console__.status(
        ":brain: Sending nominate call on [white]{}[/white] ...".format(
            basednode.network
        )
    ):
        try:
            success = basednode._do_nominate(
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if success == True:
                basedai.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                basedai.logging.success(
                    prefix="Become Delegate",
                    sufix="<green>Finalized: </green>" + str(success),
                )

            # Raises NominationError if False
            return success

        except Exception as e:
            basedai.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            basedai.logging.warning(
                prefix="Set weights", sufix="<red>Failed: </red>" + str(e)
            )
        except NominationError as e:
            basedai.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            basedai.logging.warning(
                prefix="Set weights", sufix="<red>Failed: </red>" + str(e)
            )

    return False


def delegate_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    delegate_ss58: Optional[str] = None,
    amount: Union[Balance, float] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""Delegates the specified amount of stake to the passed delegate.

    Args:
        wallet (basedai.wallet): Basedai wallet object.
        delegate_ss58 (Optional[str]): The ``ss58`` address of the delegate.
        amount (Union[Balance, float]): Amount to stake as basedai balance, or ``float`` interpreted as Based.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool): If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.

    Raises:
        NotRegisteredError: If the wallet is has not been memorized.
        NotDelegateError: If the computekey is not a delegate on the chain.
    """
    # Decrypt keys,
    wallet.personalkey
    if not basednode.is_computekey_delegate(delegate_ss58):
        raise NotDelegateError("Computekey: {} is not a delegate.".format(delegate_ss58))

    # Get state.
    my_prev_personalkey_balance = basednode.get_balance(wallet.personalkey.ss58_address)
    delegate_take = basednode.get_delegate_take(delegate_ss58)
    delegate_owner = basednode.get_computekey_owner(delegate_ss58)
    my_prev_delegated_stake = basednode.get_stake_for_personalkey_and_computekey(
        personalkey_ss58=wallet.personalkeypub.ss58_address, computekey_ss58=delegate_ss58
    )

    # Convert to basedai.Balance
    if amount == None:
        # Stake it all.
        staking_balance = basedai.Balance.from_based(my_prev_personalkey_balance.based)
    elif not isinstance(amount, basedai.Balance):
        staking_balance = basedai.Balance.from_based(amount)
    else:
        staking_balance = amount

    # Remove existential balance to keep key alive.
    if staking_balance > basedai.Balance.from_rao(1000):
        staking_balance = staking_balance - basedai.Balance.from_rao(1000)
    else:
        staking_balance = staking_balance

    # Check enough balance to stake.
    if staking_balance > my_prev_personalkey_balance:
        basedai.__console__.print(
            ":cross_mark: [red]Not enough balance[/red]:[bold white]\n  balance:{}\n  amount: {}\n  personalkey: {}[/bold white]".format(
                my_prev_personalkey_balance, staking_balance, wallet.name
            )
        )
        return False

    # Ask before moving on.
    if prompt:
        if not Confirm.ask(
            "Do you want to delegate:[bold white]\n  amount: {}\n  to: {}\n owner: {}[/bold white]".format(
                staking_balance, delegate_ss58, delegate_owner
            )
        ):
            return False

    try:
        with basedai.__console__.status(
            ":brain: Staking to: [bold white]{}[/bold white] ...".format(
                basednode.network
            )
        ):
            staking_response: bool = basednode._do_delegation(
                wallet=wallet,
                delegate_ss58=delegate_ss58,
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
                new_balance = basednode.get_balance(address=wallet.personalkey.ss58_address)
                block = basednode.get_current_block()
                new_delegate_stake = basednode.get_stake_for_personalkey_and_computekey(
                    personalkey_ss58=wallet.personalkeypub.ss58_address,
                    computekey_ss58=delegate_ss58,
                    block=block,
                )  # Get current stake

                basedai.__console__.print(
                    "Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        my_prev_personalkey_balance, new_balance
                    )
                )
                basedai.__console__.print(
                    "Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        my_prev_delegated_stake, new_delegate_stake
                    )
                )
                return True
        else:
            basedai.__console__.print(
                ":cross_mark: [red]Failed[/red]: Error unknown."
            )
            return False

    except NotRegisteredError as e:
        basedai.__console__.print(
            ":cross_mark: [red]Computekey: {} has not been memorized.[/red]".format(
                wallet.computekey_str
            )
        )
        return False
    except StakeError as e:
        basedai.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
        return False


def undelegate_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    delegate_ss58: Optional[str] = None,
    amount: Union[Balance, float] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""Un-delegates stake from the passed delegate.

    Args:
        wallet (basedai.wallet): Basedai wallet object.
        delegate_ss58 (Optional[str]): The ``ss58`` address of the delegate.
        amount (Union[Balance, float]): Amount to unstake as basedai balance, or ``float`` interpreted as Based.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool): If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.

    Raises:
        NotRegisteredError: If the wallet has been memorized by the chain.
        NotDelegateError: If the computekey is not a delegate on the chain.
    """
    # Decrypt keys,
    wallet.personalkey
    if not basednode.is_computekey_delegate(delegate_ss58):
        raise NotDelegateError("Computekey: {} is not a delegate.".format(delegate_ss58))

    # Get state.
    my_prev_personalkey_balance = basednode.get_balance(wallet.personalkey.ss58_address)
    delegate_take = basednode.get_delegate_take(delegate_ss58)
    delegate_owner = basednode.get_computekey_owner(delegate_ss58)
    my_prev_delegated_stake = basednode.get_stake_for_personalkey_and_computekey(
        personalkey_ss58=wallet.personalkeypub.ss58_address, computekey_ss58=delegate_ss58
    )

    # Convert to basedai.Balance
    if amount == None:
        # Stake it all.
        unstaking_balance = basedai.Balance.from_based(my_prev_delegated_stake.based)

    elif not isinstance(amount, basedai.Balance):
        unstaking_balance = basedai.Balance.from_based(amount)

    else:
        unstaking_balance = amount

    # Check enough stake to unstake.
    if unstaking_balance > my_prev_delegated_stake:
        basedai.__console__.print(
            ":cross_mark: [red]Not enough delegated stake[/red]:[bold white]\n  stake:{}\n  amount: {}\n personalkey: {}[/bold white]".format(
                my_prev_delegated_stake, unstaking_balance, wallet.name
            )
        )
        return False

    # Ask before moving on.
    if prompt:
        if not Confirm.ask(
            "Do you want to un-delegate:[bold white]\n  amount: {}\n  from: {}\n  owner: {}[/bold white]".format(
                unstaking_balance, delegate_ss58, delegate_owner
            )
        ):
            return False

    try:
        with basedai.__console__.status(
            ":brain: Unstaking from: [bold white]{}[/bold white] ...".format(
                basednode.network
            )
        ):
            staking_response: bool = basednode._do_undelegation(
                wallet=wallet,
                delegate_ss58=delegate_ss58,
                amount=unstaking_balance,
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
                new_balance = basednode.get_balance(address=wallet.personalkey.ss58_address)
                block = basednode.get_current_block()
                new_delegate_stake = basednode.get_stake_for_personalkey_and_computekey(
                    personalkey_ss58=wallet.personalkeypub.ss58_address,
                    computekey_ss58=delegate_ss58,
                    block=block,
                )  # Get current stake

                basedai.__console__.print(
                    "Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        my_prev_personalkey_balance, new_balance
                    )
                )
                basedai.__console__.print(
                    "Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        my_prev_delegated_stake, new_delegate_stake
                    )
                )
                return True
        else:
            basedai.__console__.print(
                ":cross_mark: [red]Failed[/red]: Error unknown."
            )
            return False

    except NotRegisteredError as e:
        basedai.__console__.print(
            ":cross_mark: [red]Computekey: {} has not been memorized.[/red]".format(
                wallet.computekey_str
            )
        )
        return False
    except StakeError as e:
        basedai.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
        return False
