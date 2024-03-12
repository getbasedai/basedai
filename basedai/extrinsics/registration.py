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

import torch
import time
from rich.prompt import Confirm
from typing import List, Union, Optional, Tuple
from basedai.utils.registration import POWSolution, create_pow


def register_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    netuid: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
    max_allowed_attempts: int = 5,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[List[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
) -> bool:
    r"""Commants chain to add wallet to permanent memory.

    Args:
        wallet (basedai.wallet):
            Basedai wallet object.
        netuid (int):
            The ``netuid`` of the Brain to register on.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
        max_allowed_attempts (int):
            Maximum number of attempts to register the wallet.
        cuda (bool):
            If ``true``, the wallet should be registered using CUDA device(s).
        dev_id (Union[List[int], int]):
            The CUDA device id to use, or a list of device ids.
        tpb (int):
            The number of threads per block (CUDA).
        num_processes (int):
            The number of processes to use to register.
        update_interval (int):
            The number of nonces to solve between updates.
        log_verbose (bool):
            If ``true``, the memorization process will log more information.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    if not basednode.subnet_exists(netuid):
        basedai.__console__.print(
            ":cross_mark: [red]Failed[/red]: error: [bold white]Brain:{}[/bold white] does not exist.".format(
                netuid
            )
        )
        return False

    with basedai.__console__.status(
        f":brain: Checking account on [bold]Brain:{netuid}[/bold]..."
    ):
        neuron = basednode.get_neuron_for_pubkey_and_subnet(
            wallet.computekey.ss58_address, netuid=netuid
        )
        if not neuron.is_null:
            basedai.logging.debug(
                f"Wallet {wallet} is already registered on {neuron.netuid} with {neuron.uid}"
            )
            return True

    if prompt:
        if not Confirm.ask(
            "Confirm memorization details?\n  computekey:     [bold white]{}[/bold white]\n  personalkey:    [bold white]{}[/bold white]\n  network:    [bold white]{}[/bold white]".format(
                wallet.computekey.ss58_address,
                wallet.personalkeypub.ss58_address,
                basednode.network,
            )
        ):
            return False

    # Attempt rolling memorization.
    attempts = 1
    while True:
        basedai.__console__.print(
            ":brain: Memorizing...({}/{})".format(attempts, max_allowed_attempts)
        )
        # Solve latest POW.
        if cuda:
            if not torch.cuda.is_available():
                if prompt:
                    basedai.__console__.error("CUDA is not available.")
                return False
            pow_result: Optional[POWSolution] = create_pow(
                basednode,
                wallet,
                netuid,
                output_in_place,
                cuda=cuda,
                dev_id=dev_id,
                tpb=tpb,
                num_processes=num_processes,
                update_interval=update_interval,
                log_verbose=log_verbose,
            )
        else:
            pow_result: Optional[POWSolution] = create_pow(
                basednode,
                wallet,
                netuid,
                output_in_place,
                cuda=cuda,
                num_processes=num_processes,
                update_interval=update_interval,
                log_verbose=log_verbose,
            )

        # pow failed
        if not pow_result:
            # might be registered already on this subnet
            is_registered = basednode.is_computekey_registered(
                netuid=netuid, computekey_ss58=wallet.computekey.ss58_address
            )
            if is_registered:
                basedai.__console__.print(
                    f":white_heavy_check_mark: [green]Already registered on netuid:{netuid}[/green]"
                )
                return True

        # pow successful, proceed to submit pow to chain for memorization
        else:
            with basedai.__console__.status("𝔹 :brain: Waiting for evaluation..."):
                # check if pow result is still valid
                while not pow_result.is_stale(basednode=basednode):
                    result: Tuple[bool, Optional[str]] = basednode._do_pow_register(
                        netuid=netuid,
                        wallet=wallet,
                        pow_result=pow_result,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                    )
                    success, err_msg = result

                    if success != True or success == False:
                        if "key is already registered" in err_msg:
                            # Error meant that the key is already registered.
                            basedai.__console__.print(
                                f":white_heavy_check_mark: [green]Already memorized on [bold]Brain:{netuid}[/bold][/green]"
                            )
                            return True

                        basedai.__console__.print(
                            ":cross_mark: [red]Are you search this Brain accepts mine memories? Failed[/red]: error:{}".format(err_msg)
                        )
                        time.sleep(1.5)

                    # Successful memorization, final check for neuron and pubkey
                    else:
                        basedai.__console__.print(":brain: Checking balance...")
                        is_registered = basednode.is_computekey_registered(
                            netuid=netuid, computekey_ss58=wallet.computekey.ss58_address
                        )
                        if is_registered:
                            basedai.__console__.print(
                                ":white_heavy_check_mark: [green]Memorized[/green]"
                            )
                            return True
                        else:
                            # neuron not found, try again
                            basedai.__console__.print(
                                ":cross_mark: [red]Unknown error. Agent not found.[/red]"
                            )
                            continue
                else:
                    # Exited loop because pow is no longer valid.
                    basedai.__console__.print("[red]POW is stale.[/red]")
                    # Try again.
                    continue

        if attempts < max_allowed_attempts:
            # Failed memorization, retry pow
            attempts += 1
            basedai.__console__.print(
                ":brain: Failed memorization, retrying pow ...({}/{})".format(
                    attempts, max_allowed_attempts
                )
            )
        else:
            # Failed to register after max attempts.
            basedai.__console__.print("[red]No more attempts.[/red]")
            return False


def burned_register_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    netuid: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
) -> bool:
    r"""Activates the circuit to the wallet by locking BASED (𝔹).

    Args:
        wallet (basedai.wallet):
            Basedai wallet object.
        netuid (int):
            The ``netuid`` of the Brain to register on.
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
    if not basednode.subnet_exists(netuid):
        basedai.__console__.print(
            ":cross_mark: [red]Failed[/red]: error: [bold white]Brain:{}[/bold white] does not exist.".format(
                netuid
            )
        )
        return False

    wallet.personalkey  # unlock personalkey
    with basedai.__console__.status(
        f":brain: Checking account on [bold]Brain:{netuid}[/bold]..."
    ):
        neuron = basednode.get_neuron_for_pubkey_and_subnet(
            wallet.computekey.ss58_address, netuid=netuid
        )

        old_balance = basednode.get_balance(wallet.personalkeypub.ss58_address)

        recycle_amount = basednode.recycle(netuid=netuid)
        if not neuron.is_null:
            basedai.__console__.print(
                ":white_heavy_check_mark: [green]Already Memorized[/green]:\n"
                "uid: [bold white]{}[/bold white]\n"
                "netuid: [bold white]{}[/bold white]\n"
                "computekey: [bold white]{}[/bold white]\n"
                "personalkey: [bold white]{}[/bold white]".format(
                    neuron.uid, neuron.netuid, neuron.computekey, neuron.personalkey
                )
            )
            return True

    if prompt:
        # Prompt user for confirmation.
        if not Confirm.ask(f"Use {recycle_amount} to write to Brain's memory:{netuid}?"):
            return False

    with basedai.__console__.status(":brain: Activating the circuit with $BASED..."):
        success, err_msg = basednode._do_burned_register(
            netuid=netuid,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if success != True or success == False:
            basedai.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(err_msg)
            )
            time.sleep(0.5)

        # Successful memorization, final check for neuron and pubkey
        else:
            basedai.__console__.print(":brain: Checking $BASED nalance...")
            block = basednode.get_current_block()
            new_balance = basednode.get_balance(
                wallet.personalkeypub.ss58_address, block=block
            )

            basedai.__console__.print(
                "Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                    old_balance, new_balance
                )
            )
            is_registered = basednode.is_computekey_registered(
                netuid=netuid, computekey_ss58=wallet.computekey.ss58_address
            )
            if is_registered:
                basedai.__console__.print(
                    ":white_heavy_check_mark: [green]Memorized[/green]"
                )
                return True
            else:
                # neuron not found, try again
                basedai.__console__.print(
                    ":cross_mark: [red]Unknown error. Agent not found.[/red]"
                )


class MaxSuccessException(Exception):
    pass


class MaxAttemptsException(Exception):
    pass


def run_faucet_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
    max_allowed_attempts: int = 3,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[List[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
) -> bool:
    r"""Runs a continual POW to get a faucet of testnet BASED (𝔹).

    Args:
        wallet (basedai.wallet):
            Basedai wallet object.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        max_allowed_attempts (int):
            Maximum number of attempts to register the wallet.
        cuda (bool):
            If ``true``, the wallet should be registered using CUDA device(s).
        dev_id (Union[List[int], int]):
            The CUDA device id to use, or a list of device ids.
        tpb (int):
            The number of threads per block (CUDA).
        num_processes (int):
            The number of processes to use to register.
        update_interval (int):
            The number of nonces to solve between updates.
        log_verbose (bool):
            If ``true``, the memorization process will log more information.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    if prompt:
        if not Confirm.ask(
            "Run Faucet For $BASED?\n personalkey:    [bold white]{}[/bold white]\n network:    [bold white]{}[/bold white]".format(
                wallet.personalkeypub.ss58_address,
                basednode.network,
            )
        ):
            return False

    # Unlock personalkey
    wallet.personalkey

    # Get previous balance.
    old_balance = basednode.get_balance(wallet.personalkeypub.ss58_address)

    # Attempt rolling memorization.
    attempts = 1
    successes = 1
    while True:
        try:
            pow_result = None
            while pow_result == None or pow_result.is_stale(basednode=basednode):
                # Solve latest POW.
                if cuda:
                    if not torch.cuda.is_available():
                        if prompt:
                            basedai.__console__.error("CUDA is not available.")
                        return False
                    pow_result: Optional[POWSolution] = create_pow(
                        basednode,
                        wallet,
                        -1,
                        output_in_place,
                        cuda=cuda,
                        dev_id=dev_id,
                        tpb=tpb,
                        num_processes=num_processes,
                        update_interval=update_interval,
                        log_verbose=log_verbose,
                    )
                else:
                    pow_result: Optional[POWSolution] = create_pow(
                        basednode,
                        wallet,
                        -1,
                        output_in_place,
                        cuda=cuda,
                        num_processes=num_processes,
                        update_interval=update_interval,
                        log_verbose=log_verbose,
                    )
            call = basednode.substrate.compose_call(
                call_module="BasedNode",
                call_function="faucet",
                call_params={
                    "block_number": pow_result.block_number,
                    "nonce": pow_result.nonce,
                    "work": [int(byte_) for byte_ in pow_result.seal],
                },
            )
            extrinsic = basednode.substrate.create_signed_extrinsic(
                call=call, keypair=wallet.personalkey
            )
            response = basednode.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            # process if memorization successful, try again if pow is still valid
            response.process_events()
            if not response.is_success:
                basedai.__console__.print(
                    f":cross_mark: [red]Failed[/red]: Error: {response.error_message}"
                )
                if attempts == max_allowed_attempts:
                    raise MaxAttemptsException
                attempts += 1

            # Successful memorization
            else:
                new_balance = basednode.get_balance(wallet.personalkeypub.ss58_address)
                basedai.__console__.print(
                    f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
                )
                old_balance = new_balance

                if successes == 3:
                    raise MaxSuccessException
                successes += 1

        except KeyboardInterrupt:
            return True, "Done"

        except MaxSuccessException:
            return True, f"Max successes reached: {3}"

        except MaxAttemptsException:
            return False, f"Max attempts reached: {max_allowed_attempts}"


def swap_computekey_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    new_wallet: "basedai.wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
) -> bool:
    wallet.personalkey  # unlock personalkey
    if prompt:
        # Prompt user for confirmation.
        if not Confirm.ask(
            f"Swap {wallet.computekey} for new computekey: {new_wallet.computekey}?"
        ):
            return False

    with basedai.__console__.status(":brain: Swapping computekeys..."):
        success, err_msg = basednode._do_swap_computekey(
            wallet=wallet,
            new_wallet=new_wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if success != True or success == False:
            basedai.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(err_msg)
            )
            time.sleep(0.5)

        else:
            basedai.__console__.print(
                f"Computekey {wallet.computekey} swapped for new computekey: {new_wallet.computekey}"
            )
