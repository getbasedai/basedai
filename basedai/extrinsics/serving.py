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
import json
import basedai
from dataclasses import asdict
import basedai.utils.networking as net
from rich.prompt import Confirm
from ..errors import MetadataError


def serve_extrinsic(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    ip: str,
    port: int,
    protocol: int,
    netuid: int,
    placeholder1: int = 0,
    placeholder2: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization=True,
    prompt: bool = False,
) -> bool:
    r"""Subscribes a Basedai endpoint to the basednode chain.

    Args:
        wallet (basedai.wallet):
            Basedai wallet object.
        ip (str):
            Endpoint host port i.e., ``192.122.31.4``.
        port (int):
            Endpoint port number i.e., ``9221``.
        protocol (int):
            An ``int`` representation of the protocol.
        netuid (int):
            The network uid to serve on.
        placeholder1 (int):
            A placeholder for future use.
        placeholder2 (int):
            A placeholder for future use.
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
    # Decrypt computekey
    wallet.computekey
    params: "BrainportServeCallParams" = {
        "version": basedai.__version_as_int__,
        "ip": net.ip_to_int(ip),
        "port": port,
        "ip_type": net.ip_version(ip),
        "netuid": netuid,
        "computekey": wallet.computekey.ss58_address,
        "personalkey": wallet.personalkeypub.ss58_address,
        "protocol": protocol,
        "placeholder1": placeholder1,
        "placeholder2": placeholder2,
    }
    basedai.logging.debug("Checking brainport ...")
    neuron = basednode.get_neuron_for_pubkey_and_subnet(
        wallet.computekey.ss58_address, netuid=netuid
    )
    neuron_up_to_date = not neuron.is_null and params == {
        "version": neuron.brainport_info.version,
        "ip": net.ip_to_int(neuron.brainport_info.ip),
        "port": neuron.brainport_info.port,
        "ip_type": neuron.brainport_info.ip_type,
        "netuid": neuron.netuid,
        "computekey": neuron.computekey,
        "personalkey": neuron.personalkey,
        "protocol": neuron.brainport_info.protocol,
        "placeholder1": neuron.brainport_info.placeholder1,
        "placeholder2": neuron.brainport_info.placeholder2,
    }
    output = params.copy()
    output["personalkey"] = wallet.personalkeypub.ss58_address
    output["computekey"] = wallet.computekey.ss58_address
    if neuron_up_to_date:
        basedai.logging.debug(
            f"Brainport already served on: BrainportInfo({wallet.computekey.ss58_address},{ip}:{port}) "
        )
        return True

    if prompt:
        output = params.copy()
        output["personalkey"] = wallet.personalkeypub.ss58_address
        output["computekey"] = wallet.computekey.ss58_address
        if not Confirm.ask(
            "Do you want to serve brainport:\n  [bold white]{}[/bold white]".format(
                json.dumps(output, indent=4, sort_keys=True)
            )
        ):
            return False

    basedai.logging.debug(
        f"Serving brainport with: BrainportInfo({wallet.computekey.ss58_address},{ip}:{port}) -> {basednode.network}:{netuid}"
    )
    success, error_message = basednode._do_serve_brainport(
        wallet=wallet,
        call_params=params,
        wait_for_finalization=wait_for_finalization,
        wait_for_inclusion=wait_for_inclusion,
    )

    if wait_for_inclusion or wait_for_finalization:
        if success == True:
            basedai.logging.debug(
                f"Brainport served with: BrainportInfo({wallet.computekey.ss58_address},{ip}:{port}) on {basednode.network}:{netuid} "
            )
            return True
        else:
            basedai.logging.debug(
                f"Brainport failed to served with error: {error_message} "
            )
            return False
    else:
        return True


def serve_brainport_extrinsic(
    basednode: "basedai.basednode",
    netuid: int,
    brainport: "basedai.Brainport",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
) -> bool:
    r"""Serves the brainport to the network.

    Args:
        netuid ( int ):
            The ``netuid`` being served on.
        brainport (basedai.Brainport):
            Brainport to serve.
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
    brainport.wallet.computekey
    brainport.wallet.personalkeypub
    external_port = brainport.external_port

    # ---- Get external ip ----
    if brainport.external_ip == None:
        try:
            external_ip = net.get_external_ip()
            basedai.__console__.print(
                ":white_heavy_check_mark: [green]Found external ip: {}[/green]".format(
                    external_ip
                )
            )
            basedai.logging.success(
                prefix="External IP", sufix="<blue>{}</blue>".format(external_ip)
            )
        except Exception as E:
            raise RuntimeError(
                "Unable to attain your external ip. Check your internet connection. error: {}".format(
                    E
                )
            ) from E
    else:
        external_ip = brainport.external_ip

    # ---- Subscribe to chain ----
    serve_success = basednode.serve(
        wallet=brainport.wallet,
        ip=external_ip,
        port=external_port,
        netuid=netuid,
        protocol=4,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        prompt=prompt,
    )
    return serve_success


def publish_metadata(
    basednode: "basedai.basednode",
    wallet: "basedai.wallet",
    netuid: int,
    type: str,
    data: str,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> bool:
    """
    Publishes metadata on the Basedai network using the specified wallet and network identifier.

    Args:
        basednode (basedai.basednode):
            The basednode instance representing the Basedai blockchain connection.
        wallet (basedai.wallet):
            The wallet object used for authentication in the transaction.
        netuid (int):
            Network UID on which the metadata is to be published.
        type (str):
            The data type of the information being submitted. It should be one of the following: ``'Sha256'``, ``'Blake256'``, ``'Keccak256'``, or ``'Raw0-128'``. This specifies the format or hashing algorithm used for the data.
        data (str):
            The actual metadata content to be published. This should be formatted or hashed according to the ``type`` specified. (Note: max ``str`` length is 128 bytes)
        wait_for_inclusion (bool, optional):
            If ``True``, the function will wait for the extrinsic to be included in a block before returning. Defaults to ``False``.
        wait_for_finalization (bool, optional):
            If ``True``, the function will wait for the extrinsic to be finalized on the chain before returning. Defaults to ``True``.

    Returns:
        bool:
            ``True`` if the metadata was successfully published (and finalized if specified). ``False`` otherwise.

    Raises:
        MetadataError:
            If there is an error in submitting the extrinsic or if the response from the blockchain indicates failure.
    """

    wallet.computekey

    with basednode.substrate as substrate:
        call = substrate.compose_call(
            call_module="Commitments",
            call_function="set_commitment",
            call_params={"netuid": netuid, "info": {"fields": [[{f"{type}": data}]]}},
        )

        extrinsic = substrate.create_signed_extrinsic(call=call, keypair=wallet.computekey)
        response = substrate.submit_extrinsic(
            extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
        # We only wait here if we expect finalization.
        if not wait_for_finalization and not wait_for_inclusion:
            return True
        response.process_events()
        if response.is_success:
            return True
        else:
            raise MetadataError(response.error_message)


from retry import retry
from typing import Optional


def get_metadata(self, netuid: int, computekey: str, block: Optional[int] = None) -> str:
    @retry(delay=2, tries=3, backoff=2, max_delay=4)
    def make_substrate_call_with_retry():
        with self.substrate as substrate:
            return substrate.query(
                module="Commitments",
                storage_function="CommitmentOf",
                params=[netuid, computekey],
                block_hash=None if block == None else substrate.get_block_hash(block),
            )

    commit_data = make_substrate_call_with_retry()
    return commit_data.value
