# The MIT License (MIT)
# Copyright © 2024 Saul Finney
# 
# Copyright © 2023 Based Labs

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

from __future__ import annotations

import asyncio
import uuid
import time
import torch
import aiohttp
import basedai
from fastapi import Response
from typing import Union, Optional, List, Union, AsyncGenerator, Any


class brainrequester(torch.nn.Module):
    """
    The Brainrequester class, inheriting from PyTorch's Module class, represents the abstracted implementation of a network client module.

    In the brain analogy, brainrequesters receive signals
    from other neurons (in this case, network servers or brainports), and the Brainrequester class here is designed
    to send requests to those endpoint to recieve inputs.

    This class includes a wallet or keypair used for signing messages, and methods for making
    HTTP requests to the network servers. It also provides functionalities such as logging
    network requests and processing server responses.

    Args:
        keypair: The wallet or keypair used for signing messages.
        external_ip (str): The external IP address of the local system.
        brainresponder_history (list): A list of Brainresponder objects representing the historical responses.

    Methods:
        __str__(): Returns a string representation of the Brainrequester object.
        __repr__(): Returns a string representation of the Brainrequester object, acting as a fallback for __str__().
        query(self, *args, **kwargs) -> Union[basedai.Brainresponder, List[basedai.Brainresponder]]:
            Makes synchronous requests to one or multiple target Brainports and returns responses.

        forward(self, brainports, brainresponder=basedai.Brainresponder(), timeout=12, deserialize=True, run_async=True, streaming=False) -> basedai.Brainresponder:
            Asynchronously sends requests to one or multiple Brainports and collates their responses.

        call(self, target_brainport, brainresponder=basedai.Brainresponder(), timeout=12.0, deserialize=True) -> basedai.Brainresponder:
            Asynchronously sends a request to a specified Brainport and processes the response.

        call_stream(self, target_brainport, brainresponder=basedai.Brainresponder(), timeout=12.0, deserialize=True) -> AsyncGenerator[basedai.Brainresponder, None]:
            Sends a request to a specified Brainport and yields an AsyncGenerator that contains streaming response chunks before finally yielding the filled Brainresponder as the final element.

        preprocess_brainresponder_for_request(self, target_brainport_info, brainresponder, timeout=12.0) -> basedai.Brainresponder:
            Preprocesses the brainresponder for making a request, including building headers and signing.

        process_server_response(self, server_response, json_response, local_brainresponder):
            Processes the server response, updates the local brainresponder state, and merges headers.

        close_session(self):
            Synchronously closes the internal aiohttp client session.

        aclose_session(self):
            Asynchronously closes the internal aiohttp client session.

    NOTE:
        When working with async `aiohttp <https://github.com/aio-libs/aiohttp>`_ client sessions, it is recommended to use a context manager.

    Example with a context manager::

        >>> aysnc with brainrequester(wallet = basedai.wallet()) as d:
        >>>     print(d)
        >>>     d( <brainport> ) # ping brainport
        >>>     d( [<brainports>] ) # ping multiple
        >>>     d( basedai.brainport(), basedai.Brainresponder )

    However, you are able to safely call :func:`brainrequester.query()` without a context manager in a synchronous setting.

    Example without a context manager::

        >>> d = brainrequester(wallet = basedai.wallet() )
        >>> print(d)
        >>> d( <brainport> ) # ping brainport
        >>> d( [<brainports>] ) # ping multiple
        >>> d( basedai.brainport(), basedai.Brainresponder )
    """

    def __init__(
        self, wallet: Optional[Union[basedai.wallet, basedai.keypair]] = None
    ):
        """
        Initializes the Brainrequester object, setting up essential properties.

        Args:
            wallet (Optional[Union['basedai.wallet', 'basedai.keypair']], optional):
                The user's wallet or keypair used for signing messages. Defaults to ``None``, in which case a new :func:`basedai.wallet().computekey` is generated and used.
        """
        # Initialize the parent class
        super(brainrequester, self).__init__()

        # Unique identifier for the instance
        self.uuid = str(uuid.uuid1())

        # Get the external IP
        self.external_ip = basedai.utils.networking.get_external_ip()

        # If a wallet or keypair is provided, use its computekey. If not, generate a new one.
        self.keypair = (
            wallet.computekey if isinstance(wallet, basedai.wallet) else wallet
        ) or basedai.wallet().computekey

        self.brainresponder_history: list = []

        self._session: aiohttp.ClientSession = None

    @property
    async def session(self) -> aiohttp.ClientSession:
        """
        An asynchronous property that provides access to the internal `aiohttp <https://github.com/aio-libs/aiohttp>`_ client session.

        This property ensures the management of HTTP connections in an efficient way. It lazily
        initializes the `aiohttp.ClientSession <https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.ClientSession>`_ on its first use. The session is then reused for subsequent
        HTTP requests, offering performance benefits by reusing underlying connections.

        This is used internally by the brainrequester when querying brainports, and should not be used directly
        unless absolutely necessary for your application.

        Returns:
            aiohttp.ClientSession: The active `aiohttp <https://github.com/aio-libs/aiohttp>`_ client session instance. If no session exists, a
            new one is created and returned. This session is used for asynchronous HTTP requests within
            the brainrequester, adhering to the async nature of the network interactions in the Basedai framework.

        Example usage::

            import basedai as bt                    # Import basedai
            wallet = bt.wallet( ... )                 # Initialize a wallet
            brainrequester = bt.brainrequester( wallet )          # Initialize a brainrequester instance with the wallet

            async with (await brainrequester.session).post( # Use the session to make an HTTP POST request
                url,                                  # URL to send the request to
                headers={...},                        # Headers dict to be sent with the request
                json={...},                           # JSON body data to be sent with the request
                timeout=10,                           # Timeout duration in seconds
            ) as response:
                json_response = await response.json() # Extract the JSON response from the server

        """
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    def close_session(self):
        """
        Closes the internal `aiohttp <https://github.com/aio-libs/aiohttp>`_ client session synchronously.

        This method ensures the proper closure and cleanup of the aiohttp client session, releasing any
        resources like open connections and internal buffers. It is crucial for preventing resource leakage
        and should be called when the brainrequester instance is no longer in use, especially in synchronous contexts.

        Note:
            This method utilizes asyncio's event loop to close the session asynchronously from a synchronous context. It is advisable to use this method only when asynchronous context management is not feasible.

        Usage:
            When finished with brainrequester in a synchronous context
            :func:`brainrequester_instance.close_session()`.
        """
        if self._session:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._session.close())
            self._session = None

    async def aclose_session(self):
        """
        Asynchronously closes the internal `aiohttp <https://github.com/aio-libs/aiohttp>`_ client session.

        This method is the asynchronous counterpart to the :func:`close_session` method. It should be used in
        asynchronous contexts to ensure that the aiohttp client session is closed properly. The method
        releases resources associated with the session, such as open connections and internal buffers,
        which is essential for resource management in asynchronous applications.

        Usage:
            When finished with brainrequester in an asynchronous context
            await :func:`brainrequester_instance.aclose_session()`.

        Example::

            async with brainrequester_instance:
                # Operations using brainrequester
                pass
            # The session will be closed automatically after the above block
        """
        if self._session:
            await self._session.close()
            self._session = None

    def _get_endpoint_url(self, target_brainport, request_name):
        """
        Constructs the endpoint URL for a network request to a target brainport.

        This internal method generates the full HTTP URL for sending a request to the specified brainport. The
        URL includes the IP address and port of the target brainport, along with the specific request name. It
        differentiates between requests to the local system (using '0.0.0.0') and external systems.

        Args:
            target_brainport: The target brainport object containing IP and port information.
            request_name: The specific name of the request being made.

        Returns:
            str: A string representing the complete HTTP URL for the request.
        """
        endpoint = (
            f"0.0.0.0:{str(target_brainport.port)}"
            if target_brainport.ip == str(self.external_ip)
            else f"{target_brainport.ip}:{str(target_brainport.port)}"
        )
        return f"http://{endpoint}/{request_name}"

    def _handle_request_errors(self, brainresponder, request_name, exception):
        """
        Handles exceptions that occur during network requests, updating the brainresponder with appropriate status codes and messages.

        This method interprets different types of exceptions and sets the corresponding status code and
        message in the brainresponder object. It covers common network errors such as connection issues and timeouts.

        Args:
            brainresponder: The brainresponder object associated with the request.
            request_name: The name of the request during which the exception occurred.
            exception: The exception object caught during the request.

        Note:
            This method updates the brainresponder object in-place.
        """
        if isinstance(exception, aiohttp.ClientConnectorError):
            brainresponder.brainrequester.status_code = "503"
            brainresponder.brainrequester.status_message = f"Service at {brainresponder.brainport.ip}:{str(brainresponder.brainport.port)}/{request_name} unavailable."
        elif isinstance(exception, asyncio.TimeoutError):
            brainresponder.brainrequester.status_code = "408"
            brainresponder.brainrequester.status_message = (
                f"Timedout after {brainresponder.timeout} seconds."
            )
        else:
            brainresponder.brainrequester.status_code = "422"
            brainresponder.brainrequester.status_message = (
                f"Failed to parse response object with error: {str(exception)}"
            )

    def _log_outgoing_request(self, brainresponder):
        """
        Logs information about outgoing requests for debugging purposes.

        This internal method logs key details about each outgoing request, including the size of the
        request, the name of the brainresponder, the brainport's details, and a success indicator. This information
        is crucial for monitoring and debugging network activity within the Basedai network.

        To turn on debug messages, set the environment variable BASEDAI_DEBUG to ``1``, or call the basedai debug method like so::

            import basedai
            basedai.debug()

        Args:
            brainresponder: The brainresponder object representing the request being sent.
        """
        basedai.logging.debug(
            f"brainrequester | --> | {brainresponder.get_total_size()} B | {brainresponder.name} | {brainresponder.brainport.computekey} | {brainresponder.brainport.ip}:{str(brainresponder.brainport.port)} | 0 | Success"
        )

    def _log_incoming_response(self, brainresponder):
        """
        Logs information about incoming responses for debugging and monitoring.

        Similar to :func:`_log_outgoing_request`, this method logs essential details of the incoming responses,
        including the size of the response, brainresponder name, brainport details, status code, and status message.
        This logging is vital for troubleshooting and understanding the network interactions in Basedai.

        Args:
            brainresponder: The brainresponder object representing the received response.
        """
        basedai.logging.debug(
            f"brainrequester | <-- | {brainresponder.get_total_size()} B | {brainresponder.name} | {brainresponder.brainport.computekey} | {brainresponder.brainport.ip}:{str(brainresponder.brainport.port)} | {brainresponder.brainrequester.status_code} | {brainresponder.brainrequester.status_message}"
        )

    def query(
        self, *args, **kwargs
    ) -> Union[
        basedai.Brainresponder,
        List[basedai.Brainresponder],
        basedai.StreamingBrainresponder,
        List[basedai.StreamingBrainresponder],
    ]:
        """
        Makes a synchronous request to multiple target Brainports and returns the server responses.

        Cleanup is automatically handled and sessions are closed upon completed requests.

        Args:
            brainports (Union[List[Union['basedai.BrainportInfo', 'basedai.brainport']], Union['basedai.BrainportInfo', 'basedai.brainport']]):
                The list of target Brainport information.
            brainresponder (basedai.Brainresponder, optional): The Brainresponder object. Defaults to :func:`basedai.Brainresponder()`.
            timeout (float, optional): The request timeout duration in seconds.
                Defaults to ``12.0`` seconds.
        Returns:
            Union[basedai.Brainresponder, List[basedai.Brainresponder]]: If a single target brainport is provided, returns the response from that brainport. If multiple target brainports are provided, returns a list of responses from all target brainports.
        """
        result = None
        try:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self.forward(*args, **kwargs))
        except:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result = loop.run_until_complete(self.forward(*args, **kwargs))
            new_loop.close()
        finally:
            self.close_session()
            return result

    async def forward(
        self,
        brainports: Union[
            List[Union[basedai.BrainportInfo, basedai.brainport]],
            Union[basedai.BrainportInfo, basedai.brainport],
        ],
        brainresponder: basedai.Brainresponder = basedai.Brainresponder(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ) -> List[Union[AsyncGenerator[Any], bittenst.Brainresponder, basedai.StreamingBrainresponder]]:
        """
        Asynchronously sends requests to one or multiple Brainports and collates their responses.

        This function acts as a bridge for sending multiple requests concurrently or sequentially
        based on the provided parameters. It checks the type of the target Brainports, preprocesses
        the requests, and then sends them off. After getting the responses, it processes and
        collates them into a unified format.

        When querying an Brainport that sends a single response, this function returns a Brainresponder object
        containing the response data. If multiple Brainports are queried, a list of Brainresponder objects is
        returned, each containing the response from the corresponding Brainport.

        For example::

            >>> ...
            >>> wallet = basedai.wallet()                   # Initialize a wallet
            >>> brainresponder = basedai.Brainresponder(...)              # Create a brainresponder object that contains query data
            >>> dendrte = basedai.brainrequester(wallet = wallet) # Initialize a brainrequester instance
            >>> brainports = stem.brainports                       # Create a list of brainports to query
            >>> responses = await brainrequester(brainports, brainresponder)    # Send the query to all brainports and await the responses

        When querying an Brainport that sends back data in chunks using the Brainrequester, this function
        returns an AsyncGenerator that yields each chunk as it is received. The generator can be
        iterated over to process each chunk individually.

        For example::

            >>> ...
            >>> dendrte = basedai.brainrequester(wallet = wallet)
            >>> async for chunk in brainrequester.forward(brainports, brainresponder, timeout, deserialize, run_async, streaming):
            >>>     # Process each chunk here
            >>>     print(chunk)

        Args:
            brainports (Union[List[Union['basedai.BrainportInfo', 'basedai.brainport']], Union['basedai.BrainportInfo', 'basedai.brainport']]):
                The target Brainports to send requests to. Can be a single Brainport or a list of Brainports.
            brainresponder (basedai.Brainresponder, optional): The Brainresponder object encapsulating the data. Defaults to a new :func:`basedai.Brainresponder` instance.
            timeout (float, optional): Maximum duration to wait for a response from an Brainport in seconds. Defaults to ``12.0``.
            deserialize (bool, optional): Determines if the received response should be deserialized. Defaults to ``True``.
            run_async (bool, optional): If ``True``, sends requests concurrently. Otherwise, sends requests sequentially. Defaults to ``True``.
            streaming (bool, optional): Indicates if the response is expected to be in streaming format. Defaults to ``False``.

        Returns:
            Union[AsyncGenerator, basedai.Brainresponder, List[basedai.Brainresponder]]: If a single Brainport is targeted, returns its response.
            If multiple Brainports are targeted, returns a list of their responses.
        """
        is_list = True
        # If a single brainport is provided, wrap it in a list for uniform processing
        if not isinstance(brainports, list):
            is_list = False
            brainports = [brainports]

        # Check if brainresponder is an instance of the StreamingBrainresponder class or if streaming flag is set.
        is_streaming_subclass = issubclass(
            brainresponder.__class__, basedai.StreamingBrainresponder
        )
        if streaming != is_streaming_subclass:
            basedai.logging.warning(
                f"Argument streaming is {streaming} while issubclass(brainresponder, StreamingBrainresponder) is {brainresponder.__class__.__name__}. This may cause unexpected behavior."
            )
        streaming = is_streaming_subclass or streaming

        async def query_all_brainports(
            is_stream: bool,
        ) -> Union[AsyncGenerator[Any], bittenst.Brainresponder, basedai.StreamingBrainresponder]:
            """
            Handles the processing of requests to all targeted brainports, accommodating both streaming and non-streaming responses.

            This function manages the concurrent or sequential dispatch of requests to a list of brainports.
            It utilizes the ``is_stream`` parameter to determine the mode of response handling (streaming
            or non-streaming). For each brainport, it calls ``single_brainport_response`` and aggregates the responses.

            Args:
                is_stream (bool): Flag indicating whether the brainport responses are expected to be streamed.
                If ``True``, responses are handled in streaming mode.

            Returns:
                List[Union[AsyncGenerator, basedai.Brainresponder, basedai.StreamingBrainresponder]]: A list
                containing the responses from each brainport. The type of each response depends on the
                streaming mode and the type of brainresponder used.
            """

            async def single_brainport_response(
                target_brainport,
            ) -> Union[
                AsyncGenerator[Any], bittenst.Brainresponder, basedai.StreamingBrainresponder
            ]:
                """
                Manages the request and response process for a single brainport, supporting both streaming and non-streaming modes.

                This function is responsible for initiating a request to a single brainport. Depending on the
                ``is_stream`` flag, it either uses ``call_stream`` for streaming responses or ``call`` for
                standard responses. The function handles the response processing, catering to the specifics
                of streaming or non-streaming data.

                Args:
                    target_brainport: The target brainport object to which the request is to be sent. This object contains the necessary information like IP address and port to formulate the request.

                Returns:
                    Union[AsyncGenerator, basedai.Brainresponder, basedai.StreamingBrainresponder]: The response
                    from the targeted brainport. In streaming mode, an AsyncGenerator is returned, yielding
                    data chunks. In non-streaming mode, a Brainresponder or StreamingBrainresponder object is returned
                    containing the response.
                """
                if is_stream:
                    # If in streaming mode, return the async_generator
                    return self.call_stream(
                        target_brainport=target_brainport,
                        brainresponder=brainresponder.copy(),
                        timeout=timeout,
                        deserialize=deserialize,
                    )
                else:
                    # If not in streaming mode, simply call the brainport and get the response.
                    return await self.call(
                        target_brainport=target_brainport,
                        brainresponder=brainresponder.copy(),
                        timeout=timeout,
                        deserialize=deserialize,
                    )

            # If run_async flag is False, get responses one by one.
            if not run_async:
                return [
                    await single_brainport_response(target_brainport) for target_brainport in brainports
                ]
            # If run_async flag is True, get responses concurrently using asyncio.gather().
            return await asyncio.gather(
                *(single_brainport_response(target_brainport) for target_brainport in brainports)
            )

        # Get responses for all brainports.
        responses = await query_all_brainports(streaming)
        # Return the single response if only one brainport was targeted, else return all responses
        if len(responses) == 1 and not is_list:
            return responses[0]
        else:
            return responses

    async def call(
        self,
        target_brainport: Union[basedai.BrainportInfo, basedai.brainport],
        brainresponder: basedai.Brainresponder = basedai.Brainresponder(),
        timeout: float = 12.0,
        deserialize: bool = True,
    ) -> basedai.Brainresponder:
        """
        Asynchronously sends a request to a specified Brainport and processes the response.

        This function establishes a connection with a specified Brainport, sends the encapsulated
        data through the Brainresponder object, waits for a response, processes it, and then
        returns the updated Brainresponder object.

        Args:
            target_brainport (Union['basedai.BrainportInfo', 'basedai.brainport']): The target Brainport to send the request to.
            brainresponder (basedai.Brainresponder, optional): The Brainresponder object encapsulating the data. Defaults to a new :func:`basedai.Brainresponder` instance.
            timeout (float, optional): Maximum duration to wait for a response from the Brainport in seconds. Defaults to ``12.0``.
            deserialize (bool, optional): Determines if the received response should be deserialized. Defaults to ``True``.

        Returns:
            basedai.Brainresponder: The Brainresponder object, updated with the response data from the Brainport.
        """

        # Record start time
        start_time = time.time()
        target_brainport = (
            target_brainport.info()
            if isinstance(target_brainport, basedai.brainport)
            else target_brainport
        )

        # Build request endpoint from the brainresponder class
        request_name = brainresponder.__class__.__name__
        url = self._get_endpoint_url(target_brainport, request_name=request_name)

        # Preprocess brainresponder for making a request
        brainresponder = self.preprocess_brainresponder_for_request(target_brainport, brainresponder, timeout)

        try:
            # Log outgoing request
            self._log_outgoing_request(brainresponder)

            # Make the HTTP POST request
            async with (await self.session).post(
                url,
                headers=brainresponder.to_headers(),
                json=brainresponder.dict(),
                timeout=timeout,
            ) as response:
                # Extract the JSON response from the server
                json_response = await response.json()
                # Process the server response and fill brainresponder
                self.process_server_response(response, json_response, brainresponder)

            # Set process time and log the response
            brainresponder.brainrequester.process_time = str(time.time() - start_time)

        except Exception as e:
            self._handle_request_errors(brainresponder, request_name, e)

        finally:
            self._log_incoming_response(brainresponder)

            # Log brainresponder event history
            self.brainresponder_history.append(
                basedai.Brainresponder.from_headers(brainresponder.to_headers())
            )

            # Return the updated brainresponder object after deserializing if requested
            if deserialize:
                return brainresponder.deserialize()
            else:
                return brainresponder

    async def call_stream(
        self,
        target_brainport: Union[basedai.BrainportInfo, basedai.brainport],
        brainresponder: basedai.Brainresponder = basedai.Brainresponder(),
        timeout: float = 12.0,
        deserialize: bool = True,
    ) -> AsyncGenerator[Any]:
        """
        Sends a request to a specified Brainport and yields streaming responses.

        Similar to ``call``, but designed for scenarios where the Brainport sends back data in
        multiple chunks or streams. The function yields each chunk as it is received. This is
        useful for processing large responses piece by piece without waiting for the entire
        data to be transmitted.

        Args:
            target_brainport (Union['basedai.BrainportInfo', 'basedai.brainport']): The target Brainport to send the request to.
            brainresponder (basedai.Brainresponder, optional): The Brainresponder object encapsulating the data. Defaults to a new :func:`basedai.Brainresponder` instance.
            timeout (float, optional): Maximum duration to wait for a response (or a chunk of the response) from the Brainport in seconds. Defaults to ``12.0``.
            deserialize (bool, optional): Determines if each received chunk should be deserialized. Defaults to ``True``.

        Yields:
            object: Each yielded object contains a chunk of the arbitrary response data from the Brainport.
            basedai.Brainresponder: After the AsyncGenerator has been exhausted, yields the final filled Brainresponder.
        """

        # Record start time
        start_time = time.time()
        target_brainport = (
            target_brainport.info()
            if isinstance(target_brainport, basedai.brainport)
            else target_brainport
        )

        # Build request endpoint from the brainresponder class
        request_name = brainresponder.__class__.__name__
        endpoint = (
            f"0.0.0.0:{str(target_brainport.port)}"
            if target_brainport.ip == str(self.external_ip)
            else f"{target_brainport.ip}:{str(target_brainport.port)}"
        )
        url = f"http://{endpoint}/{request_name}"

        # Preprocess brainresponder for making a request
        brainresponder = self.preprocess_brainresponder_for_request(target_brainport, brainresponder, timeout)

        try:
            # Log outgoing request
            self._log_outgoing_request(brainresponder)

            # Make the HTTP POST request
            async with (await self.session).post(
                url,
                headers=brainresponder.to_headers(),
                json=brainresponder.dict(),
                timeout=timeout,
            ) as response:
                # Use brainresponder subclass' process_streaming_response method to yield the response chunks
                async for chunk in brainresponder.process_streaming_response(response):
                    yield chunk  # Yield each chunk as it's processed
                json_response = brainresponder.extract_response_json(response)

                # Process the server response
                self.process_server_response(response, json_response, brainresponder)

            # Set process time and log the response
            brainresponder.brainrequester.process_time = str(time.time() - start_time)

        except Exception as e:
            self._handle_request_errors(brainresponder, request_name, e)

        finally:
            self._log_incoming_response(brainresponder)

            # Log brainresponder event history
            self.brainresponder_history.append(
                basedai.Brainresponder.from_headers(brainresponder.to_headers())
            )

            # Return the updated brainresponder object after deserializing if requested
            if deserialize:
                yield brainresponder.deserialize()
            else:
                yield brainresponder

    def preprocess_brainresponder_for_request(
        self,
        target_brainport_info: basedai.BrainportInfo,
        brainresponder: basedai.Brainresponder,
        timeout: float = 12.0,
    ) -> basedai.Brainresponder:
        """
        Preprocesses the brainresponder for making a request. This includes building
        headers for Brainrequester and Brainport and signing the request.

        Args:
            target_brainport_info (basedai.BrainportInfo): The target brainport information.
            brainresponder (basedai.Brainresponder): The brainresponder object to be preprocessed.
            timeout (float, optional): The request timeout duration in seconds.
                Defaults to ``12.0`` seconds.

        Returns:
            basedai.Brainresponder: The preprocessed brainresponder.
        """
        # Set the timeout for the brainresponder
        brainresponder.timeout = str(timeout)

        # Build the Brainrequester headers using the local system's details
        brainresponder.brainrequester = basedai.TerminalInfo(
            **{
                "ip": str(self.external_ip),
                "version": str(basedai.__version_as_int__),
                "nonce": f"{time.monotonic_ns()}",
                "uuid": str(self.uuid),
                "computekey": str(self.keypair.ss58_address),
            }
        )

        # Build the Brainport headers using the target brainport's details
        brainresponder.brainport = basedai.TerminalInfo(
            **{
                "ip": str(target_brainport_info.ip),
                "port": str(target_brainport_info.port),
                "computekey": str(target_brainport_info.computekey),
            }
        )

        # Sign the request using the brainrequester, brainport info, and the brainresponder body hash
        message = f"{brainresponder.brainrequester.nonce}.{brainresponder.brainrequester.computekey}.{brainresponder.brainport.computekey}.{brainresponder.brainrequester.uuid}.{brainresponder.body_hash}"
        brainresponder.brainrequester.signature = f"0x{self.keypair.sign(message).hex()}"

        return brainresponder

    def process_server_response(
        self,
        server_response: Response,
        json_response: dict,
        local_brainresponder: basedai.Brainresponder,
    ):
        """
        Processes the server response, updates the local brainresponder state with the
        server's state and merges headers set by the server.

        Args:
            server_response (object): The `aiohttp <https://github.com/aio-libs/aiohttp>`_ response object from the server.
            json_response (dict): The parsed JSON response from the server.
            local_brainresponder (basedai.Brainresponder): The local brainresponder object to be updated.

        Raises:
            None: But errors in attribute setting are silently ignored.
        """
        # Check if the server responded with a successful status code
        if server_response.status == 200:
            # If the response is successful, overwrite local brainresponder state with
            # server's state only if the protocol allows mutation. To prevent overwrites,
            # the protocol must set allow_mutation = False
            server_brainresponder = local_brainresponder.__class__(**json_response)
            for key in local_brainresponder.dict().keys():
                try:
                    # Set the attribute in the local brainresponder from the corresponding
                    # attribute in the server brainresponder
                    setattr(local_brainresponder, key, getattr(server_brainresponder, key))
                except:
                    # Ignore errors during attribute setting
                    pass

        # Extract server headers and overwrite None values in local brainresponder headers
        server_headers = basedai.Brainresponder.from_headers(server_response.headers)

        # Merge brainrequester headers
        local_brainresponder.brainrequester.__dict__.update(
            {
                **local_brainresponder.brainrequester.dict(exclude_none=True),
                **server_headers.brainrequester.dict(exclude_none=True),
            }
        )

        # Merge brainport headers
        local_brainresponder.brainport.__dict__.update(
            {
                **local_brainresponder.brainport.dict(exclude_none=True),
                **server_headers.brainport.dict(exclude_none=True),
            }
        )

        # Update the status code and status message of the brainrequester to match the brainport
        local_brainresponder.brainrequester.status_code = local_brainresponder.brainport.status_code
        local_brainresponder.brainrequester.status_message = local_brainresponder.brainport.status_message

    def __str__(self) -> str:
        """
        Returns a string representation of the Brainrequester object.

        Returns:
            str: The string representation of the Brainrequester object in the format :func:`brainrequester(<user_wallet_address>)`.
        """
        return "brainrequester({})".format(self.keypair.ss58_address)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Brainrequester object, acting as a fallback for :func:`__str__()`.

        Returns:
            str: The string representation of the Brainrequester object in the format :func:`brainrequester(<user_wallet_address>)`.
        """
        return self.__str__()

    async def __aenter__(self):
        """
        Asynchronous context manager entry method.

        Enables the use of the ``async with`` statement with the Brainrequester instance. When entering the context,
        the current instance of the class is returned, making it accessible within the asynchronous context.

        Returns:
            Brainrequester: The current instance of the Brainrequester class.

        Usage::

            async with Brainrequester() as brainrequester:
                await brainrequester.some_async_method()
        """
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """
        Asynchronous context manager exit method.

        Ensures proper cleanup when exiting the ``async with`` context. This method will close the `aiohttp <https://github.com/aio-libs/aiohttp>`_ client session
        asynchronously, releasing any tied resources.

        Args:
            exc_type (Type[BaseException], optional): The type of exception that was raised.
            exc_value (BaseException, optional): The instance of exception that was raised.
            traceback (TracebackType, optional): A traceback object encapsulating the call stack at the point where the exception was raised.

        Usage::

            async with bt.brainrequester( wallet ) as brainrequester:
                await brainrequester.some_async_method()

        Note:
            This automatically closes the session by calling :func:`__aexit__` after the context closes.
        """
        await self.aclose_session()

    def __del__(self):
        """
        Brainrequester destructor.

        This method is invoked when the Brainrequester instance is about to be destroyed. The destructor ensures that the
        aiohttp client session is closed before the instance is fully destroyed, releasing any remaining resources.

        Note:
            Relying on the destructor for cleanup can be unpredictable. It is recommended to explicitly close sessions using the provided methods or the ``async with`` context manager.

        Usage::

            brainrequester = Brainrequester()
            # ... some operations ...
            del brainrequester  # This will implicitly invoke the __del__ method and close the session.
        """
        asyncio.run(self.aclose_session())
