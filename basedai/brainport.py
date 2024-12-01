""" Create and initialize Brain ports, which the forward and backward requests from other agents.
"""

# The MIT License (MIT)
# Copyright © 2024 Sean Wellington
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

import os
import uuid
import copy
import json
import time
import base64
import asyncio
import inspect
import uvicorn
import argparse
import traceback
import threading
import basedai
import contextlib

from inspect import signature, Signature, Parameter
from fastapi.responses import JSONResponse
from substrateinterface import Keypair
from fastapi import FastAPI, APIRouter, Request, Response, Depends
from starlette.types import Scope, Message
from starlette.responses import Response
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from typing import Dict, Optional, Tuple, Union, List, Callable, Any

from basedai.errors import (
    InvalidRequestNameError,
    BrainresponderParsingError,
    UnknownBrainresponderError,
    NotVerifiedException,
    BlacklistedException,
    PriorityException,
    RunException,
    PostProcessException,
    InternalServerError,
)


class FastAPIThreadedServer(uvicorn.Server):
    """
    The ``FastAPIThreadedServer`` class is a specialized server implementation for the Brainport server in the Basedai network.

    It extends the functionality of :func:`uvicorn.Server` to run the FastAPI application in a separate thread, allowing the Brainport server to handle HTTP requests concurrently and non-blocking.

    This class is designed to facilitate the integration of FastAPI with the Brainport's asynchronous architecture, ensuring efficient and scalable handling of network requests.

    Importance and Functionality
        Threaded Execution
            The class allows the FastAPI application to run in a separate thread, enabling concurrent handling of HTTP requests which is crucial for the performance and scalability of the Brainport server.

        Seamless Integration
            By running FastAPI in a threaded manner, this class ensures seamless integration of FastAPI's capabilities with the Brainport server's asynchronous and multi-threaded architecture.

        Controlled Server Management
            The methods start and stop provide controlled management of the server's lifecycle, ensuring that the server can be started and stopped as needed, which is vital for maintaining the Brainport server's reliability and availability.

        Signal Handling
            Overriding the default signal handlers prevents potential conflicts with the Brainport server's main application flow, ensuring stable operation in various network conditions.

    Use Cases
        Starting the Server
            When the Brainport server is initialized, it can use this class to start the FastAPI application in a separate thread, enabling it to begin handling HTTP requests immediately.

        Stopping the Server
            During shutdown or maintenance of the Brainport server, this class can be used to stop the FastAPI application gracefully, ensuring that all resources are properly released.

    Args:
        should_exit (bool): Flag to indicate whether the server should stop running.
        is_running (bool): Flag to indicate whether the server is currently running.

    The server overrides the default signal handlers to prevent interference with the main application flow and provides methods to start and stop the server in a controlled manner.
    """

    should_exit: bool = False
    is_running: bool = False

    def install_signal_handlers(self):
        """
        Overrides the default signal handlers provided by ``uvicorn.Server``. This method is essential to ensure that the signal handling in the threaded server does not interfere with the main application's flow, especially in a complex asynchronous environment like the Brainport server.
        """
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        """
        Manages the execution of the server in a separate thread, allowing the FastAPI application to run asynchronously without blocking the main thread of the Brainport server. This method is a key component in enabling concurrent request handling in the Brainport server.

        Yields:
            None: This method yields control back to the caller while the server is running in the background thread.
        """
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

    def _wrapper_run(self):
        """
        A wrapper method for the :func:`run_in_thread` context manager. This method is used internally by the ``start`` method to initiate the server's execution in a separate thread.
        """
        with self.run_in_thread():
            while not self.should_exit:
                time.sleep(1e-3)

    def start(self):
        """
        Starts the FastAPI server in a separate thread if it is not already running. This method sets up the server to handle HTTP requests concurrently, enabling the Brainport server to efficiently manage
        incoming network requests.

        The method ensures that the server starts running in a non-blocking manner, allowing the Brainport server to continue its other operations seamlessly.
        """
        if not self.is_running:
            self.should_exit = False
            thread = threading.Thread(target=self._wrapper_run, daemon=True)
            thread.start()
            self.is_running = True

    def stop(self):
        """
        Signals the FastAPI server to stop running. This method sets the :func:`should_exit` flag to ``True``, indicating that the server should cease its operations and exit the running thread.

        Stopping the server is essential for controlled shutdowns and resource management in the Brainport server, especially during maintenance or when redeploying with updated configurations.
        """
        if self.is_running:
            self.should_exit = True


class brainport:
    """
    The ``brainport`` class in Basedai is a fundamental component that serves as the server-side interface for a neuron within the Basedai network.

    This class is responsible for managing
    incoming requests from other neurons and implements various mechanisms to ensure efficient
    and secure network interactions.

    An brainport relies on a FastAPI router to create endpoints for different message types. These
    endpoints are crucial for handling various request types that a neuron might receive. The
    class is designed to be flexible and customizable, allowing users to specify custom rules
    for forwarding, blacklisting, prioritizing, and verifying incoming requests. The class also
    includes internal mechanisms to manage a thread pool, supporting concurrent handling of
    requests with defined priority levels.

    Methods in this class are equipped to deal with incoming requests from various scenarios in the
    network and serve as the server face for a neuron. It accepts multiple arguments, like wallet,
    configuration parameters, ip address, server binding  port, external ip, external port and max
    workers. Key methods involve managing and operating the FastAPI application router, including
    the attachment and operation of endpoints.

    Key Features:

    - FastAPI router integration for endpoint creation and management.
    - Customizable request handling including forwarding, blacklisting, and prioritization.
    - Verification of incoming requests against custom-defined functions.
    - Thread pool management for concurrent request handling.
    - Command-line argument support for user-friendly program interaction.

    Example Usage::

        import basedai
        # Define your custom brainresponder class
        class MySyanpse( basedai.Brainresponder ):
            input: int = 1
            output: int = None

        # Define a custom request forwarding function using your brainresponder class
        def forward( brainresponder: MySyanpse ) -> MySyanpse:
            # Apply custom logic to brainresponder and return it
            brainresponder.output = 2
            return brainresponder

        # Define a custom request verification function
        def verify_my_brainresponder( brainresponder: MySyanpse ):
            # Apply custom verification logic to brainresponder
            # Optionally raise Exception
            assert brainresponder.input == 1
            ...

        # Define a custom request blacklist fucntion
        def blacklist_my_brainresponder( brainresponder: MySyanpse ) -> bool:
            # Apply custom blacklist
            return False ( if non blacklisted ) or True ( if blacklisted )

        # Define a custom request priority fucntion
        def prioritize_my_synape( brainresponder: MySyanpse ) -> float:
            # Apply custom priority
            return 1.0

        # Initialize Brainport object with a custom configuration
        my_brainport = basedai.brainport(
            config=my_config,
            wallet=my_wallet,
            port=9090,
            ip="192.0.2.0",
            external_ip="203.0.113.0",
            external_port=7070
        )

        # Attach the endpoint with the specified verification and forward functions.
        my_brainport.attach(
            forward_fn = forward_my_brainresponder,
            verify_fn = verify_my_brainresponder,
            blacklist_fn = blacklist_my_brainresponder,
            priority_fn = prioritize_my_synape
        )

        # Serve and start your brainport.
        my_brainport.serve(
            netuid = ...
            basednode = ...
        ).start()

        # If you have multiple forwarding functions, you can chain attach them.
        my_brainport.attach(
            forward_fn = forward_my_brainresponder,
            verify_fn = verify_my_brainresponder,
            blacklist_fn = blacklist_my_brainresponder,
            priority_fn = prioritize_my_synape
        ).attach(
            forward_fn = forward_my_brainresponder_2,
            verify_fn = verify_my_brainresponder_2,
            blacklist_fn = blacklist_my_brainresponder_2,
            priority_fn = prioritize_my_synape_2
        ).serve(
            netuid = ...
            basednode = ...
        ).start()

    Args:
        wallet (basedai.wallet, optional): Wallet with computekey and personalkeypub.
        config (basedai.config, optional): Configuration parameters for the brainport.
        port (int, optional): Port for server binding.
        ip (str, optional): Binding IP address.
        external_ip (str, optional): External IP address to broadcast.
        external_port (int, optional): External port to broadcast.
        max_workers (int, optional): Number of active threads for request handling.

    Returns:
        basedai.brainport: An instance of the brainport class configured as per the provided arguments.

    Note:
        This class is a core part of Basedai's decentralized network for machine intelligence,
        allowing neurons to communicate effectively and securely.

    Importance and Functionality
        Endpoint Registration
            This method dynamically registers API endpoints based on the Brainresponder used, allowing the Brainport to respond to specific types of requests and brainresponders.

        Customization of Request Handling
            By attaching different functions, the Brainport can customize how it
            handles, verifies, prioritizes, and potentially blocks incoming requests, making it adaptable to various network scenarios.

        Security and Efficiency
            The method contributes to both the security (via verification and blacklisting) and efficiency (via prioritization) of request handling, which are crucial in a decentralized network environment.

        Flexibility
            The ability to define custom functions for different aspects of request handling provides great flexibility, allowing the Brainport to be tailored to specific needs and use cases within the Basedai network.

        Error Handling and Validation
            The method ensures that the attached functions meet the required
            signatures, providing error handling to prevent runtime issues.

    """

    def info(self) -> "basedai.BrainportInfo":
        """Returns the brainport info object associated with this brainport."""
        return basedai.BrainportInfo(
            version=basedai.__version_as_int__,
            ip=self.external_ip,
            ip_type=4,
            port=self.external_port,
            computekey=self.wallet.computekey.ss58_address,
            personalkey=self.wallet.personalkeypub.ss58_address,
            protocol=4,
            placeholder1=0,
            placeholder2=0,
        )

    def __init__(
        self,
        wallet: "basedai.wallet" = None,
        config: Optional["basedai.config"] = None,
        port: Optional[int] = None,
        ip: Optional[str] = None,
        external_ip: Optional[str] = None,
        external_port: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> "basedai.brainport":
        r"""Creates a new basedai.Brainport object from passed arguments.
        Args:
            config (:obj:`Optional[basedai.config]`, `optional`):
                basedai.brainport.config()
            wallet (:obj:`Optional[basedai.wallet]`, `optional`):
                basedai wallet with computekey and personalkeypub.
            port (:type:`Optional[int]`, `optional`):
                Binding port.
            ip (:type:`Optional[str]`, `optional`):
                Binding ip.
            external_ip (:type:`Optional[str]`, `optional`):
                The external ip of the server to broadcast to the network.
            external_port (:type:`Optional[int]`, `optional`):
                The external port of the server to broadcast to the network.
            max_workers (:type:`Optional[int]`, `optional`):
                Used to create the threadpool if not passed, specifies the number of active threads servicing requests.
        """
        # Build and check config.
        if config is None:
            config = brainport.config()
        config = copy.deepcopy(config)
        config.brainport.ip = ip or config.brainport.get("ip", basedai.defaults.brainport.ip)
        config.brainport.port = port or config.brainport.get("port", basedai.defaults.brainport.port)
        config.brainport.external_ip = external_ip or config.brainport.get(
            "external_ip", basedai.defaults.brainport.external_ip
        )
        config.brainport.external_port = external_port or config.brainport.get(
            "external_port", basedai.defaults.brainport.external_port
        )
        config.brainport.max_workers = max_workers or config.brainport.get(
            "max_workers", basedai.defaults.brainport.max_workers
        )
        brainport.check_config(config)
        self.config = config

        # Get wallet or use default.
        self.wallet = wallet or basedai.wallet()

        # Build brainport objects.
        self.uuid = str(uuid.uuid1())
        self.ip = self.config.brainport.ip
        self.port = self.config.brainport.port
        self.external_ip = (
            self.config.brainport.external_ip
            if self.config.brainport.external_ip != None
            else basedai.utils.networking.get_external_ip()
        )
        self.external_port = (
            self.config.brainport.external_port
            if self.config.brainport.external_port != None
            else self.config.brainport.port
        )
        self.full_address = str(self.config.brainport.ip) + ":" + str(self.config.brainport.port)
        self.started = False

        # Build middleware
        self.thread_pool = basedai.PriorityThreadPoolExecutor(
            max_workers=self.config.brainport.max_workers
        )
        self.nonces = {}

        # Request default functions.
        self.forward_class_types = {}
        self.blacklist_fns = {}
        self.priority_fns = {}
        self.forward_fns = {}
        self.verify_fns = {}
        self.required_hash_fields = {}

        # Instantiate FastAPI
        self.app = FastAPI()
        log_level = "trace" if basedai.logging.__trace_on__ else "critical"
        self.fast_config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.config.brainport.port, log_level=log_level
        )
        self.fast_server = FastAPIThreadedServer(config=self.fast_config)
        self.router = APIRouter()
        self.app.include_router(self.router)

        # Build ourselves as the middleware.
        self.app.add_middleware(BrainportMiddleware, brainport=self)

        # Attach default forward.
        def ping(r: basedai.Brainresponder) -> basedai.Brainresponder:
            return r

        self.attach(
            forward_fn=ping, verify_fn=None, blacklist_fn=None, priority_fn=None
        )

    def attach(
        self,
        forward_fn: Callable,
        blacklist_fn: Callable = None,
        priority_fn: Callable = None,
        verify_fn: Callable = None,
    ) -> "basedai.brainport":
        """

        Attaches custom functions to the Brainport server for handling incoming requests. This method enables
        the Brainport to define specific behaviors for request forwarding, verification, blacklisting, and
        prioritization, thereby customizing its interaction within the Basedai network.

        Registers an API endpoint to the FastAPI application router.
        It uses the name of the first argument of the :func:`forward_fn` function as the endpoint name.

        The attach method in the Basedai framework's brainport class is a crucial function for registering
        API endpoints to the Brainport's FastAPI application router. This method allows the Brainport server to
        define how it handles incoming requests by attaching functions for forwarding, verifying,
        blacklisting, and prioritizing requests. It's a key part of customizing the server's behavior
        and ensuring efficient and secure handling of requests within the Basedai network.

        Args:
            forward_fn (Callable): Function to be called when the API endpoint is accessed. It should have at least one argument.
            blacklist_fn (Callable, optional): Function to filter out undesired requests. It should take the same arguments as :func:`forward_fn` and return a boolean value. Defaults to ``None``, meaning no blacklist filter will be used.
            priority_fn (Callable, optional): Function to rank requests based on their priority. It should take the same arguments as :func:`forward_fn` and return a numerical value representing the request's priority. Defaults to ``None``, meaning no priority sorting will be applied.
            verify_fn (Callable, optional): Function to verify requests. It should take the same arguments as :func:`forward_fn` and return a boolean value. If ``None``, :func:`self.default_verify` function will be used.

        Note:
            The methods :func:`forward_fn`, :func:`blacklist_fn`, :func:`priority_fn`, and :func:`verify_fn` should be designed to receive the same parameters.

        Raises:
            AssertionError: If :func:`forward_fn` does not have the signature: ``forward( brainresponder: YourBrainresponder ) -> brainresponder``.
            AssertionError: If :func:`blacklist_fn` does not have the signature: ``blacklist( brainresponder: YourBrainresponder ) -> bool``.
            AssertionError: If :func:`priority_fn` does not have the signature: ``priority( brainresponder: YourBrainresponder ) -> float``.
            AssertionError: If :func:`verify_fn` does not have the signature: ``verify( brainresponder: YourBrainresponder ) -> None``.

        Returns:
            self: Returns the instance of the BrainportServer class for potential method chaining.

        Example Usage::

            def forward_custom(brainresponder: MyCustomBrainresponder) -> MyCustomBrainresponder:
                # Custom logic for processing the request
                return brainresponder

            def blacklist_custom(brainresponder: MyCustomBrainresponder) -> Tuple[bool, str]:
                return True, "Allowed!"

            def priority_custom(brainresponder: MyCustomBrainresponder) -> float:
                return 1.0

            def verify_custom(brainresponder: MyCustomBrainresponder):
                # Custom logic for verifying the request
                pass

            my_brainport = basedai.brainport(...)
            my_brainport.attach(forward_fn=forward_custom, verify_fn=verify_custom)

        Note:
            The :func:`attach` method is fundamental in setting up the Brainport server's request handling capabilities,
            enabling it to participate effectively and securely in the Basedai network. The flexibility
            offered by this method allows developers to tailor the Brainport's behavior to specific requirements and
            use cases.
        """

        # Assert 'forward_fn' has exactly one argument
        forward_sig = signature(forward_fn)
        assert (
            len(list(forward_sig.parameters)) == 1
        ), "The passed function must have exactly one argument"

        # Obtain the class of the first argument of 'forward_fn'
        request_class = forward_sig.parameters[
            list(forward_sig.parameters)[0]
        ].annotation

        # Assert that the first argument of 'forward_fn' is a subclass of 'basedai.Brainresponder'
        assert issubclass(
            request_class, basedai.Brainresponder
        ), "The argument of forward_fn must inherit from basedai.Brainresponder"

        # Obtain the class name of the first argument of 'forward_fn'
        request_name = forward_sig.parameters[
            list(forward_sig.parameters)[0]
        ].annotation.__name__

        # Add the endpoint to the router, making it available on both GET and POST methods
        self.router.add_api_route(
            f"/{request_name}",
            forward_fn,
            methods=["GET", "POST"],
            dependencies=[Depends(self.verify_body_integrity)],
        )
        self.app.include_router(self.router)

        # Expected signatures for 'blacklist_fn', 'priority_fn' and 'verify_fn'
        blacklist_sig = Signature(
            [
                Parameter(
                    "brainresponder",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=forward_sig.parameters[
                        list(forward_sig.parameters)[0]
                    ].annotation,
                )
            ],
            return_annotation=Tuple[bool, str],
        )
        priority_sig = Signature(
            [
                Parameter(
                    "brainresponder",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=forward_sig.parameters[
                        list(forward_sig.parameters)[0]
                    ].annotation,
                )
            ],
            return_annotation=float,
        )
        verify_sig = Signature(
            [
                Parameter(
                    "brainresponder",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=forward_sig.parameters[
                        list(forward_sig.parameters)[0]
                    ].annotation,
                )
            ],
            return_annotation=None,
        )

        # Check the signature of blacklist_fn, priority_fn and verify_fn if they are provided
        if blacklist_fn:
            assert (
                signature(blacklist_fn) == blacklist_sig
            ), "The blacklist_fn function must have the signature: blacklist( brainresponder: {} ) -> Tuple[bool, str]".format(
                request_name
            )
        if priority_fn:
            assert (
                signature(priority_fn) == priority_sig
            ), "The priority_fn function must have the signature: priority( brainresponder: {} ) -> float".format(
                request_name
            )
        if verify_fn:
            assert (
                signature(verify_fn) == verify_sig
            ), "The verify_fn function must have the signature: verify( brainresponder: {} ) -> None".format(
                request_name
            )

        # Store functions in appropriate attribute dictionaries
        self.forward_class_types[request_name] = forward_sig.parameters[
            list(forward_sig.parameters)[0]
        ].annotation
        self.blacklist_fns[request_name] = blacklist_fn
        self.priority_fns[request_name] = priority_fn
        self.verify_fns[request_name] = (
            verify_fn or self.default_verify
        )  # Use 'default_verify' if 'verify_fn' is None
        self.forward_fns[request_name] = forward_fn

        # Parse required hash fields from the forward function protocol defaults
        required_hash_fields = request_class.__dict__["__fields__"][
            "required_hash_fields"
        ].default
        self.required_hash_fields[request_name] = required_hash_fields

        return self

    @classmethod
    def config(cls) -> "basedai.config":
        """
        Parses the command-line arguments to form a Basedai configuration object.

        Returns:
            basedai.config: Configuration object with settings from command-line arguments.
        """
        parser = argparse.ArgumentParser()
        brainport.add_args(parser)  # Add specific brainport-related arguments
        return basedai.config(parser, args=[])

    @classmethod
    def help(cls):
        """
        Prints the help text (list of command-line arguments and their descriptions) to stdout.
        """
        parser = argparse.ArgumentParser()
        brainport.add_args(parser)  # Add specific brainport-related arguments
        print(cls.__new__.__doc__)  # Print docstring of the class
        parser.print_help()  # Print parser's help text

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None):
        """
        Adds BrainportServer-specific command-line arguments to the argument parser.

        Args:
            parser (argparse.ArgumentParser): Argument parser to which the arguments will be added.
            prefix (str, optional): Prefix to add to the argument names. Defaults to None.

        Note:
            Environment variables are used to define default values for the arguments.
        """
        prefix_str = "" if prefix is None else prefix + "."
        try:
            # Get default values from environment variables or use default values
            default_brainport_port = os.getenv("BT_BRAINPORT_PORT") or 8091
            default_brainport_ip = os.getenv("BT_BRAINPORT_IP") or "[::]"
            default_brainport_external_port = os.getenv("BT_BRAINPORT_EXTERNAL_PORT") or None
            default_brainport_external_ip = os.getenv("BT_BRAINPORT_EXTERNAL_IP") or None
            default_brainport_max_workers = os.getenv("BT_BRAINPORT_MAX_WORERS") or 10

            # Add command-line arguments to the parser
            parser.add_argument(
                "--" + prefix_str + "brainport.port",
                type=int,
                help="The local port this brainport endpoint is bound to. i.e. 8091",
                default=default_brainport_port,
            )
            parser.add_argument(
                "--" + prefix_str + "brainport.ip",
                type=str,
                help="""The local ip this brainport binds to. ie. [::]""",
                default=default_brainport_ip,
            )
            parser.add_argument(
                "--" + prefix_str + "brainport.external_port",
                type=int,
                required=False,
                help="""The public port this brainport broadcasts to the network. i.e. 8091""",
                default=default_brainport_external_port,
            )
            parser.add_argument(
                "--" + prefix_str + "brainport.external_ip",
                type=str,
                required=False,
                help="""The external ip this brainport broadcasts to the network to. ie. [::]""",
                default=default_brainport_external_ip,
            )
            parser.add_argument(
                "--" + prefix_str + "brainport.max_workers",
                type=int,
                help="""The maximum number connection handler threads working simultaneously on this endpoint.
                        The grpc server distributes new worker threads to service requests up to this number.""",
                default=default_brainport_max_workers,
            )

        except argparse.ArgumentError:
            # Exception handling for re-parsing arguments
            pass

    async def verify_body_integrity(self, request: Request):
        """
        The ``verify_body_integrity`` method in the Basedai framework is a key security function within the
        Brainport server's middleware. It is responsible for ensuring the integrity of the body of incoming HTTP
        requests.

        It asynchronously verifies the integrity of the body of a request by comparing the hash of required fields
        with the corresponding hashes provided in the request headers. This method is critical for ensuring
        that the incoming request payload has not been altered or tampered with during transmission, establishing
        a level of trust and security between the sender and receiver in the network.

        Args:
            request (Request): The incoming FastAPI request object containing both headers and the request body.

        Returns:
            dict: Returns the parsed body of the request as a dictionary if all the hash comparisons match,
                indicating that the body is intact and has not been tampered with.

        Raises:
            JSONResponse: Raises a JSONResponse with a 400 status code if any of the hash comparisons fail,
                        indicating a potential integrity issue with the incoming request payload.
                        The response includes the detailed error message specifying which field has a hash mismatch.

        This method performs several key functions:

        1. Decoding and loading the request body for inspection.
        2. Gathering required field names for hash comparison from the Brainport configuration.
        3. Loading and parsing the request body into a dictionary.
        4. Reconstructing the Brainresponder object and recomputing the hash for verification and logging.
        5. Comparing the recomputed hash with the hash provided in the request headers for verification.

        Note:
            The integrity verification is an essential step in ensuring the security of the data exchange
            within the Basedai network. It helps prevent tampering and manipulation of data during transit,
            thereby maintaining the reliability and trust in the network communication.
        """
        # Await and load the request body so we can inspect it
        body = await request.body()
        request_body = body.decode() if isinstance(body, bytes) else body

        # Gather the required field names from the brainport's required_hash_fields dict
        request_name = request.url.path.split("/")[1]
        required_hash_fields = self.required_hash_fields[request_name]

        # Load the body dict and check if all required field hashes match
        body_dict = json.loads(request_body)

        # Reconstruct the brainresponder object from the body dict and recompute the hash
        syn = self.forward_class_types[request_name](**body_dict)
        parsed_body_hash = syn.body_hash  # Rehash the body from request

        body_hash = request.headers.get("computed_body_hash", "")
        if parsed_body_hash != body_hash:
            raise ValueError(
                f"Hash mismatch between header body hash {body_hash} and parsed body hash {parsed_body_hash}"
            )

        # If body is good, return the parsed body so that it can be passed onto the route function
        return body_dict

    @classmethod
    def check_config(cls, config: "basedai.config"):
        """
        This method checks the configuration for the brainport's port and wallet.

        Args:
            config (basedai.config): The config object holding brainport settings.

        Raises:
            AssertionError: If the brainport or external ports are not in range [1024, 65535]
        """
        assert (
            config.brainport.port > 1024 and config.brainport.port < 65535
        ), "Brainport port must be in range [1024, 65535]"

        assert config.brainport.external_port is None or (
            config.brainport.external_port > 1024 and config.brainport.external_port < 65535
        ), "External port must be in range [1024, 65535]"

    def to_string(self):
        """
        Provides a human-readable representation of the BrainportInfo for this Brainport.
        """
        return self.info().to_string()

    def __str__(self) -> str:
        """
        Provides a human-readable representation of the Brainport instance.
        """
        return "Brainport({}, {}, {}, {}, {})".format(
            self.ip,
            self.port,
            self.wallet.computekey.ss58_address,
            "started" if self.started else "stopped",
            list(self.forward_fns.keys()),
        )

    def __repr__(self) -> str:
        """
        Provides a machine-readable (unambiguous) representation of the Brainport instance.
        It is made identical to __str__ in this case.
        """
        return self.__str__()

    def __del__(self):
        """
        This magic method is called when the Brainport object is about to be destroyed.
        It ensures that the Brainport server shuts down properly.
        """
        self.stop()

    def start(self) -> "basedai.brainport":
        """
        Starts the Brainport server and its underlying FastAPI server thread, transitioning the state of the
        Brainport instance to ``started``. This method initiates the server's ability to accept and process
        incoming network requests, making it an active participant in the Basedai network.

        The start method triggers the FastAPI server associated with the Brainport to begin listening for
        incoming requests. It is a crucial step in making the neuron represented by this Brainport operational
        within the Basedai network.

        Returns:
            basedai.brainport: The Brainport instance in the 'started' state.

        Example::

            my_brainport = basedai.brainport(...)
            ... # setup brainport, attach functions, etc.
            my_brainport.start()  # Starts the brainport server

        Note:
            After invoking this method, the Brainport is ready to handle requests as per its configured endpoints and custom logic.
        """
        self.fast_server.start()
        self.started = True
        return self

    def stop(self) -> "basedai.brainport":
        """
        Stops the Brainport server and its underlying GRPC server thread, transitioning the state of the Brainport
        instance to ``stopped``. This method ceases the server's ability to accept new network requests,
        effectively removing the neuron's server-side presence in the Basedai network.

        By stopping the FastAPI server, the Brainport ceases to listen for incoming requests, and any existing
        connections are gracefully terminated. This function is typically used when the neuron is being
        shut down or needs to temporarily go offline.

        Returns:
            basedai.brainport: The Brainport instance in the 'stopped' state.

        Example::

            my_brainport = basedai.brainport(...)
            my_brainport.start()
            ...
            my_brainport.stop()  # Stops the brainport server


        Note:
            It is advisable to ensure that all ongoing processes or requests are completed or properly handled before invoking this method.
        """
        self.fast_server.stop()
        self.started = False
        return self

    def serve(
        self, netuid: int, basednode: basedai.basednode = None
    ) -> "basedai.brainport":
        """
        Serves the Brainport on the specified basednode connection using the configured wallet. This method
        registers the Brainport with a specific subnet within the Basedai network, identified by the ``netuid``.
        It links the Brainport to the broader network, allowing it to participate in the decentralized exchange
        of information.

        Args:
            netuid (int): The unique identifier of the subnet to register on. This ID is essential for the Brainport to correctly position itself within the Basedai network topology.
            basednode (basedai.basednode, optional): The basednode connection to use for serving. If not provided, a new connection is established based on default configurations.

        Returns:
            basedai.brainport: The Brainport instance that is now actively serving on the specified basednode.

        Example::

            my_brainport = basedai.brainport(...)
            basednode = bt.basednode(network="local") # Local by default
            my_brainport.serve(netuid=1, basednode=basednode)  # Serves the brainport on subnet with netuid 1

        Note:
            The ``serve`` method is crucial for integrating the Brainport into the Basedai network, allowing it
            to start receiving and processing requests from other neurons.
        """
        if basednode == None:
            basednode = basedai.basednode()
        basednode.serve_brainport(netuid=netuid, brainport=self)
        return self

    async def default_verify(self, brainresponder: basedai.Brainresponder):
        """
        This method is used to verify the authenticity of a received message using a digital signature.

        It ensures that the message was not tampered with and was sent by the expected sender.

        The :func:`default_verify` method in the Basedai framework is a critical security function within the
        Brainport server. It is designed to authenticate incoming messages by verifying their digital
        signatures. This verification ensures the integrity of the message and confirms that it was
        indeed sent by the claimed sender. The method plays a pivotal role in maintaining the trustworthiness
        and reliability of the communication within the Basedai network.

        Key Features
            Security Assurance
                The default_verify method is crucial for ensuring the security of the Basedai network. By verifying digital signatures, it guards against unauthorized access
                and data manipulation.

            Preventing Replay Attacks
                The method checks for increasing nonce values, which is a vital
                step in preventing replay attacks. A replay attack involves an adversary reusing or
                delaying the transmission of a valid data transmission to deceive the receiver.

            Authenticity and Integrity Checks
                By verifying that the message's digital signature matches
                its content, the method ensures the message's authenticity (it comes from the claimed
                sender) and integrity (it hasn't been altered during transmission).

            Trust in Communication
                This method fosters trust in the network communication. Neurons
                (nodes in the Basedai network) can confidently interact, knowing that the messages they
                receive are genuine and have not been tampered with.

            Cryptographic Techniques
                The method's reliance on asymmetric encryption techniques is a
                cornerstone of modern cryptographic security, ensuring that only entities with the correct
                cryptographic keys can participate in secure communication.

        Args:
            brainresponder: basedai.Brainresponder
                basedai request brainresponder.

        Raises:
            Exception: If the ``receiver_computekey`` doesn't match with ``self.receiver_computekey``.
            Exception: If the nonce is not larger than the previous nonce for the same endpoint key.
            Exception: If the signature verification fails.

        After successful verification, the nonce for the given endpoint key is updated.

        Note:
            The verification process assumes the use of an asymmetric encryption algorithm,
            where the sender signs the message with their private key and the receiver verifies the
            signature using the sender's public key.
        """
        # Build the keypair from the brainrequester_computekey
        keypair = Keypair(ss58_address=brainresponder.brainrequester.computekey)

        # Build the signature messages.
        message = f"{brainresponder.brainrequester.nonce}.{brainresponder.brainrequester.computekey}.{self.wallet.computekey.ss58_address}.{brainresponder.brainrequester.uuid}.{brainresponder.computed_body_hash}"

        # Build the unique endpoint key.
        endpoint_key = f"{brainresponder.brainrequester.computekey}:{brainresponder.brainrequester.uuid}"

        # Check the nonce from the endpoint key.
        if endpoint_key in self.nonces.keys():
            # Ensure the nonce increases.
            if brainresponder.brainrequester.nonce <= self.nonces[endpoint_key]:
                raise Exception("Nonce is too small")

        if not keypair.verify(message, brainresponder.brainrequester.signature):
            raise Exception(
                f"Signature mismatch with {message} and {brainresponder.brainrequester.signature}"
            )

        # Success
        self.nonces[endpoint_key] = brainresponder.brainrequester.nonce


def create_error_response(brainresponder: basedai.Brainresponder):
    return JSONResponse(
        status_code=int(brainresponder.brainport.status_code),
        headers=brainresponder.to_headers(),
        content={"message": brainresponder.brainport.status_message},
    )


def log_and_handle_error(
    brainresponder: basedai.brainresponder, exception: Exception, status_code: int, start_time: int
):
    # Display the traceback for user clarity.
    basedai.logging.trace(f"Forward exception: {traceback.format_exc()}")

    # Set the status code of the brainresponder to the given status code.
    error_type = exception.__class__.__name__
    error_message = str(exception)
    detailed_error_message = f"{error_type}: {error_message}"

    # Log the detailed error message for internal use
    basedai.logging.error(detailed_error_message)

    # Set a user-friendly error message
    brainresponder.brainport.status_code = str(status_code)
    brainresponder.brainport.status_message = error_message

    # Calculate the processing time by subtracting the start time from the current time.
    brainresponder.brainport.process_time = str(time.time() - start_time)

    return brainresponder


class BrainportMiddleware(BaseHTTPMiddleware):
    """
    The `BrainportMiddleware` class is a key component in the Brainport server, responsible for processing all incoming requests.

    It handles the essential tasks of verifying requests, executing blacklist checks,
    running priority functions, and managing the logging of messages and errors. Additionally, the class
    is responsible for updating the headers of the response and executing the requested functions.

    This middleware acts as an intermediary layer in request handling, ensuring that each request is
    processed according to the defined rules and protocols of the Basedai network. It plays a pivotal
    role in maintaining the integrity and security of the network communication.

    Args:
        app (FastAPI): An instance of the FastAPI application to which this middleware is attached.
        brainport (basedai.brainport): The Brainport instance that will process the requests.

    The middleware operates by intercepting incoming requests, performing necessary preprocessing
    (like verification and priority assessment), executing the request through the Brainport's endpoints, and
    then handling any postprocessing steps such as response header updating and logging.
    """

    def __init__(self, app: "BrainportMiddleware", brainport: "basedai.brainport"):
        """
        Initialize the BrainportMiddleware class.

        Args:
        app (object): An instance of the application where the middleware processor is used.
        brainport (object): The brainport instance used to process the requests.
        """
        super().__init__(app)
        self.brainport = brainport

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Request:
        """
        Asynchronously processes incoming HTTP requests and returns the corresponding responses. This
        method acts as the central processing unit of the BrainportMiddleware, handling each step in the
        request lifecycle.

        Args:
            request (Request): The incoming HTTP request to be processed.
            call_next (RequestResponseEndpoint): A callable that processes the request and returns a response.

        Returns:
            Response: The HTTP response generated after processing the request.

        This method performs several key functions:

        1. Request Preprocessing: Sets up Brainresponder object from request headers and fills necessary information.
        2. Logging: Logs the start of request processing.
        3. Blacklist Checking: Verifies if the request is blacklisted.
        4. Request Verification: Ensures the authenticity and integrity of the request.
        5. Priority Assessment: Evaluates and assigns priority to the request.
        6. Request Execution: Calls the next function in the middleware chain to process the request.
        7. Response Postprocessing: Updates response headers and logs the end of the request processing.

        The method also handles exceptions and errors that might occur during each stage, ensuring that
        appropriate responses are returned to the client.
        """
        # Records the start time of the request processing.
        start_time = time.time()

        try:
            # Set up the brainresponder from its headers.
            brainresponder: basedai.Brainresponder = await self.preprocess(request)

            # Logs the start of the request processing
            basedai.logging.debug(
                f"brainport     | <-- | {request.headers.get('content-length', -1)} B | {brainresponder.name} | {brainresponder.brainrequester.computekey} | {brainresponder.brainrequester.ip}:{brainresponder.brainrequester.port} | 200 | Success "
            )

            # Call the blacklist function
            await self.blacklist(brainresponder)

            # Call verify and return the verified request
            await self.verify(brainresponder)

            # Call the priority function
            await self.priority(brainresponder)

            # Call the run function
            response = await self.run(brainresponder, call_next, request)

            # Call the postprocess function
            response = await self.postprocess(brainresponder, response, start_time)

        # Handle errors related to preprocess.
        except InvalidRequestNameError as e:
            if "brainresponder" not in locals():
                brainresponder: basedai.Brainresponder = basedai.Brainresponder()
            log_and_handle_error(brainresponder, e, 400, start_time)
            response = create_error_response(brainresponder)

        except BrainresponderParsingError as e:
            if "brainresponder" not in locals():
                brainresponder = basedai.Brainresponder()
            log_and_handle_error(brainresponder, e, 400, start_time)
            response = create_error_response(brainresponder)

        except UnknownBrainresponderError as e:
            if "brainresponder" not in locals():
                brainresponder = basedai.Brainresponder()
            log_and_handle_error(brainresponder, e, 404, start_time)
            response = create_error_response(brainresponder)

        # Handle errors related to verify.
        except NotVerifiedException as e:
            log_and_handle_error(brainresponder, e, 401, start_time)
            response = create_error_response(brainresponder)

        # Handle errors related to blacklist.
        except BlacklistedException as e:
            log_and_handle_error(brainresponder, e, 403, start_time)
            response = create_error_response(brainresponder)

        # Handle errors related to priority.
        except PriorityException as e:
            log_and_handle_error(brainresponder, e, 503, start_time)
            response = create_error_response(brainresponder)

        # Handle errors related to run.
        except RunException as e:
            log_and_handle_error(brainresponder, e, 500, start_time)
            response = create_error_response(brainresponder)

        # Handle errors related to postprocess.
        except PostProcessException as e:
            log_and_handle_error(brainresponder, e, 500, start_time)
            response = create_error_response(brainresponder)

        # Handle all other errors.
        except Exception as e:
            log_and_handle_error(brainresponder, InternalServerError(str(e)), 500, start_time)
            response = create_error_response(brainresponder)

        # Logs the end of request processing and returns the response
        finally:
            # Log the details of the processed brainresponder, including total size, name, computekey, IP, port,
            # status code, and status message, using the debug level of the logger.
            basedai.logging.debug(
                f"brainport     | --> | {response.headers.get('content-length', -1)} B | {brainresponder.name} | {brainresponder.brainrequester.computekey} | {brainresponder.brainrequester.ip}:{brainresponder.brainrequester.port}  | {brainresponder.brainport.status_code} | {brainresponder.brainport.status_message}"
            )

            # Return the response to the requester.
            return response

    async def preprocess(self, request: Request) -> basedai.Brainresponder:
        """
        Performs the initial processing of the incoming request. This method is responsible for
        extracting relevant information from the request and setting up the Brainresponder object, which
        represents the state and context of the request within the Brainport server.

        Args:
            request (Request): The incoming request to be preprocessed.

        Returns:
            basedai.Brainresponder: The Brainresponder object representing the preprocessed state of the request.

        The preprocessing involves:

        1. Extracting the request name from the URL path.
        2. Creating a Brainresponder instance from the request headers using the appropriate class type.
        3. Filling in the Brainport and Brainrequester information into the Brainresponder object.
        4. Signing the Brainresponder from the Brainport side using the wallet computekey.

        This method sets the foundation for the subsequent steps in the request handling process,
        ensuring that all necessary information is encapsulated within the Brainresponder object.
        """
        # Extracts the request name from the URL path.
        try:
            request_name = request.url.path.split("/")[1]
        except:
            raise InvalidRequestNameError(
                f"Improperly formatted request. Could not parser request {request.url.path}."
            )

        # Creates a brainresponder instance from the headers using the appropriate forward class type
        # based on the request name obtained from the URL path.
        request_brainresponder = self.brainport.forward_class_types.get(request_name)
        if request_brainresponder is None:
            raise UnknownBrainresponderError(
                f"Brainresponder name '{request_name}' not found. Available brainresponders {list(self.brainport.forward_class_types.keys())}"
            )

        try:
            brainresponder = request_brainresponder.from_headers(request.headers)
        except Exception as e:
            raise BrainresponderParsingError(
                f"Improperly formatted request. Could not parse headers {request.headers} into brainresponder of type {request_name}."
            )
        brainresponder.name = request_name

        # Fills the local brainport information into the brainresponder.
        brainresponder.brainport.__dict__.update(
            {
                "version": str(basedai.__version_as_int__),
                "uuid": str(self.brainport.uuid),
                "nonce": f"{time.monotonic_ns()}",
                "status_message": "Success",
                "status_code": "100",
            }
        )

        # Fills the brainrequester information into the brainresponder.
        brainresponder.brainrequester.__dict__.update(
            {"port": str(request.client.port), "ip": str(request.client.host)}
        )

        # Signs the brainresponder from the brainport side using the wallet computekey.
        message = f"{brainresponder.brainport.nonce}.{brainresponder.brainrequester.computekey}.{brainresponder.brainport.computekey}.{brainresponder.brainport.uuid}"
        brainresponder.brainport.signature = f"0x{self.brainport.wallet.computekey.sign(message).hex()}"

        # Return the setup brainresponder.
        return brainresponder

    async def verify(self, brainresponder: basedai.Brainresponder):
        """
        Verifies the authenticity and integrity of the request. This method ensures that the incoming
        request meets the predefined security and validation criteria.

        Args:
            brainresponder (basedai.Brainresponder): The Brainresponder object representing the request.

        Raises:
            Exception: If the verification process fails due to unmet criteria or security concerns.

        The verification process involves:

        1. Retrieving the specific verification function for the request's Brainresponder type.
        2. Executing the verification function and handling any exceptions that arise.

        Successful verification allows the request to proceed further in the processing pipeline, while
        failure results in an appropriate exception being raised.
        """
        # Start of the verification process. Verification is the process where we ensure that
        # the incoming request is from a trusted source or fulfills certain requirements.
        # We get a specific verification function from 'verify_fns' dictionary that corresponds
        # to our request's name. Each request name (brainresponder name) has its unique verification function.
        verify_fn = self.brainport.verify_fns.get(brainresponder.name)

        # If a verification function exists for the request's name
        if verify_fn:
            try:
                # We attempt to run the verification function using the brainresponder instance
                # created from the request. If this function runs without throwing an exception,
                # it means that the verification was successful.
                (
                    await verify_fn(brainresponder)
                    if inspect.iscoroutinefunction(verify_fn)
                    else verify_fn(brainresponder)
                )
            except Exception as e:
                # If there was an exception during the verification process, we log that
                # there was a verification exception.
                basedai.logging.trace(f"Verify exception {str(e)}")

                # We set the status code of the brainresponder to "401" which denotes an unauthorized access.
                brainresponder.brainport.status_code = "401"

                # We raise an exception to stop the process and return the error to the requester.
                # The error message includes the original exception message.
                raise NotVerifiedException(f"Not Verified with error: {str(e)}")

    async def blacklist(self, brainresponder: basedai.Brainresponder):
        """
        Checks if the request should be blacklisted. This method ensures that requests from disallowed
        sources or with malicious intent are blocked from processing. This can be extremely useful for
        preventing spam or other forms of abuse. The blacklist is a list of keys or identifiers that
        are prohibited from accessing certain resources.

        Args:
            brainresponder (basedai.Brainresponder): The Brainresponder object representing the request.

        Raises:
            Exception: If the request is found in the blacklist.

        The blacklist check involves:

        1. Retrieving the blacklist checking function for the request's Brainresponder type.
        2. Executing the check and handling the case where the request is blacklisted.

        If a request is blacklisted, it is blocked, and an exception is raised to halt further processing.
        """
        # A blacklist is a list of keys or identifiers
        # that are prohibited from accessing certain resources.
        # We retrieve the blacklist checking function from the 'blacklist_fns' dictionary
        # that corresponds to the request's name (brainresponder name).
        blacklist_fn = self.brainport.blacklist_fns.get(brainresponder.name)

        # If a blacklist checking function exists for the request's name
        if blacklist_fn:
            # We execute the blacklist checking function using the brainresponder instance as input.
            # If the function returns True, it means that the key or identifier is blacklisted.
            blacklisted, reason = (
                await blacklist_fn(brainresponder)
                if inspect.iscoroutinefunction(blacklist_fn)
                else blacklist_fn(brainresponder)
            )
            if blacklisted:
                # We log that the key or identifier is blacklisted.
                basedai.logging.trace(f"Blacklisted: {blacklisted}, {reason}")

                # We set the status code of the brainresponder to "403" which indicates a forbidden access.
                brainresponder.brainport.status_code = "403"

                # We raise an exception to halt the process and return the error message to the requester.
                raise BlacklistedException(f"Forbidden. Key is blacklisted: {reason}.")

    async def priority(self, brainresponder: basedai.Brainresponder):
        """
        Executes the priority function for the request. This method assesses and assigns a priority
        level to the request, determining its urgency and importance in the processing queue.

        Args:
            brainresponder (basedai.Brainresponder): The Brainresponder object representing the request.

        Raises:
            Exception: If the priority assessment process encounters issues, such as timeouts.

        The priority function plays a crucial role in managing the processing load and ensuring that
        critical requests are handled promptly.
        """
        # Retrieve the priority function from the 'priority_fns' dictionary that corresponds
        # to the request's name (brainresponder name).
        priority_fn = self.brainport.priority_fns.get(brainresponder.name)

        async def submit_task(
            executor: basedai.threadpool, priority: float
        ) -> Tuple[float, Any]:
            """
            Submits the given priority function to the specified executor for asynchronous execution.
            The function will run in the provided executor and return the priority value along with the result.

            Args:
                executor: The executor in which the priority function will be run.
                priority: The priority function to be executed.

            Returns:
                tuple: A tuple containing the priority value and the result of the priority function execution.
            """
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(executor, lambda: priority)
            result = await future
            return priority, result

        # If a priority function exists for the request's name
        if priority_fn:
            try:
                # Execute the priority function and get the priority value.
                priority = (
                    await priority_fn(brainresponder)
                    if inspect.iscoroutinefunction(priority_fn)
                    else priority_fn(brainresponder)
                )

                # Submit the task to the thread pool for execution with the given priority.
                # The submit_task function will handle the execution and return the result.
                _, result = await submit_task(self.brainport.thread_pool, priority)

            except TimeoutError as e:
                # If the execution of the priority function exceeds the timeout,
                # it raises an exception to handle the timeout error.
                basedai.logging.trace(f"TimeoutError: {str(e)}")

                # Set the status code of the brainresponder to "408" which indicates a timeout error.
                brainresponder.brainport.status_code = "408"

                # Raise an exception to stop the process and return an appropriate error message to the requester.
                raise PriorityException(f"Response timeout after: {brainresponder.timeout}s")

    async def run(
        self,
        brainresponder: basedai.Brainresponder,
        call_next: RequestResponseEndpoint,
        request: Request,
    ) -> Response:
        """
        Executes the requested function as part of the request processing pipeline. This method calls
        the next function in the middleware chain to process the request and generate a response.

        Args:
            brainresponder (basedai.Brainresponder): The Brainresponder object representing the request.
            call_next (RequestResponseEndpoint): The next function in the middleware chain to process requests.
            request (Request): The original HTTP request.

        Returns:
            Response: The HTTP response generated by processing the request.

        This method is a critical part of the request lifecycle, where the actual processing of the
        request takes place, leading to the generation of a response.
        """
        try:
            # The requested function is executed by calling the 'call_next' function,
            # passing the original request as an argument. This function processes the request
            # and returns the response.
            response = await call_next(request)

        except Exception as e:
            # If an exception occurs during the execution of the requested function,
            # it is caught and handled here.

            # Log the exception for debugging purposes.
            basedai.logging.trace(f"Run exception: {str(e)}")

            # Set the status code of the brainresponder to "500" which indicates an internal server error.
            brainresponder.brainport.status_code = "500"

            # Raise an exception to stop the process and return an appropriate error message to the requester.
            raise RunException(f"Internal server error with error: {str(e)}")

        # Return the starlet response
        return response

    async def postprocess(
        self, brainresponder: basedai.Brainresponder, response: Response, start_time: float
    ) -> Response:
        """
        Performs the final processing on the response before sending it back to the client. This method
        updates the response headers and logs the end of the request processing.

        Args:
            brainresponder (basedai.Brainresponder): The Brainresponder object representing the request.
            response (Response): The response generated by processing the request.
            start_time (float): The timestamp when the request processing started.

        Returns:
            Response: The final HTTP response, with updated headers, ready to be sent back to the client.

        Postprocessing is the last step in the request handling process, ensuring that the response is
        properly formatted and contains all necessary information.
        """
        # Set the status code of the brainresponder to "200" which indicates a successful response.
        brainresponder.brainport.status_code = "200"

        # Set the status message of the brainresponder to "Success".
        brainresponder.brainport.status_message = "Success"

        try:
            # Update the response headers with the headers from the brainresponder.
            updated_headers = brainresponder.to_headers()
            response.headers.update(updated_headers)
        except Exception as e:
            # If there is an exception during the response header update, we log the exception.
            raise PostProcessException(
                f"Error while parsing or updating response headers. Postprocess exception: {str(e)}."
            )

        # Calculate the processing time by subtracting the start time from the current time.
        brainresponder.brainport.process_time = str(time.time() - start_time)

        return response
