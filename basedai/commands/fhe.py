import argparse
import basedai
import json
import asyncio
import websockets
import os
import tenseal as ts
from phe import paillier
import ollama
from typing import Any, Dict
import logging
import secrets
import numpy as np
import json
import websockets

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FHEError(Exception):
    """Base exception for FHE-related errors."""
    pass

class FHERunCommand:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        fhe_run_parser = parser.add_parser('run', help='Run FHE operations')
        fhe_run_parser.add_argument('--address', type=str, required=True, help='Address that signed the work')
        fhe_run_parser.add_argument('--balance', type=float, required=True, help='Minimum balance required')
        fhe_run_parser.add_argument('--library', type=str, choices=['tenseal', 'paillier'], required=True, help='FHE library to use')
        fhe_run_parser.add_argument('--operation', type=str, choices=['square', 'add', 'multiply', 'mean', 'variance'], required=True, help='FHE operation to perform')
        fhe_run_parser.add_argument('--peer', type=str, required=True, help='Peer address to receive encrypted data from')
        fhe_run_parser.add_argument('--use_cerberus', action='store_true', help='Use Cerberus Squeezing optimization')
        fhe_run_parser.add_argument('--squeeze_rate', type=float, default=0.1, help='Cerberus Squeeze rate')
        fhe_run_parser.add_argument('--use_secure_container', action='store_true', help='Use secure compute container for FHE tasks')

    @classmethod
    def run(cls, cli):
        try:
            address = cli.config.address
            balance = cli.config.balance
            library = cli.config.library
            operation = cli.config.operation
            peer = cli.config.peer
            use_cerberus = cli.config.use_cerberus
            squeeze_rate = cli.config.squeeze_rate
            use_secure_container = cli.config.use_secure_container

            logger.info(f"Running FHE command for address: {address}")
            logger.info(f"Minimum balance: {balance}")
            logger.info(f"Using FHE library: {library}")
            logger.info(f"Operation: {operation}")
            logger.info(f"Receiving data from peer: {peer}")
            logger.info(f"Using Cerberus Squeezing: {use_cerberus}")
            if use_cerberus:
                logger.info(f"Squeeze rate: {squeeze_rate}")
            logger.info(f"Using secure compute container: {use_secure_container}")

            # Receive encrypted data from the peer
            encrypted_data = cls.receive_encrypted_data(peer)

            if use_secure_container:
                result = cls.run_in_secure_container(encrypted_data, library, operation, use_cerberus, squeeze_rate)
            else:
                if library == 'tenseal':
                    result = cls.run_tenseal(encrypted_data, operation, use_cerberus, squeeze_rate)
                elif library == 'paillier':
                    result = cls.run_paillier(encrypted_data, operation, use_cerberus, squeeze_rate)

            logger.info(f"FHE operation result: {result}")

            # Send the result back to the peer
            cls.send_result_to_peer(peer, result)

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            raise FHEError(f"FHE operation failed: {str(e)}")

    @classmethod
    def run_in_secure_container(cls, encrypted_data, library, operation, use_cerberus, squeeze_rate):
        logger.info("Setting up secure container for FHE operation")

        # 1. Set up a secure container using Docker
        container_name = f"fhe_container_{library}_{operation}"
        docker_image = f"fhe_{library}:latest"  # Assume we have pre-built Docker images for each FHE library

        try:
            # Pull the Docker image if not present
            os.system(f"docker pull {docker_image}")

            # Run the container
            container_id = os.popen(f"docker run -d --name {container_name} {docker_image}").read().strip()

            # 2. Send the encrypted data and operation parameters to the container
            cls.send_data_to_container(container_id, encrypted_data, operation, use_cerberus, squeeze_rate)

            # 3. Run the FHE operation inside the container
            os.system(f"docker exec {container_id} python /app/run_fhe.py")

            # 4. Receive the encrypted result from the container
            result = cls.receive_result_from_container(container_id)

            # Clean up: stop and remove the container
            os.system(f"docker stop {container_id}")
            os.system(f"docker rm {container_id}")

            # 5. Return the result
            return result

        except Exception as e:
            logger.error(f"Error in secure container execution: {str(e)}")
            raise FHEError(f"Secure container execution failed: {str(e)}")

    @classmethod
    def send_data_to_container(cls, container_id, encrypted_data, operation, use_cerberus, squeeze_rate):
        # Serialize and send data to the container
        data = {
            "encrypted_data": encrypted_data,
            "operation": operation,
            "use_cerberus": use_cerberus,
            "squeeze_rate": squeeze_rate
        }
        serialized_data = json.dumps(data)
        os.system(f"docker exec {container_id} bash -c 'echo \'{serialized_data}\' > /app/input_data.json'")

    @classmethod
    def receive_result_from_container(cls, container_id):
        # Receive and deserialize result from the container
        result_json = os.popen(f"docker exec {container_id} cat /app/output_data.json").read()
        return json.loads(result_json)["result"]

    @staticmethod
    def run_tenseal(encrypted_data, operation, use_cerberus, squeeze_rate):
        import tenseal as ts
        import numpy as np

        context = ts.context_from(encrypted_data['context'])
        x = ts.ckks_vector_from(context, encrypted_data['encrypted_data'])

        if use_cerberus:
            # Apply Cerberus Squeezing
            entropy = np.ones(len(x))
            squeeze_threshold = 0.5

        if operation == 'square':
            result = x.square()
        elif operation == 'add':
            result = x + x
        elif operation == 'multiply':
            result = x * x
        elif operation == 'mean':
            result = x.sum() / len(x)
        elif operation == 'variance':
            mean = x.sum() / len(x)
            var = ((x - mean).square().sum()) / len(x)
            result = var

        if use_cerberus:
            # Apply Cerberus Squeezing
            for i in range(len(result)):
                if entropy[i] > squeeze_threshold:
                    result[i] *= x[i]
                entropy[i] *= np.exp(-squeeze_rate * abs(float(result[i])))

        return result.serialize()

    @staticmethod
    def run_paillier(encrypted_data, operation, use_cerberus, squeeze_rate):
        from phe import paillier
        import numpy as np

        public_key, private_key = paillier.generate_paillier_keypair()
        x = [public_key.encrypt(v) for v in encrypted_data['encrypted_data']]

        if use_cerberus:
            # Apply Cerberus Squeezing
            entropy = np.ones(len(x))
            squeeze_threshold = 0.5

        if operation == 'square':
            result = [xi * xi for xi in x]
        elif operation == 'add':
            result = sum(x)
        elif operation == 'multiply':
            result = x[0]
            for xi in x[1:]:
                result *= xi
        elif operation == 'mean':
            result = sum(x) / len(x)
        elif operation == 'variance':
            mean = sum(x) / len(x)
            var = sum((xi - mean)**2 for xi in x) / len(x)
            result = var

        if use_cerberus:
            # Apply Cerberus Squeezing
            for i in range(len(result)):
                if entropy[i] > squeeze_threshold:
                    result[i] *= x[i]
                entropy[i] *= np.exp(-squeeze_rate * abs(float(private_key.decrypt(result[i]))))

        return private_key.decrypt(result)

    @staticmethod
    async def receive_encrypted_data(peer: str):
        try:
            async with websockets.connect(f"ws://{peer}") as websocket:
                logger.info(f"Connected to peer: {peer}")
                await websocket.send("REQUEST_ENCRYPTED_DATA")
                response = await websocket.recv()
                data = json.loads(response)
                logger.info(f"Received encrypted data from peer: {peer}")
                return data
        except Exception as e:
            logger.error(f"Error receiving data from peer {peer}: {str(e)}")
            return None

    @staticmethod
    async def send_result_to_peer(peer: str, result):
        try:
            async with websockets.connect(f"ws://{peer}") as websocket:
                logger.info(f"Connected to peer: {peer}")
                await websocket.send(json.dumps({"result": result}))
                logger.info(f"Sent result to peer: {peer}")
                logger.debug(f"Result: {result}")
        except Exception as e:
            logger.error(f"Error sending result to peer {peer}: {str(e)}")

    @staticmethod
    def run_tenseal(encrypted_data: dict, operation: str):
        try:
            # Deserialize the context
            context = ts.context_from(encrypted_data['context'])

            # Decrypt the received data
            x = ts.ckks_vector_from(context, encrypted_data['encrypted_data'])

            if operation == 'square':
                result = x.square()
            elif operation == 'add':
                result = x + x  # Example of adding the vector to itself
            elif operation == 'multiply':
                result = x * x  # Example of multiplying the vector by itself
            elif operation == 'mean':
                result = x.sum() / len(x)
            elif operation == 'variance':
                mean = x.sum() / len(x)
                var = ((x - mean).square().sum()) / len(x)
                result = var
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            return result.serialize()
        except Exception as e:
            logger.error(f"TenSEAL operation failed: {str(e)}")
            raise FHEError(f"TenSEAL operation failed: {str(e)}")

    @staticmethod
    def run_paillier(encrypted_data: dict, operation: str):
        try:
            # In a real scenario, you would need to securely exchange public keys
            # This is a simplified example
            public_key, private_key = paillier.generate_paillier_keypair()

            # Decrypt the received data
            x = [public_key.encrypt(v) for v in encrypted_data['encrypted_data']]

            if operation == 'square':
                result = [xi * xi for xi in x]
            elif operation == 'add':
                result = sum(x)
            elif operation == 'multiply':
                result = x[0]
                for xi in x[1:]:
                    result *= xi
            elif operation == 'mean':
                result = sum(x) / len(x)
            elif operation == 'variance':
                mean = sum(x) / len(x)
                var = sum((xi - mean)**2 for xi in x) / len(x)
                result = var
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            return private_key.decrypt(result)
        except Exception as e:
            logger.error(f"Paillier operation failed: {str(e)}")
            raise FHEError(f"Paillier operation failed: {str(e)}")

    @classmethod
    def run_in_secure_container(cls, encrypted_data, library, operation, use_cerberus, squeeze_rate):
        # TODO: Implement secure container logic
        # This method should:
        # 1. Set up a secure container (e.g., using Docker or a trusted execution environment)
        # 2. Send the encrypted data and operation parameters to the container
        # 3. Run the FHE operation inside the container
        # 4. Receive the encrypted result from the container
        # 5. Return the result

        logger.info("Running FHE operation in secure container")
        # Placeholder implementation
        if library == 'tenseal':
            return cls.run_tenseal(encrypted_data, operation, use_cerberus, squeeze_rate)
        elif library == 'paillier':
            return cls.run_paillier(encrypted_data, operation, use_cerberus, squeeze_rate)

class FHEConfigCommand:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        fhe_config_parser = parser.add_parser('config', help='Configure FHE server')
        fhe_config_parser.add_argument('--discovery_server', type=str, required=True, help='Discovery server address')
        fhe_config_parser.add_argument('--port', type=int, required=True, help='Port to listen on')
        fhe_config_parser.add_argument('--ollama_model', type=str, default='llama2', help='Ollama model to use')
        fhe_config_parser.add_argument('--name', type=str, required=True, help='Name of this FHE server')

    @classmethod
    def run(cls, cli):
        try:
            discovery_server = cli.config.discovery_server
            port = cli.config.port
            ollama_model = cli.config.ollama_model
            name = cli.config.name

            config = {
                "discovery_server": discovery_server,
                "port": port,
                "ollama_model": ollama_model,
                "name": name
            }

            config_path = 'fhe_config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f)

            logger.info(f"FHE configuration saved to {config_path}")

            # Register this server with the discovery server
            cls.register_with_discovery_server(config)
        except Exception as e:
            logger.error(f"Failed to save FHE configuration: {str(e)}")
            raise FHEError(f"Failed to save FHE configuration: {str(e)}")

    @staticmethod
    def register_with_discovery_server(config):
        # TODO: Implement the actual registration process
        logger.info(f"Registering FHE server '{config['name']}' with discovery server at {config['discovery_server']}")

class FHEStartServerCommand:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        fhe_start_server_parser = parser.add_parser('start_server', help='Start FHE server')
        fhe_start_server_parser.add_argument('--config', type=str, default='fhe_config.json', help='Path to the configuration file')

    @classmethod
    def run(cls, cli):
        try:
            config_path = cli.config.config

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file {config_path} not found. Please run 'basedcli fhe config' first.")

            with open(config_path, 'r') as f:
                config = json.load(f)

            port = config['port']
            ollama_model = config.get('ollama_model', 'llama2')
            name = config['name']

            logger.info(f"Starting FHE server '{name}' on port {port}")
            asyncio.get_event_loop().run_until_complete(cls.start_server(port, ollama_model))
        except Exception as e:
            logger.error(f"Failed to start FHE server: {str(e)}")
            raise FHEError(f"Failed to start FHE server: {str(e)}")

class FHEDiscoverCommand:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        fhe_discover_parser = parser.add_parser('discover', help='Discover FHE servers')
        fhe_discover_parser.add_argument('--discovery_server', type=str, required=True, help='Discovery server address')
        fhe_discover_parser.add_argument('--num_threads', type=int, default=4, help='Number of threads for parallel discovery')

    @classmethod
    def run(cls, cli):
        try:
            discovery_server = cli.config.discovery_server
            num_threads = cli.config.num_threads
            servers = cls.discover_fhe_servers(discovery_server, num_threads)

            print("Available FHE servers:")
            for server in servers:
                print(f"Name: {server['name']}, Address: {server['address']}, Port: {server['port']}, Capacity: {server['capacity']}")
        except Exception as e:
            logger.error(f"Failed to discover FHE servers: {str(e)}")
            raise FHEError(f"Failed to discover FHE servers: {str(e)}")

    @classmethod
    def discover_fhe_servers(cls, discovery_server, num_threads):
        import threading
        import random
        from btdht import DHT

        def discover_chunk(chunk, results):
            dht = DHT()
            dht.start()
            for i in chunk:
                try:
                    # Use BitTorrent DHT to find nodes
                    nodes = dht.get_peers(f"basedai:fhe:server:{i}")
                    if nodes:
                        for node in nodes:
                            ip, port = node
                            # Verify if the node is running BasedAI
                            if cls.verify_basedai_node(ip, port):
                                server = {
                                    "name": f"FHE Server {i}",
                                    "address": ip,
                                    "port": port,
                                    "capacity": cls.get_node_capacity(ip, port)
                                }
                                results.append(server)
                except Exception as e:
                    basedai.logging.error(f"Error discovering FHE server {i}: {str(e)}")
            dht.stop()

        all_servers = []
        threads = []
        chunk_size = 10 // num_threads
        for i in range(num_threads):
            chunk = range(i * chunk_size, (i + 1) * chunk_size)
            thread = threading.Thread(target=discover_chunk, args=(chunk, all_servers))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return sorted(all_servers, key=lambda x: x['capacity'], reverse=True)

    @classmethod
    def verify_basedai_node(cls, ip, port):
        # TODO: Implement verification logic
        # This should check if the node is running BasedAI software
        return True

    @classmethod
    def get_node_capacity(cls, ip, port):
        # TODO: Implement capacity retrieval
        # This should query the node for its current capacity
        return random.randint(50, 100)

    @staticmethod
    async def start_server(port: int, ollama_model: str):
        async def handle_connection(websocket, path):
            try:
                async for message in websocket:
                    logger.info(f"Received message: {message}")

                    # Process the message using Ollama
                    response = ollama.generate(model=ollama_model, prompt=message)

                    # Encrypt the response before sending
                    encrypted_response = FHEStartServerCommand.encrypt_response(response['response'])

                    # Send the encrypted Ollama response back
                    await websocket.send(json.dumps(encrypted_response))
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {str(e)}")

        try:
            server = await websockets.serve(handle_connection, "localhost", port)
            logger.info(f"Server started on ws://localhost:{port}")
            await server.wait_closed()
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {str(e)}")
            raise FHEError(f"Failed to start WebSocket server: {str(e)}")

    @staticmethod
    def encrypt_response(response: str) -> Dict[str, Any]:
        # This is a placeholder for actual FHE encryption
        # In a real implementation, you would use one of the FHE libraries to encrypt the response
        # For demonstration, we'll use TenSEAL for actual FHE encryption
        try:
            context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
            context.global_scale = 2**40
            context.generate_galois_keys()

            # Convert the response string to a list of ASCII values
            ascii_values = [ord(char) for char in response]

            # Encrypt the ASCII values
            encrypted_vector = ts.ckks_vector(context, ascii_values)

            # Serialize the encrypted vector
            serialized_vector = encrypted_vector.serialize()

            return {
                "encrypted_data": serialized_vector.hex(),
                "context": context.serialize().hex()
            }
        except Exception as e:
            logger.error(f"FHE encryption failed: {str(e)}")
            raise FHEError(f"FHE encryption failed: {str(e)}")
