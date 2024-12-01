import argparse
import basedai
import json
import asyncio
import websockets
import os
import tenseal as ts
import pyfhel
from phe import paillier
import ollama
from typing import Any, Dict
import logging
import secrets
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FHEError(Exception):
    """Base exception for FHE-related errors."""
    pass

class FHERunCommand:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--address', type=str, required=True, help='Address that signed the work')
        parser.add_argument('--balance', type=float, required=True, help='Minimum balance required')
        parser.add_argument('--command', type=float, required=True, help='FHE command to run (the initial value)')
        parser.add_argument('--library', type=str, choices=['tenseal', 'pyfhel', 'paillier'], required=True, help='FHE library to use')
        parser.add_argument('--operation', type=str, choices=['square', 'add', 'multiply', 'mean', 'variance'], required=True, help='FHE operation to perform')
        parser.add_argument('--value', type=float, nargs='+', help='Additional value(s) for operations')

    @classmethod
    def run(cls, cli):
        try:
            address = cli.config.address
            balance = cli.config.balance
            command = cli.config.command
            library = cli.config.library
            operation = cli.config.operation
            value = cli.config.value

            logger.info(f"Running FHE command for address: {address}")
            logger.info(f"Minimum balance: {balance}")
            logger.info(f"Initial value: {command}")
            logger.info(f"Using FHE library: {library}")
            logger.info(f"Operation: {operation}")

            if operation in ['add', 'multiply', 'mean', 'variance'] and value is None:
                raise ValueError("Value(s) must be provided for add, multiply, mean, or variance operations")

            if library == 'tenseal':
                result = cls.run_tenseal(command, operation, value)
            elif library == 'pyfhel':
                result = cls.run_pyfhel(command, operation, value)
            elif library == 'paillier':
                result = cls.run_paillier(command, operation, value)

            logger.info(f"FHE operation result: {result}")

            # Example instructions for each operation
            cls.print_example_instructions()

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            raise FHEError(f"FHE operation failed: {str(e)}")

    @staticmethod
    def print_example_instructions():
        examples = {
            "square": "basedcli fhe run --address <your_address> --balance <min_balance> --command 5 --library tenseal --operation square",
            "add": "basedcli fhe run --address <your_address> --balance <min_balance> --command 5 --library concrete --operation add --value 3",
            "multiply": "basedcli fhe run --address <your_address> --balance <min_balance> --command 5 --library pyfhel --operation multiply --value 3",
            "mean": "basedcli fhe run --address <your_address> --balance <min_balance> --command 5 --library paillier --operation mean --value 3 4 5",
            "variance": "basedcli fhe run --address <your_address> --balance <min_balance> --command 5 --library tenseal --operation variance --value 3 4 5",
        }

        print("\nExample FHE operation commands:")
        for operation, example in examples.items():
            print(f"{operation.capitalize()}:")
            print(f"  {example}\n")

    @staticmethod
    def run_tenseal(command: str, operation: str, value: list = None):
        try:
            context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
            context.global_scale = 2**40
            context.generate_galois_keys()

            x = ts.ckks_vector(context, [float(command)])
            if operation == 'square':
                result = x.square()
            elif operation == 'add':
                y = ts.ckks_vector(context, value)
                result = x + y
            elif operation == 'multiply':
                y = ts.ckks_vector(context, value)
                result = x * y
            elif operation == 'mean':
                y = ts.ckks_vector(context, value)
                result = (x + y.sum()) / (len(value) + 1)
            elif operation == 'variance':
                y = ts.ckks_vector(context, value)
                mean = (x + y.sum()) / (len(value) + 1)
                var = ((x - mean).square() + ((y - mean).square()).sum()) / (len(value) + 1)
                result = var
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            decrypted_result = result.decrypt()
            logger.info(f"TenSEAL result: {operation}({command}, {value}) = {decrypted_result}")
        except Exception as e:
            logger.error(f"TenSEAL operation failed: {str(e)}")
            raise FHEError(f"TenSEAL operation failed: {str(e)}")

    @staticmethod
    def run_pyfhel(command: str, operation: str, value: list = None):
        try:
            HE = pyfhel.Pyfhel()
            HE.contextGen(scheme='bfv', n=2**14, t_bits=20)
            HE.keyGen()
            HE.relinKeyGen()
            HE.rotateKeyGen()

            x = HE.encryptFrac(float(command))
            if operation == 'square':
                result = x * x
            elif operation == 'add':
                y = [HE.encryptFrac(v) for v in value]
                result = x + sum(y)
            elif operation == 'multiply':
                y = [HE.encryptFrac(v) for v in value]
                result = x * HE.cumProd(y)
            elif operation == 'mean':
                y = [HE.encryptFrac(v) for v in value]
                result = (x + sum(y)) / (len(value) + 1)
            elif operation == 'variance':
                y = [HE.encryptFrac(v) for v in value]
                mean = (x + sum(y)) / (len(value) + 1)
                var = ((x - mean)**2 + sum((yi - mean)**2 for yi in y)) / (len(value) + 1)
                result = var
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            decrypted_result = HE.decryptFrac(result)
            logger.info(f"Pyfhel result: {operation}({command}, {value}) = {decrypted_result}")
        except Exception as e:
            logger.error(f"Pyfhel operation failed: {str(e)}")
            raise FHEError(f"Pyfhel operation failed: {str(e)}")

    @staticmethod
    def run_paillier(command: str, operation: str, value: list = None):
        try:
            public_key, private_key = paillier.generate_paillier_keypair()
            x = public_key.encrypt(float(command))
            if operation == 'square':
                result = x * x
            elif operation == 'add':
                y = [public_key.encrypt(v) for v in value]
                result = x + sum(y)
            elif operation == 'multiply':
                result = x
                for v in value:
                    result *= v
            elif operation == 'mean':
                y = [public_key.encrypt(v) for v in value]
                result = (x + sum(y)) / (len(value) + 1)
            elif operation == 'variance':
                y = [public_key.encrypt(v) for v in value]
                mean = (x + sum(y)) / (len(value) + 1)
                var = ((x - mean)**2 + sum((yi - mean)**2 for yi in y)) / (len(value) + 1)
                result = var
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            decrypted_result = private_key.decrypt(result)
            logger.info(f"Paillier result: {operation}({command}, {value}) = {decrypted_result}")
        except Exception as e:
            logger.error(f"Paillier operation failed: {str(e)}")
            raise FHEError(f"Paillier operation failed: {str(e)}")

class FHEConfigCommand:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--discovery_server', type=str, required=True, help='Discovery server address')
        parser.add_argument('--port', type=int, required=True, help='Port to listen on')
        parser.add_argument('--ollama_model', type=str, default='llama2', help='Ollama model to use')
        parser.add_argument('--name', type=str, required=True, help='Name of this FHE server')

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
        parser.add_argument('--config', type=str, default='fhe_config.json', help='Path to the configuration file')

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
        parser.add_argument('--discovery_server', type=str, required=True, help='Discovery server address')

    @classmethod
    def run(cls, cli):
        try:
            discovery_server = cli.config.discovery_server
            servers = cls.discover_fhe_servers(discovery_server)
            
            print("Available FHE servers:")
            for server in servers:
                print(f"Name: {server['name']}, Address: {server['address']}, Port: {server['port']}")
        except Exception as e:
            logger.error(f"Failed to discover FHE servers: {str(e)}")
            raise FHEError(f"Failed to discover FHE servers: {str(e)}")

    @staticmethod
    def discover_fhe_servers(discovery_server):
        # TODO: Implement the actual discovery process
        # This is a placeholder implementation
        return [
            {"name": "FHE Server 1", "address": "fhe1.example.com", "port": 8080},
            {"name": "FHE Server 2", "address": "fhe2.example.com", "port": 8081},
        ]

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
