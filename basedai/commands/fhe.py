import argparse
import basedai
from basedai.commands import BasedCommand
import json
import asyncio
import websockets
import os
import tenseal as ts
import concrete.numpy as cnp
import pyfhel
from phe import paillier
import ollama

class FHERunCommand(BasedCommand):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--address', type=str, required=True, help='Address that signed the work')
        parser.add_argument('--balance', type=float, required=True, help='Minimum balance required')
        parser.add_argument('--command', type=str, required=True, help='FHE command to run')
        parser.add_argument('--library', type=str, choices=['tenseal', 'concrete', 'pyfhel', 'paillier'], required=True, help='FHE library to use')

    @classmethod
    def run(cls, cli):
        address = cli.config.address
        balance = cli.config.balance
        command = cli.config.command
        library = cli.config.library

        print(f"Running FHE command for address: {address}")
        print(f"Minimum balance: {balance}")
        print(f"Command: {command}")
        print(f"Using FHE library: {library}")

        if library == 'tenseal':
            cls.run_tenseal(command)
        elif library == 'concrete':
            cls.run_concrete(command)
        elif library == 'pyfhel':
            cls.run_pyfhel(command)
        elif library == 'paillier':
            cls.run_paillier(command)

    @staticmethod
    def run_tenseal(command):
        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        context.global_scale = 2**40
        context.generate_galois_keys()

        # Example: Encrypt a number, square it, and decrypt
        x = ts.ckks_vector(context, [float(command)])
        result = x.square()
        decrypted_result = result.decrypt()[0]
        print(f"TenSEAL result: {command}^2 = {decrypted_result}")

    @staticmethod
    def run_concrete(command):
        compiler = cnp.Compiler({"p_error": 1/2**10})
        
        @compiler.compile(inputset=[(3.14,)])
        def square(x):
            return x ** 2

        circuit = square.encrypt_run_decrypt(float(command))
        print(f"Concrete result: {command}^2 = {circuit}")

    @staticmethod
    def run_pyfhel(command):
        HE = pyfhel.Pyfhel()
        HE.contextGen(scheme='bfv', n=2**14, t_bits=20)
        HE.keyGen()

        x = HE.encryptInt(int(command))
        result = x * x
        decrypted_result = HE.decryptInt(result)
        print(f"Pyfhel result: {command}^2 = {decrypted_result}")

    @staticmethod
    def run_paillier(command):
        public_key, private_key = paillier.generate_paillier_keypair()
        x = public_key.encrypt(float(command))
        result = x * x
        decrypted_result = private_key.decrypt(result)
        print(f"Paillier result: {command}^2 = {decrypted_result}")

class FHEConfigCommand(BasedCommand):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--discovery_server', type=str, required=True, help='Discovery server address')
        parser.add_argument('--port', type=int, required=True, help='Port to listen on')
        parser.add_argument('--ollama_model', type=str, default='llama2', help='Ollama model to use')

    @classmethod
    def run(cls, cli):
        discovery_server = cli.config.discovery_server
        port = cli.config.port
        ollama_model = cli.config.ollama_model

        config = {
            "discovery_server": discovery_server,
            "port": port,
            "ollama_model": ollama_model
        }

        with open('fhe_config.json', 'w') as f:
            json.dump(config, f)

        print(f"FHE configuration saved to fhe_config.json")

class FHEStartServerCommand(BasedCommand):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--config', type=str, default='fhe_config.json', help='Path to the configuration file')

    @classmethod
    def run(cls, cli):
        config_path = cli.config.config

        if not os.path.exists(config_path):
            print(f"Configuration file {config_path} not found. Please run 'basedcli fhe config' first.")
            return

        with open(config_path, 'r') as f:
            config = json.load(f)

        port = config['port']
        ollama_model = config.get('ollama_model', 'llama2')

        print(f"Starting FHE server on port {port}")
        asyncio.get_event_loop().run_until_complete(cls.start_server(port, ollama_model))

    @staticmethod
    async def start_server(port, ollama_model):
        async def handle_connection(websocket, path):
            try:
                async for message in websocket:
                    print(f"Received message: {message}")
                    
                    # Process the message using Ollama
                    response = ollama.generate(model=ollama_model, prompt=message)
                    
                    # Send the Ollama response back
                    await websocket.send(f"Ollama response: {response['response']}")
            except websockets.exceptions.ConnectionClosed:
                pass

        server = await websockets.serve(handle_connection, "localhost", port)
        print(f"Server started on ws://localhost:{port}")
        await server.wait_closed()
