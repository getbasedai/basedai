import argparse
import basedai
from basedai.commands import BasedCommand
import json
import asyncio
import websockets
import os

class FHERunCommand(BasedCommand):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--address', type=str, required=True, help='Address that signed the work')
        parser.add_argument('--balance', type=float, required=True, help='Minimum balance required')
        parser.add_argument('--command', type=str, required=True, help='FHE command to run')

    @classmethod
    def run(cls, cli):
        address = cli.config.address
        balance = cli.config.balance
        command = cli.config.command

        # Here you would implement the logic to verify the address signature and balance
        # For demonstration, we'll just print the values
        print(f"Running FHE command for address: {address}")
        print(f"Minimum balance: {balance}")
        print(f"Command: {command}")

        # Here you would implement the actual FHE command execution
        print("Executing FHE command...")

class FHEConfigCommand(BasedCommand):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--discovery_server', type=str, required=True, help='Discovery server address')
        parser.add_argument('--port', type=int, required=True, help='Port to listen on')

    @classmethod
    def run(cls, cli):
        discovery_server = cli.config.discovery_server
        port = cli.config.port

        config = {
            "discovery_server": discovery_server,
            "port": port
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

        print(f"Starting FHE server on port {port}")
        asyncio.get_event_loop().run_until_complete(cls.start_server(port))

    @staticmethod
    async def start_server(port):
        async def handle_connection(websocket, path):
            try:
                async for message in websocket:
                    # Here you would implement the logic to handle incoming FHE commands
                    print(f"Received message: {message}")
                    # For demonstration, we'll just echo the message back
                    await websocket.send(f"Received: {message}")
            except websockets.exceptions.ConnectionClosed:
                pass

        server = await websockets.serve(handle_connection, "localhost", port)
        print(f"Server started on ws://localhost:{port}")
        await server.wait_closed()
