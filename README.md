<div align="center">

# **BasedAI** 
[![Research](https://img.shields.io/badge/arXiv-2403.01008v1-red.svg)](https://arxiv.org/abs/2403.01008)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
---
[Documentation](https://docs.getbased.ai/) • [Token](https://app.uniswap.org/explore/tokens/ethereum/0x44971abf0251958492fee97da3e5c5ada88b9185) • [Twitter](https://twitter.com/getbasedai)

</div>

BasedAI is a decentralized network built to enable end-to-end private computations on distributed GPU infrastructure. The flagship 'Brain' of BasedAI is allows for Fully Homomorphic Encryption (FHE) of Large Language Models (LLMs), effectively creating Zero-Knowledge LLMs (ZK-LLMs).

For detailed understanding of the BasedAI network and its groundbreaking approach with ZK-LLMs, please refer to our [research paper](https://arxiv.org/abs/2403.01008).

# Install

1. To get started with BasedAI:

```bash
# pip install basedai
```

# Getting Started 

In the BasedAI ecosystem, users participate through an integrated wallet system that supports operations with BasedAI's native token, $BASED. The wallet enables users to stake tokens on different Brains in the ecosystem and earn rewards for contributions.

## Wallet Management

1. Create a new wallet through the BasedAI CLI:
```bash
# create a wallet  
basedcli wallet new_personalkey
```

2. Interact with your wallet via the BasedAI CLI:
```bash
# check the balances of your wallet 
basedcli wallet balance
```

3. Join as a Brain miner or validator with a compute key:
```bash
# create a new compute wallet for agents operating on your behalf in the network
basedcli wallet new_computekey
```

For further instructions on token management and participation in the network, please refer to the [official documentation](https://docs.getbased.ai/).

## Configuring and Deploying FHE Operations for ZK-LLM

BasedAI introduces Zero-Knowledge Large Language Models (ZK-LLMs) using Fully Homomorphic Encryption (FHE). Here's how to configure and deploy FHE operations:

1. Configure the FHE server:
```bash
basedcli fhe config --discovery_server <discovery_server_address> --port <port_number> --ollama_model <model_name> --name <server_name>
```
This command sets up the FHE server configuration, including the discovery server address, port to listen on, Ollama model to use, and a name for your FHE server.

2. Start the FHE server:
```bash
basedcli fhe start_server --config fhe_config.json
```
This command starts the FHE server using the configuration file created in the previous step.

3. Discover available FHE servers:
```bash
basedcli fhe discover --discovery_server <discovery_server_address>
```
Use this command to find available FHE servers in the network.

4. Run FHE operations:
```bash
basedcli fhe run --address <your_address> --balance <min_balance> --command <initial_value> --library <fhe_library> --operation <operation_type> --value <additional_values>
```
This command allows you to perform FHE operations such as square, add, multiply, mean, and variance on encrypted data.

Example FHE operations:
- Square operation:
  ```bash
  basedcli fhe run --address <your_address> --balance <min_balance> --command 5 --library tenseal --operation square
  ```
- Addition operation:
  ```bash
  basedcli fhe run --address <your_address> --balance <min_balance> --command 5 --library concrete --operation add --value 3
  ```
- Mean calculation:
  ```bash
  basedcli fhe run --address <your_address> --balance <min_balance> --command 5 --library paillier --operation mean --value 3 4 5
  ```

These FHE operations enable secure, privacy-preserving computations on encrypted data, forming the foundation of ZK-LLMs in the BasedAI network.

For more detailed information on FHE operations and ZK-LLMs, please refer to our [research paper](https://arxiv.org/abs/2403.01008) and the [official documentation](https://docs.getbased.ai/).

## Using the CLI

The BasedAI Command Line Interface (CLI) will be the primary tool for interacting with Brains, managing wallets, participating in computation tasks as miners or validators, and engaging in the token economy of BasedAI.

For a list of possible commands:
```bash
basedcli help
```

## The BasedAI PIP Package

The BasedAI package includes essential tools included `basedai` (the Python operations package for developers) and `basedcli` (for setting up mining, validation activities, and querying ZK-LLMs). Learn more in the [official documentation](https://docs.getbased.ai). 

## Complex Payment Types and Service Broadcasting

BasedAI now supports more complex payment types and enhanced service broadcasting. Here's how to use these new features:

### Accepting Complex Payments

To accept a payment with complex payment types:

```python
wallet = basedai.wallet()
payment_info = {
    'amount': 100.0,
    'payer_address': '0x1234...',
    'payment_type': 'subscription',
    'additional_data': {
        'duration': '1 month',
        'start_date': '2024-09-18'
    }
}
success = wallet.accept_payment(payment_info)
```

### Broadcasting Service Details

To broadcast detailed service information:

```python
broadcaster = basedai.Broadcaster(wallet)
service_info = {
    'name': 'Advanced Data Processing',
    'cost': 50.0,
    'payment_types': ['instant', 'subscription', 'escrow'],
    'description': 'High-performance data processing service',
    'availability': '24/7',
    'performance_metrics': {
        'avg_response_time': 0.5,
        'success_rate': 0.99
    }
}
success = broadcaster.broadcast_service_details(service_info)
```

### Updating Service Status

To update the status of a service:

```python
status = {
    'available': True,
    'capacity': 100,
    'response_time': 0.3,
    'queue_length': 5
}
success = broadcaster.update_service_status('Advanced Data Processing', status)
```

### Broadcasting Node Metrics

To broadcast various metrics about your node:

```python
metrics = {
    'capacity': 500,
    'uptime': 720.5,
    'success_rate': 0.995,
    'average_response_time': 0.4,
    'supported_services': ['Data Processing', 'Machine Learning']
}
success = broadcaster.broadcast_node_metrics(metrics)
```

## Governance and Voting

BasedAI embraces decentralized decision-making through Pepecoin-linked "Brains". Owners of GigaBrains, representing significant staked contributions, can participate in voting on key network decisions, upholding the ecosystem's democratic governance model.

For voting procedures:
```bash
# vote for a proposal
basedcli brains vote <proposal-id> <option>
```

## For More Information 

For further information and updates on BasedAI, please visit the [official website](https://getbased.ai) or consult the initial [research paper](https://getbased.ai/whitepaper). 

## License
The MIT License (MIT)
Copyright © 2024 Based Labs 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

