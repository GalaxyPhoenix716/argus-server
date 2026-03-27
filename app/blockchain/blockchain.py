#final
from web3 import Web3

w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
abi = [
			{
				"inputs": [
					{
						"internalType": "string",
						"name": "eventType",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "dataHash",
						"type": "string"
					}
				],
				"name": "addLog",
				"outputs": [],
				"stateMutability": "nonpayable",
				"type": "function"
			},
			{
				"inputs": [
					{
						"internalType": "uint256",
						"name": "index",
						"type": "uint256"
					}
				],
				"name": "getLog",
				"outputs": [
					{
						"internalType": "string",
						"name": "",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "",
						"type": "string"
					},
					{
						"internalType": "uint256",
						"name": "",
						"type": "uint256"
					}
				],
				"stateMutability": "view",
				"type": "function"
			},
			{
				"inputs": [],
				"name": "getLogCount",
				"outputs": [
					{
						"internalType": "uint256",
						"name": "",
						"type": "uint256"
					}
				],
				"stateMutability": "view",
				"type": "function"
			},
			{
				"inputs": [
					{
						"internalType": "uint256",
						"name": "",
						"type": "uint256"
					}
				],
				"name": "logs",
				"outputs": [
					{
						"internalType": "string",
						"name": "eventType",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "dataHash",
						"type": "string"
					},
					{
						"internalType": "uint256",
						"name": "timestamp",
						"type": "uint256"
					}
				],
				"stateMutability": "view",
				"type": "function"
			}
		]
address = w3.to_checksum_address("0x5fbdb2315678afecb367f032d93f642f64180aa3")
contract = w3.eth.contract(address=address,abi=abi)


def add_log(event_type: str, encrypted_data: str):
    tx_hash = contract.functions.addLog(
        event_type,
        encrypted_data
    ).transact({"from": w3.eth.accounts[0]})

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt


def get_log(index: int):
    return contract.functions.getLog(index).call()


def no_of_logs():
    return contract.functions.getLogCount().call()
    
