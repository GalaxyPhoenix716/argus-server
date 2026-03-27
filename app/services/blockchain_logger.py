"""
Blockchain Logger Service

Logs critical security events to blockchain for immutable audit trail.
Integrates with Ethereum/Polygon networks via Web3.py.
"""

from typing import Dict, Optional, Any
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

try:
    from web3 import Web3
    from web3.exceptions import TransactionNotFound
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logging.warning("Web3.py not available. Blockchain logging disabled.")

from app.ai.threat_classifier import ThreatClassification
from app.ai.explainability_engine import ExplainabilityResult

logger = logging.getLogger(__name__)


class BlockchainNetwork(str, Enum):
    """Supported blockchain networks"""
    ETHEREUM_MAINNET = "ethereum_mainnet"
    ETHEREUM_TESTNET = "ethereum_testnet"
    POLYGON_MAINNET = "polygon_mainnet"
    POLYGON_TESTNET = "polygon_testnet"
    PRIVATE_CHAIN = "private_chain"


class LogLevel(str, Enum):
    """Blockchain log levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class BlockchainEvent:
    """Event to be logged to blockchain"""
    event_id: str
    timestamp: datetime
    channel_id: str
    event_type: str
    log_level: LogLevel
    data: Dict[str, Any]
    network: BlockchainNetwork
    tx_hash: Optional[str] = None
    block_number: Optional[int] = None
    gas_used: Optional[int] = None
    status: str = "pending"


class BlockchainLogger:
    """
    Blockchain logging service for critical security events.

    Logs high-confidence attack detections to blockchain for immutable
    audit trail and regulatory compliance.
    """

    def __init__(
        self,
        network: BlockchainNetwork = BlockchainNetwork.POLYGON_TESTNET,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        contract_address: Optional[str] = None,
        gas_limit: int = 100000
    ):
        """
        Initialize blockchain logger.

        Args:
            network: Blockchain network to use
            rpc_url: RPC endpoint URL
            private_key: Private key for signing transactions
            contract_address: Smart contract address
            gas_limit: Gas limit for transactions
        """
        self.network = network
        self.rpc_url = rpc_url
        self.private_key = private_key
        self.contract_address = contract_address
        self.gas_limit = gas_limit

        # Initialize Web3
        self.w3 = None
        self.account = None
        self.contract = None

        # Event queue for batch logging
        self.event_queue = []
        self.batch_size = 10
        self.batch_timeout = 30  # seconds

        if WEB3_AVAILABLE:
            self._initialize_web3()
        else:
            logger.warning("Blockchain logging disabled: Web3.py not available")

    def _initialize_web3(self):
        """Initialize Web3 connection"""
        if not self.rpc_url:
            # Default RPC URLs for test networks
            rpc_urls = {
                BlockchainNetwork.ETHEREUM_TESTNET: "https://sepolia.infura.io/v3/",
                BlockchainNetwork.POLYGON_TESTNET: "https://rpc.ankr.com/polygon_mumbai",
            }
            self.rpc_url = rpc_urls.get(self.network, "http://localhost:8545")

        try:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

            # Check connection
            if not self.w3.is_connected():
                raise ConnectionError(f"Failed to connect to {self.rpc_url}")

            logger.info(f"Connected to blockchain: {self.network} at {self.rpc_url}")

            # Initialize account if private key provided
            if self.private_key:
                self.account = self.w3.eth.account.from_key(self.private_key)
                logger.info(f"Loaded account: {self.account.address}")

            # Initialize contract (simplified - would load actual ABI in production)
            if self.contract_address:
                # Placeholder contract interaction
                logger.info(f"Contract address: {self.contract_address}")

        except Exception as e:
            logger.error(f"Failed to initialize Web3: {e}")
            self.w3 = None

    async def log_critical_event(
        self,
        event_type: str,
        channel_id: str,
        classification: ThreatClassification,
        explanation: Optional[ExplainabilityResult] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Log critical security event to blockchain.

        Args:
            event_type: Type of event (attack, failure, anomaly)
            channel_id: Channel identifier
            classification: Threat classification result
            explanation: Optional explanation
            metadata: Additional metadata

        Returns:
            Transaction hash if successful, None otherwise
        """
        if not self._is_ready():
            logger.warning("Blockchain logger not ready")
            return None

        # Create event data
        event_data = {
            'channel_id': channel_id,
            'threat_class': classification.threat_class,
            'confidence': classification.confidence,
            'risk_score': classification.risk_score,
            'specific_type': classification.specific_threat_type,
            'top_features': classification.top_features,
            'timestamp': datetime.now().isoformat()
        }

        if explanation:
            event_data['explanation'] = {
                'pattern_type': explanation.pattern_type,
                'confidence_explanation': explanation.confidence_explanation,
                'reason': explanation.reason[:500]  # Truncate for gas efficiency
            }

        if metadata:
            event_data.update(metadata)

        # Determine log level
        if classification.threat_class == 'attack' and classification.confidence > 0.9:
            log_level = LogLevel.CRITICAL
        elif classification.threat_class == 'failure':
            log_level = LogLevel.WARNING
        else:
            log_level = LogLevel.INFO

        # Create blockchain event
        event = BlockchainEvent(
            event_id=f"{channel_id}_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            channel_id=channel_id,
            event_type=event_type,
            log_level=log_level,
            data=event_data,
            network=self.network
        )

        # Log to blockchain
        tx_hash = await self._submit_transaction(event)

        if tx_hash:
            event.tx_hash = tx_hash
            event.status = "confirmed"
            logger.info(f"Logged event to blockchain: {tx_hash}")
        else:
            event.status = "failed"
            logger.error(f"Failed to log event to blockchain")

        return tx_hash

    async def _submit_transaction(self, event: BlockchainEvent) -> Optional[str]:
        """Submit transaction to blockchain"""
        if not self.w3 or not self.account:
            return None

        try:
            # Prepare transaction data (simplified)
            # In production, would encode event data for smart contract
            event_json = str(event.data)

            # Estimate gas
            gas_estimate = self.w3.eth.estimate_gas({
                'to': self.contract_address if self.contract_address else self.account.address,
                'data': event_json.encode('utf-8')
            })

            # Build transaction
            transaction = {
                'chainId': 80001 if 'polygon' in self.network.value else 11155111,  # Mumbai/Sepolia
                'from': self.account.address,
                'to': self.contract_address if self.contract_address else self.account.address,
                'gas': min(gas_estimate + 10000, self.gas_limit),
                'gasPrice': self.w3.eth.gas_price,
                'value': 0,
                'data': event_json.encode('utf-8'),
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
            }

            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction,
                self.private_key
            )

            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt.status == 1:
                event.block_number = receipt.blockNumber
                event.gas_used = receipt.gasUsed
                logger.info(f"Transaction confirmed: {tx_hash.hex()}")
                return tx_hash.hex()
            else:
                logger.error(f"Transaction failed: {tx_hash.hex()}")
                return None

        except Exception as e:
            logger.error(f"Transaction error: {e}", exc_info=True)
            return None

    def _is_ready(self) -> bool:
        """Check if blockchain logger is ready"""
        return (
            WEB3_AVAILABLE and
            self.w3 is not None and
            self.w3.is_connected() and
            self.account is not None
        )

    async def batch_log_events(self, events: list) -> Dict[str, str]:
        """
        Log multiple events in batch for gas efficiency.

        Args:
            events: List of blockchain events

        Returns:
            Dictionary mapping event_id to transaction hash
        """
        results = {}

        for event in events:
            tx_hash = await self._submit_transaction(event)
            results[event.event_id] = tx_hash

        return results

    def get_transaction_status(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a transaction.

        Args:
            tx_hash: Transaction hash

        Returns:
            Transaction status dict or None
        """
        if not self.w3:
            return None

        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            return {
                'tx_hash': tx_hash,
                'block_number': receipt.blockNumber,
                'status': 'confirmed' if receipt.status == 1 else 'failed',
                'gas_used': receipt.gasUsed,
                'confirmations': self.w3.eth.block_number - receipt.blockNumber
            }
        except TransactionNotFound:
            return {'tx_hash': tx_hash, 'status': 'pending'}
        except Exception as e:
            logger.error(f"Error getting transaction status: {e}")
            return None

    def verify_event_log(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        Verify event was successfully logged to blockchain.

        Args:
            tx_hash: Transaction hash

        Returns:
            Event data or None
        """
        if not self.w3:
            return None

        try:
            # Get transaction
            tx = self.w3.eth.get_transaction(tx_hash)

            # Decode event data (simplified)
            # In production, would decode using contract ABI
            event_data = tx['input'].decode('utf-8')

            return {
                'tx_hash': tx_hash,
                'from': tx['from'],
                'to': tx['to'],
                'data': event_data,
                'block_number': tx['blockNumber'],
                'gas_used': tx['gas']
            }
        except Exception as e:
            logger.error(f"Error verifying event log: {e}")
            return None

    def get_network_info(self) -> Optional[Dict[str, Any]]:
        """Get blockchain network information"""
        if not self.w3:
            return None

        try:
            return {
                'network': self.network.value,
                'chain_id': self.w3.eth.chain_id,
                'block_number': self.w3.eth.block_number,
                'gas_price': self.w3.eth.gas_price,
                'account_balance': self.w3.eth.get_balance(self.account.address) if self.account else 0
            }
        except Exception as e:
            logger.error(f"Error getting network info: {e}")
            return None

    def estimate_gas_cost(self, event: BlockchainEvent) -> Optional[int]:
        """Estimate gas cost for logging an event"""
        if not self.w3:
            return None

        try:
            event_json = str(event.data)
            gas_estimate = self.w3.eth.estimate_gas({
                'to': self.contract_address if self.contract_address else self.account.address,
                'data': event_json.encode('utf-8')
            })

            # Calculate cost in wei
            gas_price = self.w3.eth.gas_price
            cost_wei = gas_estimate * gas_price

            # Convert to ether
            cost_ether = self.w3.from_wei(cost_wei, 'ether')

            return int(cost_wei)

        except Exception as e:
            logger.error(f"Error estimating gas cost: {e}")
            return None

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of blockchain logger.

        Returns:
            Health check results
        """
        results = {
            'web3_available': WEB3_AVAILABLE,
            'web3_initialized': self.w3 is not None,
            'connected': self._is_ready() if self.w3 else False,
            'account_loaded': self.account is not None,
            'contract_configured': self.contract_address is not None
        }

        if self.w3 and self.w3.is_connected():
            try:
                results['block_number'] = self.w3.eth.block_number
                results['network_info'] = self.get_network_info()
            except Exception as e:
                results['error'] = str(e)

        results['ready'] = all([
            WEB3_AVAILABLE,
            self.w3 is not None,
            self._is_ready()
        ])

        return results