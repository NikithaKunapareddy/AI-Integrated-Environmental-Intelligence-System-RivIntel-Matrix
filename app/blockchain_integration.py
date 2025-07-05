"""
Advanced Blockchain Integration and Distributed Ledger System
RivIntel Matrix - Environmental Intelligence System
Author: Nikitha Kunapareddy

This module provides blockchain integration for environmental data integrity,
smart contracts for automated environmental compliance, and distributed
ledger technology for immutable environmental records.
"""

import asyncio
import json
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
import ecdsa
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
import sqlite3
import aiosqlite
from web3 import Web3, HTTPProvider
from eth_account import Account
import ipfshttpclient
import requests
from merkletools import MerkleTools
import pandas as pd
import numpy as np


@dataclass
class EnvironmentalTransaction:
    """Represents an environmental data transaction"""
    transaction_id: str
    timestamp: datetime
    sensor_id: str
    data_type: str
    sensor_data: Dict[str, Any]
    location: Dict[str, float]  # lat, lng
    previous_hash: str
    signature: str
    validator_id: str
    compliance_status: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of transaction data"""
        data_string = json.dumps(
            {
                'transaction_id': self.transaction_id,
                'timestamp': self.timestamp.isoformat(),
                'sensor_id': self.sensor_id,
                'data_type': self.data_type,
                'sensor_data': self.sensor_data,
                'location': self.location,
                'previous_hash': self.previous_hash
            },
            sort_keys=True
        )
        return hashlib.sha256(data_string.encode()).hexdigest()


@dataclass
class EnvironmentalBlock:
    """Represents a block in the environmental blockchain"""
    block_id: str
    timestamp: datetime
    previous_block_hash: str
    merkle_root: str
    transactions: List[EnvironmentalTransaction]
    nonce: int
    difficulty: int
    miner_id: str
    block_hash: str
    
    def to_dict(self) -> Dict:
        return {
            'block_id': self.block_id,
            'timestamp': self.timestamp.isoformat(),
            'previous_block_hash': self.previous_block_hash,
            'merkle_root': self.merkle_root,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'nonce': self.nonce,
            'difficulty': self.difficulty,
            'miner_id': self.miner_id,
            'block_hash': self.block_hash
        }


class EnvironmentalBlockchain:
    """
    Blockchain system for environmental data integrity and compliance tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.chain: List[EnvironmentalBlock] = []
        self.pending_transactions: List[EnvironmentalTransaction] = []
        self.difficulty = config.get('mining_difficulty', 4)
        self.block_size = config.get('max_transactions_per_block', 100)
        self.mining_reward = config.get('mining_reward', 10.0)
        
        # Cryptographic keys
        self.private_key = None
        self.public_key = None
        self.initialize_keys()
        
        # Database for persistent storage
        self.db_path = config.get('db_path', 'environmental_blockchain.db')
        
        # Smart contract integration
        self.web3 = None
        self.contract_address = None
        self.contract_abi = None
        self.initialize_ethereum_connection()
        
        # IPFS for large data storage
        self.ipfs_client = None
        self.initialize_ipfs()
        
    def initialize_keys(self):
        """Initialize cryptographic keys for signing"""
        try:
            # Generate RSA key pair
            key = RSA.generate(2048)
            self.private_key = key
            self.public_key = key.publickey()
            
            self.logger.info("Cryptographic keys initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing keys: {e}")
            
    def initialize_ethereum_connection(self):
        """Initialize Ethereum blockchain connection"""
        try:
            ethereum_url = self.config.get('ethereum_url', 'http://localhost:8545')
            self.web3 = Web3(HTTPProvider(ethereum_url))
            
            if self.web3.isConnected():
                self.logger.info("Connected to Ethereum blockchain")
            else:
                self.logger.warning("Failed to connect to Ethereum blockchain")
                
            # Load smart contract
            self.contract_address = self.config.get('contract_address')
            self.contract_abi = self.config.get('contract_abi')
            
        except Exception as e:
            self.logger.error(f"Error initializing Ethereum connection: {e}")
            
    def initialize_ipfs(self):
        """Initialize IPFS client for distributed storage"""
        try:
            ipfs_url = self.config.get('ipfs_url', '/ip4/127.0.0.1/tcp/5001')
            self.ipfs_client = ipfshttpclient.connect(ipfs_url)
            
            self.logger.info("IPFS client initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing IPFS: {e}")
            
    async def initialize_database(self):
        """Initialize SQLite database for blockchain storage"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    block_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    previous_block_hash TEXT,
                    merkle_root TEXT,
                    nonce INTEGER,
                    difficulty INTEGER,
                    miner_id TEXT,
                    block_hash TEXT,
                    block_data TEXT
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    block_id TEXT,
                    timestamp TEXT,
                    sensor_id TEXT,
                    data_type TEXT,
                    sensor_data TEXT,
                    location TEXT,
                    previous_hash TEXT,
                    signature TEXT,
                    validator_id TEXT,
                    compliance_status TEXT,
                    FOREIGN KEY (block_id) REFERENCES blocks (block_id)
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS compliance_records (
                    record_id TEXT PRIMARY KEY,
                    transaction_id TEXT,
                    regulation_type TEXT,
                    compliance_score REAL,
                    violations TEXT,
                    remediation_actions TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
                )
            """)
            
            await db.commit()
            
    def create_genesis_block(self) -> EnvironmentalBlock:
        """Create the first block in the blockchain"""
        genesis_transaction = EnvironmentalTransaction(
            transaction_id="genesis",
            timestamp=datetime.now(),
            sensor_id="genesis_sensor",
            data_type="genesis",
            sensor_data={"message": "Genesis block for RivIntel Environmental Blockchain"},
            location={"lat": 0.0, "lng": 0.0},
            previous_hash="0",
            signature="genesis_signature",
            validator_id="genesis_validator",
            compliance_status="compliant"
        )
        
        merkle_tree = MerkleTools()
        merkle_tree.add_leaf(genesis_transaction.calculate_hash())
        merkle_tree.make_tree()
        merkle_root = merkle_tree.get_merkle_root().hex()
        
        genesis_block = EnvironmentalBlock(
            block_id="genesis",
            timestamp=datetime.now(),
            previous_block_hash="0",
            merkle_root=merkle_root,
            transactions=[genesis_transaction],
            nonce=0,
            difficulty=self.difficulty,
            miner_id="genesis_miner",
            block_hash=""
        )
        
        genesis_block.block_hash = self.calculate_block_hash(genesis_block)
        return genesis_block
        
    def add_transaction(self, transaction: EnvironmentalTransaction) -> bool:
        """Add a new environmental transaction to pending transactions"""
        try:
            # Validate transaction
            if not self.validate_transaction(transaction):
                self.logger.warning(f"Invalid transaction: {transaction.transaction_id}")
                return False
                
            # Sign transaction
            transaction.signature = self.sign_transaction(transaction)
            
            # Add to pending transactions
            self.pending_transactions.append(transaction)
            
            self.logger.info(f"Transaction added: {transaction.transaction_id}")
            
            # Auto-mine if enough transactions
            if len(self.pending_transactions) >= self.block_size:
                asyncio.create_task(self.mine_pending_transactions())
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding transaction: {e}")
            return False
            
    def validate_transaction(self, transaction: EnvironmentalTransaction) -> bool:
        """Validate environmental transaction"""
        try:
            # Check required fields
            if not all([
                transaction.transaction_id,
                transaction.sensor_id,
                transaction.data_type,
                transaction.sensor_data,
                transaction.location
            ]):
                return False
                
            # Validate sensor data format
            if not isinstance(transaction.sensor_data, dict):
                return False
                
            # Validate location
            location = transaction.location
            if not isinstance(location, dict) or 'lat' not in location or 'lng' not in location:
                return False
                
            # Check lat/lng ranges
            lat, lng = location['lat'], location['lng']
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                return False
                
            # Validate sensor data values
            if not self.validate_sensor_data(transaction.sensor_data, transaction.data_type):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating transaction: {e}")
            return False
            
    def validate_sensor_data(self, sensor_data: Dict, data_type: str) -> bool:
        """Validate sensor data based on data type"""
        validation_rules = {
            'water_quality': {
                'ph': (0, 14),
                'dissolved_oxygen': (0, 20),
                'turbidity': (0, 1000),
                'temperature': (-10, 50),
                'conductivity': (0, 5000)
            },
            'air_quality': {
                'pm2_5': (0, 500),
                'pm10': (0, 600),
                'ozone': (0, 1000),
                'co': (0, 100),
                'no2': (0, 200)
            },
            'weather': {
                'temperature': (-50, 60),
                'humidity': (0, 100),
                'pressure': (800, 1200),
                'wind_speed': (0, 200),
                'rainfall': (0, 1000)
            }
        }
        
        if data_type not in validation_rules:
            return True  # Allow unknown data types
            
        rules = validation_rules[data_type]
        
        for param, value in sensor_data.items():
            if param in rules:
                min_val, max_val = rules[param]
                try:
                    numeric_value = float(value)
                    if not (min_val <= numeric_value <= max_val):
                        self.logger.warning(f"Invalid {param} value: {value}")
                        return False
                except (ValueError, TypeError):
                    self.logger.warning(f"Non-numeric {param} value: {value}")
                    return False
                    
        return True
        
    def sign_transaction(self, transaction: EnvironmentalTransaction) -> str:
        """Sign transaction with private key"""
        try:
            transaction_hash = transaction.calculate_hash()
            hash_obj = SHA256.new(transaction_hash.encode())
            signature = pkcs1_15.new(self.private_key).sign(hash_obj)
            return signature.hex()
            
        except Exception as e:
            self.logger.error(f"Error signing transaction: {e}")
            return ""
            
    def verify_transaction_signature(self, transaction: EnvironmentalTransaction) -> bool:
        """Verify transaction signature"""
        try:
            transaction_hash = transaction.calculate_hash()
            hash_obj = SHA256.new(transaction_hash.encode())
            signature = bytes.fromhex(transaction.signature)
            
            pkcs1_15.new(self.public_key).verify(hash_obj, signature)
            return True
            
        except Exception:
            return False
            
    async def mine_pending_transactions(self, miner_id: str = "default_miner") -> Optional[EnvironmentalBlock]:
        """Mine pending transactions into a new block"""
        if not self.pending_transactions:
            return None
            
        self.logger.info(f"Mining block with {len(self.pending_transactions)} transactions")
        
        # Get transactions to mine
        transactions_to_mine = self.pending_transactions[:self.block_size]
        self.pending_transactions = self.pending_transactions[self.block_size:]
        
        # Calculate Merkle root
        merkle_tree = MerkleTools()
        for tx in transactions_to_mine:
            merkle_tree.add_leaf(tx.calculate_hash())
        merkle_tree.make_tree()
        merkle_root = merkle_tree.get_merkle_root().hex()
        
        # Create new block
        previous_block_hash = self.get_latest_block().block_hash if self.chain else "0"
        
        new_block = EnvironmentalBlock(
            block_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            previous_block_hash=previous_block_hash,
            merkle_root=merkle_root,
            transactions=transactions_to_mine,
            nonce=0,
            difficulty=self.difficulty,
            miner_id=miner_id,
            block_hash=""
        )
        
        # Proof of Work mining
        start_time = time.time()
        while not self.is_valid_proof(new_block):
            new_block.nonce += 1
            
            # Prevent infinite mining
            if time.time() - start_time > 300:  # 5 minutes timeout
                self.logger.warning("Mining timeout reached")
                break
                
        new_block.block_hash = self.calculate_block_hash(new_block)
        
        # Add block to chain
        if await self.add_block(new_block):
            mining_time = time.time() - start_time
            self.logger.info(f"Block mined successfully in {mining_time:.2f} seconds")
            
            # Store in IPFS if available
            if self.ipfs_client:
                await self.store_block_in_ipfs(new_block)
                
            return new_block
            
        return None
        
    def calculate_block_hash(self, block: EnvironmentalBlock) -> str:
        """Calculate SHA-256 hash of block"""
        block_string = json.dumps({
            'block_id': block.block_id,
            'timestamp': block.timestamp.isoformat(),
            'previous_block_hash': block.previous_block_hash,
            'merkle_root': block.merkle_root,
            'nonce': block.nonce,
            'difficulty': block.difficulty,
            'miner_id': block.miner_id
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
        
    def is_valid_proof(self, block: EnvironmentalBlock) -> bool:
        """Check if block hash meets difficulty requirement"""
        block_hash = self.calculate_block_hash(block)
        return block_hash.startswith('0' * self.difficulty)
        
    async def add_block(self, block: EnvironmentalBlock) -> bool:
        """Add validated block to the blockchain"""
        try:
            # Validate block
            if not await self.validate_block(block):
                self.logger.warning(f"Invalid block: {block.block_id}")
                return False
                
            # Add to chain
            self.chain.append(block)
            
            # Store in database
            await self.store_block_in_database(block)
            
            # Update compliance records
            await self.update_compliance_records(block)
            
            # Submit to Ethereum if configured
            if self.web3 and self.contract_address:
                await self.submit_to_ethereum(block)
                
            self.logger.info(f"Block added to chain: {block.block_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding block: {e}")
            return False
            
    async def validate_block(self, block: EnvironmentalBlock) -> bool:
        """Validate block before adding to chain"""
        try:
            # Check block hash
            calculated_hash = self.calculate_block_hash(block)
            if calculated_hash != block.block_hash:
                return False
                
            # Check proof of work
            if not self.is_valid_proof(block):
                return False
                
            # Check previous block hash
            if self.chain:
                previous_block = self.get_latest_block()
                if block.previous_block_hash != previous_block.block_hash:
                    return False
                    
            # Validate all transactions
            for transaction in block.transactions:
                if not self.validate_transaction(transaction):
                    return False
                    
                if not self.verify_transaction_signature(transaction):
                    return False
                    
            # Verify Merkle root
            merkle_tree = MerkleTools()
            for tx in block.transactions:
                merkle_tree.add_leaf(tx.calculate_hash())
            merkle_tree.make_tree()
            calculated_merkle_root = merkle_tree.get_merkle_root().hex()
            
            if calculated_merkle_root != block.merkle_root:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating block: {e}")
            return False
            
    def get_latest_block(self) -> Optional[EnvironmentalBlock]:
        """Get the latest block in the chain"""
        return self.chain[-1] if self.chain else None
        
    async def store_block_in_database(self, block: EnvironmentalBlock):
        """Store block in SQLite database"""
        async with aiosqlite.connect(self.db_path) as db:
            # Store block
            await db.execute("""
                INSERT INTO blocks 
                (block_id, timestamp, previous_block_hash, merkle_root, nonce, 
                 difficulty, miner_id, block_hash, block_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block.block_id,
                block.timestamp.isoformat(),
                block.previous_block_hash,
                block.merkle_root,
                block.nonce,
                block.difficulty,
                block.miner_id,
                block.block_hash,
                json.dumps(block.to_dict())
            ))
            
            # Store transactions
            for tx in block.transactions:
                await db.execute("""
                    INSERT INTO transactions
                    (transaction_id, block_id, timestamp, sensor_id, data_type,
                     sensor_data, location, previous_hash, signature, validator_id,
                     compliance_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tx.transaction_id,
                    block.block_id,
                    tx.timestamp.isoformat(),
                    tx.sensor_id,
                    tx.data_type,
                    json.dumps(tx.sensor_data),
                    json.dumps(tx.location),
                    tx.previous_hash,
                    tx.signature,
                    tx.validator_id,
                    tx.compliance_status
                ))
                
            await db.commit()
            
    async def store_block_in_ipfs(self, block: EnvironmentalBlock):
        """Store block data in IPFS for distributed storage"""
        try:
            if not self.ipfs_client:
                return
                
            block_json = json.dumps(block.to_dict(), indent=2)
            result = self.ipfs_client.add_json(block.to_dict())
            
            self.logger.info(f"Block stored in IPFS: {result}")
            
            # Store IPFS hash in block metadata
            # This could be stored in the database or blockchain
            
        except Exception as e:
            self.logger.error(f"Error storing block in IPFS: {e}")
            
    async def update_compliance_records(self, block: EnvironmentalBlock):
        """Update compliance records for transactions in block"""
        async with aiosqlite.connect(self.db_path) as db:
            for tx in block.transactions:
                compliance_score, violations, remediation_actions = await self.assess_compliance(tx)
                
                await db.execute("""
                    INSERT INTO compliance_records
                    (record_id, transaction_id, regulation_type, compliance_score,
                     violations, remediation_actions, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    tx.transaction_id,
                    "environmental_regulations",
                    compliance_score,
                    json.dumps(violations),
                    json.dumps(remediation_actions),
                    datetime.now().isoformat()
                ))
                
            await db.commit()
            
    async def assess_compliance(self, transaction: EnvironmentalTransaction) -> Tuple[float, List[str], List[str]]:
        """Assess environmental compliance for transaction"""
        violations = []
        remediation_actions = []
        base_score = 100.0
        
        # Define compliance thresholds
        compliance_thresholds = {
            'water_quality': {
                'ph': {'min': 6.5, 'max': 8.5, 'penalty': 20},
                'dissolved_oxygen': {'min': 5.0, 'max': 100, 'penalty': 15},
                'turbidity': {'min': 0, 'max': 4.0, 'penalty': 10}
            },
            'air_quality': {
                'pm2_5': {'min': 0, 'max': 35, 'penalty': 25},
                'pm10': {'min': 0, 'max': 150, 'penalty': 20},
                'ozone': {'min': 0, 'max': 70, 'penalty': 15}
            }
        }
        
        data_type = transaction.data_type
        sensor_data = transaction.sensor_data
        
        if data_type in compliance_thresholds:
            thresholds = compliance_thresholds[data_type]
            
            for param, value in sensor_data.items():
                if param in thresholds:
                    threshold = thresholds[param]
                    try:
                        numeric_value = float(value)
                        
                        if not (threshold['min'] <= numeric_value <= threshold['max']):
                            violations.append(f"{param} value {value} exceeds compliance threshold")
                            base_score -= threshold['penalty']
                            
                            # Suggest remediation actions
                            if param == 'ph' and numeric_value < threshold['min']:
                                remediation_actions.append("Add alkaline buffer to increase pH")
                            elif param == 'ph' and numeric_value > threshold['max']:
                                remediation_actions.append("Add acidic buffer to decrease pH")
                            elif param == 'pm2_5' and numeric_value > threshold['max']:
                                remediation_actions.append("Implement air filtration systems")
                            elif param == 'turbidity' and numeric_value > threshold['max']:
                                remediation_actions.append("Implement water filtration and sedimentation")
                                
                    except (ValueError, TypeError):
                        violations.append(f"Invalid {param} value format")
                        base_score -= 5
                        
        compliance_score = max(0.0, base_score)
        return compliance_score, violations, remediation_actions
        
    async def submit_to_ethereum(self, block: EnvironmentalBlock):
        """Submit block hash to Ethereum smart contract"""
        try:
            if not self.web3 or not self.contract_address:
                return
                
            # Create contract instance
            contract = self.web3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
            
            # Prepare transaction
            account = Account.from_key(self.config.get('ethereum_private_key'))
            
            transaction = contract.functions.submitBlockHash(
                block.block_hash,
                block.timestamp.timestamp(),
                len(block.transactions)
            ).buildTransaction({
                'from': account.address,
                'nonce': self.web3.eth.getTransactionCount(account.address),
                'gas': 100000,
                'gasPrice': self.web3.toWei('20', 'gwei')
            })
            
            # Sign and send transaction
            signed_txn = account.sign_transaction(transaction)
            tx_hash = self.web3.eth.sendRawTransaction(signed_txn.rawTransaction)
            
            self.logger.info(f"Block hash submitted to Ethereum: {tx_hash.hex()}")
            
        except Exception as e:
            self.logger.error(f"Error submitting to Ethereum: {e}")
            
    async def validate_chain(self) -> bool:
        """Validate the entire blockchain"""
        try:
            for i in range(1, len(self.chain)):
                current_block = self.chain[i]
                previous_block = self.chain[i - 1]
                
                # Check block hash
                if current_block.block_hash != self.calculate_block_hash(current_block):
                    return False
                    
                # Check previous hash link
                if current_block.previous_block_hash != previous_block.block_hash:
                    return False
                    
                # Validate block
                if not await self.validate_block(current_block):
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating chain: {e}")
            return False
            
    async def get_transaction_history(self, sensor_id: str) -> List[EnvironmentalTransaction]:
        """Get transaction history for a specific sensor"""
        history = []
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT * FROM transactions WHERE sensor_id = ? ORDER BY timestamp
            """, (sensor_id,)) as cursor:
                async for row in cursor:
                    tx = EnvironmentalTransaction(
                        transaction_id=row[0],
                        timestamp=datetime.fromisoformat(row[2]),
                        sensor_id=row[3],
                        data_type=row[4],
                        sensor_data=json.loads(row[5]),
                        location=json.loads(row[6]),
                        previous_hash=row[7],
                        signature=row[8],
                        validator_id=row[9],
                        compliance_status=row[10]
                    )
                    history.append(tx)
                    
        return history
        
    async def get_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """Generate compliance report for date range"""
        report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_transactions': 0,
            'compliant_transactions': 0,
            'violations': [],
            'average_compliance_score': 0.0,
            'compliance_by_sensor': {}
        }
        
        async with aiosqlite.connect(self.db_path) as db:
            # Get transaction count
            async with db.execute("""
                SELECT COUNT(*) FROM transactions 
                WHERE timestamp BETWEEN ? AND ?
            """, (start_date.isoformat(), end_date.isoformat())) as cursor:
                report['total_transactions'] = (await cursor.fetchone())[0]
                
            # Get compliance data
            async with db.execute("""
                SELECT cr.compliance_score, cr.violations, t.sensor_id
                FROM compliance_records cr
                JOIN transactions t ON cr.transaction_id = t.transaction_id
                WHERE cr.timestamp BETWEEN ? AND ?
            """, (start_date.isoformat(), end_date.isoformat())) as cursor:
                scores = []
                async for row in cursor:
                    score = row[0]
                    violations = json.loads(row[1])
                    sensor_id = row[2]
                    
                    scores.append(score)
                    
                    if score >= 80:  # Compliance threshold
                        report['compliant_transactions'] += 1
                        
                    if violations:
                        report['violations'].extend(violations)
                        
                    # Track by sensor
                    if sensor_id not in report['compliance_by_sensor']:
                        report['compliance_by_sensor'][sensor_id] = []
                    report['compliance_by_sensor'][sensor_id].append(score)
                    
                if scores:
                    report['average_compliance_score'] = sum(scores) / len(scores)
                    
        return report


# Smart Contract Integration
class EnvironmentalSmartContract:
    """
    Smart contract integration for automated environmental compliance
    """
    
    def __init__(self, web3_instance: Web3, contract_address: str, contract_abi: List):
        self.web3 = web3_instance
        self.contract = web3_instance.eth.contract(
            address=contract_address,
            abi=contract_abi
        )
        self.logger = logging.getLogger(__name__)
        
    async def deploy_contract(self, deployer_private_key: str) -> str:
        """Deploy environmental compliance smart contract"""
        # This would contain the actual contract deployment logic
        # For demo purposes, we'll return a mock address
        return "0x" + "a" * 40
        
    async def register_sensor(self, sensor_id: str, location: Dict, owner_address: str):
        """Register a new environmental sensor"""
        try:
            account = Account.from_key(owner_address)
            
            transaction = self.contract.functions.registerSensor(
                sensor_id,
                int(location['lat'] * 1000000),  # Convert to fixed point
                int(location['lng'] * 1000000),
                owner_address
            ).buildTransaction({
                'from': account.address,
                'nonce': self.web3.eth.getTransactionCount(account.address),
                'gas': 100000,
                'gasPrice': self.web3.toWei('20', 'gwei')
            })
            
            signed_txn = account.sign_transaction(transaction)
            tx_hash = self.web3.eth.sendRawTransaction(signed_txn.rawTransaction)
            
            self.logger.info(f"Sensor registered: {sensor_id}, tx: {tx_hash.hex()}")
            
        except Exception as e:
            self.logger.error(f"Error registering sensor: {e}")
            
    async def submit_compliance_violation(self, transaction_id: str, violation_details: Dict):
        """Submit compliance violation to smart contract"""
        # Implementation would interact with smart contract
        pass
        
    async def trigger_automated_response(self, sensor_id: str, violation_type: str):
        """Trigger automated response for compliance violation"""
        # Implementation would trigger smart contract functions
        pass


# Demo and testing functions
async def demo_environmental_blockchain():
    """Demonstrate environmental blockchain capabilities"""
    config = {
        'mining_difficulty': 3,
        'max_transactions_per_block': 5,
        'mining_reward': 10.0,
        'db_path': 'demo_environmental_blockchain.db',
        'ethereum_url': 'http://localhost:8545',
        'ipfs_url': '/ip4/127.0.0.1/tcp/5001'
    }
    
    # Initialize blockchain
    blockchain = EnvironmentalBlockchain(config)
    await blockchain.initialize_database()
    
    # Create genesis block
    genesis_block = blockchain.create_genesis_block()
    await blockchain.add_block(genesis_block)
    
    # Add some environmental transactions
    for i in range(10):
        transaction = EnvironmentalTransaction(
            transaction_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            sensor_id=f"sensor_{i % 3}",
            data_type="water_quality",
            sensor_data={
                'ph': 7.0 + np.random.normal(0, 0.5),
                'dissolved_oxygen': 8.0 + np.random.normal(0, 1.0),
                'turbidity': 2.0 + np.random.normal(0, 0.5),
                'temperature': 20.0 + np.random.normal(0, 2.0)
            },
            location={
                'lat': 40.7128 + np.random.normal(0, 0.01),
                'lng': -74.0060 + np.random.normal(0, 0.01)
            },
            previous_hash="",
            signature="",
            validator_id=f"validator_{i % 2}",
            compliance_status="pending"
        )
        
        blockchain.add_transaction(transaction)
        await asyncio.sleep(0.1)
        
    # Wait for mining
    await asyncio.sleep(5)
    
    # Validate chain
    is_valid = await blockchain.validate_chain()
    print(f"Blockchain is valid: {is_valid}")
    
    # Generate compliance report
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=1)
    report = await blockchain.get_compliance_report(start_date, end_date)
    print(f"Compliance report: {json.dumps(report, indent=2)}")
    
    return blockchain


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_environmental_blockchain())
