import sys
import requests
import base64
import json
import asyncio
import time
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders import message
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TxOpts, TokenAccountOpts
from solana.rpc.commitment import Processed

# Jupiter API endpoints
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_API = "https://quote-api.jup.ag/v6/swap"

# Replace with your private key
PRIVATE_KEY = Keypair.from_base58_string("")  # Replace with your base58 private key

# Token mint addresses (example: SOL to USDC)
INPUT_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC (for selling into)
OUTPUT_MINT = sys.argv[1]  # Token mint address to sell

# Slippage (in basis points, e.g., 50 = 0.5%)
SLIPPAGE_BPS = 10

# RPC URL
RPC_URL = "https://mainnet.helius-rpc.com/?api-key="  # Replace with your preferred RPC

print(f"OUTPUT_MINT received: {OUTPUT_MINT}")

async def get_token_balance():
    """Fetches the full balance of the given token for the user's wallet."""
    async with AsyncClient(RPC_URL) as solana_client:
        payer_pubkey = Pubkey.from_string(str(PRIVATE_KEY.pubkey()))
        mint_pubkey = Pubkey.from_string(OUTPUT_MINT)  # Convert OUTPUT_MINT to Pubkey

        # Correct usage of TokenAccountOpts
        opts = TokenAccountOpts(mint=mint_pubkey)
        response = await solana_client.get_token_accounts_by_owner(payer_pubkey, opts)

        if not response.value:
            print(f"No balance found for token {OUTPUT_MINT}")
            return 0
        
        # Fix: Correctly extract the token account pubkey
        token_account = response.value[0].pubkey  
        
        balance_response = await solana_client.get_token_account_balance(token_account)
        balance = int(balance_response.value.amount)
        print(f"🔹 Token Balance for {OUTPUT_MINT}: {balance} units")
        return balance

async def swap():
    start_time = time.time()

    # Initialize keypair
    keypair = PRIVATE_KEY
    payer_pubkey = str(keypair.pubkey())

    try:
        # Get the full balance of the token before swapping
        balance = await get_token_balance()
        if balance == 0:
            raise Exception("No balance available for the given token")

        # Step 1: Fetch swap quote from Jupiter
        print("Fetching swap quote from Jupiter...")
        quote_params = {
            "inputMint": OUTPUT_MINT,
            "outputMint": INPUT_MINT,
            "amount": balance,  # Use the full balance dynamically
            "slippageBps": SLIPPAGE_BPS,
            "userPublicKey": payer_pubkey,
        }
        quote_response = requests.get(JUPITER_QUOTE_API, params=quote_params).json()

        if "error" in quote_response:
            raise Exception(f"Failed to fetch quote: {quote_response['error']}")

        print("Quote received:", quote_response)

        # Step 2: Prepare swap transaction
        print("Preparing swap transaction...")
        swap_payload = {
            "quoteResponse": quote_response,
            "userPublicKey": payer_pubkey,
            "wrapUnwrapSOL": True,  # Wrap SOL if needed
        }
        swap_response = requests.post(JUPITER_SWAP_API, json=swap_payload).json()

        if "error" in swap_response:
            raise Exception(f"Failed to prepare swap: {swap_response['error']}")

        print("Swap transaction prepared:", swap_response)

        # Step 3: Decode and sign the transaction
        swap_transaction = swap_response["swapTransaction"]
        raw_transaction = VersionedTransaction.from_bytes(base64.b64decode(swap_transaction))
        signature = keypair.sign_message(message.to_bytes_versioned(raw_transaction.message))
        signed_txn = VersionedTransaction.populate(raw_transaction.message, [signature])

        # Step 4: Send the signed transaction
        print("Signing and sending transaction...")
        solana_client = AsyncClient(RPC_URL, commitment=Processed)
        send_time = time.time()
        result = await solana_client.send_raw_transaction(txn=bytes(signed_txn), opts=TxOpts(skip_preflight=False, preflight_commitment=Processed))
        await solana_client.close()

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Extract and print the transaction ID
        transaction_id = json.loads(result.to_json())['result']
        print("Transaction ID:", transaction_id)
        print("Transaction URL:", f"https://explorer.solana.com/tx/{transaction_id}")
        print(f"Swap completed in {elapsed_time:.2f} seconds")
        print(f"Transaction finished in {end_time - send_time:.2f} seconds")

    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Swap failed:", str(e))
        print(f"Time elapsed before failure: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(swap())
