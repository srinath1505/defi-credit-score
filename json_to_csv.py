import json
import csv

# Input and output paths
input_path = "user-wallet-transactions.json"
output_path = "user-wallet-transactions.csv"

# Load JSON data
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Flatten each record
def flatten(record):
    flat = {
        'id': record.get('_id', {}).get('$oid', ''),
        'userWallet': record.get('userWallet', ''),
        'network': record.get('network', ''),
        'protocol': record.get('protocol', ''),
        'txHash': record.get('txHash', ''),
        'logId': record.get('logId', ''),
        'timestamp': record.get('timestamp', ''),
        'blockNumber': record.get('blockNumber', ''),
        'action': record.get('action', ''),
        'type': record.get('actionData', {}).get('type', ''),
        'amount': record.get('actionData', {}).get('amount', ''),
        'assetSymbol': record.get('actionData', {}).get('assetSymbol', ''),
        'assetPriceUSD': record.get('actionData', {}).get('assetPriceUSD', ''),
        'poolId': record.get('actionData', {}).get('poolId', ''),
        'userId': record.get('actionData', {}).get('userId', ''),
        'createdAt': record.get('createdAt', {}).get('$date', ''),
        'updatedAt': record.get('updatedAt', {}).get('$date', ''),
    }
    return flat

# Flatten all records
flattened_data = [flatten(rec) for rec in data]

# Write to CSV
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
    writer.writeheader()
    writer.writerows(flattened_data)

print(f"✅ Converted JSON to CSV: {output_path}")
