def scan_transaction(tx):
    print(f"🛡️ Scanning Transaction ${tx['amount']}...")
    return False

def main():
    is_fraud = scan_transaction({"amount": 49.99, "location": "US"})
    print(f"✓ Fraud Flag Status: {'ALERT 🚨' if is_fraud else 'CLEAN 🟢'}")

if __name__ == "__main__":
    main()
