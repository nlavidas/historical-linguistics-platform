import os
from twilio.rest import Client

def check_twilio():
    required_vars = ['TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_FROM_NUMBER']
    if not all(os.getenv(k) for k in required_vars):
        return {"status": "error", "message": "Missing Twilio env vars"}
    
    try:
        client = Client(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))
        message = client.messages.create(
            to=os.getenv('TWILIO_TO_NUMBER', '+306948066777'),
            from_=os.getenv('TWILIO_FROM_NUMBER'),
            body="[TEST] Twilio configuration verified"
        )
        return {"status": "success", "sid": message.sid}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    result = check_twilio()
    print(f"Twilio check: {result['status'].upper()}")
    if result["status"] == "error":
        print(f"Details: {result['message']}")
    else:
        print(f"SMS SID: {result['sid']}")
