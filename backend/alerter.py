import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Africa's Talking Credentials
# Set these in your environment variables for production
AT_USERNAME = os.getenv('AT_USERNAME', 'sandbox')
AT_API_KEY = os.getenv('AT_API_KEY', 'mock_key_only')

# If MOCK_MODE is True, it will just print the SMS instead of sending a real network request.
MOCK_MODE = AT_API_KEY == 'mock_key_only'

def send_alert_sms(phone_numbers, message):
    """
    Dispatches SMS alerts via Africa's Talking API.
    """
    if MOCK_MODE:
        logger.info(f"\n[MOCK SMS] To: {phone_numbers}\n[MOCK SMS] Message: {message}\n")
        return {"status": "success", "mock": True}
        
    url = "https://api.africastalking.com/version1/messaging"
    headers = {
        "ApiKey": AT_API_KEY,
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    
    data = {
        "username": AT_USERNAME,
        "to": ",".join(phone_numbers),
        "message": message
    }
    
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        logger.info(f"SMS Alert sent successfully to {len(phone_numbers)} recipients.")
        return response.json()
    except Exception as e:
        logger.error(f"Failed to send SMS alert: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    send_alert_sms(["+254700000000"], "URGENT: Turkana County has entered High Water Scarcity Risk for next month.")
