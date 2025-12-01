#!/usr/bin/env python3
"""
SECURE SMS/VIBER NOTIFICATION SERVICE
========================================
Sends SMS/Viber alerts for platform status, security events, and second password delivery.
Mobile: +30 6948066777
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path
from datetime import datetime
import time
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sms_security.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecureSMSNotifier:
    """Secure SMS/Viber notification service with authentication"""

    def __init__(self):
        self.mobile_number = "+306948066777"
        self.config_file = Path(__file__).parent / "sms_config.json"
        self.load_config()

    def load_config(self):
        """Load SMS service configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration - will need API keys
            self.config = {
                "twilio_sid": os.getenv("TWILIO_SID", ""),
                "twilio_token": os.getenv("TWILIO_TOKEN", ""),
                "twilio_from": os.getenv("TWILIO_FROM", ""),
                "viber_token": os.getenv("VIBER_TOKEN", ""),
                "enabled_services": ["sms"],  # sms, viber, or both
                "alert_types": {
                    "platform_start": True,
                    "platform_stop": True,
                    "security_alert": True,
                    "system_error": True,
                    "daily_report": False
                }
            }
            self.save_config()

    def save_config(self):
        """Save configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def generate_second_password(self):
        """Generate a secure second password"""
        timestamp = str(int(time.time()))
        secret = "corpus_platform_2025_secure"
        combined = f"{timestamp}_{secret}_{self.mobile_number}"
        password = hashlib.sha256(combined.encode()).hexdigest()[:8].upper()
        return password, timestamp

    def send_sms(self, message, priority="normal"):
        """Send SMS via Twilio"""
        if not all([self.config.get("twilio_sid"), self.config.get("twilio_token"), self.config.get("twilio_from")]):
            logger.warning("Twilio not configured - SMS sending disabled")
            return False

        try:
            from twilio.rest import Client
            client = Client(self.config["twilio_sid"], self.config["twilio_token"])

            # Add priority indicator
            if priority == "urgent":
                message = f"🚨 URGENT: {message}"
            elif priority == "security":
                message = f"🔒 SECURITY: {message}"

            message = client.messages.create(
                body=message[:160],  # SMS limit
                from_=self.config["twilio_from"],
                to=self.mobile_number
            )

            logger.info(f"SMS sent to {self.mobile_number}: {message.sid}")
            return True

        except Exception as e:
            logger.error(f"SMS sending failed: {e}")
            return False

    def send_viber(self, message, priority="normal"):
        """Send Viber message"""
        if not self.config.get("viber_token"):
            logger.warning("Viber not configured")
            return False

        try:
            # Viber Business Messages API
            url = "https://chatapi.viber.com/pa/send_message"
            headers = {
                "X-Viber-Auth-Token": self.config["viber_token"],
                "Content-Type": "application/json"
            }

            data = {
                "receiver": self.mobile_number,
                "type": "text",
                "text": message[:1000]  # Viber limit
            }

            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                logger.info(f"Viber message sent to {self.mobile_number}")
                return True
            else:
                logger.error(f"Viber send failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Viber sending failed: {e}")
            return False

    def send_notification(self, message, alert_type="info", priority="normal"):
        """Send notification via configured services"""
        if not self.config["alert_types"].get(alert_type, True):
            return  # Alert type disabled

        sent_count = 0

        # Send via SMS if enabled
        if "sms" in self.config["enabled_services"]:
            if self.send_sms(message, priority):
                sent_count += 1

        # Send via Viber if enabled
        if "viber" in self.config["enabled_services"]:
            if self.send_viber(message, priority):
                sent_count += 1

        return sent_count > 0

    def send_second_password(self):
        """Generate and send second password"""
        password, timestamp = self.generate_second_password()
        message = f"Your Corpus Platform second password: {password} (expires in 5 minutes)"

        if self.send_notification(message, "security", "urgent"):
            # Store the password temporarily for verification
            self.store_temp_password(password, timestamp)
            return password
        return None

    def store_temp_password(self, password, timestamp):
        """Store temporary password for verification"""
        temp_file = Path(__file__).parent / ".temp_password"
        with open(temp_file, 'w') as f:
            json.dump({
                "password": password,
                "timestamp": timestamp,
                "expires": int(timestamp) + 300  # 5 minutes
            }, f)

    def verify_second_password(self, input_password):
        """Verify second password"""
        temp_file = Path(__file__).parent / ".temp_password"
        if not temp_file.exists():
            return False

        try:
            with open(temp_file, 'r') as f:
                data = json.load(f)

            # Check expiration
            if time.time() > data["expires"]:
                temp_file.unlink()  # Remove expired file
                return False

            # Check password
            if input_password.upper() == data["password"]:
                temp_file.unlink()  # Remove used password
                return True

        except Exception as e:
            logger.error(f"Password verification error: {e}")

        return False

    def alert_platform_start(self):
        """Alert when platform starts"""
        message = "Corpus Platform: System STARTED and SECURED"
        return self.send_notification(message, "platform_start", "normal")

    def alert_platform_stop(self):
        """Alert when platform stops"""
        message = "Corpus Platform: System STOPPED"
        return self.send_notification(message, "platform_stop", "normal")

    def alert_security_event(self, event):
        """Alert for security events"""
        message = f"Corpus Platform SECURITY: {event}"
        return self.send_notification(message, "security_alert", "urgent")

    def alert_system_error(self, error):
        """Alert for system errors"""
        message = f"Corpus Platform ERROR: {error[:100]}"
        return self.send_notification(message, "system_error", "urgent")

if __name__ == "__main__":
    # CLI interface
    import argparse

    parser = argparse.ArgumentParser(description="Secure SMS/Viber Notifier")
    parser.add_argument("action", choices=["start", "stop", "error", "password", "test"])
    parser.add_argument("--message", help="Custom message for error alerts")

    args = parser.parse_args()

    notifier = SecureSMSNotifier()

    if args.action == "start":
        success = notifier.alert_platform_start()
        print(f"Start alert sent: {success}")

    elif args.action == "stop":
        success = notifier.alert_platform_stop()
        print(f"Stop alert sent: {success}")

    elif args.action == "error":
        message = args.message or "Unknown system error"
        success = notifier.alert_system_error(message)
        print(f"Error alert sent: {success}")

    elif args.action == "password":
        password = notifier.send_second_password()
        if password:
            print(f"Second password sent: {password}")
        else:
            print("Failed to send second password")

    elif args.action == "test":
        success = notifier.send_notification("Test message from Corpus Platform", "info", "normal")
        print(f"Test message sent: {success}")
