#!/usr/bin/env python3
"""
SECURE MONITORING SERVICE WITH SMS ALERTS
==========================================
Enhanced monitoring with SMS/Viber notifications for security events.
Mobile alerts to: +30 6948066777
"""

import os
import sys
import signal
import time
import logging
from pathlib import Path
import subprocess
import psutil

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from secure_sms_notifier import SecureSMSNotifier
    sms_notifier = SecureSMSNotifier()
except ImportError:
    sms_notifier = None
    print("Warning: SMS notifier not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('secure_monitoring.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecureMonitoringService:
    """Monitoring service with SMS security alerts"""

    def __init__(self):
        self.running = True
        self.last_alert_time = 0
        self.alert_cooldown = 300  # 5 minutes between duplicate alerts

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received")
        self.running = False

        if sms_notifier:
            sms_notifier.alert_platform_stop()

    def check_service_health(self, service_name):
        """Check if a systemd service is healthy"""
        try:
            result = subprocess.run(
                ['systemctl', 'is-active', service_name],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Service check failed for {service_name}: {e}")
            return False

    def check_process_health(self, process_pattern):
        """Check if processes matching pattern are running"""
        try:
            result = subprocess.run(
                ['pgrep', '-f', process_pattern],
                capture_output=True, timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Process check failed for {process_pattern}: {e}")
            return False

    def get_system_resources(self):
        """Get system resource usage"""
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu_percent': cpu,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3)
            }
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return None

    def check_security_events(self):
        """Check for security-related events"""
        alerts = []

        # Check for failed login attempts
        try:
            result = subprocess.run(
                ['journalctl', '-u', 'sshd', '--since', '1 hour ago', '-g', 'Failed'],
                capture_output=True, text=True, timeout=10
            )
            failed_attempts = result.stdout.count('Failed password')
            if failed_attempts > 5:
                alerts.append(f"Multiple SSH login failures: {failed_attempts}")
        except Exception as e:
            logger.error(f"SSH security check failed: {e}")

        # Check for unusual processes
        try:
            suspicious_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if any(susp in ' '.join(proc.info['cmdline'] or []) for susp in ['nc', 'netcat', 'nmap', 'hydra']):
                        suspicious_processes.append(proc.info['name'])
                except:
                    continue

            if suspicious_processes:
                alerts.append(f"Suspicious processes detected: {', '.join(suspicious_processes)}")
        except Exception as e:
            logger.error(f"Process security check failed: {e}")

        return alerts

    def send_alert(self, message, priority="normal"):
        """Send alert with cooldown to prevent spam"""
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return  # Too soon since last alert

        logger.warning(f"ALERT: {message}")

        if sms_notifier:
            success = sms_notifier.send_notification(message, "system_error", priority)
            if success:
                self.last_alert_time = current_time
                logger.info("SMS alert sent successfully")
            else:
                logger.error("Failed to send SMS alert")

    def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        logger.info("Running monitoring cycle...")

        # Check system services
        services_to_check = ['monitoring', 'nginx', 'ollama']
        for service in services_to_check:
            if not self.check_service_health(service):
                self.send_alert(f"Service {service} is not running", "urgent")

        # Check critical processes
        processes_to_check = [
            ('collection', 'DIACHRONIC_MULTILINGUAL_COLLECTOR'),
            ('annotation', 'annotation_worker'),
            ('web_panel', 'secure_web_panel')
        ]
        for name, pattern in processes_to_check:
            if not self.check_process_health(pattern):
                self.send_alert(f"Critical process {name} is not running", "urgent")

        # Check system resources
        resources = self.get_system_resources()
        if resources:
            if resources['cpu_percent'] > 90:
                self.send_alert(f"High CPU usage: {resources['cpu_percent']:.1f}%", "normal")
            if resources['memory_percent'] > 90:
                self.send_alert(f"High memory usage: {resources['memory_percent']:.1f}%", "normal")
            if resources['disk_percent'] > 95:
                self.send_alert(f"Critical disk usage: {resources['disk_percent']:.1f}%", "urgent")

        # Check security events
        security_alerts = self.check_security_events()
        for alert in security_alerts:
            self.send_alert(f"Security: {alert}", "urgent")

        logger.info("Monitoring cycle completed")

    def run(self):
        """Main monitoring loop"""
        logger.info("Secure monitoring service starting...")

        # Send startup alert
        if sms_notifier:
            sms_notifier.alert_platform_start()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        try:
            while self.running:
                self.run_monitoring_cycle()
                time.sleep(60)  # Check every minute

        except Exception as e:
            logger.error(f"Monitoring service error: {e}")
            if sms_notifier:
                sms_notifier.alert_system_error(f"Monitoring service crashed: {str(e)}")

        logger.info("Secure monitoring service stopped")

if __name__ == "__main__":
    service = SecureMonitoringService()
    service.run()
