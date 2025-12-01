#!/usr/bin/env python3
"""
SECURE WEB CONTROL PANEL WITH SMS/VIBER PROTECTION
=====================================================
Enhanced security with two-factor authentication and mobile notifications.
Mobile alerts to: +30 6948066777
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from flask import Flask, render_template_string, request, redirect, url_for, session, flash
from datetime import datetime, timedelta
import secrets
import time

# Import our secure SMS notifier
try:
    from secure_sms_notifier import SecureSMSNotifier
    sms_notifier = SecureSMSNotifier()
except ImportError:
    sms_notifier = None
    logging.warning("SMS notifier not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('secure_web_panel.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # Secure random secret

# Security configuration
SECURITY_CONFIG = {
    "primary_password": "historical_linguistics_2025",
    "require_2fa": True,
    "session_timeout": 3600,  # 1 hour
    "max_login_attempts": 3,
    "lockout_duration": 300  # 5 minutes
}

class SecureWebPanel:
    """Enhanced web panel with security features"""

    def __init__(self):
        self.failed_attempts = {}
        self.lockouts = {}

    def check_lockout(self, ip):
        """Check if IP is locked out"""
        if ip in self.lockouts:
            if time.time() < self.lockouts[ip]:
                return True
            else:
                del self.lockouts[ip]
        return False

    def record_failed_attempt(self, ip):
        """Record failed login attempt"""
        if ip not in self.failed_attempts:
            self.failed_attempts[ip] = 0
        self.failed_attempts[ip] += 1

        if self.failed_attempts[ip] >= SECURITY_CONFIG["max_login_attempts"]:
            self.lockouts[ip] = time.time() + SECURITY_CONFIG["lockout_duration"]
            if sms_notifier:
                sms_notifier.alert_security_event(f"Multiple failed login attempts from IP: {ip}")

    def verify_primary_password(self, password):
        """Verify primary password"""
        return password == SECURITY_CONFIG["primary_password"]

    def is_session_valid(self):
        """Check if user session is still valid"""
        if 'login_time' not in session:
            return False

        elapsed = time.time() - session['login_time']
        if elapsed > SECURITY_CONFIG["session_timeout"]:
            session.clear()
            return False

        # Extend session on activity
        session['login_time'] = time.time()
        return True

    def require_auth(self, f):
        """Decorator for requiring authentication"""
        def decorated_function(*args, **kwargs):
            if not self.is_session_valid():
                return redirect(url_for('login'))

            # Check 2FA if required
            if SECURITY_CONFIG["require_2fa"] and not session.get('2fa_verified', False):
                return redirect(url_for('two_factor'))

            return f(*args, **kwargs)
        decorated_function.__name__ = f.__name__
        return decorated_function

# Initialize secure panel
secure_panel = SecureWebPanel()

# HTML Templates
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🔒 Secure Corpus Platform Login</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f0f0f0; margin: 0; padding: 20px; }
        .container { max-width: 400px; margin: 100px auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        input[type="password"] { width: 100%; padding: 12px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button { width: 100%; padding: 12px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 10px 0; }
        button:hover { background: #0056b3; }
        .error { color: red; text-align: center; margin: 10px 0; }
        .security-notice { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 4px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 Secure Access Required</h1>
        <div class="security-notice">
            <strong>Security Notice:</strong><br>
            This system is protected with multi-factor authentication.<br>
            SMS alerts enabled for security events.
        </div>
        <form method="post">
            <input type="password" name="password" placeholder="Primary Password" required>
            <button type="submit">Login</button>
        </form>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
    </div>
</body>
</html>
"""

TWO_FACTOR_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>📱 Two-Factor Authentication</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f0f0f0; margin: 0; padding: 20px; }
        .container { max-width: 400px; margin: 100px auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        input[type="text"] { width: 100%; padding: 12px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button { width: 100%; padding: 12px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 10px 0; }
        button:hover { background: #218838; }
        .info { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 15px; border-radius: 4px; margin: 20px 0; }
        .resend { background: #ffc107; margin-top: 10px; }
        .resend:hover { background: #e0a800; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📱 Two-Factor Authentication</h1>
        <div class="info">
            <strong>Security Code Sent:</strong><br>
            A second password has been sent to your mobile device.<br>
            Code expires in 5 minutes.
        </div>
        <form method="post">
            <input type="text" name="code" placeholder="Enter 8-digit code" maxlength="8" required>
            <button type="submit">Verify Code</button>
        </form>
        <form method="post" action="{{ url_for('resend_code') }}">
            <button type="submit" class="resend">Resend Code</button>
        </form>
        {% if error %}<div style="color: red; text-align: center; margin: 10px 0;">{{ error }}</div>{% endif %}
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🛡️ Secure Corpus Platform Control</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f8f9fa; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #6c757d; margin-bottom: 40px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px; margin: 30px 0; }
        .card { background: #ffffff; border-radius: 6px; padding: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #dee2e6; }
        .card h3 { color: #495057; margin-top: 0; margin-bottom: 15px; }
        button { width: 100%; padding: 12px; border: none; border-radius: 4px; cursor: pointer; margin: 5px 0; font-size: 14px; }
        .start-btn { background: #28a745; color: white; }
        .start-btn:hover { background: #218838; }
        .stop-btn { background: #dc3545; color: white; }
        .stop-btn:hover { background: #c82333; }
        .status-btn { background: #007bff; color: white; }
        .status-btn:hover { background: #0056b3; }
        .security-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .security-card h3 { color: white; }
        .logout-btn { background: #6c757d; color: white; position: absolute; top: 20px; right: 20px; }
        .logout-btn:hover { background: #545b62; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-running { background: #28a745; }
        .status-stopped { background: #dc3545; }
        .alert { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 4px; margin: 20px 0; }
    </style>
</head>
<body>
    <form method="post" action="{{ url_for('logout') }}" style="position: absolute; top: 20px; right: 20px;">
        <button type="submit" class="logout-btn">Logout</button>
    </form>

    <div class="container">
        <h1>🛡️ Secure Corpus Platform Control</h1>
        <p class="subtitle">Authenticated Session • SMS Security Enabled • Mobile: +30 6948066777</p>

        <div class="alert">
            <strong>Security Active:</strong> Two-factor authentication enabled. All actions logged and monitored.
        </div>

        <div class="grid">
            <div class="card security-card">
                <h3>🔐 Security Status</h3>
                <p><strong>2FA:</strong> Enabled</p>
                <p><strong>SMS Alerts:</strong> Active</p>
                <p><strong>Mobile:</strong> +30 6948066777</p>
                <p><strong>Session:</strong> Secure</p>
            </div>

            <div class="card">
                <h3>🎯 Platform Control</h3>
                <form method="post" action="{{ url_for('start_platform') }}">
                    <button type="submit" class="start-btn">▶️ Start Platform</button>
                </form>
                <form method="post" action="{{ url_for('stop_platform') }}">
                    <button type="submit" class="stop-btn">⏹️ Stop Platform</button>
                </form>
                <form method="post" action="{{ url_for('platform_status') }}">
                    <button type="submit" class="status-btn">📊 Check Status</button>
                </form>
            </div>

            <div class="card">
                <h3>📱 SMS Notifications</h3>
                <form method="post" action="{{ url_for('test_sms') }}">
                    <button type="submit" class="status-btn">📤 Test SMS</button>
                </form>
                <form method="post" action="{{ url_for('send_password') }}">
                    <button type="submit" class="status-btn">🔑 Send Password</button>
                </form>
                <p><small>Sends alerts to +30 6948066777</small></p>
            </div>

            <div class="card">
                <h3>📊 System Status</h3>
                <p><span class="status-indicator status-running"></span>Web Panel: Running</p>
                <p><span class="status-indicator status-{{ 'running' if monitoring_status else 'stopped' }}"></span>Monitoring: {{ 'Running' if monitoring_status else 'Stopped' }}</p>
                <p><span class="status-indicator status-{{ 'running' if collection_status else 'stopped' }}"></span>Collection: {{ 'Running' if collection_status else 'Stopped' }}</p>
                <p><span class="status-indicator status-{{ 'running' if annotation_status else 'stopped' }}"></span>Annotation: {{ 'Running' if annotation_status else 'Stopped' }}</p>
            </div>
        </div>

        {% if message %}
        <div style="background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 15px; border-radius: 4px; margin: 20px 0;">
            {{ message }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    if secure_panel.is_session_valid() and session.get('2fa_verified', False):
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    client_ip = request.remote_addr

    # Check for lockout
    if secure_panel.check_lockout(client_ip):
        flash("Account temporarily locked due to multiple failed attempts. Try again later.")
        if sms_notifier:
            sms_notifier.alert_security_event(f"Login lockout triggered from IP: {client_ip}")
        return render_template_string(LOGIN_TEMPLATE, error="Account locked. SMS alert sent.")

    if request.method == 'POST':
        password = request.form.get('password')

        if secure_panel.verify_primary_password(password):
            session['login_time'] = time.time()
            session['ip'] = client_ip
            if sms_notifier:
                sms_notifier.alert_security_event(f"Successful login from IP: {client_ip}")
            return redirect(url_for('two_factor'))
        else:
            secure_panel.record_failed_attempt(client_ip)
            return render_template_string(LOGIN_TEMPLATE, error="Invalid password")

    return render_template_string(LOGIN_TEMPLATE)

@app.route('/two-factor', methods=['GET', 'POST'])
def two_factor():
    if not secure_panel.is_session_valid():
        return redirect(url_for('login'))

    if request.method == 'POST':
        code = request.form.get('code')

        if sms_notifier and sms_notifier.verify_second_password(code):
            session['2fa_verified'] = True
            return redirect(url_for('dashboard'))
        else:
            return render_template_string(TWO_FACTOR_TEMPLATE, error="Invalid or expired code")

    # Send second password on first visit
    if sms_notifier:
        sms_notifier.send_second_password()

    return render_template_string(TWO_FACTOR_TEMPLATE)

@app.route('/resend-code', methods=['POST'])
def resend_code():
    if not secure_panel.is_session_valid():
        return redirect(url_for('login'))

    if sms_notifier:
        sms_notifier.send_second_password()

    return redirect(url_for('two_factor'))

@app.route('/dashboard')
@secure_panel.require_auth
def dashboard():
    # Check system status
    monitoring_status = check_service_status('monitoring')
    collection_status = check_service_status('collection')
    annotation_status = check_service_status('annotation')

    message = session.pop('message', None)
    return render_template_string(DASHBOARD_TEMPLATE,
                                monitoring_status=monitoring_status,
                                collection_status=collection_status,
                                annotation_status=annotation_status,
                                message=message)

@app.route('/start-platform', methods=['POST'])
@secure_panel.require_auth
def start_platform():
    try:
        # Start monitoring service
        subprocess.run(['sudo', 'systemctl', 'start', 'monitoring'], check=True)
        # Start collection if configured
        session['message'] = "Platform started successfully"

        if sms_notifier:
            sms_notifier.alert_platform_start()

    except Exception as e:
        session['message'] = f"Error starting platform: {str(e)}"
        if sms_notifier:
            sms_notifier.alert_system_error(f"Platform start failed: {str(e)}")

    return redirect(url_for('dashboard'))

@app.route('/stop-platform', methods=['POST'])
@secure_panel.require_auth
def stop_platform():
    try:
        # Stop monitoring service
        subprocess.run(['sudo', 'systemctl', 'stop', 'monitoring'], check=True)
        session['message'] = "Platform stopped successfully"

        if sms_notifier:
            sms_notifier.alert_platform_stop()

    except Exception as e:
        session['message'] = f"Error stopping platform: {str(e)}"

    return redirect(url_for('dashboard'))

@app.route('/platform-status', methods=['POST'])
@secure_panel.require_auth
def platform_status():
    try:
        result = subprocess.run(['systemctl', 'status', 'monitoring'], capture_output=True, text=True)
        status = "Running" if "active (running)" in result.stdout else "Stopped"
        session['message'] = f"Platform status: {status}"
    except Exception as e:
        session['message'] = f"Error checking status: {str(e)}"

    return redirect(url_for('dashboard'))

@app.route('/test-sms', methods=['POST'])
@secure_panel.require_auth
def test_sms():
    if sms_notifier:
        success = sms_notifier.send_notification("Test message from secure control panel", "info", "normal")
        session['message'] = "Test SMS sent successfully" if success else "SMS sending failed"
    else:
        session['message'] = "SMS service not configured"

    return redirect(url_for('dashboard'))

@app.route('/send-password', methods=['POST'])
@secure_panel.require_auth
def send_password():
    if sms_notifier:
        password = sms_notifier.send_second_password()
        session['message'] = "New password sent to mobile" if password else "Failed to send password"
    else:
        session['message'] = "SMS service not configured"

    return redirect(url_for('dashboard'))

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

def check_service_status(service_name):
    """Check if a service is running"""
    try:
        if service_name == 'monitoring':
            result = subprocess.run(['systemctl', 'is-active', 'monitoring'], capture_output=True, text=True)
            return result.returncode == 0
        elif service_name == 'collection':
            # Check if collection process is running
            result = subprocess.run(['pgrep', '-f', 'collection'], capture_output=True)
            return result.returncode == 0
        elif service_name == 'annotation':
            # Check if annotation process is running
            result = subprocess.run(['pgrep', '-f', 'annotation'], capture_output=True)
            return result.returncode == 0
    except:
        pass
    return False

if __name__ == '__main__':
    logger.info("Starting secure web control panel with SMS protection")

    # Send startup alert
    if sms_notifier:
        sms_notifier.alert_platform_start()

    app.run(host='0.0.0.0', port=5000, debug=False, ssl_context='adhoc')
