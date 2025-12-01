# 🔒 SECURE CORPUS PLATFORM WITH SMS PROTECTION
## Enhanced Security Features

### 🛡️ Security Overview
The Corpus Platform now includes enterprise-grade security with SMS/Viber mobile notifications and two-factor authentication. All critical events are monitored and alerted to the designated mobile number: **+30 6948066777**.

### 📱 SMS/Viber Notification System

#### Features:
- **Real-time Alerts**: Platform start/stop, security events, system errors
- **Two-Factor Authentication**: Second password sent via SMS for login
- **Mobile Control**: Remote monitoring and control via SMS commands
- **Emergency Alerts**: Critical system failures and security breaches

#### Supported Services:
- **Twilio SMS**: Primary SMS delivery service
- **Viber Business Messages**: Alternative messaging platform
- **Configurable**: Enable SMS, Viber, or both

### 🔐 Authentication & Access Control

#### Two-Factor Authentication (2FA):
1. **Primary Password**: `historical_linguistics_2025`
2. **Second Password**: 8-digit code sent to mobile (+30 6948066777)
3. **Session Management**: Automatic logout after 1 hour of inactivity
4. **IP Tracking**: Login attempts logged and monitored

#### Security Features:
- **Failed Login Protection**: Account lockout after 3 failed attempts
- **Session Security**: Secure session handling with encryption
- **Activity Monitoring**: All actions logged with timestamps
- **Mobile Verification**: SMS confirmation for critical operations

### 🚨 Alert System

#### Alert Types:
- **Platform Events**: Start/stop notifications
- **Security Alerts**: Failed logins, suspicious activity
- **System Errors**: Service failures, resource issues
- **Emergency Alerts**: Critical system problems

#### Alert Priorities:
- **Normal**: Routine notifications
- **Urgent**: Important system events
- **Security**: Authentication and access events

### 🛠️ Setup Instructions

#### 1. Configure SMS Service
```bash
# Run the setup script
chmod +x setup_secure_sms.sh
./setup_secure_sms.sh
```

#### 2. Required API Keys
- **Twilio**: Account SID, Auth Token, Phone Number
- **Viber**: Business API Token (optional)

#### 3. Start Secure Services
```bash
# Start secure web panel
python3 secure_web_panel.py &

# Start secure monitoring
sudo systemctl start monitoring
sudo systemctl enable monitoring
```

### 📊 Access Points

#### Secure Web Control Panel
- **URL**: `https://corpus-platform.nlavid.as` (with SSL)
- **Authentication**: Primary password + SMS 2FA
- **Features**: Full platform control with security monitoring

#### Mobile SMS Commands
Send SMS to control the platform:
- `STATUS` - Get system status
- `START` - Start platform services
- `STOP` - Stop platform services
- `PASSWORD` - Request new second password

### 🔍 Security Monitoring

#### Real-time Monitoring:
- **Service Health**: All critical services monitored
- **Resource Usage**: CPU, memory, disk alerts
- **Security Events**: Login attempts, suspicious processes
- **Network Security**: SSH and connection monitoring

#### Automated Responses:
- **Self-healing**: Automatic service restarts
- **Alert Escalation**: SMS alerts for critical issues
- **Security Lockdown**: Automatic responses to threats

### 📋 Daily Security Report

The system generates daily security summaries sent via SMS:
- Login attempts and successes
- System health status
- Security events
- Platform performance metrics

### 🚨 Emergency Procedures

#### Security Breach Response:
1. **Immediate Alert**: SMS sent to +30 6948066777
2. **System Lockdown**: Automatic service isolation
3. **Evidence Collection**: Logs secured for analysis
4. **Manual Verification**: Admin confirmation required

#### System Recovery:
1. **SMS Verification**: Confirm admin identity
2. **Secure Access**: 2FA required for recovery operations
3. **Service Restoration**: Step-by-step system recovery
4. **Security Audit**: Post-incident analysis

### 📞 Contact & Support

#### Emergency Contact:
- **Mobile**: +30 6948066777
- **SMS Alerts**: All security events
- **24/7 Monitoring**: Automated system watch

#### Security Team:
- **Primary Admin**: Nikolaos Lavidas
- **Emergency Response**: SMS-verified commands
- **System Access**: 2FA protected

### ✅ Security Compliance

#### Standards Met:
- **Two-Factor Authentication**: Industry standard 2FA
- **Encrypted Communications**: SSL/TLS protected
- **Access Logging**: Comprehensive audit trails
- **Mobile Verification**: SMS-based identity confirmation

#### Best Practices:
- **Principle of Least Privilege**: Minimal required permissions
- **Defense in Depth**: Multiple security layers
- **Continuous Monitoring**: Real-time security assessment
- **Incident Response**: Automated alert and response system

---

**🔒 SECURE STATUS: ACTIVE**
**📱 SMS PROTECTION: ENABLED**
**📞 EMERGENCY CONTACT: +30 6948066777**
