import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add parent directory to path to import the main module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_web_panel import app, panel, lightside_service, automation_jobs

class TestCorpusPanel(unittest.TestCase):
    def setUp(self):
        """Set up test client and configure app for testing."""
        app.config['TESTING'] = True
        self.client = app.test_client()
        
        # Create a test database
        self.test_db = tempfile.NamedTemporaryFile(delete=False)
        panel.db_path = self.test_db.name
        
        # Initialize test data
        self.test_data = {
            'username': 'testuser',
            'password': 'testpass123',
            'content': 'Test corpus content',
            'language': 'greek'
        }
        
        # Mock Twilio client
        self.twilio_patcher = patch('unified_web_panel.TwilioClient')
        self.mock_twilio = self.twilio_patcher.start()
        
    def tearDown(self):
        """Clean up after tests."""
        self.twilio_patcher.stop()
        os.unlink(self.test_db.name)
        
    def test_01_initial_setup(self):
        """Verify initial setup and configuration."""
        self.assertTrue(hasattr(panel, 'db_path'))
        self.assertTrue(hasattr(panel, 'is_authenticated'))
        
    def test_02_login_page(self):
        """Test login page accessibility."""
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)
        
    def test_03_authentication_flow(self):
        """Test the complete authentication flow."""
        # Test successful login
        with self.client as c:
            response = c.post('/login', data=dict(
                username='admin',
                password=panel.password
            ), follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Dashboard', response.data)
            
        # Test failed login
        response = self.client.post('/login', data=dict(
            username='admin',
            password='wrongpassword'
        ), follow_redirects=True)
        self.assertIn(b'Invalid credentials', response.data)
        
    def test_04_twilio_integration(self):
        """Test Twilio SMS functionality."""
        # Test with Twilio configured
        with patch.dict(os.environ, {
            'TWILIO_ACCOUNT_SID': 'test',
            'TWILIO_AUTH_TOKEN': 'test',
            'TWILIO_FROM_NUMBER': '+1234567890'
        }):
            response = self.client.get('/debug/status')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertTrue(data['twilio_ready'])
            
    def test_05_automation_jobs(self):
        """Test automation job submission and tracking."""
        # Test LightSide training job
        response = self.client.post('/automation/lightside', 
                                 follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Automation', response.data)
        
        # Verify job was added to the queue
        self.assertGreater(len(automation_jobs), 0)
        
    def test_06_api_endpoints(self):
        """Test all API endpoints."""
        # Test /api/stats
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        stats = json.loads(response.data)
        self.assertIn('total_items', stats)
        
        # Test /debug/status
        response = self.client.get('/debug/status')
        self.assertEqual(response.status_code, 200)
        status = json.loads(response.data)
        self.assertIn('twilio_ready', status)
        
    def test_07_error_handling(self):
        """Test error conditions and edge cases."""
        # Test invalid route
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
        
        # Test unauthorized API access
        with self.client.session_transaction() as sess:
            sess.clear()
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 401)

if __name__ == '__main__':
    unittest.main()
