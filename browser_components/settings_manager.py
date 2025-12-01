"""
Settings Manager Component
Professional interface for platform configuration and settings
"""

import streamlit as st
import json
import os
from typing import Dict, Any
from datetime import datetime

class SettingsManager:
    """Platform settings and configuration manager"""
    
    def __init__(self):
        self.config_file = "/root/corpus_platform/config.json"
        self.default_config = {
            'general': {
                'platform_name': 'Diachronic Linguistics Research Platform',
                'default_language': 'grc',
                'annotation_standard': 'UD',
                'max_workers': 4,
                'log_level': 'INFO',
                'auto_backup': True,
                'backup_interval_hours': 24
            },
            'parser': {
                'batch_size': 32,
                'timeout_seconds': 60,
                'cache_enabled': True,
                'cache_size_mb': 500,
                'default_models': ['stanza', 'spacy', 'cltk']
            },
            'database': {
                'connection_pool_size': 20,
                'query_timeout_seconds': 30,
                'auto_vacuum': True,
                'backup_on_shutdown': True
            },
            'api': {
                'rate_limit_per_minute': 100,
                'max_request_size_mb': 10,
                'cors_enabled': True,
                'authentication_required': False
            },
            'monitoring': {
                'metrics_retention_days': 30,
                'alert_email_enabled': False,
                'alert_email': '',
                'performance_threshold_ms': 5000
            },
            'ui': {
                'theme': 'light',
                'items_per_page': 50,
                'show_advanced_options': False,
                'enable_experimental_features': False
            }
        }
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            return self.default_config.copy()
        except:
            return self.default_config.copy()
    
    def save_config(self, config: Dict) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except:
            return False
    
    def render_general_settings(self, config: Dict) -> Dict:
        """Render general settings panel"""
        st.subheader("General Settings")
        
        general = config.get('general', self.default_config['general'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            general['platform_name'] = st.text_input(
                "Platform Name",
                value=general.get('platform_name', 'Diachronic Linguistics Research Platform')
            )
            
            general['default_language'] = st.selectbox(
                "Default Language",
                options=['grc', 'la', 'sa', 'got', 'cu', 'cop'],
                index=['grc', 'la', 'sa', 'got', 'cu', 'cop'].index(general.get('default_language', 'grc')),
                format_func=lambda x: {
                    'grc': 'Ancient Greek',
                    'la': 'Latin',
                    'sa': 'Sanskrit',
                    'got': 'Gothic',
                    'cu': 'Old Church Slavonic',
                    'cop': 'Coptic'
                }.get(x, x)
            )
            
            general['annotation_standard'] = st.selectbox(
                "Annotation Standard",
                options=['UD', 'PROIEL', 'AGDT', 'Custom'],
                index=['UD', 'PROIEL', 'AGDT', 'Custom'].index(general.get('annotation_standard', 'UD'))
            )
        
        with col2:
            general['max_workers'] = st.number_input(
                "Max Worker Processes",
                min_value=1,
                max_value=32,
                value=general.get('max_workers', 4)
            )
            
            general['log_level'] = st.selectbox(
                "Log Level",
                options=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                index=['DEBUG', 'INFO', 'WARNING', 'ERROR'].index(general.get('log_level', 'INFO'))
            )
            
            general['auto_backup'] = st.checkbox(
                "Enable Auto Backup",
                value=general.get('auto_backup', True)
            )
            
            if general['auto_backup']:
                general['backup_interval_hours'] = st.number_input(
                    "Backup Interval (hours)",
                    min_value=1,
                    max_value=168,
                    value=general.get('backup_interval_hours', 24)
                )
        
        return general
    
    def render_parser_settings(self, config: Dict) -> Dict:
        """Render parser settings panel"""
        st.subheader("Parser Settings")
        
        parser = config.get('parser', self.default_config['parser'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            parser['batch_size'] = st.slider(
                "Batch Size",
                min_value=1,
                max_value=128,
                value=parser.get('batch_size', 32)
            )
            
            parser['timeout_seconds'] = st.number_input(
                "Parse Timeout (seconds)",
                min_value=10,
                max_value=600,
                value=parser.get('timeout_seconds', 60)
            )
            
            parser['cache_enabled'] = st.checkbox(
                "Enable Parse Cache",
                value=parser.get('cache_enabled', True)
            )
        
        with col2:
            if parser['cache_enabled']:
                parser['cache_size_mb'] = st.number_input(
                    "Cache Size (MB)",
                    min_value=100,
                    max_value=5000,
                    value=parser.get('cache_size_mb', 500)
                )
            
            parser['default_models'] = st.multiselect(
                "Default Models",
                options=['stanza', 'spacy', 'cltk', 'udpipe', 'trankit'],
                default=parser.get('default_models', ['stanza', 'spacy', 'cltk'])
            )
        
        return parser
    
    def render_database_settings(self, config: Dict) -> Dict:
        """Render database settings panel"""
        st.subheader("Database Settings")
        
        database = config.get('database', self.default_config['database'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            database['connection_pool_size'] = st.slider(
                "Connection Pool Size",
                min_value=5,
                max_value=100,
                value=database.get('connection_pool_size', 20)
            )
            
            database['query_timeout_seconds'] = st.number_input(
                "Query Timeout (seconds)",
                min_value=5,
                max_value=300,
                value=database.get('query_timeout_seconds', 30)
            )
        
        with col2:
            database['auto_vacuum'] = st.checkbox(
                "Enable Auto Vacuum",
                value=database.get('auto_vacuum', True)
            )
            
            database['backup_on_shutdown'] = st.checkbox(
                "Backup on Shutdown",
                value=database.get('backup_on_shutdown', True)
            )
        
        return database
    
    def render_api_settings(self, config: Dict) -> Dict:
        """Render API settings panel"""
        st.subheader("API Settings")
        
        api = config.get('api', self.default_config['api'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            api['rate_limit_per_minute'] = st.number_input(
                "Rate Limit (requests/minute)",
                min_value=10,
                max_value=1000,
                value=api.get('rate_limit_per_minute', 100)
            )
            
            api['max_request_size_mb'] = st.number_input(
                "Max Request Size (MB)",
                min_value=1,
                max_value=100,
                value=api.get('max_request_size_mb', 10)
            )
        
        with col2:
            api['cors_enabled'] = st.checkbox(
                "Enable CORS",
                value=api.get('cors_enabled', True)
            )
            
            api['authentication_required'] = st.checkbox(
                "Require Authentication",
                value=api.get('authentication_required', False)
            )
        
        return api
    
    def render_monitoring_settings(self, config: Dict) -> Dict:
        """Render monitoring settings panel"""
        st.subheader("Monitoring Settings")
        
        monitoring = config.get('monitoring', self.default_config['monitoring'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            monitoring['metrics_retention_days'] = st.number_input(
                "Metrics Retention (days)",
                min_value=7,
                max_value=365,
                value=monitoring.get('metrics_retention_days', 30)
            )
            
            monitoring['performance_threshold_ms'] = st.number_input(
                "Performance Alert Threshold (ms)",
                min_value=1000,
                max_value=30000,
                value=monitoring.get('performance_threshold_ms', 5000)
            )
        
        with col2:
            monitoring['alert_email_enabled'] = st.checkbox(
                "Enable Email Alerts",
                value=monitoring.get('alert_email_enabled', False)
            )
            
            if monitoring['alert_email_enabled']:
                monitoring['alert_email'] = st.text_input(
                    "Alert Email Address",
                    value=monitoring.get('alert_email', '')
                )
        
        return monitoring
    
    def render_ui_settings(self, config: Dict) -> Dict:
        """Render UI settings panel"""
        st.subheader("Interface Settings")
        
        ui = config.get('ui', self.default_config['ui'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            ui['theme'] = st.selectbox(
                "Theme",
                options=['light', 'dark', 'auto'],
                index=['light', 'dark', 'auto'].index(ui.get('theme', 'light'))
            )
            
            ui['items_per_page'] = st.number_input(
                "Items Per Page",
                min_value=10,
                max_value=200,
                value=ui.get('items_per_page', 50)
            )
        
        with col2:
            ui['show_advanced_options'] = st.checkbox(
                "Show Advanced Options",
                value=ui.get('show_advanced_options', False)
            )
            
            ui['enable_experimental_features'] = st.checkbox(
                "Enable Experimental Features",
                value=ui.get('enable_experimental_features', False)
            )
        
        return ui
    
    def render(self):
        """Main render method for settings manager"""
        st.header("Settings")
        
        # Load current config
        config = self.load_config()
        
        # Create tabs for different settings sections
        tabs = st.tabs([
            "General",
            "Parser",
            "Database",
            "API",
            "Monitoring",
            "Interface",
            "Import/Export"
        ])
        
        # General settings
        with tabs[0]:
            config['general'] = self.render_general_settings(config)
        
        # Parser settings
        with tabs[1]:
            config['parser'] = self.render_parser_settings(config)
        
        # Database settings
        with tabs[2]:
            config['database'] = self.render_database_settings(config)
        
        # API settings
        with tabs[3]:
            config['api'] = self.render_api_settings(config)
        
        # Monitoring settings
        with tabs[4]:
            config['monitoring'] = self.render_monitoring_settings(config)
        
        # UI settings
        with tabs[5]:
            config['ui'] = self.render_ui_settings(config)
        
        # Import/Export
        with tabs[6]:
            st.subheader("Configuration Import/Export")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Export Configuration**")
                config_json = json.dumps(config, indent=2)
                st.download_button(
                    "Download Configuration",
                    data=config_json,
                    file_name=f"platform_config_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            
            with col2:
                st.write("**Import Configuration**")
                uploaded_file = st.file_uploader("Upload configuration file", type=['json'])
                
                if uploaded_file is not None:
                    try:
                        imported_config = json.load(uploaded_file)
                        if st.button("Apply Imported Configuration"):
                            config = imported_config
                            st.success("Configuration imported successfully")
                    except:
                        st.error("Invalid configuration file")
        
        # Save button
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            if st.button("Save Settings", type="primary", use_container_width=True):
                if self.save_config(config):
                    st.success("Settings saved successfully")
                else:
                    st.error("Failed to save settings")
        
        with col2:
            if st.button("Reset to Defaults", use_container_width=True):
                config = self.default_config.copy()
                st.info("Settings reset to defaults (not saved)")
