#!/usr/bin/env python3
"""
Professional Browser Interface for Diachronic Linguistics Research Platform
Main application entry point
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import interface components with error handling
try:
    from browser_components.corpus_browser import CorpusBrowser
except ImportError:
    CorpusBrowser = None

try:
    from browser_components.analysis_studio import AnalysisStudio
except ImportError:
    AnalysisStudio = None

try:
    from browser_components.research_dashboard import ResearchDashboard
except ImportError:
    ResearchDashboard = None

try:
    from browser_components.syntactic_tools import SyntacticTools
except ImportError:
    SyntacticTools = None

try:
    from browser_components.valency_explorer import ValencyExplorer
except ImportError:
    ValencyExplorer = None

try:
    from browser_components.monitoring_panel import MonitoringPanel
except ImportError:
    MonitoringPanel = None

try:
    from browser_components.settings_manager import SettingsManager
except ImportError:
    SettingsManager = None

try:
    from browser_components.documentation import DocumentationViewer
except ImportError:
    DocumentationViewer = None

# Configure page
st.set_page_config(
    page_title="Diachronic Linguistics Research Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/nlavidas/historical-linguistics-platform',
        'Report a bug': "https://github.com/nlavidas/historical-linguistics-platform/issues",
        'About': "Diachronic Linguistics Research Platform v2.0"
    }
)

# Professional CSS styling
st.markdown("""
<style>
    /* Professional clean interface */
    .main { 
        padding: 0rem 1rem; 
        background-color: #fafafa;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 2px; 
        background-color: #f0f0f0;
        padding: 0.5rem;
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] { 
        height: 45px; 
        padding: 0 24px;
        background-color: white;
        border-radius: 6px 6px 0 0;
        border: 1px solid #ddd;
        font-weight: 500;
        color: #333;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0066cc;
        color: white;
        border-color: #0066cc;
    }
    
    /* Headers */
    h1 {
        color: #1a1a1a;
        font-weight: 300;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #333;
        font-weight: 400;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #555;
        font-weight: 500;
        margin-top: 1rem;
    }
    
    /* Metrics styling */
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton button {
        background-color: #0066cc;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 4px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton button:hover {
        background-color: #0052a3;
    }
    
    /* Data tables */
    .dataframe {
        font-size: 14px;
        border: 1px solid #ddd;
    }
    
    .dataframe thead th {
        background-color: #f5f5f5;
        font-weight: 600;
        text-align: left;
        padding: 8px;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Professional input fields */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 8px 12px;
        font-size: 14px;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {
        border-color: #0066cc;
        box-shadow: 0 0 0 2px rgba(0,102,204,0.1);
    }
</style>
""", unsafe_allow_html=True)

class DiachronicLinguisticsPlatform:
    """Main application class for the browser interface"""
    
    def __init__(self):
        self.initialize_session_state()
        self.components = self.load_components()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'current_corpus' not in st.session_state:
            st.session_state.current_corpus = None
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        if 'active_project' not in st.session_state:
            st.session_state.active_project = None
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'default_language': 'grc',
                'annotation_standard': 'UD',
                'treebank_format': 'PROIEL'
            }
    
    def load_components(self):
        """Load all interface components"""
        components = {}
        
        if CorpusBrowser:
            components['corpus_browser'] = CorpusBrowser()
        if AnalysisStudio:
            components['analysis_studio'] = AnalysisStudio()
        if ResearchDashboard:
            components['research_dashboard'] = ResearchDashboard()
        if SyntacticTools:
            components['syntactic_tools'] = SyntacticTools()
        if ValencyExplorer:
            components['valency_explorer'] = ValencyExplorer()
        if MonitoringPanel:
            components['monitoring_panel'] = MonitoringPanel()
        if SettingsManager:
            components['settings_manager'] = SettingsManager()
        if DocumentationViewer:
            components['documentation'] = DocumentationViewer()
        
        return components
    
    def render_header(self):
        """Render application header"""
        col1, col2, col3 = st.columns([1, 4, 1])
        
        with col2:
            st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Diachronic Linguistics Research Platform</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #666; margin-top: 0;'>Professional Computational Historical Linguistics</p>", unsafe_allow_html=True)
    
    def render_main_navigation(self):
        """Render main navigation tabs"""
        tabs = st.tabs([
            "Corpus Browser",
            "Analysis Studio",
            "Research Dashboard",
            "Syntactic Tools",
            "Valency Explorer",
            "System Monitoring",
            "Settings",
            "Documentation"
        ])
        
        with tabs[0]:
            if 'corpus_browser' in self.components:
                self.components['corpus_browser'].render()
            else:
                st.info("Corpus Browser component not available")
        
        with tabs[1]:
            if 'analysis_studio' in self.components:
                self.components['analysis_studio'].render()
            else:
                st.info("Analysis Studio component not available")
        
        with tabs[2]:
            if 'research_dashboard' in self.components:
                self.components['research_dashboard'].render()
            else:
                st.info("Research Dashboard component not available")
        
        with tabs[3]:
            if 'syntactic_tools' in self.components:
                self.components['syntactic_tools'].render()
            else:
                st.info("Syntactic Tools component not available")
        
        with tabs[4]:
            if 'valency_explorer' in self.components:
                self.components['valency_explorer'].render()
            else:
                st.info("Valency Explorer component not available")
        
        with tabs[5]:
            if 'monitoring_panel' in self.components:
                self.components['monitoring_panel'].render()
            else:
                st.info("Monitoring Panel component not available")
        
        with tabs[6]:
            if 'settings_manager' in self.components:
                self.components['settings_manager'].render()
            else:
                st.info("Settings Manager component not available")
        
        with tabs[7]:
            if 'documentation' in self.components:
                self.components['documentation'].render()
            else:
                st.info("Documentation component not available")
    
    def run(self):
        """Run the main application"""
        self.render_header()
        self.render_main_navigation()

# Main execution
if __name__ == "__main__":
    app = DiachronicLinguisticsPlatform()
    app.run()
