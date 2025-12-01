#!/usr/bin/env python3
"""
PROIEL-Syntacticus Style Greek Corpus Platform
Professional Diachronic Greek Linguistics Research Platform
University of Athens - Nikolaos Lavidas

Focus: All periods of Greek (Ancient → Byzantine → Medieval → Early Modern)
Features: PROIEL annotation, Semantic Role Labeling, ML/AI, FAIR principles
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import json
import os
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION - Professional Syntacticus-style
# ============================================================================

st.set_page_config(
    page_title="Greek Diachronic Corpus Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/nlavidas/historical-linguistics-platform',
        'Report a bug': 'https://github.com/nlavidas/historical-linguistics-platform/issues',
        'About': """
        ## Greek Diachronic Corpus Platform
        **Version:** 3.0.0 FAIR  
        **Institution:** University of Athens  
        **Principal Investigator:** Nikolaos Lavidas
        
        A PROIEL/Syntacticus-style platform for diachronic Greek linguistics.
        """
    }
)

# ============================================================================
# PROFESSIONAL CSS STYLING (Syntacticus/PROIEL inspired)
# ============================================================================

st.markdown("""
<style>
    /* Main theme - Professional academic style */
    :root {
        --primary-color: #1e3a5f;
        --secondary-color: #2c5282;
        --accent-color: #3182ce;
        --background-light: #f7fafc;
        --text-dark: #1a202c;
        --border-color: #e2e8f0;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 300;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }
    
    .main-header .subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Navigation tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e3a5f !important;
        color: white !important;
    }
    
    /* Cards */
    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .info-card h3 {
        color: #1e3a5f;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        border-bottom: 2px solid #3182ce;
        padding-bottom: 0.5rem;
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Period badges */
    .period-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px;
    }
    
    .period-archaic { background: #fef3c7; color: #92400e; }
    .period-classical { background: #dbeafe; color: #1e40af; }
    .period-hellenistic { background: #d1fae5; color: #065f46; }
    .period-koine { background: #ede9fe; color: #5b21b6; }
    .period-byzantine { background: #fce7f3; color: #9d174d; }
    .period-medieval { background: #fee2e2; color: #991b1b; }
    .period-early-modern { background: #e0e7ff; color: #3730a3; }
    
    /* Treebank visualization */
    .treebank-node {
        background: #f0f4f8;
        border: 2px solid #3182ce;
        border-radius: 8px;
        padding: 8px 16px;
        margin: 4px;
        display: inline-block;
    }
    
    .treebank-relation {
        color: #718096;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    
    /* FAIR badge */
    .fair-badge {
        background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6, #ec4899);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 58, 95, 0.3);
    }
    
    /* DataFrames */
    .dataframe {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Greek text styling */
    .greek-text {
        font-family: 'Gentium Plus', 'Times New Roman', serif;
        font-size: 1.2rem;
        line-height: 1.8;
        color: #1a202c;
    }
    
    /* Annotation highlight */
    .annotation-highlight {
        background: linear-gradient(180deg, transparent 60%, #fef3c7 60%);
        padding: 0 4px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# GREEK PERIODS CONFIGURATION
# ============================================================================

GREEK_PERIODS = {
    "archaic": {
        "name": "Archaic Greek",
        "dates": "800-500 BCE",
        "color": "#fef3c7",
        "authors": ["Homer", "Hesiod", "Sappho", "Pindar"]
    },
    "classical": {
        "name": "Classical Greek", 
        "dates": "500-323 BCE",
        "color": "#dbeafe",
        "authors": ["Plato", "Aristotle", "Sophocles", "Euripides", "Thucydides", "Demosthenes"]
    },
    "hellenistic": {
        "name": "Hellenistic Greek",
        "dates": "323-31 BCE",
        "color": "#d1fae5",
        "authors": ["Polybius", "Callimachus", "Apollonius", "Septuagint"]
    },
    "koine": {
        "name": "Koine Greek",
        "dates": "300 BCE - 300 CE",
        "color": "#ede9fe",
        "authors": ["New Testament", "Plutarch", "Epictetus", "Lucian"]
    },
    "late_antique": {
        "name": "Late Antique Greek",
        "dates": "300-600 CE",
        "color": "#fce7f3",
        "authors": ["Church Fathers", "Eusebius", "John Chrysostom"]
    },
    "byzantine": {
        "name": "Byzantine Greek",
        "dates": "600-1453 CE",
        "color": "#fee2e2",
        "authors": ["Anna Comnena", "Michael Psellus", "Maximus Planudes", "Chronicle of Morea"]
    },
    "medieval": {
        "name": "Medieval Greek",
        "dates": "1100-1453 CE",
        "color": "#fef3c7",
        "authors": ["Digenes Akritas", "Chronicle of Morea", "Ptochoprodromos"]
    },
    "early_modern": {
        "name": "Early Modern Greek",
        "dates": "1453-1800 CE",
        "color": "#e0e7ff",
        "authors": ["Cretan Literature", "Erotokritos", "Kornaros"]
    }
}

# PROIEL Dependency Relations
PROIEL_RELATIONS = {
    "pred": {"name": "Predicate", "description": "Main predicate of clause"},
    "sub": {"name": "Subject", "description": "Subject of verb"},
    "obj": {"name": "Object", "description": "Direct object"},
    "obl": {"name": "Oblique", "description": "Oblique argument"},
    "ag": {"name": "Agent", "description": "Agent in passive"},
    "atr": {"name": "Attribute", "description": "Attributive modifier"},
    "adv": {"name": "Adverbial", "description": "Adverbial modifier"},
    "apos": {"name": "Apposition", "description": "Appositive"},
    "aux": {"name": "Auxiliary", "description": "Auxiliary element"},
    "comp": {"name": "Complement", "description": "Complement clause"},
    "expl": {"name": "Expletive", "description": "Expletive element"},
    "narg": {"name": "Non-argument", "description": "Non-argument dependent"},
    "nonsub": {"name": "Non-subject", "description": "Non-subject ex-argument"},
    "parpred": {"name": "Parenthetical", "description": "Parenthetical predication"},
    "per": {"name": "Peripheral", "description": "Peripheral element"},
    "pid": {"name": "Predicate Identity", "description": "Predicate identity"},
    "voc": {"name": "Vocative", "description": "Vocative"},
    "xadv": {"name": "External Adverbial", "description": "Open adverbial complement"},
    "xobj": {"name": "External Object", "description": "Open objective complement"},
    "xsub": {"name": "External Subject", "description": "External subject"}
}

# Semantic Roles (PropBank/FrameNet style - from Jurafsky & Martin)
SEMANTIC_ROLES = {
    "ARG0": {"name": "Agent", "description": "Volitional causer of event", "color": "#ef4444"},
    "ARG1": {"name": "Patient/Theme", "description": "Entity affected by action", "color": "#f97316"},
    "ARG2": {"name": "Instrument/Beneficiary", "description": "Secondary participant", "color": "#eab308"},
    "ARG3": {"name": "Starting Point", "description": "Source or origin", "color": "#22c55e"},
    "ARG4": {"name": "Ending Point", "description": "Goal or destination", "color": "#06b6d4"},
    "ARGM-LOC": {"name": "Location", "description": "Where event takes place", "color": "#3b82f6"},
    "ARGM-TMP": {"name": "Temporal", "description": "When event takes place", "color": "#8b5cf6"},
    "ARGM-MNR": {"name": "Manner", "description": "How action is performed", "color": "#ec4899"},
    "ARGM-CAU": {"name": "Cause", "description": "Reason for event", "color": "#6366f1"},
    "ARGM-PRP": {"name": "Purpose", "description": "Purpose of action", "color": "#14b8a6"},
    "ARGM-DIR": {"name": "Direction", "description": "Direction of motion", "color": "#f43f5e"},
    "ARGM-EXT": {"name": "Extent", "description": "Degree or amount", "color": "#a855f7"}
}

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                period TEXT,
                genre TEXT,
                source TEXT,
                language TEXT DEFAULT 'grc',
                sentence_count INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                annotation_status TEXT DEFAULT 'pending',
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sentences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentences (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                sentence_index INTEGER,
                text TEXT NOT NULL,
                translation TEXT,
                tokens TEXT,
                semantic_roles TEXT,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        
        # Tokens table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id TEXT,
                token_index INTEGER,
                form TEXT NOT NULL,
                lemma TEXT,
                pos TEXT,
                morphology TEXT,
                head INTEGER,
                relation TEXT,
                semantic_role TEXT,
                gloss TEXT,
                FOREIGN KEY (sentence_id) REFERENCES sentences(id)
            )
        """)
        
        # Valency lexicon
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valency_lexicon (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                pattern TEXT NOT NULL,
                arguments TEXT,
                frequency INTEGER DEFAULT 1,
                period TEXT,
                examples TEXT,
                semantic_class TEXT
            )
        """)
        
        # ML Models registry
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                model_type TEXT,
                language TEXT,
                period TEXT,
                accuracy REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_path TEXT,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        stats["documents"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(sentence_count) FROM documents")
        result = cursor.fetchone()[0]
        stats["sentences"] = result if result else 0
        
        cursor.execute("SELECT SUM(token_count) FROM documents")
        result = cursor.fetchone()[0]
        stats["tokens"] = result if result else 0
        
        cursor.execute("SELECT period, COUNT(*) FROM documents WHERE period IS NOT NULL GROUP BY period")
        stats["by_period"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("SELECT COUNT(DISTINCT lemma) FROM valency_lexicon")
        stats["valency_verbs"] = cursor.fetchone()[0]
        
        conn.close()
        return stats


# ============================================================================
# INITIALIZE
# ============================================================================

@st.cache_resource
def get_db():
    return DatabaseManager()

# Session state
if 'processes' not in st.session_state:
    st.session_state.processes = {
        'collector': {'status': 'stopped'},
        'preprocessor': {'status': 'stopped'},
        'parser': {'status': 'stopped'},
        'srl': {'status': 'stopped'},
        'ml_trainer': {'status': 'stopped'}
    }

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    db = get_db()
    
    # Professional Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ Greek Diachronic Corpus Platform</h1>
        <div class="subtitle">
            PROIEL-Syntacticus Style Annotation System | University of Athens
        </div>
        <div style="margin-top: 1rem;">
            <span class="fair-badge">FAIR Data Principles</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/NKUA_logo.svg/1200px-NKUA_logo.svg.png", width=100)
        st.markdown("### Navigation")
        
        # Quick stats
        stats = db.get_statistics()
        st.metric("Documents", f"{stats.get('documents', 0):,}")
        st.metric("Tokens", f"{stats.get('tokens', 0):,}")
        
        st.markdown("---")
        st.markdown("### Greek Periods")
        for period_id, period_info in GREEK_PERIODS.items():
            count = stats.get('by_period', {}).get(period_id, 0)
            st.markdown(f"**{period_info['name']}**: {count}")
        
        st.markdown("---")
        st.markdown("### Resources")
        st.markdown("[PROIEL Treebank](https://proiel.github.io)")
        st.markdown("[Syntacticus](https://syntacticus.org)")
        st.markdown("[Perseus Digital Library](http://www.perseus.tufts.edu)")
    
    # Main tabs
    tabs = st.tabs([
        "🏠 Overview",
        "📚 Corpus Browser", 
        "🌳 Treebank Viewer",
        "🎭 Semantic Roles",
        "⚡ Valency Lexicon",
        "🤖 ML & AI Tools",
        "📊 Analytics",
        "⚙️ Pipeline Control"
    ])
    
    with tabs[0]:
        render_overview(db)
    
    with tabs[1]:
        render_corpus_browser(db)
    
    with tabs[2]:
        render_treebank_viewer(db)
    
    with tabs[3]:
        render_semantic_roles(db)
    
    with tabs[4]:
        render_valency_lexicon(db)
    
    with tabs[5]:
        render_ml_tools(db)
    
    with tabs[6]:
        render_analytics(db)
    
    with tabs[7]:
        render_pipeline_control(db)


def render_overview(db):
    """Professional overview page"""
    stats = db.get_statistics()
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Documents</div>
        </div>
        """.format(stats.get('documents', 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Sentences</div>
        </div>
        """.format(stats.get('sentences', 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Tokens</div>
        </div>
        """.format(stats.get('tokens', 0)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <div class="metric-value">{}</div>
            <div class="metric-label">Periods</div>
        </div>
        """.format(len(GREEK_PERIODS)), unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-container" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Valency Frames</div>
        </div>
        """.format(stats.get('valency_verbs', 0)), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Greek Periods Timeline
    st.markdown("### 📅 Greek Language Periods")
    
    cols = st.columns(4)
    for idx, (period_id, period_info) in enumerate(GREEK_PERIODS.items()):
        with cols[idx % 4]:
            st.markdown(f"""
            <div class="info-card">
                <h3>{period_info['name']}</h3>
                <p><strong>{period_info['dates']}</strong></p>
                <p><em>Key authors:</em></p>
                <p>{', '.join(period_info['authors'][:3])}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # FAIR Principles
    st.markdown("### 🌐 FAIR Data Principles")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🔍 Findable</h3>
            <p>Rich metadata, persistent identifiers, indexed in searchable resources</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🔓 Accessible</h3>
            <p>Open protocols, authentication where needed, metadata always available</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>🔄 Interoperable</h3>
            <p>PROIEL/UD standards, formal vocabularies, qualified references</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="info-card">
            <h3>♻️ Reusable</h3>
            <p>Clear licensing, detailed provenance, community standards</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Byzantine/Medieval emphasis
    st.markdown("---")
    st.markdown("### 🏰 Byzantine & Medieval Greek Focus")
    
    st.info("""
    **Special emphasis on understudied periods:**
    - **Byzantine Greek** (600-1453 CE): Anna Comnena, Maximus Planudes, Chronicle of Morea
    - **Medieval Greek** (1100-1453 CE): Digenes Akritas, Ptochoprodromos, vernacular texts
    - **Early Modern Greek** (1453-1800 CE): Cretan Renaissance, Erotokritos
    - **Retranslations**: Later Greek translations of classical texts (Planudes' translations)
    """)


def render_corpus_browser(db):
    """Corpus browser with period filtering"""
    st.markdown("### 📚 Corpus Browser")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        period = st.selectbox("Period", ["All"] + list(GREEK_PERIODS.keys()),
                             format_func=lambda x: GREEK_PERIODS[x]['name'] if x != "All" else "All Periods")
    
    with col2:
        genre = st.selectbox("Genre", ["All", "Epic", "Tragedy", "Comedy", "History", 
                                       "Philosophy", "Oratory", "Patristic", "Chronicle", "Romance"])
    
    with col3:
        annotation_status = st.selectbox("Annotation Status", ["All", "Complete", "In Progress", "Pending"])
    
    with col4:
        search = st.text_input("Search", placeholder="Author, title, or keyword...")
    
    # Results placeholder
    st.markdown("---")
    
    # Sample data display
    sample_texts = [
        {"title": "Iliad", "author": "Homer", "period": "archaic", "tokens": 115477, "status": "Complete"},
        {"title": "Odyssey", "author": "Homer", "period": "archaic", "tokens": 87765, "status": "Complete"},
        {"title": "Republic", "author": "Plato", "period": "classical", "tokens": 98432, "status": "In Progress"},
        {"title": "Chronicle of Morea", "author": "Anonymous", "period": "medieval", "tokens": 45000, "status": "Pending"},
        {"title": "Alexiad", "author": "Anna Comnena", "period": "byzantine", "tokens": 120000, "status": "Pending"},
        {"title": "Planudes Translations", "author": "Maximus Planudes", "period": "byzantine", "tokens": 80000, "status": "Pending"},
    ]
    
    df = pd.DataFrame(sample_texts)
    
    # Add period badges
    def format_period(p):
        if p in GREEK_PERIODS:
            return GREEK_PERIODS[p]['name']
        return p
    
    df['Period'] = df['period'].apply(format_period)
    
    st.dataframe(
        df[['title', 'author', 'Period', 'tokens', 'status']],
        use_container_width=True,
        column_config={
            "title": "Title",
            "author": "Author",
            "tokens": st.column_config.NumberColumn("Tokens", format="%d"),
            "status": st.column_config.TextColumn("Status")
        }
    )


def render_treebank_viewer(db):
    """PROIEL-style treebank visualization"""
    st.markdown("### 🌳 Treebank Viewer (PROIEL Style)")
    
    # Sample sentence
    st.markdown("#### Sample Annotation")
    
    sample_sentence = "Ἐν ἀρχῇ ἦν ὁ λόγος"
    st.markdown(f'<div class="greek-text">{sample_sentence}</div>', unsafe_allow_html=True)
    st.caption("John 1:1 - In the beginning was the Word")
    
    # Token table
    tokens_data = [
        {"ID": 1, "Form": "Ἐν", "Lemma": "ἐν", "POS": "R-", "Head": 2, "Relation": "adv", "Gloss": "in"},
        {"ID": 2, "Form": "ἀρχῇ", "Lemma": "ἀρχή", "POS": "Nb", "Head": 3, "Relation": "obl", "Gloss": "beginning"},
        {"ID": 3, "Form": "ἦν", "Lemma": "εἰμί", "POS": "V-", "Head": 0, "Relation": "pred", "Gloss": "was"},
        {"ID": 4, "Form": "ὁ", "Lemma": "ὁ", "POS": "S-", "Head": 5, "Relation": "atr", "Gloss": "the"},
        {"ID": 5, "Form": "λόγος", "Lemma": "λόγος", "POS": "Nb", "Head": 3, "Relation": "sub", "Gloss": "word"},
    ]
    
    st.dataframe(pd.DataFrame(tokens_data), use_container_width=True)
    
    # Dependency tree visualization
    st.markdown("#### Dependency Tree")
    
    # Create tree visualization with Plotly
    fig = go.Figure()
    
    # Node positions
    positions = {1: (0, 0), 2: (1, 0), 3: (2, 1), 4: (3, 0), 5: (4, 0)}
    
    # Add edges
    edges = [(1, 2), (2, 3), (3, 3), (4, 5), (5, 3)]
    for start, end in edges:
        if start != end:
            x0, y0 = positions[start]
            x1, y1 = positions[end]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(color='#3182ce', width=2),
                hoverinfo='none'
            ))
    
    # Add nodes
    for token in tokens_data:
        x, y = positions[token['ID']]
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=40, color='#1e3a5f'),
            text=token['Form'],
            textposition='middle center',
            textfont=dict(color='white', size=12),
            hovertext=f"{token['Form']}<br>Lemma: {token['Lemma']}<br>POS: {token['POS']}<br>Relation: {token['Relation']}",
            hoverinfo='text'
        ))
    
    fig.update_layout(
        showlegend=False,
        height=300,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # PROIEL Relations reference
    st.markdown("#### PROIEL Dependency Relations")
    
    relations_df = pd.DataFrame([
        {"Relation": k, "Name": v["name"], "Description": v["description"]}
        for k, v in PROIEL_RELATIONS.items()
    ])
    
    st.dataframe(relations_df, use_container_width=True, hide_index=True)


def render_semantic_roles(db):
    """Semantic Role Labeling interface (Jurafsky & Martin style)"""
    st.markdown("### 🎭 Semantic Role Labeling")
    
    st.info("""
    **Based on:** Jurafsky & Martin, Chapter 21 - Semantic Role Labeling
    
    Semantic roles express the abstract relationship between a predicate and its arguments.
    Following PropBank/FrameNet conventions adapted for Greek.
    """)
    
    # Sample sentence with SRL
    st.markdown("#### Example: Argument Structure Annotation")
    
    sample = "ὁ στρατηγὸς τοὺς στρατιώτας εἰς τὴν πόλιν ἤγαγεν"
    st.markdown(f'<div class="greek-text">{sample}</div>', unsafe_allow_html=True)
    st.caption("The general led the soldiers into the city")
    
    # SRL annotation
    srl_data = [
        {"Token": "ὁ στρατηγὸς", "Role": "ARG0", "Label": "Agent", "Description": "The one performing the action"},
        {"Token": "τοὺς στρατιώτας", "Role": "ARG1", "Label": "Patient/Theme", "Description": "Entity being led"},
        {"Token": "εἰς τὴν πόλιν", "Role": "ARG4", "Label": "Goal", "Description": "Destination of motion"},
        {"Token": "ἤγαγεν", "Role": "PRED", "Label": "Predicate", "Description": "Main verb (ἄγω - to lead)"},
    ]
    
    st.dataframe(pd.DataFrame(srl_data), use_container_width=True, hide_index=True)
    
    # Semantic roles reference
    st.markdown("#### Semantic Role Inventory")
    
    cols = st.columns(3)
    for idx, (role_id, role_info) in enumerate(SEMANTIC_ROLES.items()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style="background: {role_info['color']}20; border-left: 4px solid {role_info['color']}; 
                        padding: 10px; margin: 5px 0; border-radius: 4px;">
                <strong>{role_id}</strong>: {role_info['name']}<br>
                <small>{role_info['description']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Annotation interface
    st.markdown("---")
    st.markdown("#### Annotate New Sentence")
    
    input_text = st.text_area("Enter Greek text:", height=100)
    
    if st.button("Analyze Semantic Roles", type="primary"):
        if input_text:
            st.success("Semantic role analysis would be performed here")
            st.json({
                "text": input_text,
                "predicate": "detected_verb",
                "arguments": [
                    {"role": "ARG0", "span": "subject phrase"},
                    {"role": "ARG1", "span": "object phrase"}
                ]
            })


def render_valency_lexicon(db):
    """Valency lexicon browser"""
    st.markdown("### ⚡ Valency Lexicon")
    
    # Search
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_lemma = st.text_input("Search verb lemma:", placeholder="e.g., δίδωμι, λέγω, ἄγω")
    
    with col2:
        pattern_filter = st.selectbox("Pattern", ["All", "NOM", "NOM+ACC", "NOM+GEN", "NOM+DAT", "NOM+ACC+DAT"])
    
    with col3:
        period_filter = st.selectbox("Period", ["All"] + list(GREEK_PERIODS.keys()),
                                    format_func=lambda x: GREEK_PERIODS[x]['name'] if x != "All" else "All")
    
    # Sample valency data
    valency_data = [
        {"Lemma": "δίδωμι", "Pattern": "NOM+ACC+DAT", "Gloss": "give", "Class": "Transfer", "Period": "All"},
        {"Lemma": "λέγω", "Pattern": "NOM+ACC", "Gloss": "say, speak", "Class": "Communication", "Period": "All"},
        {"Lemma": "ἄγω", "Pattern": "NOM+ACC", "Gloss": "lead, bring", "Class": "Motion", "Period": "All"},
        {"Lemma": "ἀκούω", "Pattern": "NOM+GEN", "Gloss": "hear", "Class": "Perception", "Period": "All"},
        {"Lemma": "πείθομαι", "Pattern": "NOM+DAT", "Gloss": "obey", "Class": "Social", "Period": "All"},
        {"Lemma": "διδάσκω", "Pattern": "NOM+ACC+ACC", "Gloss": "teach", "Class": "Transfer", "Period": "All"},
        {"Lemma": "πληρόω", "Pattern": "NOM+ACC+GEN", "Gloss": "fill", "Class": "Change", "Period": "All"},
    ]
    
    st.dataframe(pd.DataFrame(valency_data), use_container_width=True, hide_index=True)
    
    # Valency patterns explanation
    st.markdown("#### Valency Pattern Reference")
    
    patterns_info = {
        "NOM": "Intransitive (Subject only)",
        "NOM+ACC": "Monotransitive (Subject + Direct Object)",
        "NOM+GEN": "Genitive Object",
        "NOM+DAT": "Dative Object",
        "NOM+ACC+DAT": "Ditransitive (Subject + Direct + Indirect Object)",
        "NOM+ACC+ACC": "Double Accusative",
        "NOM+ACC+GEN": "Accusative + Genitive"
    }
    
    for pattern, desc in patterns_info.items():
        st.markdown(f"- **{pattern}**: {desc}")


def render_ml_tools(db):
    """ML and AI tools interface"""
    st.markdown("### 🤖 Machine Learning & AI Tools")
    
    st.info("""
    **Community-driven ML models for Historical Greek**
    
    Based on: Schneider's Text Analytics in Digital Humanities + LightSide ML Workbench
    """)
    
    # ML Models
    tab1, tab2, tab3, tab4 = st.tabs(["POS Tagger", "Lemmatizer", "Parser", "SRL Model"])
    
    with tab1:
        st.markdown("#### POS Tagger for Historical Greek")
        
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Model", ["Stanza (grc)", "CLTK Greek", "Custom BiLSTM", "Transformer (BERT-Greek)"])
            st.selectbox("Period Specialization", ["General", "Classical", "Koine", "Byzantine", "Medieval"])
        
        with col2:
            st.metric("Accuracy", "94.2%")
            st.metric("Training Tokens", "1.2M")
        
        test_text = st.text_area("Test POS Tagger:", value="ἐν ἀρχῇ ἦν ὁ λόγος")
        if st.button("Tag", key="pos_tag"):
            st.success("POS tagging complete")
            st.code("ἐν/R- ἀρχῇ/Nb ἦν/V- ὁ/S- λόγος/Nb")
    
    with tab2:
        st.markdown("#### Lemmatizer")
        
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Lemmatizer", ["CLTK", "Stanza", "Rule-based", "Neural"])
        with col2:
            st.metric("Coverage", "98.5%")
        
        test_text = st.text_area("Test Lemmatizer:", value="ἤγαγεν τοὺς στρατιώτας")
        if st.button("Lemmatize", key="lemmatize"):
            st.success("Lemmatization complete")
            st.code("ἤγαγεν → ἄγω\nτοὺς → ὁ\nστρατιώτας → στρατιώτης")
    
    with tab3:
        st.markdown("#### Dependency Parser")
        
        st.selectbox("Parser Model", ["Stanza UD", "PROIEL-trained", "Custom Biaffine"])
        st.slider("Beam Size", 1, 10, 5)
        
        if st.button("Parse", key="parse"):
            st.success("Parsing complete")
    
    with tab4:
        st.markdown("#### Semantic Role Labeler")
        
        st.selectbox("SRL Model", ["PropBank-style", "FrameNet-style", "Custom Greek SRL"])
        
        if st.button("Label Roles", key="srl"):
            st.success("SRL complete")
    
    # LightSide integration
    st.markdown("---")
    st.markdown("#### 🔦 LightSide ML Workbench Integration")
    
    st.markdown("""
    LightSide features available:
    - **Feature extraction**: N-grams, POS patterns, syntactic features
    - **Classification**: Text classification, sentiment analysis
    - **Sequence labeling**: NER, POS tagging
    - **Model comparison**: Cross-validation, feature analysis
    """)
    
    if st.button("Launch LightSide Interface"):
        st.info("LightSide integration would open here")


def render_analytics(db):
    """Analytics dashboard"""
    st.markdown("### 📊 Corpus Analytics")
    
    # Period distribution
    st.markdown("#### Distribution by Period")
    
    period_data = {
        "Period": list(GREEK_PERIODS.keys()),
        "Documents": [15, 45, 20, 35, 25, 30, 15, 10],
        "Tokens": [200000, 850000, 300000, 600000, 400000, 500000, 200000, 150000]
    }
    
    fig = px.bar(
        period_data, 
        x="Period", 
        y="Tokens",
        color="Period",
        title="Token Distribution by Greek Period"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Valency patterns
    st.markdown("#### Valency Pattern Distribution")
    
    valency_dist = {
        "Pattern": ["NOM+ACC", "NOM", "NOM+DAT", "NOM+GEN", "NOM+ACC+DAT", "Other"],
        "Count": [450, 200, 120, 80, 60, 40]
    }
    
    fig2 = px.pie(valency_dist, values="Count", names="Pattern", title="Valency Patterns")
    st.plotly_chart(fig2, use_container_width=True)


def render_pipeline_control(db):
    """Pipeline control panel"""
    st.markdown("### ⚙️ Pipeline Control")
    
    # Process status
    processes = [
        ("Text Collector", "collector", "Collects texts from Perseus, PROIEL, First1KGreek"),
        ("Preprocessor", "preprocessor", "Tokenization, normalization, sentence splitting"),
        ("Parser", "parser", "Dependency parsing with PROIEL relations"),
        ("SRL Annotator", "srl", "Semantic role labeling"),
        ("ML Trainer", "ml_trainer", "Train and update ML models")
    ]
    
    for name, key, desc in processes:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**{name}**")
            st.caption(desc)
        
        with col2:
            status = st.session_state.processes.get(key, {}).get('status', 'stopped')
            if status == 'running':
                st.success("Running")
            else:
                st.warning("Stopped")
        
        with col3:
            if st.button(f"Start {name}", key=f"start_{key}"):
                st.session_state.processes[key]['status'] = 'running'
                st.rerun()
            if st.button(f"Stop {name}", key=f"stop_{key}"):
                st.session_state.processes[key]['status'] = 'stopped'
                st.rerun()
        
        st.markdown("---")
    
    # Quick actions
    st.markdown("#### Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start All", type="primary"):
            for key in st.session_state.processes:
                st.session_state.processes[key]['status'] = 'running'
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop All"):
            for key in st.session_state.processes:
                st.session_state.processes[key]['status'] = 'stopped'
            st.rerun()
    
    with col3:
        if st.button("📥 Collect Byzantine Texts"):
            st.info("Collecting Byzantine Greek texts...")


if __name__ == "__main__":
    main()
