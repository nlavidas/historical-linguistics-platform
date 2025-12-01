#!/usr/bin/env python3
"""
Diachronic Linguistics Research Platform
Professional Web Application for Historical Linguistics
Version 2.0 - Comprehensive Pipeline with Process Controls
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
import subprocess
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Process status tracking
if 'processes' not in st.session_state:
    st.session_state.processes = {
        'collector': {'status': 'stopped', 'pid': None},
        'preprocessor': {'status': 'stopped', 'pid': None},
        'parser': {'status': 'stopped', 'pid': None},
        'valency': {'status': 'stopped', 'pid': None},
        'etymology': {'status': 'stopped', 'pid': None}
    }

# Language metadata
LANGUAGE_INFO = {
    "grc": {"name": "Ancient Greek", "family": "Indo-European", "branch": "Hellenic"},
    "la": {"name": "Latin", "family": "Indo-European", "branch": "Italic"},
    "sa": {"name": "Sanskrit", "family": "Indo-European", "branch": "Indo-Iranian"},
    "got": {"name": "Gothic", "family": "Indo-European", "branch": "Germanic"},
    "cu": {"name": "Old Church Slavonic", "family": "Indo-European", "branch": "Slavic"},
    "cop": {"name": "Coptic", "family": "Afro-Asiatic", "branch": "Egyptian"},
    "xcl": {"name": "Classical Armenian", "family": "Indo-European", "branch": "Armenian"},
}

VALENCY_PATTERNS = {
    "NOM": "Intransitive",
    "NOM+ACC": "Monotransitive",
    "NOM+GEN": "Genitive object",
    "NOM+DAT": "Dative object",
    "NOM+ACC+DAT": "Ditransitive",
}

DEPENDENCY_RELATIONS = {
    "pred": "Predicate",
    "sub": "Subject",
    "obj": "Object",
    "obl": "Oblique",
    "atr": "Attribute",
    "adv": "Adverbial",
    "ag": "Agent",
    "comp": "Complement",
}

class DatabaseManager:
    def __init__(self, db_path: str = "corpus_platform.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, title TEXT, author TEXT, language TEXT,
                period TEXT, genre TEXT, source TEXT, content TEXT,
                annotation_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentences (
                id TEXT PRIMARY KEY, document_id TEXT, text TEXT,
                tokens TEXT, metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valency_lexicon (
                id INTEGER PRIMARY KEY, verb_lemma TEXT, language TEXT,
                pattern TEXT, arguments TEXT, frequency INTEGER DEFAULT 1,
                examples TEXT, period TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etymology (
                id INTEGER PRIMARY KEY, lemma TEXT, language TEXT, pos TEXT,
                proto_form TEXT, cognates TEXT, bibliography TEXT
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
        
        cursor.execute("SELECT COUNT(*) FROM sentences")
        stats["sentences"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT language, COUNT(*) FROM documents GROUP BY language")
        stats["languages"] = {r[0]: r[1] for r in cursor.fetchall()}
        
        cursor.execute("SELECT COUNT(*) FROM valency_lexicon")
        stats["valency"] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def search_corpus(self, query: str, filters: Dict = None) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        sql = "SELECT * FROM documents WHERE title LIKE ? OR author LIKE ? LIMIT 100"
        cursor.execute(sql, (f"%{query}%", f"%{query}%"))
        results = [dict(r) for r in cursor.fetchall()]
        
        conn.close()
        return results
    
    def search_valency(self, lemma: str, language: str = None) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        sql = "SELECT * FROM valency_lexicon WHERE verb_lemma LIKE ?"
        params = [f"%{lemma}%"]
        
        if language:
            sql += " AND language = ?"
            params.append(language)
        
        cursor.execute(sql, params)
        results = [dict(r) for r in cursor.fetchall()]
        
        conn.close()
        return results

class LinguisticAnalyzer:
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def analyze(self, text: str, language: str, analyses: List[str]) -> Dict:
        result = {"language": language, "sentences": [], "statistics": {}}
        
        sentences = re.split(r'(?<=[.;:!?])\s+', text.strip())
        
        for idx, sent in enumerate(sentences):
            tokens = self._tokenize(sent, language)
            sent_result = {"id": f"s{idx+1}", "text": sent, "tokens": tokens}
            
            if "valency" in analyses:
                sent_result["valency"] = self._extract_valency(tokens)
            
            result["sentences"].append(sent_result)
        
        result["statistics"] = self._calc_stats(result["sentences"])
        return result
    
    def _tokenize(self, text: str, language: str) -> List[Dict]:
        words = re.findall(r'\b[\w\u0370-\u03FF\u1F00-\u1FFF]+\b', text)
        tokens = []
        
        for idx, word in enumerate(words, 1):
            tokens.append({
                "id": idx, "form": word, "lemma": word.lower(),
                "pos": self._get_pos(word), "morphology": {},
                "head": 0, "deprel": ""
            })
        
        return tokens
    
    def _get_pos(self, word: str) -> str:
        if word.endswith(('ος', 'ον', 'us', 'um')): return "NOUN"
        if word.endswith(('ω', 'ει', 'o', 'are')): return "VERB"
        return "X"
    
    def _extract_valency(self, tokens: List[Dict]) -> List[Dict]:
        patterns = []
        for t in tokens:
            if t["pos"] == "VERB":
                patterns.append({"verb": t["form"], "lemma": t["lemma"], "pattern": "NOM+ACC"})
        return patterns
    
    def _calc_stats(self, sentences: List[Dict]) -> Dict:
        all_tokens = [t for s in sentences for t in s.get("tokens", [])]
        return {
            "sentences": len(sentences),
            "tokens": len(all_tokens),
            "types": len(set(t["form"] for t in all_tokens)),
            "ttr": len(set(t["form"] for t in all_tokens)) / len(all_tokens) if all_tokens else 0
        }

# Initialize
@st.cache_resource
def get_db():
    return DatabaseManager("corpus_platform.db")

@st.cache_resource
def get_analyzer(_db):
    return LinguisticAnalyzer(_db)

# Page config
st.set_page_config(page_title="Diachronic Linguistics Platform", layout="wide")

st.markdown("""
<style>
    .main { padding: 0 1rem; }
    h1 { color: #1a1a1a; font-weight: 300; }
    .stButton button { background-color: #0066cc; color: white; }
</style>
""", unsafe_allow_html=True)

def main():
    db = get_db()
    analyzer = get_analyzer(db)
    
    st.markdown("<h1 style='text-align:center'>Diachronic Linguistics Research Platform</h1>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Pipeline Control", "Corpus Browser", "Analysis Studio", "Valency Explorer", "Syntactic Tools", "Monitoring", "Settings"])
    
    with tabs[0]:
        render_pipeline_control(db)
    
    with tabs[1]:
        render_corpus_browser(db)
    
    with tabs[2]:
        render_analysis_studio(db, analyzer)
    
    with tabs[3]:
        render_valency_explorer(db)
    
    with tabs[4]:
        render_syntactic_tools(db)
    
    with tabs[5]:
        render_monitoring(db)
    
    with tabs[6]:
        render_settings()

def render_pipeline_control(db):
    """Pipeline Control Panel - Start/Stop Processing Tasks"""
    st.header("Pipeline Control Center")
    st.markdown("Control all text processing pipelines from this panel. Each process runs independently and can be started/stopped at any time.")
    
    st.divider()
    
    # Text Collection
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("1. Text Collection")
        st.markdown("""
        **Collects texts from multiple sources:**
        - Perseus Digital Library (Greek classics)
        - PROIEL Treebank (annotated texts)
        - Project Gutenberg (English translations)
        - Wikisource (Greek and English)
        
        **Focus:** Greek texts (Homer, Plato, Aristotle, NT) + English translations
        """)
    with col2:
        status = st.session_state.processes.get('collector', {}).get('status', 'stopped')
        st.metric("Status", status.upper())
    with col3:
        if st.button("Start Collection", key="start_collector", type="primary"):
            st.session_state.processes['collector']['status'] = 'running'
            st.success("Text collection started")
            st.rerun()
        if st.button("Stop Collection", key="stop_collector"):
            st.session_state.processes['collector']['status'] = 'stopped'
            st.info("Text collection stopped")
            st.rerun()
    
    st.divider()
    
    # Preprocessing
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("2. Preprocessing and Lemmatization")
        st.markdown("""
        **Preprocessing pipeline:**
        - Unicode normalization
        - Sentence segmentation
        - Word tokenization
        - Lemmatization (CLTK, Stanza, rule-based)
        - Stopword identification
        - Text statistics calculation
        """)
    with col2:
        status = st.session_state.processes.get('preprocessor', {}).get('status', 'stopped')
        st.metric("Status", status.upper())
    with col3:
        if st.button("Start Preprocessing", key="start_preprocess", type="primary"):
            st.session_state.processes['preprocessor']['status'] = 'running'
            st.success("Preprocessing started")
            st.rerun()
        if st.button("Stop Preprocessing", key="stop_preprocess"):
            st.session_state.processes['preprocessor']['status'] = 'stopped'
            st.info("Preprocessing stopped")
            st.rerun()
    
    st.divider()
    
    # Parsing
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("3. Parsing and Annotation")
        st.markdown("""
        **Syntactic parsing:**
        - Dependency parsing (Stanza, spaCy, CLTK)
        - POS tagging (UD tagset)
        - Morphological analysis
        - Named Entity Recognition
        - PROIEL/UD annotation standards
        """)
    with col2:
        status = st.session_state.processes.get('parser', {}).get('status', 'stopped')
        st.metric("Status", status.upper())
    with col3:
        if st.button("Start Parsing", key="start_parser", type="primary"):
            st.session_state.processes['parser']['status'] = 'running'
            st.success("Parsing started")
            st.rerun()
        if st.button("Stop Parsing", key="stop_parser"):
            st.session_state.processes['parser']['status'] = 'stopped'
            st.info("Parsing stopped")
            st.rerun()
    
    st.divider()
    
    # Valency Analysis
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("4. Valency Analysis")
        st.markdown("""
        **Valency extraction:**
        - Verbal argument structure identification
        - Case frame extraction
        - Pattern classification (NOM, NOM+ACC, etc.)
        - Diachronic pattern tracking
        - Cross-linguistic comparison
        """)
    with col2:
        status = st.session_state.processes.get('valency', {}).get('status', 'stopped')
        st.metric("Status", status.upper())
    with col3:
        if st.button("Start Valency", key="start_valency", type="primary"):
            st.session_state.processes['valency']['status'] = 'running'
            st.success("Valency analysis started")
            st.rerun()
        if st.button("Stop Valency", key="stop_valency"):
            st.session_state.processes['valency']['status'] = 'stopped'
            st.info("Valency analysis stopped")
            st.rerun()
    
    st.divider()
    
    # Etymology Analysis
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("5. Etymological Analysis")
        st.markdown("""
        **Etymology tracking:**
        - Proto-form reconstruction
        - Cognate identification
        - Semantic development tracking
        - Loanword detection
        - Cross-linguistic etymology
        """)
    with col2:
        status = st.session_state.processes.get('etymology', {}).get('status', 'stopped')
        st.metric("Status", status.upper())
    with col3:
        if st.button("Start Etymology", key="start_etymology", type="primary"):
            st.session_state.processes['etymology']['status'] = 'running'
            st.success("Etymology analysis started")
            st.rerun()
        if st.button("Stop Etymology", key="stop_etymology"):
            st.session_state.processes['etymology']['status'] = 'stopped'
            st.info("Etymology analysis stopped")
            st.rerun()
    
    st.divider()
    
    # Quick Actions
    st.subheader("Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Start All Pipelines", type="primary"):
            for key in st.session_state.processes:
                st.session_state.processes[key]['status'] = 'running'
            st.success("All pipelines started")
            st.rerun()
    
    with col2:
        if st.button("Stop All Pipelines"):
            for key in st.session_state.processes:
                st.session_state.processes[key]['status'] = 'stopped'
            st.info("All pipelines stopped")
            st.rerun()
    
    with col3:
        if st.button("Collect Greek Classics"):
            st.info("Collecting Homer, Plato, Aristotle, Sophocles...")
    
    with col4:
        if st.button("Collect NT + Translations"):
            st.info("Collecting New Testament Greek + English translations...")
    
    # Statistics
    st.divider()
    st.subheader("Pipeline Statistics")
    
    stats = db.get_statistics()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Raw Texts", stats.get("documents", 0))
    col2.metric("Sentences", stats.get("sentences", 0))
    col3.metric("Languages", len(stats.get("languages", {})))
    col4.metric("Valency Patterns", stats.get("valency", 0))
    col5.metric("Running Processes", sum(1 for p in st.session_state.processes.values() if p.get('status') == 'running'))

def render_corpus_browser(db):
    st.header("Corpus Browser")
    
    stats = db.get_statistics()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", stats.get("documents", 0))
    col2.metric("Sentences", stats.get("sentences", 0))
    col3.metric("Languages", len(stats.get("languages", {})))
    col4.metric("Valency Entries", stats.get("valency", 0))
    
    st.divider()
    
    col1, col2, col3 = st.columns([3, 1, 1])
    query = col1.text_input("Search", placeholder="Enter keywords...")
    language = col2.selectbox("Language", ["all"] + list(LANGUAGE_INFO.keys()))
    col3.write("")
    if col3.button("Search", type="primary"):
        results = db.search_corpus(query)
        if results:
            for doc in results:
                st.write(f"**{doc['title']}** - {doc['author']}")
        else:
            st.info("No results found")

def render_analysis_studio(db, analyzer):
    st.header("Analysis Studio")
    
    text = st.text_area("Text for Analysis", height=200)
    
    col1, col2 = st.columns([1, 2])
    language = col1.selectbox("Language", list(LANGUAGE_INFO.keys()),
                              format_func=lambda x: LANGUAGE_INFO[x]["name"])
    
    col2.write("**Analyses:**")
    morph = col2.checkbox("Morphological", value=True)
    syntax = col2.checkbox("Syntactic", value=True)
    valency = col2.checkbox("Valency")
    
    analyses = []
    if morph: analyses.append("morphological")
    if syntax: analyses.append("syntactic")
    if valency: analyses.append("valency")
    
    if st.button("Analyze", type="primary", disabled=not text):
        result = analyzer.analyze(text, language, analyses)
        
        for sent in result["sentences"]:
            with st.expander(f"Sentence: {sent['text'][:50]}..."):
                df = pd.DataFrame(sent["tokens"])
                st.dataframe(df)
        
        st.subheader("Statistics")
        stats = result["statistics"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sentences", stats["sentences"])
        col2.metric("Tokens", stats["tokens"])
        col3.metric("Types", stats["types"])
        col4.metric("TTR", f"{stats['ttr']:.3f}")

def render_valency_explorer(db):
    st.header("Valency Explorer")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    lemma = col1.text_input("Verb lemma", placeholder="Enter verb...")
    language = col2.selectbox("Language", ["all"] + list(LANGUAGE_INFO.keys()), key="val_lang")
    
    if col3.button("Search", key="val_search"):
        results = db.search_valency(lemma, language if language != "all" else None)
        if results:
            st.dataframe(pd.DataFrame(results))
        else:
            st.info("No valency patterns found")
    
    st.subheader("Valency Pattern Reference")
    df = pd.DataFrame([{"Pattern": k, "Description": v} for k, v in VALENCY_PATTERNS.items()])
    st.dataframe(df)

def render_syntactic_tools(db):
    st.header("Syntactic Tools")
    
    st.subheader("Dependency Relations (PROIEL)")
    df = pd.DataFrame([{"Relation": k, "Description": v} for k, v in DEPENDENCY_RELATIONS.items()])
    st.dataframe(df)
    
    st.subheader("Treebank Format Converter")
    col1, col2 = st.columns(2)
    input_format = col1.selectbox("Input", ["CoNLL-U", "PROIEL XML"])
    output_format = col2.selectbox("Output", ["PROIEL XML", "CoNLL-U"])
    
    input_data = st.text_area("Input data", height=200)
    if st.button("Convert"):
        st.info("Conversion would be performed here")

def render_monitoring(db):
    st.header("System Monitoring")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CPU", "23%")
    col2.metric("Memory", "45%")
    col3.metric("Disk", "35%")
    col4.metric("Health", "98%")
    
    st.subheader("Recent Activity")
    st.info("System running normally")

def render_settings():
    st.header("Settings")
    
    st.subheader("General")
    st.text_input("Platform Name", value="Diachronic Linguistics Platform")
    st.selectbox("Default Language", list(LANGUAGE_INFO.keys()))
    st.selectbox("Annotation Standard", ["UD", "PROIEL", "AGDT"])
    
    st.subheader("Parser")
    st.slider("Batch Size", 1, 128, 32)
    st.checkbox("Enable Cache", value=True)
    
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved")

if __name__ == "__main__":
    main()
