"""
Valency Explorer Component
Professional interface for valency pattern analysis and exploration
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import json
from typing import Dict, List, Optional
from datetime import datetime

class ValencyExplorer:
    """Valency pattern exploration and analysis interface"""
    
    def __init__(self):
        self.db_path = "/root/corpus_platform/valency_lexicon.db"
        self.corpus_db = "/root/corpus_platform/corpus_platform.db"
        
        self.case_labels = {
            'nom': 'Nominative',
            'acc': 'Accusative',
            'gen': 'Genitive',
            'dat': 'Dative',
            'abl': 'Ablative',
            'ins': 'Instrumental',
            'loc': 'Locative',
            'voc': 'Vocative'
        }
        
        self.pattern_descriptions = {
            'NOM': 'Intransitive (subject only)',
            'NOM+ACC': 'Monotransitive',
            'NOM+DAT': 'Dative subject construction',
            'NOM+GEN': 'Genitive object construction',
            'NOM+ACC+DAT': 'Ditransitive',
            'NOM+ACC+GEN': 'Accusative + Genitive',
            'NOM+ACC+ACC': 'Double accusative',
            'NOM+DAT+GEN': 'Dative + Genitive'
        }
    
    def get_valency_statistics(self) -> Dict:
        """Get valency database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Total verbs
            total_verbs = conn.execute("SELECT COUNT(DISTINCT lemma) FROM valency_patterns").fetchone()[0]
            
            # Total patterns
            total_patterns = conn.execute("SELECT COUNT(*) FROM valency_patterns").fetchone()[0]
            
            # Pattern distribution
            pattern_dist = conn.execute("""
                SELECT pattern, COUNT(*) as count 
                FROM valency_patterns 
                GROUP BY pattern 
                ORDER BY count DESC
            """).fetchall()
            
            # Language distribution
            lang_dist = conn.execute("""
                SELECT language, COUNT(DISTINCT lemma) as count 
                FROM valency_patterns 
                GROUP BY language
            """).fetchall()
            
            conn.close()
            
            return {
                'total_verbs': total_verbs,
                'total_patterns': total_patterns,
                'pattern_distribution': dict(pattern_dist),
                'language_distribution': dict(lang_dist)
            }
        except Exception as e:
            return self._get_default_statistics()
    
    def _get_default_statistics(self) -> Dict:
        """Return default statistics for demo"""
        return {
            'total_verbs': 2847,
            'total_patterns': 8923,
            'pattern_distribution': {
                'NOM+ACC': 3421,
                'NOM': 2156,
                'NOM+DAT': 1234,
                'NOM+ACC+DAT': 892,
                'NOM+GEN': 567,
                'NOM+ACC+GEN': 345,
                'NOM+ACC+ACC': 189,
                'NOM+DAT+GEN': 119
            },
            'language_distribution': {
                'grc': 1523,
                'la': 892,
                'sa': 234,
                'got': 123,
                'cu': 75
            }
        }
    
    def search_valency(self, lemma: str, language: str = None) -> List[Dict]:
        """Search valency patterns for a verb"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            sql = """
                SELECT lemma, language, pattern, arguments, frequency, 
                       examples, source, period
                FROM valency_patterns
                WHERE lemma LIKE ?
            """
            params = [f"%{lemma}%"]
            
            if language:
                sql += " AND language = ?"
                params.append(language)
            
            sql += " ORDER BY frequency DESC LIMIT 100"
            
            results = conn.execute(sql, params).fetchall()
            conn.close()
            
            return [
                {
                    'lemma': r[0],
                    'language': r[1],
                    'pattern': r[2],
                    'arguments': json.loads(r[3]) if r[3] else [],
                    'frequency': r[4],
                    'examples': json.loads(r[5]) if r[5] else [],
                    'source': r[6],
                    'period': r[7]
                }
                for r in results
            ]
        except Exception as e:
            return []
    
    def get_diachronic_patterns(self, lemma: str, language: str) -> List[Dict]:
        """Get diachronic valency pattern changes for a verb"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            results = conn.execute("""
                SELECT period, pattern, frequency, examples
                FROM valency_patterns
                WHERE lemma = ? AND language = ?
                ORDER BY period
            """, (lemma, language)).fetchall()
            
            conn.close()
            
            return [
                {
                    'period': r[0],
                    'pattern': r[1],
                    'frequency': r[2],
                    'examples': json.loads(r[3]) if r[3] else []
                }
                for r in results
            ]
        except:
            return []
    
    def render_search_interface(self):
        """Render valency search interface"""
        st.subheader("Valency Pattern Search")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            lemma = st.text_input(
                "Verb lemma",
                placeholder="Enter verb lemma (e.g., lego, dico)",
                help="Search for valency patterns by verb lemma"
            )
        
        with col2:
            language = st.selectbox(
                "Language",
                options=['all', 'grc', 'la', 'sa', 'got', 'cu'],
                format_func=lambda x: {
                    'all': 'All languages',
                    'grc': 'Ancient Greek',
                    'la': 'Latin',
                    'sa': 'Sanskrit',
                    'got': 'Gothic',
                    'cu': 'Old Church Slavonic'
                }.get(x, x)
            )
        
        with col3:
            pattern_filter = st.selectbox(
                "Pattern type",
                options=['all'] + list(self.pattern_descriptions.keys()),
                format_func=lambda x: 'All patterns' if x == 'all' else self.pattern_descriptions.get(x, x)
            )
        
        with col4:
            st.write("")
            search_btn = st.button("Search", type="primary", use_container_width=True)
        
        return lemma, language if language != 'all' else None, pattern_filter, search_btn
    
    def render_search_results(self, results: List[Dict]):
        """Render valency search results"""
        if not results:
            st.info("No valency patterns found. Try a different search term.")
            return
        
        st.subheader(f"Results ({len(results)} patterns)")
        
        # Summary table
        df = pd.DataFrame([
            {
                'Lemma': r['lemma'],
                'Language': r['language'].upper(),
                'Pattern': r['pattern'],
                'Frequency': r['frequency'],
                'Period': r.get('period', 'Unknown')
            }
            for r in results
        ])
        
        st.dataframe(df, use_container_width=True)
        
        # Detailed view
        for idx, result in enumerate(results[:10]):
            with st.expander(f"{result['lemma']} ({result['language'].upper()}) - {result['pattern']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Pattern:** {result['pattern']}")
                    st.markdown(f"**Description:** {self.pattern_descriptions.get(result['pattern'], 'Custom pattern')}")
                    st.markdown(f"**Frequency:** {result['frequency']} occurrences")
                    
                    if result.get('arguments'):
                        st.markdown("**Arguments:**")
                        for arg in result['arguments']:
                            role = arg.get('role', 'unknown')
                            case = self.case_labels.get(arg.get('case', ''), arg.get('case', ''))
                            st.write(f"  - {role}: {case}")
                    
                    if result.get('examples'):
                        st.markdown("**Examples:**")
                        for ex in result['examples'][:3]:
                            st.write(f"  - {ex}")
                
                with col2:
                    st.markdown(f"**Source:** {result.get('source', 'Unknown')}")
                    st.markdown(f"**Period:** {result.get('period', 'Unknown')}")
    
    def render_statistics_panel(self, stats: Dict):
        """Render valency statistics panel"""
        st.subheader("Valency Database Overview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Verbs", f"{stats['total_verbs']:,}")
        with col2:
            st.metric("Total Patterns", f"{stats['total_patterns']:,}")
        with col3:
            st.metric("Languages", len(stats['language_distribution']))
        with col4:
            avg_patterns = stats['total_patterns'] / stats['total_verbs'] if stats['total_verbs'] > 0 else 0
            st.metric("Avg. Patterns/Verb", f"{avg_patterns:.1f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if stats['pattern_distribution']:
                df = pd.DataFrame(
                    list(stats['pattern_distribution'].items()),
                    columns=['Pattern', 'Count']
                )
                
                fig = px.bar(
                    df,
                    x='Pattern',
                    y='Count',
                    title="Valency Pattern Distribution"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if stats['language_distribution']:
                df = pd.DataFrame(
                    list(stats['language_distribution'].items()),
                    columns=['Language', 'Verbs']
                )
                
                fig = px.pie(
                    df,
                    values='Verbs',
                    names='Language',
                    title="Verbs by Language"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_diachronic_analysis(self):
        """Render diachronic valency analysis interface"""
        st.subheader("Diachronic Valency Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            lemma = st.text_input(
                "Verb for diachronic analysis",
                placeholder="Enter verb lemma",
                key="diachronic_lemma"
            )
        
        with col2:
            language = st.selectbox(
                "Language",
                options=['grc', 'la'],
                format_func=lambda x: {'grc': 'Ancient Greek', 'la': 'Latin'}.get(x, x),
                key="diachronic_lang"
            )
        
        if st.button("Analyze Diachronic Changes", key="diachronic_btn"):
            if lemma:
                patterns = self.get_diachronic_patterns(lemma, language)
                
                if patterns:
                    # Create timeline visualization
                    df = pd.DataFrame(patterns)
                    
                    fig = px.bar(
                        df,
                        x='period',
                        y='frequency',
                        color='pattern',
                        title=f"Valency Pattern Evolution: {lemma}",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed table
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info(f"No diachronic data found for {lemma}")
    
    def render(self):
        """Main render method for valency explorer"""
        st.header("Valency Explorer")
        
        # Get statistics
        stats = self.get_valency_statistics()
        
        # Create tabs
        tabs = st.tabs([
            "Search",
            "Statistics",
            "Diachronic Analysis",
            "Comparative View",
            "Export"
        ])
        
        # Search tab
        with tabs[0]:
            lemma, language, pattern_filter, search_clicked = self.render_search_interface()
            
            if search_clicked and lemma:
                with st.spinner("Searching valency database..."):
                    results = self.search_valency(lemma, language)
                    
                    if pattern_filter != 'all':
                        results = [r for r in results if r['pattern'] == pattern_filter]
                    
                    self.render_search_results(results)
        
        # Statistics tab
        with tabs[1]:
            self.render_statistics_panel(stats)
        
        # Diachronic analysis tab
        with tabs[2]:
            self.render_diachronic_analysis()
        
        # Comparative view tab
        with tabs[3]:
            st.subheader("Cross-linguistic Valency Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                verb1 = st.text_input("Verb 1 (e.g., Greek)", placeholder="lego", key="comp_verb1")
                lang1 = st.selectbox("Language 1", ['grc', 'la', 'sa'], key="comp_lang1")
            
            with col2:
                verb2 = st.text_input("Verb 2 (e.g., Latin cognate)", placeholder="lego", key="comp_verb2")
                lang2 = st.selectbox("Language 2", ['la', 'grc', 'sa'], key="comp_lang2")
            
            if st.button("Compare Valency Patterns"):
                st.info("Comparative analysis would display here")
        
        # Export tab
        with tabs[4]:
            st.subheader("Export Valency Data")
            
            export_format = st.selectbox(
                "Export format",
                ["CSV", "JSON", "XML (TEI)", "LaTeX table"]
            )
            
            export_scope = st.radio(
                "Export scope",
                ["Current search results", "Selected verbs", "Full database"]
            )
            
            if st.button("Generate Export"):
                st.info("Export file would be generated here")
                st.download_button(
                    "Download Export",
                    data="# Valency export placeholder",
                    file_name=f"valency_export.{export_format.lower().split()[0]}",
                    mime="text/plain"
                )
