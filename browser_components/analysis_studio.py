"""
Analysis Studio Component
Professional linguistic analysis interface with full metadata support
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
import networkx as nx
from pyvis.network import Network
import tempfile
import os

class AnalysisStudio:
    """Comprehensive linguistic analysis interface"""
    
    def __init__(self):
        self.parser_api = "http://localhost:5001"
        self.annotation_standards = {
            'UD': 'Universal Dependencies',
            'PROIEL': 'PROIEL Treebank',
            'PERSEUS': 'Perseus Ancient Greek and Latin',
            'AGDT': 'Ancient Greek Dependency Treebank'
        }
        
        self.analysis_types = {
            'morphological': 'Morphological Analysis',
            'syntactic': 'Syntactic Parsing',
            'valency': 'Valency Extraction',
            'semantic': 'Semantic Analysis',
            'etymological': 'Etymology Tracking',
            'prosodic': 'Prosodic Analysis',
            'stylometric': 'Stylometric Features'
        }
    
    def perform_analysis(self, text: str, language: str, analyses: List[str]) -> Dict:
        """Perform comprehensive linguistic analysis"""
        try:
            # Call parser API
            response = requests.post(
                f"{self.parser_api}/analyze",
                json={
                    'text': text,
                    'language': language,
                    'analyses': analyses,
                    'annotation_standard': st.session_state.get('annotation_standard', 'UD')
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"Analysis failed with status {response.status_code}"}
                
        except requests.exceptions.Timeout:
            return {'error': 'Analysis timeout - text may be too long'}
        except Exception as e:
            return {'error': str(e)}
    
    def render_input_section(self) -> Tuple[str, str, List[str]]:
        """Render the input section for analysis"""
        # Text input area
        text_input = st.text_area(
            "Text for Analysis",
            height=250,
            placeholder="Enter text in Ancient Greek, Latin, Sanskrit, Gothic, or other historical languages...",
            help="Maximum 10,000 characters for optimal performance"
        )
        
        # Analysis configuration
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Language selection
            language = st.selectbox(
                "Language",
                options=[
                    'auto',
                    'grc',  # Ancient Greek
                    'la',   # Latin
                    'sa',   # Sanskrit
                    'got',  # Gothic
                    'cop',  # Coptic
                    'cu',   # Old Church Slavonic
                    'xcl',  # Classical Armenian
                    'orv',  # Old Russian
                    'ang',  # Old English
                ],
                format_func=lambda x: {
                    'auto': 'Auto-detect',
                    'grc': 'Ancient Greek',
                    'la': 'Latin',
                    'sa': 'Sanskrit',
                    'got': 'Gothic',
                    'cop': 'Coptic',
                    'cu': 'Old Church Slavonic',
                    'xcl': 'Classical Armenian',
                    'orv': 'Old Russian',
                    'ang': 'Old English'
                }.get(x, x)
            )
            
            # Annotation standard
            annotation_standard = st.selectbox(
                "Annotation Standard",
                options=list(self.annotation_standards.keys()),
                format_func=lambda x: self.annotation_standards[x]
            )
            st.session_state.annotation_standard = annotation_standard
        
        with col2:
            # Analysis types selection
            st.markdown("**Select Analyses:**")
            
            col_a, col_b = st.columns(2)
            selected_analyses = []
            
            with col_a:
                for key in ['morphological', 'syntactic', 'valency', 'semantic']:
                    if st.checkbox(self.analysis_types[key], value=key in ['morphological', 'syntactic']):
                        selected_analyses.append(key)
            
            with col_b:
                for key in ['etymological', 'prosodic', 'stylometric']:
                    if st.checkbox(self.analysis_types[key]):
                        selected_analyses.append(key)
        
        return text_input, language, selected_analyses
    
    def render_morphological_analysis(self, data: Dict):
        """Render morphological analysis results"""
        st.subheader("Morphological Analysis")
        
        for sent_idx, sentence in enumerate(data.get('sentences', [])):
            with st.expander(f"Sentence {sent_idx + 1}: {sentence.get('text', '')[:80]}..."):
                # Create morphological data table
                morph_data = []
                
                for token in sentence.get('tokens', []):
                    morph_features = token.get('morphology', {})
                    
                    morph_data.append({
                        'Form': token.get('form', ''),
                        'Lemma': token.get('lemma', ''),
                        'POS': token.get('pos', ''),
                        'Case': morph_features.get('Case', '—'),
                        'Number': morph_features.get('Number', '—'),
                        'Gender': morph_features.get('Gender', '—'),
                        'Tense': morph_features.get('Tense', '—'),
                        'Mood': morph_features.get('Mood', '—'),
                        'Voice': morph_features.get('Voice', '—'),
                        'Person': morph_features.get('Person', '—'),
                        'Degree': morph_features.get('Degree', '—')
                    })
                
                if morph_data:
                    df = pd.DataFrame(morph_data)
                    # Remove columns that are all dashes
                    df = df.loc[:, ~(df == '—').all()]
                    st.dataframe(df, use_container_width=True)
    
    def render_syntactic_analysis(self, data: Dict):
        """Render syntactic analysis with dependency visualization"""
        st.subheader("Syntactic Analysis")
        
        for sent_idx, sentence in enumerate(data.get('sentences', [])):
            with st.expander(f"Sentence {sent_idx + 1}: Dependencies"):
                # Dependency table
                dep_data = []
                tokens = sentence.get('tokens', [])
                
                for idx, token in enumerate(tokens, 1):
                    dep_data.append({
                        'ID': idx,
                        'Form': token.get('form', ''),
                        'Lemma': token.get('lemma', ''),
                        'POS': token.get('pos', ''),
                        'Head': token.get('head', 0),
                        'Relation': token.get('deprel', ''),
                        'Enhanced': token.get('deps', '')
                    })
                
                if dep_data:
                    df = pd.DataFrame(dep_data)
                    st.dataframe(df, use_container_width=True)
                
                # Dependency tree visualization
                if st.checkbox(f"Show dependency tree", key=f"tree_{sent_idx}"):
                    self.render_dependency_tree(sentence)
    
    def render_dependency_tree(self, sentence: Dict):
        """Render dependency tree visualization"""
        try:
            # Create directed graph
            G = nx.DiGraph()
            tokens = sentence.get('tokens', [])
            
            # Add nodes
            for idx, token in enumerate(tokens, 1):
                G.add_node(idx, label=f"{token['form']}\n{token['pos']}")
            
            # Add root
            G.add_node(0, label="ROOT")
            
            # Add edges
            for idx, token in enumerate(tokens, 1):
                head = token.get('head', 0)
                deprel = token.get('deprel', '')
                G.add_edge(head, idx, label=deprel)
            
            # Create visualization
            net = Network(height="400px", width="100%", directed=True)
            net.from_nx(G)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as tmp:
                net.save_graph(tmp.name)
                
                # Read and display
                with open(tmp.name, 'r') as f:
                    html = f.read()
                    st.components.v1.html(html, height=450)
            
        except Exception as e:
            st.error(f"Could not render dependency tree: {str(e)}")
    
    def render_valency_analysis(self, data: Dict):
        """Render valency analysis results"""
        st.subheader("Valency Analysis")
        
        valency_patterns = []
        
        for sentence in data.get('sentences', []):
            for valency in sentence.get('valency_patterns', []):
                valency_patterns.append({
                    'Verb': valency.get('verb', ''),
                    'Lemma': valency.get('lemma', ''),
                    'Pattern': valency.get('pattern', ''),
                    'Arguments': ', '.join([
                        f"{arg['role']}:{arg['form']}" 
                        for arg in valency.get('arguments', [])
                    ]),
                    'Frequency': valency.get('frequency', 1)
                })
        
        if valency_patterns:
            df = pd.DataFrame(valency_patterns)
            st.dataframe(df, use_container_width=True)
            
            # Pattern distribution
            pattern_counts = df['Pattern'].value_counts()
            
            fig = px.bar(
                x=pattern_counts.index,
                y=pattern_counts.values,
                title="Valency Pattern Distribution",
                labels={'x': 'Pattern', 'y': 'Occurrences'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No verbal valency patterns found in the text.")
    
    def render_analysis_statistics(self, data: Dict):
        """Render overall analysis statistics"""
        stats = data.get('statistics', {})
        
        if not stats:
            return
        
        st.subheader("Text Statistics")
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sentences", stats.get('sentence_count', 0))
            st.metric("Tokens", stats.get('token_count', 0))
        
        with col2:
            st.metric("Types", stats.get('type_count', 0))
            st.metric("Lemmas", stats.get('lemma_count', 0))
        
        with col3:
            st.metric("TTR", f"{stats.get('type_token_ratio', 0):.3f}")
            st.metric("Avg. Sent. Length", f"{stats.get('avg_sentence_length', 0):.1f}")
        
        with col4:
            st.metric("Avg. Word Length", f"{stats.get('avg_word_length', 0):.1f}")
            complexity = stats.get('morphological_complexity', 0)
            st.metric("Morph. Complexity", f"{complexity:.2f}")
        
        # POS distribution
        if stats.get('pos_distribution'):
            pos_data = pd.DataFrame(
                list(stats['pos_distribution'].items()),
                columns=['POS', 'Count']
            )
            
            fig = px.pie(
                pos_data,
                values='Count',
                names='POS',
                title="Part of Speech Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render(self):
        """Main render method for analysis studio"""
        st.header("Analysis Studio")
        
        # Input section
        text_input, language, selected_analyses = self.render_input_section()
        
        # Analysis button
        if st.button("Analyze Text", type="primary", use_container_width=True, disabled=not text_input):
            if not selected_analyses:
                st.warning("Please select at least one analysis type.")
                return
            
            # Perform analysis
            with st.spinner("Performing linguistic analysis..."):
                start_time = datetime.now()
                results = self.perform_analysis(text_input, language, selected_analyses)
                end_time = datetime.now()
                
                # Check for errors
                if 'error' in results:
                    st.error(f"Analysis error: {results['error']}")
                    return
                
                # Display results
                st.success(f"Analysis completed in {(end_time - start_time).total_seconds():.2f} seconds")
                
                # Create tabs for different analyses
                available_tabs = []
                if 'morphological' in selected_analyses:
                    available_tabs.append("Morphology")
                if 'syntactic' in selected_analyses:
                    available_tabs.append("Syntax")
                if 'valency' in selected_analyses:
                    available_tabs.append("Valency")
                available_tabs.extend(["Statistics", "Export"])
                
                tabs = st.tabs(available_tabs)
                tab_idx = 0
                
                # Morphological analysis tab
                if 'morphological' in selected_analyses:
                    with tabs[tab_idx]:
                        self.render_morphological_analysis(results)
                    tab_idx += 1
                
                # Syntactic analysis tab
                if 'syntactic' in selected_analyses:
                    with tabs[tab_idx]:
                        self.render_syntactic_analysis(results)
                    tab_idx += 1
                
                # Valency analysis tab
                if 'valency' in selected_analyses:
                    with tabs[tab_idx]:
                        self.render_valency_analysis(results)
                    tab_idx += 1
                
                # Statistics tab
                with tabs[tab_idx]:
                    self.render_analysis_statistics(results)
                tab_idx += 1
                
                # Export tab
                with tabs[tab_idx]:
                    st.subheader("Export Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        export_format = st.selectbox(
                            "Export Format",
                            ["JSON", "CoNLL-U", "PROIEL XML", "CSV", "TEI XML"]
                        )
                    
                    with col2:
                        st.write("")  # Spacer
                        if st.button("Generate Export", use_container_width=True):
                            # Generate export data
                            if export_format == "JSON":
                                export_data = json.dumps(results, indent=2, ensure_ascii=False)
                                st.download_button(
                                    "Download JSON",
                                    data=export_data,
                                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                    
                    with col3:
                        st.write("")  # Spacer
                        if st.button("Save to Project", use_container_width=True):
                            if 'analysis_history' not in st.session_state:
                                st.session_state.analysis_history = []
                            
                            st.session_state.analysis_history.append({
                                'timestamp': datetime.now().isoformat(),
                                'text_preview': text_input[:100] + "...",
                                'language': results.get('language', language),
                                'analyses': selected_analyses,
                                'results': results
                            })
                            st.success("Analysis saved to project history")
