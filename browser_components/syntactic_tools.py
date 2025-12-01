"""
Syntactic Tools Component
Professional interface for syntactic analysis with PROIEL and Syntacticus integration
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import json
import requests
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import re

class SyntacticTools:
    """Advanced syntactic analysis tools with treebank integration"""
    
    def __init__(self):
        self.treebank_formats = {
            'PROIEL': 'PROIEL Treebank Format',
            'UD': 'Universal Dependencies',
            'AGDT': 'Ancient Greek Dependency Treebank',
            'PDT': 'Prague Dependency Treebank',
            'Syntacticus': 'Syntacticus Enhanced Format'
        }
        
        self.dependency_relations = {
            # PROIEL specific relations
            'pred': 'predicate',
            'sub': 'subject',
            'obj': 'object',
            'obl': 'oblique',
            'atr': 'attribute',
            'atv': 'attributive verb',
            'adv': 'adverbial',
            'ag': 'agent',
            'comp': 'complement',
            'apos': 'apposition',
            'aux': 'auxiliary',
            'expl': 'expletive',
            'parpred': 'parenthetical predication',
            'vocr': 'vocative raised',
            'narg': 'non-argument',
            'nonsub': 'non-subject',
            'xobj': 'external object',
            'xadv': 'external adverbial'
        }
        
        self.syntacticus_api = "http://syntacticus.org/api"
        self.proiel_api = "http://localhost:5002/proiel"
    
    def load_treebank_data(self, treebank_id: str, format: str) -> Dict:
        """Load treebank data from various sources"""
        try:
            if format == 'PROIEL':
                return self.load_proiel_treebank(treebank_id)
            elif format == 'Syntacticus':
                return self.load_syntacticus_data(treebank_id)
            else:
                # Load from local database
                return self.load_local_treebank(treebank_id, format)
        except Exception as e:
            st.error(f"Error loading treebank: {str(e)}")
            return {}
    
    def load_proiel_treebank(self, treebank_id: str) -> Dict:
        """Load PROIEL format treebank"""
        try:
            response = requests.get(f"{self.proiel_api}/treebank/{treebank_id}")
            if response.status_code == 200:
                return response.json()
            else:
                return self._get_sample_proiel_data()
        except:
            return self._get_sample_proiel_data()
    
    def _get_sample_proiel_data(self) -> Dict:
        """Return sample PROIEL treebank data"""
        return {
            'id': 'proiel_sample',
            'name': 'PROIEL Sample Treebank',
            'sentences': [
                {
                    'id': 1,
                    'text': 'μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος',
                    'tokens': [
                        {
                            'id': 1,
                            'form': 'μῆνιν',
                            'lemma': 'μῆνις',
                            'pos': 'Nb',
                            'morphology': 'Case=Acc|Gender=Fem|Number=Sing',
                            'head': 2,
                            'relation': 'obj',
                            'empty_token_sort': None,
                            'citation_part': '1.1'
                        },
                        {
                            'id': 2,
                            'form': 'ἄειδε',
                            'lemma': 'ἀείδω',
                            'pos': 'V-',
                            'morphology': 'Mood=Imp|Number=Sing|Person=2|Tense=Pres|VerbForm=Fin|Voice=Act',
                            'head': 0,
                            'relation': 'pred',
                            'empty_token_sort': None,
                            'citation_part': '1.1'
                        }
                    ]
                }
            ]
        }
    
    def parse_conll(self, conll_text: str) -> List[Dict]:
        """Parse CoNLL-U format text"""
        sentences = []
        current_sentence = {'tokens': [], 'text': ''}
        
        for line in conll_text.strip().split('\n'):
            line = line.strip()
            
            if line.startswith('# text = '):
                current_sentence['text'] = line[9:]
            elif line.startswith('#'):
                continue  # Skip other comments
            elif not line:
                # End of sentence
                if current_sentence['tokens']:
                    sentences.append(current_sentence)
                    current_sentence = {'tokens': [], 'text': ''}
            else:
                # Token line
                parts = line.split('\t')
                if len(parts) >= 10 and '-' not in parts[0] and '.' not in parts[0]:
                    token = {
                        'id': int(parts[0]),
                        'form': parts[1],
                        'lemma': parts[2],
                        'upos': parts[3],
                        'xpos': parts[4],
                        'feats': parts[5],
                        'head': int(parts[6]) if parts[6] != '_' else 0,
                        'deprel': parts[7],
                        'deps': parts[8],
                        'misc': parts[9]
                    }
                    current_sentence['tokens'].append(token)
        
        # Don't forget last sentence
        if current_sentence['tokens']:
            sentences.append(current_sentence)
        
        return sentences
    
    def convert_to_proiel(self, sentences: List[Dict]) -> str:
        """Convert parsed sentences to PROIEL XML format"""
        root = ET.Element('proiel')
        source = ET.SubElement(root, 'source')
        div = ET.SubElement(source, 'div')
        
        for sent_idx, sentence in enumerate(sentences):
            sent_elem = ET.SubElement(div, 'sentence', id=str(sent_idx + 1))
            
            for token in sentence.get('tokens', []):
                token_elem = ET.SubElement(sent_elem, 'token', 
                    id=str(token['id']),
                    form=token['form'],
                    lemma=token.get('lemma', ''),
                    part_of_speech=token.get('upos', ''),
                    morphology=token.get('feats', ''),
                    head_id=str(token.get('head', 0)),
                    relation=token.get('deprel', '')
                )
        
        return ET.tostring(root, encoding='unicode')
    
    def render_treebank_browser(self):
        """Render treebank browsing interface"""
        st.subheader("Treebank Browser")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            treebank = st.selectbox(
                "Select Treebank",
                options=[
                    'proiel_grc',
                    'proiel_lat',
                    'proiel_got',
                    'proiel_xcl',
                    'proiel_cu',
                    'syntacticus_grc',
                    'agdt_perseus',
                    'custom'
                ],
                format_func=lambda x: {
                    'proiel_grc': 'PROIEL Greek NT',
                    'proiel_lat': 'PROIEL Latin Vulgate',
                    'proiel_got': 'PROIEL Gothic Bible',
                    'proiel_xcl': 'PROIEL Armenian',
                    'proiel_cu': 'PROIEL Old Church Slavonic',
                    'syntacticus_grc': 'Syntacticus Greek',
                    'agdt_perseus': 'AGDT Perseus Collection',
                    'custom': 'Custom Upload'
                }.get(x, x)
            )
        
        with col2:
            if treebank == 'custom':
                uploaded_file = st.file_uploader(
                    "Upload Treebank",
                    type=['xml', 'conllu', 'txt'],
                    help="Upload PROIEL XML, CoNLL-U, or other treebank formats"
                )
        
        with col3:
            format_type = st.selectbox(
                "Format",
                options=list(self.treebank_formats.keys()),
                format_func=lambda x: self.treebank_formats[x]
            )
        
        return treebank, format_type
    
    def render_dependency_analysis(self, sentences: List[Dict]):
        """Render dependency structure analysis"""
        st.subheader("Dependency Analysis")
        
        # Sentence selector
        if len(sentences) > 1:
            sent_idx = st.selectbox(
                "Select Sentence",
                options=range(len(sentences)),
                format_func=lambda x: f"Sentence {x+1}: {sentences[x].get('text', '')[:50]}..."
            )
        else:
            sent_idx = 0
        
        if sent_idx < len(sentences):
            sentence = sentences[sent_idx]
            
            # Dependency statistics
            col1, col2 = st.columns(2)
            
            with col1:
                # Dependency relation distribution
                dep_counts = {}
                for token in sentence.get('tokens', []):
                    rel = token.get('relation', token.get('deprel', 'unknown'))
                    dep_counts[rel] = dep_counts.get(rel, 0) + 1
                
                if dep_counts:
                    df = pd.DataFrame(
                        list(dep_counts.items()),
                        columns=['Relation', 'Count']
                    )
                    
                    fig = px.bar(
                        df,
                        x='Relation',
                        y='Count',
                        title="Dependency Relations"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Arc length distribution
                arc_lengths = []
                for token in sentence.get('tokens', []):
                    if token.get('head', 0) > 0:
                        arc_length = abs(token['id'] - token['head'])
                        arc_lengths.append(arc_length)
                
                if arc_lengths:
                    fig = px.histogram(
                        x=arc_lengths,
                        title="Dependency Arc Lengths",
                        labels={'x': 'Arc Length', 'y': 'Frequency'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Dependency table
            dep_data = []
            for token in sentence.get('tokens', []):
                dep_data.append({
                    'ID': token.get('id', ''),
                    'Form': token.get('form', ''),
                    'Lemma': token.get('lemma', ''),
                    'POS': token.get('pos', token.get('upos', '')),
                    'Head': token.get('head', 0),
                    'Relation': self.dependency_relations.get(
                        token.get('relation', token.get('deprel', '')),
                        token.get('relation', token.get('deprel', ''))
                    ),
                    'Enhanced': token.get('deps', '')
                })
            
            st.dataframe(pd.DataFrame(dep_data), use_container_width=True)
    
    def render_query_interface(self):
        """Render treebank query interface"""
        st.subheader("Treebank Query")
        
        # Query builder
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query_type = st.radio(
                "Query Type",
                ["Simple", "Advanced", "MQL (Mamba Query Language)"],
                horizontal=True
            )
            
            if query_type == "Simple":
                # Simple query interface
                form_query = st.text_input("Word form contains", placeholder="e.g., λεγ")
                lemma_query = st.text_input("Lemma equals", placeholder="e.g., λέγω")
                pos_query = st.selectbox("Part of speech", ["Any", "Verb", "Noun", "Adj", "Adv", "Prep"])
                dep_query = st.selectbox(
                    "Dependency relation",
                    ["Any"] + list(self.dependency_relations.values())
                )
                
            elif query_type == "Advanced":
                # Advanced query with multiple conditions
                st.text_area(
                    "Query conditions (JSON)",
                    placeholder='{\n  "lemma": "λέγω",\n  "pos": "V",\n  "dependent": {\n    "relation": "obj",\n    "case": "acc"\n  }\n}',
                    height=150
                )
            
            else:  # MQL
                st.text_area(
                    "MQL Query",
                    placeholder="[lemma='λέγω' & pos='V'] >obj [case='acc']",
                    height=100,
                    help="Mamba Query Language for complex syntactic patterns"
                )
        
        with col2:
            st.write("")  # Spacer
            st.write("")
            search_button = st.button("Search", type="primary", use_container_width=True)
            
            # Search scope
            st.selectbox(
                "Search in",
                ["Current treebank", "All Greek", "All treebanks"]
            )
        
        return search_button
    
    def render_statistics_panel(self, treebank_data: Dict):
        """Render treebank statistics panel"""
        st.subheader("Treebank Statistics")
        
        # Calculate statistics
        total_sentences = len(treebank_data.get('sentences', []))
        total_tokens = sum(len(s.get('tokens', [])) for s in treebank_data.get('sentences', []))
        
        # Get unique values
        all_pos = set()
        all_relations = set()
        all_lemmas = set()
        
        for sentence in treebank_data.get('sentences', []):
            for token in sentence.get('tokens', []):
                all_pos.add(token.get('pos', token.get('upos', '')))
                all_relations.add(token.get('relation', token.get('deprel', '')))
                all_lemmas.add(token.get('lemma', ''))
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sentences", f"{total_sentences:,}")
        with col2:
            st.metric("Tokens", f"{total_tokens:,}")
        with col3:
            st.metric("Unique Lemmas", f"{len(all_lemmas):,}")
        with col4:
            avg_sent_length = total_tokens / total_sentences if total_sentences > 0 else 0
            st.metric("Avg. Sent. Length", f"{avg_sent_length:.1f}")
        
        # POS tag distribution
        pos_counts = {}
        for sentence in treebank_data.get('sentences', []):
            for token in sentence.get('tokens', []):
                pos = token.get('pos', token.get('upos', 'Unknown'))
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        if pos_counts:
            col1, col2 = st.columns(2)
            
            with col1:
                df_pos = pd.DataFrame(
                    list(pos_counts.items()),
                    columns=['POS', 'Count']
                ).sort_values('Count', ascending=False).head(10)
                
                fig = px.bar(
                    df_pos,
                    x='POS',
                    y='Count',
                    title="Top 10 POS Tags"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Relation distribution
                rel_counts = {}
                for sentence in treebank_data.get('sentences', []):
                    for token in sentence.get('tokens', []):
                        rel = token.get('relation', token.get('deprel', 'unknown'))
                        rel_counts[rel] = rel_counts.get(rel, 0) + 1
                
                df_rel = pd.DataFrame(
                    list(rel_counts.items()),
                    columns=['Relation', 'Count']
                ).sort_values('Count', ascending=False).head(10)
                
                fig = px.bar(
                    df_rel,
                    x='Relation',
                    y='Count',
                    title="Top 10 Dependency Relations"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render(self):
        """Main render method for syntactic tools"""
        st.header("Syntactic Analysis Tools")
        
        # Tool selection
        tool_tabs = st.tabs([
            "Treebank Browser",
            "Dependency Analysis",
            "Query Builder",
            "Statistics",
            "Format Converter"
        ])
        
        # Treebank Browser
        with tool_tabs[0]:
            treebank_id, format_type = self.render_treebank_browser()
            
            if st.button("Load Treebank"):
                with st.spinner("Loading treebank data..."):
                    treebank_data = self.load_treebank_data(treebank_id, format_type)
                    
                    if treebank_data and 'sentences' in treebank_data:
                        st.success(f"Loaded {len(treebank_data['sentences'])} sentences")
                        
                        # Display sample
                        if st.checkbox("Show sample sentences"):
                            for i, sent in enumerate(treebank_data['sentences'][:5]):
                                st.write(f"{i+1}. {sent.get('text', 'No text available')}")
                        
                        # Store in session state
                        st.session_state.current_treebank = treebank_data
        
        # Dependency Analysis
        with tool_tabs[1]:
            if 'current_treebank' in st.session_state:
                self.render_dependency_analysis(st.session_state.current_treebank.get('sentences', []))
            else:
                st.info("Please load a treebank first in the Treebank Browser tab.")
        
        # Query Builder
        with tool_tabs[2]:
            search_clicked = self.render_query_interface()
            
            if search_clicked:
                with st.spinner("Searching treebank..."):
                    # Perform search (mock results for now)
                    st.info("Search functionality will query loaded treebanks")
                    
                    # Sample results
                    results = [
                        {
                            'sentence': 'μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος',
                            'match': 'ἄειδε',
                            'source': 'Homer, Iliad 1.1'
                        }
                    ]
                    
                    for result in results:
                        with st.expander(f"{result['source']}"):
                            st.write(result['sentence'])
                            st.write(f"Match: **{result['match']}**")
        
        # Statistics
        with tool_tabs[3]:
            if 'current_treebank' in st.session_state:
                self.render_statistics_panel(st.session_state.current_treebank)
            else:
                st.info("Please load a treebank first in the Treebank Browser tab.")
        
        # Format Converter
        with tool_tabs[4]:
            st.subheader("Treebank Format Converter")
            
            col1, col2 = st.columns(2)
            
            with col1:
                input_format = st.selectbox("Input Format", list(self.treebank_formats.keys()))
                input_text = st.text_area("Input Data", height=300, placeholder="Paste your treebank data here...")
            
            with col2:
                output_format = st.selectbox("Output Format", list(self.treebank_formats.keys()))
                
                if st.button("Convert", disabled=not input_text):
                    with st.spinner("Converting..."):
                        # Perform conversion
                        if input_format == "UD" and output_format == "PROIEL":
                            # Parse CoNLL-U and convert to PROIEL
                            sentences = self.parse_conll(input_text)
                            output = self.convert_to_proiel(sentences)
                            
                            st.text_area("Converted Output", value=output, height=300)
                            st.download_button(
                                "Download Converted File",
                                data=output,
                                file_name=f"converted_{output_format.lower()}.xml",
                                mime="application/xml"
                            )
                        else:
                            st.info(f"Conversion from {input_format} to {output_format} coming soon")
