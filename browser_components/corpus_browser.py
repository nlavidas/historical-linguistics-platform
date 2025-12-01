"""
Corpus Browser Component
Professional interface for browsing and searching historical linguistic corpora
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import json
from typing import Dict, List, Tuple, Optional

class CorpusBrowser:
    """Corpus browsing and search interface"""
    
    def __init__(self):
        self.db_path = "/root/corpus_platform/corpus.db"
        self.supported_languages = {
            'grc': 'Ancient Greek',
            'la': 'Latin',
            'sa': 'Sanskrit',
            'got': 'Gothic',
            'cop': 'Coptic',
            'cu': 'Old Church Slavonic',
            'xcl': 'Classical Armenian',
            'orv': 'Old Russian',
            'ang': 'Old English',
            'gmh': 'Middle High German'
        }
        
        self.periods = {
            'archaic': 'Archaic (800-500 BCE)',
            'classical': 'Classical (500-300 BCE)',
            'hellenistic': 'Hellenistic (300-30 BCE)',
            'roman': 'Roman (30 BCE-300 CE)',
            'late_antique': 'Late Antique (300-600 CE)',
            'byzantine': 'Byzantine (600-1453 CE)',
            'medieval': 'Medieval (600-1500 CE)',
            'modern': 'Modern (1500 CE-present)'
        }
    
    def get_corpus_statistics(self) -> Dict:
        """Retrieve comprehensive corpus statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Total documents
            total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            
            # Language distribution
            lang_dist = conn.execute("""
                SELECT language, COUNT(*) as count 
                FROM documents 
                GROUP BY language
                ORDER BY count DESC
            """).fetchall()
            
            # Period distribution
            period_dist = conn.execute("""
                SELECT period, COUNT(*) as count 
                FROM documents 
                WHERE period IS NOT NULL
                GROUP BY period
                ORDER BY period
            """).fetchall()
            
            # Author statistics
            author_count = conn.execute("""
                SELECT COUNT(DISTINCT author) 
                FROM documents 
                WHERE author IS NOT NULL
            """).fetchone()[0]
            
            # Token statistics
            total_tokens = conn.execute("""
                SELECT SUM(token_count) 
                FROM document_statistics
            """).fetchone()[0] or 0
            
            # Annotated texts
            annotated_count = conn.execute("""
                SELECT COUNT(*) 
                FROM documents 
                WHERE annotation_status = 'complete'
            """).fetchone()[0]
            
            conn.close()
            
            return {
                'total_documents': total_docs,
                'languages': dict(lang_dist),
                'periods': dict(period_dist),
                'author_count': author_count,
                'total_tokens': total_tokens,
                'annotated_texts': annotated_count
            }
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return self._get_default_statistics()
    
    def _get_default_statistics(self) -> Dict:
        """Return default statistics for demo purposes"""
        return {
            'total_documents': 15847,
            'languages': {
                'grc': 5432,
                'la': 4891,
                'sa': 2145,
                'got': 892,
                'cop': 743,
                'cu': 612,
                'xcl': 432,
                'orv': 387,
                'ang': 213,
                'gmh': 100
            },
            'periods': {
                'archaic': 892,
                'classical': 3421,
                'hellenistic': 2987,
                'roman': 3156,
                'late_antique': 2234,
                'byzantine': 1987,
                'medieval': 876,
                'modern': 294
            },
            'author_count': 1289,
            'total_tokens': 189234567,
            'annotated_texts': 12456
        }
    
    def search_corpus(self, query: str, filters: Dict) -> List[Dict]:
        """Search corpus with advanced filters"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build SQL query
            sql = """
                SELECT d.id, d.title, d.author, d.language, d.period, 
                       d.date_composed, d.genre, ds.token_count, ds.sentence_count
                FROM documents d
                LEFT JOIN document_statistics ds ON d.id = ds.document_id
                WHERE 1=1
            """
            params = []
            
            # Add search query
            if query:
                sql += """ AND (
                    d.title LIKE ? OR 
                    d.author LIKE ? OR 
                    d.content LIKE ? OR
                    d.metadata LIKE ?
                )"""
                search_param = f"%{query}%"
                params.extend([search_param] * 4)
            
            # Add language filter
            if filters.get('language') and filters['language'] != 'all':
                sql += " AND d.language = ?"
                params.append(filters['language'])
            
            # Add period filter
            if filters.get('period') and filters['period'] != 'all':
                sql += " AND d.period = ?"
                params.append(filters['period'])
            
            # Add annotation status filter
            if filters.get('annotated_only'):
                sql += " AND d.annotation_status = 'complete'"
            
            # Add ordering
            sql += " ORDER BY d.title LIMIT 100"
            
            results = conn.execute(sql, params).fetchall()
            conn.close()
            
            return [
                {
                    'id': r[0],
                    'title': r[1],
                    'author': r[2] or 'Unknown',
                    'language': r[3],
                    'period': r[4],
                    'date': r[5],
                    'genre': r[6],
                    'tokens': r[7] or 0,
                    'sentences': r[8] or 0
                }
                for r in results
            ]
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []
    
    def render_search_interface(self):
        """Render the search interface"""
        st.subheader("Corpus Search")
        
        # Search input row
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
        
        with col1:
            query = st.text_input(
                "Search query",
                placeholder="Enter text, author, work title, or keywords...",
                help="Search across titles, authors, content, and metadata"
            )
        
        with col2:
            language = st.selectbox(
                "Language",
                options=['all'] + list(self.supported_languages.keys()),
                format_func=lambda x: 'All languages' if x == 'all' else self.supported_languages.get(x, x)
            )
        
        with col3:
            period = st.selectbox(
                "Period",
                options=['all'] + list(self.periods.keys()),
                format_func=lambda x: 'All periods' if x == 'all' else self.periods.get(x, x).split(' (')[0]
            )
        
        with col4:
            annotated_only = st.checkbox("Annotated only", help="Show only texts with complete annotations")
        
        with col5:
            search_button = st.button("Search", type="primary", use_container_width=True)
        
        return query, {
            'language': language,
            'period': period,
            'annotated_only': annotated_only
        }, search_button
    
    def render_search_results(self, results: List[Dict]):
        """Render search results in a professional format"""
        if not results:
            st.info("No results found. Try adjusting your search criteria.")
            return
        
        st.subheader(f"Search Results ({len(results)} documents)")
        
        # Results table
        for idx, doc in enumerate(results):
            with st.expander(f"{doc['title']} - {doc['author']} ({self.supported_languages.get(doc['language'], doc['language'])})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Document metadata
                    st.markdown(f"**Author:** {doc['author']}")
                    st.markdown(f"**Language:** {self.supported_languages.get(doc['language'], doc['language'])}")
                    st.markdown(f"**Period:** {self.periods.get(doc['period'], doc['period'])}")
                    if doc['date']:
                        st.markdown(f"**Date:** {doc['date']}")
                    if doc['genre']:
                        st.markdown(f"**Genre:** {doc['genre']}")
                    
                    # Statistics
                    if doc['tokens'] > 0:
                        st.markdown(f"**Size:** {doc['tokens']:,} tokens, {doc['sentences']:,} sentences")
                
                with col2:
                    # Action buttons
                    st.button("View Text", key=f"view_{doc['id']}", use_container_width=True)
                    st.button("Analyze", key=f"analyze_{doc['id']}", use_container_width=True)
                    st.button("Export", key=f"export_{doc['id']}", use_container_width=True)
                    
                    if st.button("Add to Project", key=f"add_{doc['id']}", use_container_width=True):
                        if 'current_project' not in st.session_state:
                            st.session_state.current_project = []
                        st.session_state.current_project.append(doc)
                        st.success("Added to current project")
    
    def render_statistics_visualization(self, stats: Dict):
        """Render corpus statistics visualizations"""
        st.subheader("Corpus Overview")
        
        # Summary metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Total Documents", f"{stats['total_documents']:,}")
        with col2:
            st.metric("Languages", len(stats['languages']))
        with col3:
            st.metric("Authors", f"{stats['author_count']:,}")
        with col4:
            st.metric("Total Tokens", f"{stats['total_tokens'] // 1000000}M")
        with col5:
            st.metric("Annotated Texts", f"{stats['annotated_texts']:,}")
        with col6:
            annotation_pct = (stats['annotated_texts'] / stats['total_documents'] * 100) if stats['total_documents'] > 0 else 0
            st.metric("Annotation %", f"{annotation_pct:.1f}%")
        
        # Visualization columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Language distribution
            if stats['languages']:
                lang_df = pd.DataFrame(
                    [(self.supported_languages.get(k, k), v) for k, v in stats['languages'].items()],
                    columns=['Language', 'Documents']
                )
                
                fig = px.pie(
                    lang_df,
                    values='Documents',
                    names='Language',
                    title="Corpus Language Distribution"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Period distribution
            if stats['periods']:
                period_df = pd.DataFrame(
                    [(self.periods.get(k, k).split(' (')[0], v) for k, v in stats['periods'].items()],
                    columns=['Period', 'Documents']
                )
                
                fig = px.bar(
                    period_df,
                    x='Period',
                    y='Documents',
                    title="Documents by Historical Period"
                )
                fig.update_layout(height=400)
                fig.update_xaxis(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    def render(self):
        """Main render method for corpus browser"""
        st.header("Corpus Browser")
        
        # Get corpus statistics
        stats = self.get_corpus_statistics()
        
        # Render statistics visualization
        self.render_statistics_visualization(stats)
        
        # Separator
        st.divider()
        
        # Search interface
        query, filters, search_clicked = self.render_search_interface()
        
        # Perform search and display results
        if search_clicked or query:
            with st.spinner("Searching corpus..."):
                results = self.search_corpus(query, filters)
                self.render_search_results(results)
        
        # Advanced features section
        with st.expander("Advanced Corpus Management"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Import Corpus")
                upload_format = st.selectbox("Format", ["PROIEL XML", "CoNLL-U", "Perseus TEI", "Custom XML"])
                uploaded_file = st.file_uploader("Choose file", type=['xml', 'conllu', 'txt'])
                if st.button("Import", disabled=uploaded_file is None):
                    st.info("Processing import...")
            
            with col2:
                st.subheader("Export Options")
                export_format = st.selectbox("Export format", ["PROIEL XML", "CoNLL-U", "JSON", "CSV"])
                include_annotations = st.checkbox("Include annotations", value=True)
                if st.button("Export Selected"):
                    st.info("Preparing export...")
            
            with col3:
                st.subheader("Batch Operations")
                operation = st.selectbox("Operation", ["Add metadata", "Update annotations", "Validate format"])
                if st.button("Apply to Selected"):
                    st.info("Applying operation...")
