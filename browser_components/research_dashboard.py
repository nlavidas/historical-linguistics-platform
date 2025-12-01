"""
Research Dashboard Component
Professional interface for research progress tracking and analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class ResearchDashboard:
    """Research progress and analytics dashboard"""
    
    def __init__(self):
        self.corpus_db = "/root/corpus_platform/corpus_platform.db"
        self.monitoring_db = "/root/corpus_platform/monitoring.db"
        self.findings_db = "/root/corpus_platform/research_findings.db"
    
    def get_research_metrics(self) -> Dict:
        """Get comprehensive research metrics"""
        try:
            metrics = {}
            
            # Corpus metrics
            conn = sqlite3.connect(self.corpus_db)
            metrics['total_texts'] = conn.execute("SELECT COUNT(*) FROM corpus_items").fetchone()[0]
            metrics['total_words'] = conn.execute("SELECT SUM(word_count) FROM corpus_items").fetchone()[0] or 0
            metrics['annotated_texts'] = conn.execute(
                "SELECT COUNT(*) FROM corpus_items WHERE annotation_status = 'complete'"
            ).fetchone()[0]
            conn.close()
            
            # Monitoring metrics
            conn = sqlite3.connect(self.monitoring_db)
            metrics['analyses_today'] = conn.execute("""
                SELECT COUNT(*) FROM performance_metrics 
                WHERE date(timestamp) = date('now')
            """).fetchone()[0]
            metrics['total_analyses'] = conn.execute(
                "SELECT COUNT(*) FROM performance_metrics"
            ).fetchone()[0]
            conn.close()
            
            return metrics
        except Exception as e:
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict:
        """Return default metrics for demo"""
        return {
            'total_texts': 15847,
            'total_words': 189234567,
            'annotated_texts': 12456,
            'analyses_today': 247,
            'total_analyses': 89234
        }
    
    def get_activity_timeline(self, days: int = 30) -> List[Dict]:
        """Get research activity timeline"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            
            results = conn.execute("""
                SELECT date(timestamp) as date, 
                       COUNT(*) as count,
                       AVG(execution_time_ms) as avg_time
                FROM performance_metrics
                WHERE timestamp >= date('now', ?)
                GROUP BY date(timestamp)
                ORDER BY date
            """, (f'-{days} days',)).fetchall()
            
            conn.close()
            
            return [
                {'date': r[0], 'count': r[1], 'avg_time': r[2]}
                for r in results
            ]
        except:
            # Generate sample data
            dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
            return [
                {'date': d, 'count': 50 + (hash(d) % 100), 'avg_time': 200 + (hash(d) % 300)}
                for d in reversed(dates)
            ]
    
    def get_language_progress(self) -> Dict:
        """Get progress by language"""
        try:
            conn = sqlite3.connect(self.corpus_db)
            
            results = conn.execute("""
                SELECT language, 
                       COUNT(*) as total,
                       SUM(CASE WHEN annotation_status = 'complete' THEN 1 ELSE 0 END) as annotated,
                       SUM(word_count) as words
                FROM corpus_items
                GROUP BY language
            """).fetchall()
            
            conn.close()
            
            return {
                r[0]: {'total': r[1], 'annotated': r[2], 'words': r[3]}
                for r in results
            }
        except:
            return {
                'grc': {'total': 5432, 'annotated': 4521, 'words': 78234567},
                'la': {'total': 4891, 'annotated': 4012, 'words': 65234123},
                'sa': {'total': 2145, 'annotated': 1234, 'words': 23456789},
                'got': {'total': 892, 'annotated': 756, 'words': 8923456},
                'cop': {'total': 743, 'annotated': 512, 'words': 6234567}
            }
    
    def render_overview_metrics(self, metrics: Dict):
        """Render overview metrics panel"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Texts", f"{metrics['total_texts']:,}")
        with col2:
            st.metric("Total Words", f"{metrics['total_words'] // 1000000}M")
        with col3:
            st.metric("Annotated", f"{metrics['annotated_texts']:,}")
        with col4:
            st.metric("Today's Analyses", f"{metrics['analyses_today']:,}")
        with col5:
            completion = (metrics['annotated_texts'] / metrics['total_texts'] * 100) if metrics['total_texts'] > 0 else 0
            st.metric("Completion", f"{completion:.1f}%")
    
    def render_activity_chart(self, timeline: List[Dict]):
        """Render activity timeline chart"""
        if not timeline:
            st.info("No activity data available")
            return
        
        df = pd.DataFrame(timeline)
        df['date'] = pd.to_datetime(df['date'])
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=df['date'], y=df['count'], name="Analyses", marker_color='#0066cc'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['avg_time'], name="Avg. Time (ms)", 
                      line=dict(color='#ff6600', width=2)),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Research Activity (Last 30 Days)",
            height=400
        )
        fig.update_yaxes(title_text="Number of Analyses", secondary_y=False)
        fig.update_yaxes(title_text="Average Time (ms)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_language_progress(self, progress: Dict):
        """Render language progress visualization"""
        if not progress:
            st.info("No language progress data available")
            return
        
        # Create progress dataframe
        data = []
        for lang, stats in progress.items():
            completion = (stats['annotated'] / stats['total'] * 100) if stats['total'] > 0 else 0
            data.append({
                'Language': lang.upper(),
                'Total': stats['total'],
                'Annotated': stats['annotated'],
                'Completion': completion,
                'Words': stats['words']
            })
        
        df = pd.DataFrame(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df,
                x='Language',
                y=['Total', 'Annotated'],
                title="Texts by Language",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df,
                x='Language',
                y='Completion',
                title="Annotation Completion (%)",
                color='Completion',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_research_findings(self):
        """Render research findings panel"""
        st.subheader("Recent Research Findings")
        
        # Sample findings
        findings = [
            {
                'date': '2025-12-01',
                'type': 'Valency Pattern',
                'description': 'Identified 23 new ditransitive patterns in Byzantine Greek texts',
                'significance': 'High'
            },
            {
                'date': '2025-11-30',
                'type': 'Language Contact',
                'description': 'Detected Latin loanword influence on Greek verbal morphology',
                'significance': 'Medium'
            },
            {
                'date': '2025-11-29',
                'type': 'Diachronic Change',
                'description': 'Documented accusative-genitive alternation in Hellenistic period',
                'significance': 'High'
            }
        ]
        
        for finding in findings:
            with st.expander(f"{finding['date']} - {finding['type']}"):
                st.write(finding['description'])
                st.write(f"**Significance:** {finding['significance']}")
    
    def render_project_status(self):
        """Render project status panel"""
        st.subheader("Project Status")
        
        projects = [
            {
                'name': 'Greek Valency Lexicon',
                'progress': 78,
                'status': 'In Progress',
                'deadline': '2025-03-01'
            },
            {
                'name': 'Latin-Greek Contact Study',
                'progress': 45,
                'status': 'In Progress',
                'deadline': '2025-06-01'
            },
            {
                'name': 'Byzantine Corpus Annotation',
                'progress': 92,
                'status': 'Near Completion',
                'deadline': '2025-01-15'
            }
        ]
        
        for project in projects:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{project['name']}**")
                st.progress(project['progress'] / 100)
            
            with col2:
                st.write(f"{project['progress']}%")
            
            with col3:
                st.write(project['deadline'])
    
    def render(self):
        """Main render method for research dashboard"""
        st.header("Research Dashboard")
        
        # Get metrics
        metrics = self.get_research_metrics()
        timeline = self.get_activity_timeline()
        progress = self.get_language_progress()
        
        # Overview metrics
        self.render_overview_metrics(metrics)
        
        st.divider()
        
        # Create tabs
        tabs = st.tabs([
            "Activity",
            "Progress",
            "Findings",
            "Projects",
            "Reports"
        ])
        
        # Activity tab
        with tabs[0]:
            self.render_activity_chart(timeline)
        
        # Progress tab
        with tabs[1]:
            self.render_language_progress(progress)
        
        # Findings tab
        with tabs[2]:
            self.render_research_findings()
        
        # Projects tab
        with tabs[3]:
            self.render_project_status()
        
        # Reports tab
        with tabs[4]:
            st.subheader("Generate Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                report_type = st.selectbox(
                    "Report type",
                    ["Progress Summary", "Valency Analysis", "Corpus Statistics", "Full Research Report"]
                )
                
                date_range = st.date_input(
                    "Date range",
                    value=(datetime.now() - timedelta(days=30), datetime.now())
                )
            
            with col2:
                output_format = st.selectbox(
                    "Output format",
                    ["PDF", "HTML", "Markdown", "LaTeX"]
                )
                
                include_charts = st.checkbox("Include visualizations", value=True)
            
            if st.button("Generate Report", type="primary"):
                st.info("Report generation would be triggered here")
                st.download_button(
                    "Download Report",
                    data="# Research Report\n\nGenerated report content...",
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
