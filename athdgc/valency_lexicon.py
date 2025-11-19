"""
Valency Lexicon Builder for Diachronic Analysis
Builds comprehensive valency lexicons with genealogy tracking
"""

import sqlite3
from typing import Dict, List, Optional, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ValencyLexicon:
    """Build and manage valency lexicons for diachronic research"""
    
    def __init__(self, db_path: str = "valency_lexicon.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize valency lexicon database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Valency entries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS valency_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verb_lemma TEXT NOT NULL,
                language TEXT NOT NULL,
                period TEXT,
                pattern TEXT NOT NULL,
                argument_structure TEXT,
                frequency INTEGER DEFAULT 1,
                source_text TEXT,
                classification TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Diachronic changes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diachronic_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verb_lemma TEXT NOT NULL,
                period_from TEXT NOT NULL,
                period_to TEXT NOT NULL,
                change_type TEXT,
                pattern_before TEXT,
                pattern_after TEXT,
                confidence REAL,
                evidence TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Contact-induced changes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contact_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verb_lemma TEXT NOT NULL,
                source_language TEXT,
                target_language TEXT,
                pattern TEXT,
                contact_type TEXT,
                evidence TEXT,
                period TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"✓ Valency lexicon database initialized: {self.db_path}")
    
    def add_entry(self, verb_lemma: str, language: str, pattern: str, 
                  period: Optional[str] = None, source_text: Optional[str] = None,
                  classification: Optional[str] = None) -> int:
        """Add a valency entry to the lexicon"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO valency_entries 
            (verb_lemma, language, period, pattern, source_text, classification)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (verb_lemma, language, period, pattern, source_text, classification))
        
        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return entry_id
    
    def bulk_add_from_proiel(self, proiel_data: Dict, language: str, period: Optional[str] = None):
        """Bulk add entries from PROIEL processor output"""
        added = 0
        
        for pattern in proiel_data.get('valency_patterns', []):
            self.add_entry(
                verb_lemma=pattern['verb_lemma'],
                language=language,
                pattern=pattern['pattern'],
                period=period,
                source_text=proiel_data.get('source', 'Unknown')
            )
            added += 1
        
        logger.info(f"✓ Added {added} valency entries from PROIEL data")
        return added
    
    def get_entries(self, verb_lemma: Optional[str] = None, language: Optional[str] = None,
                    period: Optional[str] = None) -> List[Dict]:
        """Query valency entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM valency_entries WHERE 1=1"
        params = []
        
        if verb_lemma:
            query += " AND verb_lemma = ?"
            params.append(verb_lemma)
        if language:
            query += " AND language = ?"
            params.append(language)
        if period:
            query += " AND period = ?"
            params.append(period)
        
        cursor.execute(query, params)
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def detect_diachronic_change(self, verb_lemma: str, period_from: str, 
                                  period_to: str) -> Optional[Dict]:
        """Detect diachronic changes in valency patterns"""
        # Get patterns from both periods
        patterns_from = self.get_entries(verb_lemma=verb_lemma, period=period_from)
        patterns_to = self.get_entries(verb_lemma=verb_lemma, period=period_to)
        
        if not patterns_from or not patterns_to:
            return None
        
        # Analyze pattern changes
        patterns_from_set = set(p['pattern'] for p in patterns_from)
        patterns_to_set = set(p['pattern'] for p in patterns_to)
        
        if patterns_from_set != patterns_to_set:
            change = {
                'verb_lemma': verb_lemma,
                'period_from': period_from,
                'period_to': period_to,
                'patterns_lost': list(patterns_from_set - patterns_to_set),
                'patterns_gained': list(patterns_to_set - patterns_from_set),
                'patterns_retained': list(patterns_from_set & patterns_to_set)
            }
            
            # Record change
            self._record_change(change)
            
            return change
        
        return None
    
    def _record_change(self, change: Dict):
        """Record a detected diachronic change"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO diachronic_changes 
            (verb_lemma, period_from, period_to, change_type, pattern_before, pattern_after, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            change['verb_lemma'],
            change['period_from'],
            change['period_to'],
            'pattern_shift',
            ', '.join(change['patterns_lost']),
            ', '.join(change['patterns_gained']),
            0.8  # Default confidence
        ))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict:
        """Get lexicon statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total entries
        cursor.execute("SELECT COUNT(*) FROM valency_entries")
        stats['total_entries'] = cursor.fetchone()[0]
        
        # Unique verbs
        cursor.execute("SELECT COUNT(DISTINCT verb_lemma) FROM valency_entries")
        stats['unique_verbs'] = cursor.fetchone()[0]
        
        # Languages
        cursor.execute("SELECT DISTINCT language FROM valency_entries")
        stats['languages'] = [row[0] for row in cursor.fetchall()]
        
        # Periods
        cursor.execute("SELECT DISTINCT period FROM valency_entries WHERE period IS NOT NULL")
        stats['periods'] = [row[0] for row in cursor.fetchall()]
        
        # Detected changes
        cursor.execute("SELECT COUNT(*) FROM diachronic_changes")
        stats['detected_changes'] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def export_lexicon(self, output_path: str, format: str = 'json'):
        """Export lexicon to file"""
        import json
        
        entries = self.get_entries()
        stats = self.get_statistics()
        
        export_data = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'version': '3.0.0',
                'author': 'Nikolaos Lavidas, NKUA',
                'funding': 'HFRI'
            },
            'statistics': stats,
            'entries': entries
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Exported valency lexicon to: {output_path}")
