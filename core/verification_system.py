#!/usr/bin/env python3
"""
Verification and Anti-Hallucination System
Strict rules for data quality and human oversight

Principles:
1. NO automatic acceptance of uncertain data
2. ALL automated outputs require confidence scores
3. LOW confidence items go to human review queue
4. SOURCES must be cited for all data
5. CHANGES are logged and reversible
6. HUMAN expert has final authority
"""

import sqlite3
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIDENCE LEVELS AND THRESHOLDS
# ============================================================================

class ConfidenceLevel(Enum):
    """Confidence levels for automated outputs"""
    VERIFIED = "verified"           # Human-verified, trusted
    HIGH = "high"                   # >95% confidence, auto-accept
    MEDIUM = "medium"               # 80-95%, flag for review
    LOW = "low"                     # 50-80%, requires review
    UNCERTAIN = "uncertain"         # <50%, must be reviewed
    UNVERIFIED = "unverified"       # No confidence score

# Thresholds for automatic acceptance
CONFIDENCE_THRESHOLDS = {
    "lemmatization": 0.95,      # Very high - morphology is critical
    "pos_tagging": 0.90,        # High
    "parsing": 0.85,            # Syntax can be ambiguous
    "semantic_role": 0.80,      # SRL is complex
    "etymology": 0.70,          # Etymology often uncertain
    "valency": 0.85,            # Argument structure
    "text_collection": 0.99,    # Source texts must be exact
    "translation": 0.70,        # Translations need review
    "period_classification": 0.85,
    "genre_classification": 0.80
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class VerificationItem:
    """Item requiring verification"""
    id: str
    item_type: str              # lemma, parse, etymology, etc.
    target_id: str              # ID of the item being verified
    original_value: Any
    proposed_value: Any
    confidence: float
    source: str                 # Where did this come from
    evidence: List[str] = field(default_factory=list)
    status: str = "pending"     # pending, approved, rejected, modified
    reviewer: str = ""
    review_notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None


@dataclass
class SourceCitation:
    """Citation for data source"""
    source_type: str            # treebank, lexicon, corpus, manual
    source_name: str            # PROIEL, LSJ, Perseus, etc.
    source_id: str              # Specific reference
    url: str = ""
    accessed_at: datetime = field(default_factory=datetime.now)
    reliability: float = 1.0    # How reliable is this source


@dataclass
class AuditLogEntry:
    """Audit log for all changes"""
    id: str
    timestamp: datetime
    action: str                 # create, update, delete, verify
    item_type: str
    item_id: str
    old_value: Any
    new_value: Any
    user: str
    reason: str
    reversible: bool = True


# ============================================================================
# VERIFICATION DATABASE
# ============================================================================

class VerificationDatabase:
    """Database for verification and audit data"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self):
        """Initialize verification tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Verification queue
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verification_queue (
                id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                original_value TEXT,
                proposed_value TEXT,
                confidence REAL,
                source TEXT,
                evidence TEXT,
                status TEXT DEFAULT 'pending',
                reviewer TEXT,
                review_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP
            )
        """)
        
        # Source citations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS source_citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_id TEXT,
                url TEXT,
                accessed_at TIMESTAMP,
                reliability REAL DEFAULT 1.0
            )
        """)
        
        # Audit log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                action TEXT NOT NULL,
                item_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                user_name TEXT,
                reason TEXT,
                reversible INTEGER DEFAULT 1
            )
        """)
        
        # Verified data cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verified_data (
                id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                verified_value TEXT NOT NULL,
                verified_by TEXT NOT NULL,
                verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_citation_id INTEGER,
                notes TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_to_verification_queue(self, item: VerificationItem) -> bool:
        """Add item to verification queue"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO verification_queue
                (id, item_type, target_id, original_value, proposed_value,
                 confidence, source, evidence, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.id, item.item_type, item.target_id,
                json.dumps(item.original_value),
                json.dumps(item.proposed_value),
                item.confidence, item.source,
                json.dumps(item.evidence), item.status
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error adding to verification queue: {e}")
            return False
    
    def get_pending_items(self, item_type: str = None, limit: int = 50) -> List[Dict]:
        """Get pending verification items"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM verification_queue WHERE status = 'pending'"
        params = []
        
        if item_type:
            query += " AND item_type = ?"
            params.append(item_type)
        
        query += " ORDER BY confidence ASC, created_at ASC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        items = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return items
    
    def approve_item(self, item_id: str, reviewer: str, notes: str = "") -> bool:
        """Approve a verification item"""
        return self._update_item_status(item_id, "approved", reviewer, notes)
    
    def reject_item(self, item_id: str, reviewer: str, notes: str = "") -> bool:
        """Reject a verification item"""
        return self._update_item_status(item_id, "rejected", reviewer, notes)
    
    def modify_item(self, item_id: str, new_value: Any, reviewer: str, notes: str = "") -> bool:
        """Modify and approve a verification item"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE verification_queue
                SET status = 'modified', proposed_value = ?, reviewer = ?,
                    review_notes = ?, reviewed_at = ?
                WHERE id = ?
            """, (json.dumps(new_value), reviewer, notes, 
                  datetime.now().isoformat(), item_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error modifying item: {e}")
            return False
    
    def _update_item_status(self, item_id: str, status: str, 
                           reviewer: str, notes: str) -> bool:
        """Update item status"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE verification_queue
                SET status = ?, reviewer = ?, review_notes = ?, reviewed_at = ?
                WHERE id = ?
            """, (status, reviewer, notes, datetime.now().isoformat(), item_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error updating item status: {e}")
            return False
    
    def log_action(self, action: str, item_type: str, item_id: str,
                  old_value: Any, new_value: Any, user: str, reason: str):
        """Log an action to audit log"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            log_id = hashlib.md5(
                f"{datetime.now().isoformat()}{item_id}{action}".encode()
            ).hexdigest()[:16]
            
            cursor.execute("""
                INSERT INTO audit_log
                (id, action, item_type, item_id, old_value, new_value, user_name, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (log_id, action, item_type, item_id,
                  json.dumps(old_value), json.dumps(new_value), user, reason))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging action: {e}")
    
    def add_citation(self, item_type: str, item_id: str, citation: SourceCitation):
        """Add source citation"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO source_citations
                (item_type, item_id, source_type, source_name, source_id, url, reliability)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (item_type, item_id, citation.source_type, citation.source_name,
                  citation.source_id, citation.url, citation.reliability))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error adding citation: {e}")
    
    def get_statistics(self) -> Dict:
        """Get verification statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Queue statistics
        cursor.execute("""
            SELECT status, COUNT(*) FROM verification_queue GROUP BY status
        """)
        stats["queue_by_status"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT item_type, COUNT(*) FROM verification_queue 
            WHERE status = 'pending' GROUP BY item_type
        """)
        stats["pending_by_type"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Audit log statistics
        cursor.execute("SELECT COUNT(*) FROM audit_log")
        stats["total_actions"] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM audit_log 
            WHERE timestamp > datetime('now', '-24 hours')
        """)
        stats["actions_24h"] = cursor.fetchone()[0]
        
        conn.close()
        return stats


# ============================================================================
# VERIFICATION MANAGER
# ============================================================================

class VerificationManager:
    """Manager for verification workflow"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db = VerificationDatabase(db_path)
        self.thresholds = CONFIDENCE_THRESHOLDS
    
    def check_and_queue(self, item_type: str, target_id: str,
                       original_value: Any, proposed_value: Any,
                       confidence: float, source: str,
                       evidence: List[str] = None) -> Tuple[bool, str]:
        """
        Check if item needs verification and queue if necessary.
        Returns (auto_accepted, item_id or message)
        """
        threshold = self.thresholds.get(item_type, 0.90)
        
        # Generate item ID
        item_id = hashlib.md5(
            f"{item_type}{target_id}{proposed_value}".encode()
        ).hexdigest()[:16]
        
        # High confidence - auto accept but log
        if confidence >= threshold:
            self.db.log_action(
                "auto_accept", item_type, target_id,
                original_value, proposed_value,
                "system", f"Confidence {confidence:.2%} >= threshold {threshold:.2%}"
            )
            return True, "Auto-accepted (high confidence)"
        
        # Below threshold - queue for review
        item = VerificationItem(
            id=item_id,
            item_type=item_type,
            target_id=target_id,
            original_value=original_value,
            proposed_value=proposed_value,
            confidence=confidence,
            source=source,
            evidence=evidence or []
        )
        
        self.db.add_to_verification_queue(item)
        
        return False, item_id
    
    def require_human_verification(self, item_type: str, target_id: str,
                                   value: Any, source: str,
                                   reason: str) -> str:
        """Force item to human verification regardless of confidence"""
        item_id = hashlib.md5(
            f"{item_type}{target_id}{value}{datetime.now()}".encode()
        ).hexdigest()[:16]
        
        item = VerificationItem(
            id=item_id,
            item_type=item_type,
            target_id=target_id,
            original_value=None,
            proposed_value=value,
            confidence=0.0,  # Force review
            source=source,
            evidence=[reason]
        )
        
        self.db.add_to_verification_queue(item)
        logger.info(f"Queued for mandatory review: {item_type} - {target_id}")
        
        return item_id
    
    def get_review_queue(self, item_type: str = None) -> List[Dict]:
        """Get items needing review"""
        return self.db.get_pending_items(item_type)
    
    def approve(self, item_id: str, reviewer: str, notes: str = "") -> bool:
        """Approve an item"""
        success = self.db.approve_item(item_id, reviewer, notes)
        if success:
            logger.info(f"Item {item_id} approved by {reviewer}")
        return success
    
    def reject(self, item_id: str, reviewer: str, notes: str = "") -> bool:
        """Reject an item"""
        success = self.db.reject_item(item_id, reviewer, notes)
        if success:
            logger.info(f"Item {item_id} rejected by {reviewer}")
        return success
    
    def modify_and_approve(self, item_id: str, new_value: Any,
                          reviewer: str, notes: str = "") -> bool:
        """Modify value and approve"""
        success = self.db.modify_item(item_id, new_value, reviewer, notes)
        if success:
            logger.info(f"Item {item_id} modified and approved by {reviewer}")
        return success


# ============================================================================
# ANTI-HALLUCINATION RULES
# ============================================================================

ANTI_HALLUCINATION_RULES = """
STRICT RULES FOR DATA QUALITY AND ANTI-HALLUCINATION

1. TEXT COLLECTION
   - Only collect from verified sources (Perseus, PROIEL, TLG, First1KGreek)
   - Store original source URL and access timestamp
   - Never modify source text without explicit logging
   - Flag any OCR or encoding issues for review

2. LEMMATIZATION
   - Use established lexica (LSJ, Morpheus, PROIEL lexicon)
   - Flag unknown lemmas for human review
   - Never invent lemmas - mark as "unknown" if uncertain
   - Confidence threshold: 95%

3. MORPHOLOGICAL ANALYSIS
   - Use rule-based analysis with known paradigms
   - Flag ambiguous forms for review
   - Cite paradigm source for each analysis
   - Never guess morphology - list all possibilities

4. PARSING (DEPENDENCY)
   - Use trained models with known accuracy
   - Flag low-confidence parses for review
   - Provide alternative parses when ambiguous
   - Confidence threshold: 85%

5. SEMANTIC ROLE LABELING
   - Based on established frame inventories
   - Flag novel argument structures
   - Require human review for complex predicates
   - Confidence threshold: 80%

6. ETYMOLOGY
   - Cite established etymological dictionaries (Beekes, Chantraine, DELG)
   - Mark uncertain etymologies explicitly
   - Never invent PIE reconstructions
   - Confidence threshold: 70%

7. VALENCY / ARGUMENT STRUCTURE
   - Based on attested patterns in corpus
   - Cite source sentences for each pattern
   - Flag rare or unusual patterns
   - Confidence threshold: 85%

8. PERIOD / GENRE CLASSIFICATION
   - Based on explicit metadata from sources
   - Flag uncertain classifications
   - Never assume period without evidence

9. TRANSLATION
   - Mark all translations as requiring review
   - Cite translation source
   - Never auto-generate translations without flagging

10. GENERAL PRINCIPLES
    - When in doubt, flag for human review
    - Never delete data - only mark as deprecated
    - All changes must be logged and reversible
    - Human expert has final authority
    - Prefer "unknown" over guessing
"""


def get_rules() -> str:
    """Get anti-hallucination rules"""
    return ANTI_HALLUCINATION_RULES


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verification System")
    parser.add_argument('command', choices=['queue', 'stats', 'rules', 'approve', 'reject'],
                       help="Command to run")
    parser.add_argument('--type', '-t', help="Item type filter")
    parser.add_argument('--id', '-i', help="Item ID")
    parser.add_argument('--reviewer', '-r', help="Reviewer name")
    parser.add_argument('--notes', '-n', help="Review notes")
    
    args = parser.parse_args()
    
    manager = VerificationManager()
    
    if args.command == 'queue':
        items = manager.get_review_queue(args.type)
        print(f"Pending items: {len(items)}")
        for item in items[:10]:
            print(f"  [{item['id']}] {item['item_type']}: {item['confidence']:.2%}")
    
    elif args.command == 'stats':
        stats = manager.db.get_statistics()
        print(json.dumps(stats, indent=2))
    
    elif args.command == 'rules':
        print(get_rules())
    
    elif args.command == 'approve':
        if args.id and args.reviewer:
            manager.approve(args.id, args.reviewer, args.notes or "")
            print(f"Approved: {args.id}")
        else:
            print("Requires --id and --reviewer")
    
    elif args.command == 'reject':
        if args.id and args.reviewer:
            manager.reject(args.id, args.reviewer, args.notes or "")
            print(f"Rejected: {args.id}")
        else:
            print("Requires --id and --reviewer")


if __name__ == "__main__":
    main()
