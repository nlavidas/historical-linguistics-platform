"""
HLP Core Database - Database Abstraction Layer

This module provides a unified database interface supporting both SQLite
and PostgreSQL backends, with connection pooling, schema management,
and query building utilities.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import sqlite3
import json
import threading
import queue
import time
import hashlib
import logging
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Any, Tuple, Union, 
    Iterator, Callable, TypeVar, Generic, Type
)
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class IsolationLevel(Enum):
    """Transaction isolation levels"""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_type: DatabaseType = DatabaseType.SQLITE
    
    host: str = "localhost"
    port: int = 5432
    database: str = "hlp_corpus"
    username: str = ""
    password: str = ""
    
    sqlite_path: Optional[str] = None
    
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    
    echo: bool = False
    
    connect_timeout: int = 10
    read_timeout: int = 30
    write_timeout: int = 30
    
    def get_connection_string(self) -> str:
        """Get database connection string"""
        if self.db_type == DatabaseType.SQLITE:
            return self.sqlite_path or ":memory:"
        elif self.db_type == DatabaseType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        return ""


class ConnectionPool:
    """Thread-safe connection pool for database connections"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._pool: queue.Queue = queue.Queue(maxsize=config.pool_size)
        self._size = 0
        self._lock = threading.Lock()
        self._local = threading.local()
        
        for _ in range(config.pool_size):
            self._create_connection()
    
    def _create_connection(self) -> Any:
        """Create a new database connection"""
        if self.config.db_type == DatabaseType.SQLITE:
            conn = sqlite3.connect(
                self.config.get_connection_string(),
                check_same_thread=False,
                timeout=self.config.connect_timeout
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = -64000")
            conn.execute("PRAGMA temp_store = MEMORY")
        else:
            try:
                import psycopg2
                import psycopg2.extras
                conn = psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    connect_timeout=self.config.connect_timeout
                )
                conn.autocommit = False
            except ImportError:
                raise ImportError("psycopg2 is required for PostgreSQL support")
        
        with self._lock:
            self._size += 1
        
        return conn
    
    def get_connection(self) -> Any:
        """Get a connection from the pool"""
        try:
            conn = self._pool.get(timeout=self.config.pool_timeout)
            return conn
        except queue.Empty:
            if self._size < self.config.pool_size + self.config.max_overflow:
                return self._create_connection()
            raise TimeoutError("Connection pool exhausted")
    
    def return_connection(self, conn: Any):
        """Return a connection to the pool"""
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            conn.close()
            with self._lock:
                self._size -= 1
    
    def close_all(self):
        """Close all connections in the pool"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break
        self._size = 0
    
    @contextmanager
    def connection(self):
        """Context manager for getting a connection"""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)


class QueryBuilder:
    """SQL query builder with parameterized queries"""
    
    def __init__(self, table: str):
        self.table = table
        self._select_cols: List[str] = ["*"]
        self._where_clauses: List[str] = []
        self._where_params: List[Any] = []
        self._order_by: List[str] = []
        self._group_by: List[str] = []
        self._having: List[str] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._joins: List[str] = []
    
    def select(self, *columns: str) -> "QueryBuilder":
        """Set columns to select"""
        self._select_cols = list(columns) if columns else ["*"]
        return self
    
    def where(self, condition: str, *params: Any) -> "QueryBuilder":
        """Add WHERE clause"""
        self._where_clauses.append(condition)
        self._where_params.extend(params)
        return self
    
    def where_eq(self, column: str, value: Any) -> "QueryBuilder":
        """Add equality WHERE clause"""
        self._where_clauses.append(f"{column} = ?")
        self._where_params.append(value)
        return self
    
    def where_in(self, column: str, values: List[Any]) -> "QueryBuilder":
        """Add IN WHERE clause"""
        placeholders = ", ".join("?" * len(values))
        self._where_clauses.append(f"{column} IN ({placeholders})")
        self._where_params.extend(values)
        return self
    
    def where_like(self, column: str, pattern: str) -> "QueryBuilder":
        """Add LIKE WHERE clause"""
        self._where_clauses.append(f"{column} LIKE ?")
        self._where_params.append(pattern)
        return self
    
    def where_between(self, column: str, start: Any, end: Any) -> "QueryBuilder":
        """Add BETWEEN WHERE clause"""
        self._where_clauses.append(f"{column} BETWEEN ? AND ?")
        self._where_params.extend([start, end])
        return self
    
    def where_null(self, column: str) -> "QueryBuilder":
        """Add IS NULL WHERE clause"""
        self._where_clauses.append(f"{column} IS NULL")
        return self
    
    def where_not_null(self, column: str) -> "QueryBuilder":
        """Add IS NOT NULL WHERE clause"""
        self._where_clauses.append(f"{column} IS NOT NULL")
        return self
    
    def order_by(self, column: str, direction: str = "ASC") -> "QueryBuilder":
        """Add ORDER BY clause"""
        self._order_by.append(f"{column} {direction}")
        return self
    
    def group_by(self, *columns: str) -> "QueryBuilder":
        """Add GROUP BY clause"""
        self._group_by.extend(columns)
        return self
    
    def having(self, condition: str, *params: Any) -> "QueryBuilder":
        """Add HAVING clause"""
        self._having.append(condition)
        self._where_params.extend(params)
        return self
    
    def limit(self, limit: int) -> "QueryBuilder":
        """Set LIMIT"""
        self._limit = limit
        return self
    
    def offset(self, offset: int) -> "QueryBuilder":
        """Set OFFSET"""
        self._offset = offset
        return self
    
    def join(self, table: str, condition: str, join_type: str = "INNER") -> "QueryBuilder":
        """Add JOIN clause"""
        self._joins.append(f"{join_type} JOIN {table} ON {condition}")
        return self
    
    def left_join(self, table: str, condition: str) -> "QueryBuilder":
        """Add LEFT JOIN clause"""
        return self.join(table, condition, "LEFT")
    
    def right_join(self, table: str, condition: str) -> "QueryBuilder":
        """Add RIGHT JOIN clause"""
        return self.join(table, condition, "RIGHT")
    
    def build_select(self) -> Tuple[str, List[Any]]:
        """Build SELECT query"""
        sql = f"SELECT {', '.join(self._select_cols)} FROM {self.table}"
        
        if self._joins:
            sql += " " + " ".join(self._joins)
        
        if self._where_clauses:
            sql += " WHERE " + " AND ".join(self._where_clauses)
        
        if self._group_by:
            sql += " GROUP BY " + ", ".join(self._group_by)
        
        if self._having:
            sql += " HAVING " + " AND ".join(self._having)
        
        if self._order_by:
            sql += " ORDER BY " + ", ".join(self._order_by)
        
        if self._limit is not None:
            sql += f" LIMIT {self._limit}"
        
        if self._offset is not None:
            sql += f" OFFSET {self._offset}"
        
        return sql, self._where_params
    
    def build_count(self) -> Tuple[str, List[Any]]:
        """Build COUNT query"""
        sql = f"SELECT COUNT(*) FROM {self.table}"
        
        if self._joins:
            sql += " " + " ".join(self._joins)
        
        if self._where_clauses:
            sql += " WHERE " + " AND ".join(self._where_clauses)
        
        return sql, self._where_params
    
    def build_delete(self) -> Tuple[str, List[Any]]:
        """Build DELETE query"""
        sql = f"DELETE FROM {self.table}"
        
        if self._where_clauses:
            sql += " WHERE " + " AND ".join(self._where_clauses)
        
        return sql, self._where_params
    
    @staticmethod
    def build_insert(table: str, data: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Build INSERT query"""
        columns = list(data.keys())
        placeholders = ", ".join("?" * len(columns))
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        return sql, list(data.values())
    
    @staticmethod
    def build_insert_many(table: str, columns: List[str], rows: List[List[Any]]) -> Tuple[str, List[List[Any]]]:
        """Build INSERT query for multiple rows"""
        placeholders = ", ".join("?" * len(columns))
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        return sql, rows
    
    @staticmethod
    def build_update(table: str, data: Dict[str, Any], where: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Build UPDATE query"""
        set_clause = ", ".join(f"{k} = ?" for k in data.keys())
        where_clause = " AND ".join(f"{k} = ?" for k in where.keys())
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        params = list(data.values()) + list(where.values())
        return sql, params
    
    @staticmethod
    def build_upsert(table: str, data: Dict[str, Any], conflict_columns: List[str]) -> Tuple[str, List[Any]]:
        """Build UPSERT (INSERT OR REPLACE) query"""
        columns = list(data.keys())
        placeholders = ", ".join("?" * len(columns))
        update_clause = ", ".join(f"{k} = excluded.{k}" for k in columns if k not in conflict_columns)
        conflict_clause = ", ".join(conflict_columns)
        
        sql = f"""
            INSERT INTO {table} ({', '.join(columns)}) 
            VALUES ({placeholders})
            ON CONFLICT ({conflict_clause}) 
            DO UPDATE SET {update_clause}
        """
        return sql, list(data.values())


class TransactionManager:
    """Transaction management with savepoints"""
    
    def __init__(self, connection: Any, db_type: DatabaseType = DatabaseType.SQLITE):
        self.connection = connection
        self.db_type = db_type
        self._savepoint_counter = 0
        self._active_savepoints: List[str] = []
    
    def begin(self, isolation_level: Optional[IsolationLevel] = None):
        """Begin a transaction"""
        if self.db_type == DatabaseType.SQLITE:
            self.connection.execute("BEGIN TRANSACTION")
        else:
            if isolation_level:
                self.connection.set_session(isolation_level=isolation_level.value)
    
    def commit(self):
        """Commit the transaction"""
        self.connection.commit()
        self._active_savepoints.clear()
    
    def rollback(self):
        """Rollback the transaction"""
        self.connection.rollback()
        self._active_savepoints.clear()
    
    def savepoint(self, name: Optional[str] = None) -> str:
        """Create a savepoint"""
        if name is None:
            self._savepoint_counter += 1
            name = f"sp_{self._savepoint_counter}"
        
        self.connection.execute(f"SAVEPOINT {name}")
        self._active_savepoints.append(name)
        return name
    
    def release_savepoint(self, name: str):
        """Release a savepoint"""
        self.connection.execute(f"RELEASE SAVEPOINT {name}")
        if name in self._active_savepoints:
            self._active_savepoints.remove(name)
    
    def rollback_to_savepoint(self, name: str):
        """Rollback to a savepoint"""
        self.connection.execute(f"ROLLBACK TO SAVEPOINT {name}")
    
    @contextmanager
    def transaction(self, isolation_level: Optional[IsolationLevel] = None):
        """Context manager for transactions"""
        self.begin(isolation_level)
        try:
            yield self
            self.commit()
        except Exception:
            self.rollback()
            raise
    
    @contextmanager
    def nested_transaction(self):
        """Context manager for nested transactions using savepoints"""
        sp_name = self.savepoint()
        try:
            yield sp_name
            self.release_savepoint(sp_name)
        except Exception:
            self.rollback_to_savepoint(sp_name)
            raise


class SchemaManager:
    """Database schema management with migrations"""
    
    SCHEMA_VERSION_TABLE = "schema_version"
    
    def __init__(self, connection: Any, db_type: DatabaseType = DatabaseType.SQLITE):
        self.connection = connection
        self.db_type = db_type
        self._ensure_version_table()
    
    def _ensure_version_table(self):
        """Ensure schema version table exists"""
        cursor = self.connection.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.SCHEMA_VERSION_TABLE} (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        self.connection.commit()
    
    def get_current_version(self) -> int:
        """Get current schema version"""
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT MAX(version) FROM {self.SCHEMA_VERSION_TABLE}")
        result = cursor.fetchone()
        return result[0] if result and result[0] else 0
    
    def apply_migration(self, version: int, sql: str, description: str = ""):
        """Apply a migration"""
        current = self.get_current_version()
        if version <= current:
            logger.info(f"Migration {version} already applied")
            return
        
        cursor = self.connection.cursor()
        try:
            for statement in sql.split(";"):
                statement = statement.strip()
                if statement:
                    cursor.execute(statement)
            
            cursor.execute(
                f"INSERT INTO {self.SCHEMA_VERSION_TABLE} (version, description) VALUES (?, ?)",
                (version, description)
            )
            self.connection.commit()
            logger.info(f"Applied migration {version}: {description}")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to apply migration {version}: {e}")
            raise
    
    def create_tables(self):
        """Create all platform tables"""
        migrations = self._get_migrations()
        for version, sql, description in migrations:
            self.apply_migration(version, sql, description)
    
    def _get_migrations(self) -> List[Tuple[int, str, str]]:
        """Get all migrations"""
        return [
            (1, self._migration_001_core_tables(), "Core tables"),
            (2, self._migration_002_annotation_tables(), "Annotation tables"),
            (3, self._migration_003_valency_tables(), "Valency tables"),
            (4, self._migration_004_diachronic_tables(), "Diachronic tables"),
            (5, self._migration_005_semantic_tables(), "Semantic tables"),
            (6, self._migration_006_pipeline_tables(), "Pipeline tables"),
            (7, self._migration_007_user_tables(), "User tables"),
            (8, self._migration_008_indexes(), "Indexes"),
        ]
    
    def _migration_001_core_tables(self) -> str:
        """Core tables migration"""
        return """
            CREATE TABLE IF NOT EXISTS corpora (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                language TEXT DEFAULT 'grc',
                languages TEXT,
                version TEXT DEFAULT '1.0.0',
                license TEXT,
                citation TEXT,
                source_url TEXT,
                homepage TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                corpus_id TEXT,
                title TEXT NOT NULL,
                author TEXT,
                language TEXT DEFAULT 'grc',
                period TEXT,
                genre TEXT,
                date_composed TEXT,
                date_composed_start INTEGER,
                date_composed_end INTEGER,
                edition TEXT,
                editor TEXT,
                translator TEXT,
                source_type TEXT,
                source_url TEXT,
                source_id TEXT,
                proiel_id TEXT,
                proiel_source_id TEXT,
                sentence_count INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                word_count INTEGER DEFAULT 0,
                annotation_status TEXT DEFAULT 'pending',
                annotation_progress REAL DEFAULT 0.0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (corpus_id) REFERENCES corpora(id)
            );
            
            CREATE TABLE IF NOT EXISTS sentences (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                sentence_index INTEGER DEFAULT 0,
                text TEXT,
                translation TEXT,
                proiel_id TEXT,
                proiel_status TEXT,
                sent_id TEXT,
                token_count INTEGER DEFAULT 0,
                annotation_status TEXT DEFAULT 'pending',
                annotator TEXT,
                annotation_time TIMESTAMP,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );
            
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id TEXT NOT NULL,
                token_index INTEGER NOT NULL,
                form TEXT NOT NULL,
                lemma TEXT,
                pos TEXT,
                xpos TEXT,
                feats TEXT,
                head INTEGER,
                deprel TEXT,
                deps TEXT,
                misc TEXT,
                proiel_id TEXT,
                proiel_morph TEXT,
                span_start INTEGER,
                span_end INTEGER,
                is_multiword INTEGER DEFAULT 0,
                multiword_id TEXT,
                is_empty INTEGER DEFAULT 0,
                empty_node_id TEXT,
                annotation_status TEXT DEFAULT 'pending',
                annotator TEXT,
                annotation_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sentence_id) REFERENCES sentences(id)
            )
        """
    
    def _migration_002_annotation_tables(self) -> str:
        """Annotation tables migration"""
        return """
            CREATE TABLE IF NOT EXISTS annotation_layers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id TEXT NOT NULL,
                layer_type TEXT NOT NULL,
                layer_data TEXT,
                annotator TEXT,
                annotation_time TIMESTAMP,
                status TEXT DEFAULT 'pending',
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sentence_id) REFERENCES sentences(id)
            );
            
            CREATE TABLE IF NOT EXISTS annotation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                field_name TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                annotator TEXT,
                change_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                change_reason TEXT
            );
            
            CREATE TABLE IF NOT EXISTS annotation_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                comment_text TEXT NOT NULL,
                author TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved INTEGER DEFAULT 0,
                resolved_by TEXT,
                resolved_at TIMESTAMP
            )
        """
    
    def _migration_003_valency_tables(self) -> str:
        """Valency tables migration"""
        return """
            CREATE TABLE IF NOT EXISTS lexemes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                language TEXT NOT NULL,
                pos TEXT,
                etymology TEXT,
                cognates TEXT,
                frequency INTEGER DEFAULT 0,
                frequency_by_period TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(lemma, language)
            );
            
            CREATE TABLE IF NOT EXISTS lemma_senses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lexeme_id INTEGER NOT NULL,
                sense_id TEXT NOT NULL,
                definition TEXT NOT NULL,
                gloss TEXT,
                examples TEXT,
                semantic_field TEXT,
                frequency INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (lexeme_id) REFERENCES lexemes(id)
            );
            
            CREATE TABLE IF NOT EXISTS valency_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id TEXT NOT NULL UNIQUE,
                arguments TEXT,
                obligatory_args TEXT,
                optional_args TEXT,
                semantic_roles TEXT,
                syntactic_pattern TEXT,
                examples TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS valency_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT NOT NULL UNIQUE,
                verb_lemma TEXT NOT NULL,
                frame_id TEXT,
                language TEXT DEFAULT 'grc',
                period TEXT,
                frequency INTEGER DEFAULT 1,
                relative_frequency REAL DEFAULT 0.0,
                source_sentences TEXT,
                source_documents TEXT,
                confidence REAL DEFAULT 1.0,
                extraction_method TEXT DEFAULT 'automatic',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (frame_id) REFERENCES valency_frames(id)
            );
            
            CREATE TABLE IF NOT EXISTS valency_occurrences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT NOT NULL,
                sentence_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                verb_token_id INTEGER,
                arguments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pattern_id) REFERENCES valency_patterns(pattern_id),
                FOREIGN KEY (sentence_id) REFERENCES sentences(id),
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """
    
    def _migration_004_diachronic_tables(self) -> str:
        """Diachronic tables migration"""
        return """
            CREATE TABLE IF NOT EXISTS diachronic_periods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_code TEXT NOT NULL UNIQUE,
                period_name TEXT NOT NULL,
                language TEXT NOT NULL,
                start_year INTEGER,
                end_year INTEGER,
                description TEXT,
                parent_period TEXT,
                metadata TEXT
            );
            
            CREATE TABLE IF NOT EXISTS diachronic_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                change_type TEXT NOT NULL,
                feature TEXT NOT NULL,
                source_period TEXT NOT NULL,
                target_period TEXT NOT NULL,
                language TEXT NOT NULL,
                description TEXT,
                frequency_source REAL,
                frequency_target REAL,
                change_magnitude REAL,
                statistical_significance REAL,
                examples TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS frequency_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_type TEXT NOT NULL,
                feature_value TEXT NOT NULL,
                language TEXT NOT NULL,
                period TEXT NOT NULL,
                corpus_id TEXT,
                raw_frequency INTEGER DEFAULT 0,
                normalized_frequency REAL DEFAULT 0.0,
                total_tokens INTEGER DEFAULT 0,
                document_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (corpus_id) REFERENCES corpora(id)
            )
        """
    
    def _migration_005_semantic_tables(self) -> str:
        """Semantic tables migration"""
        return """
            CREATE TABLE IF NOT EXISTS semantic_roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id TEXT NOT NULL,
                predicate_token_id INTEGER NOT NULL,
                argument_token_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                span_start INTEGER,
                span_end INTEGER,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sentence_id) REFERENCES sentences(id)
            );
            
            CREATE TABLE IF NOT EXISTS named_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                span_start INTEGER NOT NULL,
                span_end INTEGER NOT NULL,
                text TEXT NOT NULL,
                normalized_form TEXT,
                wikidata_id TEXT,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sentence_id) REFERENCES sentences(id)
            );
            
            CREATE TABLE IF NOT EXISTS information_structure (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id TEXT NOT NULL,
                token_id INTEGER NOT NULL,
                info_status TEXT,
                topic_focus TEXT,
                contrast INTEGER DEFAULT 0,
                emphasis INTEGER DEFAULT 0,
                antecedent_id INTEGER,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sentence_id) REFERENCES sentences(id)
            );
            
            CREATE TABLE IF NOT EXISTS coreference_chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                chain_id TEXT NOT NULL,
                mentions TEXT,
                entity_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """
    
    def _migration_006_pipeline_tables(self) -> str:
        """Pipeline tables migration"""
        return """
            CREATE TABLE IF NOT EXISTS pipeline_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL UNIQUE,
                job_type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                input_data TEXT,
                output_data TEXT,
                error_message TEXT,
                progress REAL DEFAULT 0.0,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS pipeline_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                input_data TEXT,
                output_data TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES pipeline_jobs(job_id)
            );
            
            CREATE TABLE IF NOT EXISTS source_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL UNIQUE,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_url TEXT,
                access_method TEXT,
                update_schedule TEXT,
                last_updated TIMESTAMP,
                document_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS ingestion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                document_id TEXT,
                status TEXT NOT NULL,
                message TEXT,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES source_registry(source_id)
            )
        """
    
    def _migration_007_user_tables(self) -> str:
        """User tables migration"""
        return """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                email TEXT,
                full_name TEXT,
                role TEXT DEFAULT 'researcher',
                is_active INTEGER DEFAULT 1,
                two_factor_enabled INTEGER DEFAULT 0,
                phone TEXT,
                last_login TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT NOT NULL UNIQUE,
                ip_address TEXT,
                user_agent TEXT,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                entity_type TEXT,
                entity_id TEXT,
                details TEXT,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                name TEXT,
                permissions TEXT,
                expires_at TIMESTAMP,
                last_used TIMESTAMP,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """
    
    def _migration_008_indexes(self) -> str:
        """Indexes migration"""
        return """
            CREATE INDEX IF NOT EXISTS idx_documents_corpus ON documents(corpus_id);
            CREATE INDEX IF NOT EXISTS idx_documents_language ON documents(language);
            CREATE INDEX IF NOT EXISTS idx_documents_period ON documents(period);
            CREATE INDEX IF NOT EXISTS idx_documents_author ON documents(author);
            CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(annotation_status);
            
            CREATE INDEX IF NOT EXISTS idx_sentences_document ON sentences(document_id);
            CREATE INDEX IF NOT EXISTS idx_sentences_status ON sentences(annotation_status);
            
            CREATE INDEX IF NOT EXISTS idx_tokens_sentence ON tokens(sentence_id);
            CREATE INDEX IF NOT EXISTS idx_tokens_lemma ON tokens(lemma);
            CREATE INDEX IF NOT EXISTS idx_tokens_pos ON tokens(pos);
            
            CREATE INDEX IF NOT EXISTS idx_lexemes_lemma ON lexemes(lemma);
            CREATE INDEX IF NOT EXISTS idx_lexemes_language ON lexemes(language);
            
            CREATE INDEX IF NOT EXISTS idx_valency_patterns_verb ON valency_patterns(verb_lemma);
            CREATE INDEX IF NOT EXISTS idx_valency_patterns_language ON valency_patterns(language);
            CREATE INDEX IF NOT EXISTS idx_valency_patterns_period ON valency_patterns(period);
            
            CREATE INDEX IF NOT EXISTS idx_frequency_data_feature ON frequency_data(feature_type, feature_value);
            CREATE INDEX IF NOT EXISTS idx_frequency_data_period ON frequency_data(period);
            
            CREATE INDEX IF NOT EXISTS idx_semantic_roles_sentence ON semantic_roles(sentence_id);
            CREATE INDEX IF NOT EXISTS idx_named_entities_sentence ON named_entities(sentence_id);
            CREATE INDEX IF NOT EXISTS idx_info_structure_sentence ON information_structure(sentence_id);
            
            CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_status ON pipeline_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_pipeline_tasks_job ON pipeline_tasks(job_id);
            
            CREATE INDEX IF NOT EXISTS idx_audit_log_user ON audit_log(user_id);
            CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action)
        """
    
    def drop_all_tables(self):
        """Drop all tables (use with caution)"""
        cursor = self.connection.cursor()
        
        if self.db_type == DatabaseType.SQLITE:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                if table != "sqlite_sequence":
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
        
        self.connection.commit()
        logger.warning("All tables dropped")


class DatabaseManager:
    """Main database manager class"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        if config is None:
            config = DatabaseConfig()
        
        self.config = config
        self._pool: Optional[ConnectionPool] = None
        self._schema_manager: Optional[SchemaManager] = None
        self._initialized = False
    
    def initialize(self, create_tables: bool = True):
        """Initialize the database"""
        if self._initialized:
            return
        
        if self.config.db_type == DatabaseType.SQLITE and self.config.sqlite_path:
            db_dir = Path(self.config.sqlite_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
        
        self._pool = ConnectionPool(self.config)
        
        if create_tables:
            with self._pool.connection() as conn:
                self._schema_manager = SchemaManager(conn, self.config.db_type)
                self._schema_manager.create_tables()
        
        self._initialized = True
        logger.info(f"Database initialized: {self.config.db_type.value}")
    
    def close(self):
        """Close all database connections"""
        if self._pool:
            self._pool.close_all()
        self._initialized = False
    
    @contextmanager
    def connection(self):
        """Get a database connection"""
        if not self._initialized:
            self.initialize()
        
        with self._pool.connection() as conn:
            yield conn
    
    @contextmanager
    def transaction(self, isolation_level: Optional[IsolationLevel] = None):
        """Execute within a transaction"""
        with self.connection() as conn:
            tm = TransactionManager(conn, self.config.db_type)
            with tm.transaction(isolation_level):
                yield conn
    
    def execute(self, sql: str, params: Optional[List[Any]] = None) -> Any:
        """Execute a SQL statement"""
        with self.connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            conn.commit()
            return cursor
    
    def execute_many(self, sql: str, params_list: List[List[Any]]) -> Any:
        """Execute a SQL statement with multiple parameter sets"""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, params_list)
            conn.commit()
            return cursor
    
    def fetch_one(self, sql: str, params: Optional[List[Any]] = None) -> Optional[Any]:
        """Fetch a single row"""
        with self.connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            return cursor.fetchone()
    
    def fetch_all(self, sql: str, params: Optional[List[Any]] = None) -> List[Any]:
        """Fetch all rows"""
        with self.connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            return cursor.fetchall()
    
    def fetch_iter(self, sql: str, params: Optional[List[Any]] = None, 
                   batch_size: int = 1000) -> Iterator[Any]:
        """Fetch rows in batches"""
        with self.connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                yield from rows
    
    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Insert a row and return the ID"""
        sql, params = QueryBuilder.build_insert(table, data)
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            return cursor.lastrowid
    
    def insert_many(self, table: str, columns: List[str], rows: List[List[Any]]) -> int:
        """Insert multiple rows"""
        sql, params = QueryBuilder.build_insert_many(table, columns, rows)
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, params)
            conn.commit()
            return cursor.rowcount
    
    def update(self, table: str, data: Dict[str, Any], where: Dict[str, Any]) -> int:
        """Update rows"""
        sql, params = QueryBuilder.build_update(table, data, where)
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            return cursor.rowcount
    
    def delete(self, table: str, where: Dict[str, Any]) -> int:
        """Delete rows"""
        qb = QueryBuilder(table)
        for key, value in where.items():
            qb.where_eq(key, value)
        sql, params = qb.build_delete()
        
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            return cursor.rowcount
    
    def count(self, table: str, where: Optional[Dict[str, Any]] = None) -> int:
        """Count rows"""
        qb = QueryBuilder(table)
        if where:
            for key, value in where.items():
                qb.where_eq(key, value)
        sql, params = qb.build_count()
        
        result = self.fetch_one(sql, params)
        return result[0] if result else 0
    
    def exists(self, table: str, where: Dict[str, Any]) -> bool:
        """Check if rows exist"""
        return self.count(table, where) > 0
    
    def query(self, table: str) -> QueryBuilder:
        """Start a query builder"""
        return QueryBuilder(table)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            "db_type": self.config.db_type.value,
            "initialized": self._initialized
        }
        
        if self._initialized:
            stats["corpora_count"] = self.count("corpora")
            stats["documents_count"] = self.count("documents")
            stats["sentences_count"] = self.count("sentences")
            stats["tokens_count"] = self.count("tokens")
            stats["lexemes_count"] = self.count("lexemes")
            stats["valency_patterns_count"] = self.count("valency_patterns")
            
            if self._schema_manager:
                stats["schema_version"] = self._schema_manager.get_current_version()
        
        return stats
    
    def save_corpus(self, corpus: "Corpus") -> str:
        """Save a corpus to the database"""
        from hlp_core.models import Corpus
        
        data = {
            "id": corpus.id,
            "name": corpus.name,
            "description": corpus.description,
            "language": corpus.language.value,
            "languages": json.dumps([l.value for l in corpus.languages]),
            "version": corpus.version,
            "license": corpus.license,
            "citation": corpus.citation,
            "source_url": corpus.source_url,
            "homepage": corpus.homepage,
            "metadata": json.dumps(corpus.metadata),
            "updated_at": datetime.now().isoformat()
        }
        
        if self.exists("corpora", {"id": corpus.id}):
            self.update("corpora", data, {"id": corpus.id})
        else:
            data["created_at"] = datetime.now().isoformat()
            self.insert("corpora", data)
        
        return corpus.id
    
    def save_document(self, document: "Document") -> str:
        """Save a document to the database"""
        from hlp_core.models import Document
        
        data = {
            "id": document.id,
            "corpus_id": document.metadata.get("corpus_id"),
            "title": document.title,
            "author": document.author,
            "language": document.language.value,
            "period": document.period.value if document.period else None,
            "genre": document.genre.value if document.genre else None,
            "date_composed": document.date_composed,
            "date_composed_start": document.date_composed_start,
            "date_composed_end": document.date_composed_end,
            "edition": document.edition,
            "editor": document.editor,
            "translator": document.translator,
            "source_type": document.source.source_type if document.source else None,
            "source_url": document.source.source_url if document.source else None,
            "proiel_id": document.proiel_id,
            "proiel_source_id": document.proiel_source_id,
            "sentence_count": document.sentence_count,
            "token_count": document.token_count,
            "word_count": document.word_count,
            "annotation_status": document.annotation_status.value,
            "annotation_progress": document.annotation_progress,
            "metadata": json.dumps(document.metadata),
            "updated_at": datetime.now().isoformat()
        }
        
        if self.exists("documents", {"id": document.id}):
            self.update("documents", data, {"id": document.id})
        else:
            data["created_at"] = datetime.now().isoformat()
            self.insert("documents", data)
        
        return document.id
    
    def save_sentence(self, sentence: "Sentence") -> str:
        """Save a sentence to the database"""
        from hlp_core.models import Sentence
        
        data = {
            "id": sentence.id,
            "document_id": sentence.document_id,
            "sentence_index": sentence.sentence_index,
            "text": sentence.text,
            "translation": sentence.translation,
            "proiel_id": sentence.proiel_id,
            "proiel_status": sentence.proiel_status,
            "sent_id": sentence.sent_id,
            "token_count": len(sentence.tokens),
            "annotation_status": sentence.annotation_status.value,
            "annotator": sentence.annotator,
            "annotation_time": sentence.annotation_time.isoformat() if sentence.annotation_time else None,
            "metadata": json.dumps(sentence.metadata)
        }
        
        if self.exists("sentences", {"id": sentence.id}):
            self.update("sentences", data, {"id": sentence.id})
        else:
            data["created_at"] = datetime.now().isoformat()
            self.insert("sentences", data)
        
        return sentence.id
    
    def save_token(self, token: "Token", sentence_id: str) -> int:
        """Save a token to the database"""
        from hlp_core.models import Token
        
        data = {
            "sentence_id": sentence_id,
            "token_index": token.id,
            "form": token.form,
            "lemma": token.lemma,
            "pos": token.morphology.pos.value if token.morphology and token.morphology.pos else None,
            "xpos": token.morphology.proiel_morph if token.morphology else None,
            "feats": token.morphology.to_ud_string() if token.morphology else None,
            "head": token.syntax.head_id if token.syntax else None,
            "deprel": token.syntax.relation.value if token.syntax else None,
            "deps": token.syntax.to_enhanced_string() if token.syntax and token.syntax.enhanced_deps else None,
            "misc": json.dumps(token.misc) if token.misc else None,
            "proiel_id": token.proiel_id,
            "proiel_morph": token.morphology.proiel_morph if token.morphology else None,
            "span_start": token.span_start,
            "span_end": token.span_end,
            "is_multiword": 1 if token.is_multiword else 0,
            "multiword_id": token.multiword_id,
            "is_empty": 1 if token.is_empty else 0,
            "empty_node_id": token.empty_node_id,
            "annotation_status": token.annotation_status.value,
            "annotator": token.annotator,
            "annotation_time": token.annotation_time.isoformat() if token.annotation_time else None
        }
        
        return self.insert("tokens", data)
    
    def load_corpus(self, corpus_id: str) -> Optional["Corpus"]:
        """Load a corpus from the database"""
        from hlp_core.models import Corpus, Language
        
        row = self.fetch_one("SELECT * FROM corpora WHERE id = ?", [corpus_id])
        if not row:
            return None
        
        corpus = Corpus(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            language=Language(row["language"]) if row["language"] else Language.ANCIENT_GREEK,
            version=row["version"],
            license=row["license"],
            citation=row["citation"],
            source_url=row["source_url"],
            homepage=row["homepage"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )
        
        if row["languages"]:
            corpus.languages = [Language(l) for l in json.loads(row["languages"])]
        
        return corpus
    
    def load_document(self, document_id: str) -> Optional["Document"]:
        """Load a document from the database"""
        from hlp_core.models import Document, Language, Period, Genre, AnnotationStatus, SourceMetadata
        
        row = self.fetch_one("SELECT * FROM documents WHERE id = ?", [document_id])
        if not row:
            return None
        
        source = None
        if row["source_type"]:
            source = SourceMetadata(
                source_type=row["source_type"],
                source_url=row["source_url"],
                source_id=row["source_id"]
            )
        
        document = Document(
            id=row["id"],
            title=row["title"],
            author=row["author"],
            language=Language(row["language"]) if row["language"] else Language.ANCIENT_GREEK,
            period=Period(row["period"]) if row["period"] else None,
            genre=Genre(row["genre"]) if row["genre"] else None,
            source=source,
            proiel_id=row["proiel_id"],
            proiel_source_id=row["proiel_source_id"],
            date_composed=row["date_composed"],
            date_composed_start=row["date_composed_start"],
            date_composed_end=row["date_composed_end"],
            edition=row["edition"],
            editor=row["editor"],
            translator=row["translator"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            annotation_status=AnnotationStatus(row["annotation_status"]) if row["annotation_status"] else AnnotationStatus.PENDING,
            annotation_progress=row["annotation_progress"] or 0.0
        )
        
        return document
    
    def load_sentence(self, sentence_id: str) -> Optional["Sentence"]:
        """Load a sentence from the database"""
        from hlp_core.models import Sentence, Token, AnnotationStatus
        
        row = self.fetch_one("SELECT * FROM sentences WHERE id = ?", [sentence_id])
        if not row:
            return None
        
        sentence = Sentence(
            id=row["id"],
            document_id=row["document_id"],
            sentence_index=row["sentence_index"],
            text=row["text"],
            translation=row["translation"],
            proiel_id=row["proiel_id"],
            proiel_status=row["proiel_status"],
            sent_id=row["sent_id"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            annotation_status=AnnotationStatus(row["annotation_status"]) if row["annotation_status"] else AnnotationStatus.PENDING,
            annotator=row["annotator"]
        )
        
        token_rows = self.fetch_all(
            "SELECT * FROM tokens WHERE sentence_id = ? ORDER BY token_index",
            [sentence_id]
        )
        
        for token_row in token_rows:
            token = self._row_to_token(token_row)
            sentence.tokens.append(token)
        
        return sentence
    
    def _row_to_token(self, row: Any) -> "Token":
        """Convert database row to Token object"""
        from hlp_core.models import (
            Token, MorphologicalFeatures, SyntacticRelation,
            PartOfSpeech, DependencyRelation, AnnotationStatus
        )
        
        morphology = MorphologicalFeatures.from_ud_string(row["feats"] or "_")
        if row["pos"]:
            try:
                morphology.pos = PartOfSpeech(row["pos"])
            except ValueError:
                morphology.pos = PartOfSpeech.UNKNOWN
        if row["proiel_morph"]:
            morphology.proiel_morph = row["proiel_morph"]
        
        syntax = None
        if row["head"] is not None and row["deprel"]:
            try:
                relation = DependencyRelation(row["deprel"])
            except ValueError:
                relation = DependencyRelation.UNKNOWN
            
            enhanced_deps = []
            if row["deps"]:
                for dep in row["deps"].split("|"):
                    if ":" in dep:
                        h, r = dep.split(":", 1)
                        try:
                            enhanced_deps.append((int(h), r))
                        except ValueError:
                            pass
            
            syntax = SyntacticRelation(
                head_id=row["head"],
                relation=relation,
                enhanced_deps=enhanced_deps
            )
        
        return Token(
            id=row["token_index"],
            form=row["form"],
            lemma=row["lemma"],
            morphology=morphology,
            syntax=syntax,
            misc=json.loads(row["misc"]) if row["misc"] else {},
            proiel_id=row["proiel_id"],
            span_start=row["span_start"],
            span_end=row["span_end"],
            is_multiword=bool(row["is_multiword"]),
            multiword_id=row["multiword_id"],
            is_empty=bool(row["is_empty"]),
            empty_node_id=row["empty_node_id"],
            annotation_status=AnnotationStatus(row["annotation_status"]) if row["annotation_status"] else AnnotationStatus.PENDING,
            annotator=row["annotator"]
        )
    
    def search_documents(
        self,
        language: Optional[str] = None,
        period: Optional[str] = None,
        genre: Optional[str] = None,
        author: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List["Document"]:
        """Search documents with filters"""
        qb = self.query("documents")
        
        if language:
            qb.where_eq("language", language)
        if period:
            qb.where_eq("period", period)
        if genre:
            qb.where_eq("genre", genre)
        if author:
            qb.where_like("author", f"%{author}%")
        if status:
            qb.where_eq("annotation_status", status)
        
        qb.order_by("created_at", "DESC")
        qb.limit(limit)
        qb.offset(offset)
        
        sql, params = qb.build_select()
        rows = self.fetch_all(sql, params)
        
        documents = []
        for row in rows:
            doc = self.load_document(row["id"])
            if doc:
                documents.append(doc)
        
        return documents
    
    def search_sentences(
        self,
        document_id: Optional[str] = None,
        text_pattern: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List["Sentence"]:
        """Search sentences with filters"""
        qb = self.query("sentences")
        
        if document_id:
            qb.where_eq("document_id", document_id)
        if text_pattern:
            qb.where_like("text", f"%{text_pattern}%")
        if status:
            qb.where_eq("annotation_status", status)
        
        qb.order_by("document_id", "ASC")
        qb.order_by("sentence_index", "ASC")
        qb.limit(limit)
        qb.offset(offset)
        
        sql, params = qb.build_select()
        rows = self.fetch_all(sql, params)
        
        sentences = []
        for row in rows:
            sent = self.load_sentence(row["id"])
            if sent:
                sentences.append(sent)
        
        return sentences
    
    def get_valency_patterns(
        self,
        verb_lemma: Optional[str] = None,
        language: Optional[str] = None,
        period: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get valency patterns with filters"""
        qb = self.query("valency_patterns")
        
        if verb_lemma:
            qb.where_eq("verb_lemma", verb_lemma)
        if language:
            qb.where_eq("language", language)
        if period:
            qb.where_eq("period", period)
        
        qb.order_by("frequency", "DESC")
        qb.limit(limit)
        
        sql, params = qb.build_select()
        rows = self.fetch_all(sql, params)
        
        patterns = []
        for row in rows:
            patterns.append({
                "pattern_id": row["pattern_id"],
                "verb_lemma": row["verb_lemma"],
                "frame_id": row["frame_id"],
                "language": row["language"],
                "period": row["period"],
                "frequency": row["frequency"],
                "relative_frequency": row["relative_frequency"],
                "confidence": row["confidence"],
                "extraction_method": row["extraction_method"]
            })
        
        return patterns
    
    def get_frequency_data(
        self,
        feature_type: str,
        feature_value: Optional[str] = None,
        language: Optional[str] = None,
        period: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get frequency data for diachronic analysis"""
        qb = self.query("frequency_data")
        qb.where_eq("feature_type", feature_type)
        
        if feature_value:
            qb.where_eq("feature_value", feature_value)
        if language:
            qb.where_eq("language", language)
        if period:
            qb.where_eq("period", period)
        
        qb.order_by("period", "ASC")
        
        sql, params = qb.build_select()
        rows = self.fetch_all(sql, params)
        
        return [dict(row) for row in rows]


def get_default_db_manager() -> DatabaseManager:
    """Get default database manager instance"""
    try:
        from config import config
        db_path = str(config.corpus_db_path)
    except ImportError:
        import os
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "hlp_corpus.db")
    
    config = DatabaseConfig(
        db_type=DatabaseType.SQLITE,
        sqlite_path=db_path
    )
    
    manager = DatabaseManager(config)
    manager.initialize()
    
    return manager
