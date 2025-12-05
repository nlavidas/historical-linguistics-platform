"""
HLP Valency Lexicon Builder - Build and Manage Valency Lexicons

This module provides utilities for building, managing, and exporting
valency lexicons from extracted patterns.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Iterator
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from hlp_valency.pattern_extractor import (
    ExtractedFrame, Argument, ArgumentType, FrameType, ExtractionResult
)
from hlp_valency.pattern_normalization import (
    PatternNormalizer, NormalizationConfig, NormalizedPattern, NormalizedArgument
)
from hlp_core.models import (
    Language, Period, Case, ValencyFrame, ValencyPattern
)

logger = logging.getLogger(__name__)


@dataclass
class FrameInstance:
    """A single instance of a valency frame"""
    sentence_id: str
    document_id: str
    
    verb_form: str
    
    source_text: Optional[str] = None
    
    voice: Optional[str] = None
    tense: Optional[str] = None
    mood: Optional[str] = None
    
    period: Optional[Period] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "sentence_id": self.sentence_id,
            "document_id": self.document_id,
            "verb_form": self.verb_form,
            "source_text": self.source_text,
            "voice": self.voice,
            "tense": self.tense,
            "mood": self.mood,
            "period": self.period.value if self.period else None
        }


@dataclass
class PatternEntry:
    """Entry for a specific valency pattern"""
    pattern_string: str
    canonical_form: str
    
    arguments: List[NormalizedArgument]
    
    frame_type: FrameType
    
    frequency: int = 0
    
    instances: List[FrameInstance] = field(default_factory=list)
    
    period_distribution: Dict[str, int] = field(default_factory=dict)
    
    voice_distribution: Dict[str, int] = field(default_factory=dict)
    
    confidence: float = 1.0
    
    notes: List[str] = field(default_factory=list)
    
    def add_instance(self, instance: FrameInstance):
        """Add an instance of this pattern"""
        self.instances.append(instance)
        self.frequency += 1
        
        if instance.period:
            period_key = instance.period.value
            self.period_distribution[period_key] = self.period_distribution.get(period_key, 0) + 1
        
        if instance.voice:
            self.voice_distribution[instance.voice] = self.voice_distribution.get(instance.voice, 0) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pattern_string": self.pattern_string,
            "canonical_form": self.canonical_form,
            "arguments": [a.to_dict() for a in self.arguments],
            "frame_type": self.frame_type.value,
            "frequency": self.frequency,
            "period_distribution": self.period_distribution,
            "voice_distribution": self.voice_distribution,
            "confidence": self.confidence,
            "notes": self.notes,
            "instance_count": len(self.instances)
        }


@dataclass
class LexiconEntry:
    """Entry for a verb in the valency lexicon"""
    lemma: str
    
    patterns: Dict[str, PatternEntry] = field(default_factory=dict)
    
    total_frequency: int = 0
    
    primary_pattern: Optional[str] = None
    
    frame_types: Set[FrameType] = field(default_factory=set)
    
    glosses: Dict[str, str] = field(default_factory=dict)
    
    notes: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_pattern(self, pattern: NormalizedPattern, instance: FrameInstance):
        """Add a pattern to this entry"""
        key = pattern.canonical_form
        
        if key not in self.patterns:
            self.patterns[key] = PatternEntry(
                pattern_string=pattern.pattern_string,
                canonical_form=pattern.canonical_form,
                arguments=pattern.arguments,
                frame_type=pattern.frame_type
            )
        
        self.patterns[key].add_instance(instance)
        self.total_frequency += 1
        self.frame_types.add(pattern.frame_type)
        
        self._update_primary_pattern()
    
    def _update_primary_pattern(self):
        """Update the primary (most frequent) pattern"""
        if self.patterns:
            self.primary_pattern = max(
                self.patterns.keys(),
                key=lambda k: self.patterns[k].frequency
            )
    
    def get_pattern_distribution(self) -> Dict[str, float]:
        """Get distribution of patterns"""
        if self.total_frequency == 0:
            return {}
        
        return {
            k: v.frequency / self.total_frequency
            for k, v in self.patterns.items()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "lemma": self.lemma,
            "patterns": {k: v.to_dict() for k, v in self.patterns.items()},
            "total_frequency": self.total_frequency,
            "primary_pattern": self.primary_pattern,
            "frame_types": [ft.value for ft in self.frame_types],
            "glosses": self.glosses,
            "notes": self.notes,
            "pattern_count": len(self.patterns)
        }


@dataclass
class ValencyLexicon:
    """A complete valency lexicon"""
    name: str
    language: Language
    
    entries: Dict[str, LexiconEntry] = field(default_factory=dict)
    
    version: str = "1.0.0"
    
    description: Optional[str] = None
    
    source_corpora: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_entry(self, lemma: str) -> LexiconEntry:
        """Add or get entry for lemma"""
        if lemma not in self.entries:
            self.entries[lemma] = LexiconEntry(lemma=lemma)
        return self.entries[lemma]
    
    def get_entry(self, lemma: str) -> Optional[LexiconEntry]:
        """Get entry for lemma"""
        return self.entries.get(lemma)
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __iter__(self) -> Iterator[LexiconEntry]:
        return iter(self.entries.values())
    
    def __contains__(self, lemma: str) -> bool:
        return lemma in self.entries
    
    @property
    def total_patterns(self) -> int:
        """Get total number of patterns"""
        return sum(len(e.patterns) for e in self.entries.values())
    
    @property
    def total_instances(self) -> int:
        """Get total number of instances"""
        return sum(e.total_frequency for e in self.entries.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get lexicon statistics"""
        frame_type_counts = defaultdict(int)
        pattern_frequencies = []
        
        for entry in self.entries.values():
            for ft in entry.frame_types:
                frame_type_counts[ft.value] += 1
            for pattern in entry.patterns.values():
                pattern_frequencies.append(pattern.frequency)
        
        return {
            "entry_count": len(self.entries),
            "total_patterns": self.total_patterns,
            "total_instances": self.total_instances,
            "frame_type_distribution": dict(frame_type_counts),
            "avg_patterns_per_entry": self.total_patterns / len(self.entries) if self.entries else 0,
            "avg_frequency_per_pattern": sum(pattern_frequencies) / len(pattern_frequencies) if pattern_frequencies else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "language": self.language.value,
            "version": self.version,
            "description": self.description,
            "source_corpora": self.source_corpora,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "statistics": self.get_statistics(),
            "entries": {k: v.to_dict() for k, v in self.entries.items()}
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValencyLexicon:
        """Create from dictionary"""
        lexicon = cls(
            name=data["name"],
            language=Language(data["language"]),
            version=data.get("version", "1.0.0"),
            description=data.get("description"),
            source_corpora=data.get("source_corpora", [])
        )
        
        for lemma, entry_data in data.get("entries", {}).items():
            entry = lexicon.add_entry(lemma)
            entry.total_frequency = entry_data.get("total_frequency", 0)
            entry.primary_pattern = entry_data.get("primary_pattern")
            entry.glosses = entry_data.get("glosses", {})
            entry.notes = entry_data.get("notes", [])
            
            for ft_str in entry_data.get("frame_types", []):
                try:
                    entry.frame_types.add(FrameType(ft_str))
                except ValueError:
                    pass
        
        return lexicon
    
    @classmethod
    def from_json(cls, json_string: str) -> ValencyLexicon:
        """Create from JSON string"""
        data = json.loads(json_string)
        return cls.from_dict(data)


class LexiconBuilder:
    """Builds valency lexicons from extracted frames"""
    
    def __init__(
        self,
        language: Language = Language.ANCIENT_GREEK,
        normalization_config: Optional[NormalizationConfig] = None
    ):
        self.language = language
        self.normalizer = PatternNormalizer(normalization_config)
        self._lexicon: Optional[ValencyLexicon] = None
    
    def build_from_extraction_result(
        self,
        result: ExtractionResult,
        lexicon_name: str = "valency_lexicon",
        corpus_name: Optional[str] = None
    ) -> ValencyLexicon:
        """Build lexicon from extraction result"""
        self._lexicon = ValencyLexicon(
            name=lexicon_name,
            language=self.language,
            description=f"Valency lexicon built from {len(result.frames)} frames"
        )
        
        if corpus_name:
            self._lexicon.source_corpora.append(corpus_name)
        
        for frame in result.frames:
            self._add_frame(frame)
        
        return self._lexicon
    
    def build_from_frames(
        self,
        frames: List[ExtractedFrame],
        lexicon_name: str = "valency_lexicon"
    ) -> ValencyLexicon:
        """Build lexicon from list of frames"""
        self._lexicon = ValencyLexicon(
            name=lexicon_name,
            language=self.language,
            description=f"Valency lexicon built from {len(frames)} frames"
        )
        
        for frame in frames:
            self._add_frame(frame)
        
        return self._lexicon
    
    def _add_frame(self, frame: ExtractedFrame):
        """Add a frame to the lexicon"""
        if not self._lexicon:
            return
        
        normalized = self.normalizer.normalize_frame(frame)
        
        entry = self._lexicon.add_entry(normalized.verb_lemma)
        
        instance = FrameInstance(
            sentence_id=frame.sentence_id or "",
            document_id=frame.document_id or "",
            verb_form=frame.verb_form,
            source_text=frame.source_text,
            voice=frame.voice,
            tense=frame.tense,
            mood=frame.mood
        )
        
        entry.add_pattern(normalized, instance)
    
    def add_frame(self, frame: ExtractedFrame):
        """Add a single frame to current lexicon"""
        if self._lexicon is None:
            self._lexicon = ValencyLexicon(
                name="valency_lexicon",
                language=self.language
            )
        
        self._add_frame(frame)
    
    def get_lexicon(self) -> Optional[ValencyLexicon]:
        """Get the built lexicon"""
        return self._lexicon
    
    def export_to_file(self, file_path: Union[str, Path], format: str = "json"):
        """Export lexicon to file"""
        if not self._lexicon:
            raise ValueError("No lexicon to export")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self._lexicon.to_json())
        elif format == "tsv":
            self._export_tsv(file_path)
        elif format == "xml":
            self._export_xml(file_path)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _export_tsv(self, file_path: Path):
        """Export to TSV format"""
        lines = ["lemma\tpattern\tfrequency\tframe_type\targuments"]
        
        for entry in self._lexicon:
            for pattern in entry.patterns.values():
                args_str = "; ".join(a.to_string() for a in pattern.arguments)
                line = f"{entry.lemma}\t{pattern.pattern_string}\t{pattern.frequency}\t{pattern.frame_type.value}\t{args_str}"
                lines.append(line)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def _export_xml(self, file_path: Path):
        """Export to XML format"""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append(f'<lexicon name="{self._lexicon.name}" language="{self._lexicon.language.value}">')
        
        for entry in self._lexicon:
            lines.append(f'  <entry lemma="{entry.lemma}">')
            
            for pattern in entry.patterns.values():
                lines.append(f'    <pattern canonical="{pattern.canonical_form}" frequency="{pattern.frequency}">')
                lines.append(f'      <frame_type>{pattern.frame_type.value}</frame_type>')
                lines.append('      <arguments>')
                
                for arg in pattern.arguments:
                    case_attr = f' case="{arg.case.value}"' if arg.case else ''
                    prep_attr = f' preposition="{arg.preposition}"' if arg.preposition else ''
                    lines.append(f'        <argument type="{arg.arg_type.value}"{case_attr}{prep_attr}/>')
                
                lines.append('      </arguments>')
                lines.append('    </pattern>')
            
            lines.append('  </entry>')
        
        lines.append('</lexicon>')
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def build_lexicon(
    frames: List[ExtractedFrame],
    language: Language = Language.ANCIENT_GREEK,
    lexicon_name: str = "valency_lexicon"
) -> ValencyLexicon:
    """Build lexicon from frames"""
    builder = LexiconBuilder(language)
    return builder.build_from_frames(frames, lexicon_name)


def merge_lexicons(
    lexicons: List[ValencyLexicon],
    name: str = "merged_lexicon"
) -> ValencyLexicon:
    """Merge multiple lexicons"""
    if not lexicons:
        raise ValueError("No lexicons to merge")
    
    merged = ValencyLexicon(
        name=name,
        language=lexicons[0].language,
        description=f"Merged from {len(lexicons)} lexicons"
    )
    
    for lexicon in lexicons:
        merged.source_corpora.extend(lexicon.source_corpora)
        
        for entry in lexicon:
            merged_entry = merged.add_entry(entry.lemma)
            
            for pattern_key, pattern in entry.patterns.items():
                if pattern_key not in merged_entry.patterns:
                    merged_entry.patterns[pattern_key] = PatternEntry(
                        pattern_string=pattern.pattern_string,
                        canonical_form=pattern.canonical_form,
                        arguments=pattern.arguments.copy(),
                        frame_type=pattern.frame_type
                    )
                
                merged_entry.patterns[pattern_key].frequency += pattern.frequency
                merged_entry.patterns[pattern_key].instances.extend(pattern.instances)
                
                for period, count in pattern.period_distribution.items():
                    merged_entry.patterns[pattern_key].period_distribution[period] = \
                        merged_entry.patterns[pattern_key].period_distribution.get(period, 0) + count
            
            merged_entry.total_frequency += entry.total_frequency
            merged_entry.frame_types.update(entry.frame_types)
            merged_entry._update_primary_pattern()
    
    return merged


def export_lexicon(
    lexicon: ValencyLexicon,
    file_path: Union[str, Path],
    format: str = "json"
):
    """Export lexicon to file"""
    builder = LexiconBuilder(lexicon.language)
    builder._lexicon = lexicon
    builder.export_to_file(file_path, format)
