"""
Influential Texts Registry - Catalog of influential texts across all periods

This module provides a comprehensive registry of influential texts for:
- Ancient Greek (Homer, Hesiod, Tragedians, Philosophers)
- Hellenistic/Koine Greek (Septuagint, New Testament, Polybius)
- Byzantine Greek (Malalas, Psellos, Anna Komnene, Planudes)
- Medieval Latin (Vulgate, Church Fathers)
- Old English (Beowulf, Alfred)
- Middle English (Chaucer, Wycliffe)
- Early Modern English (Shakespeare, KJV)

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class InfluenceType(Enum):
    LITERARY = "literary"
    RELIGIOUS = "religious"
    PHILOSOPHICAL = "philosophical"
    HISTORICAL = "historical"
    SCIENTIFIC = "scientific"
    LEGAL = "legal"
    LINGUISTIC = "linguistic"


class Period(Enum):
    ARCHAIC_GREEK = "Archaic Greek (800-500 BCE)"
    CLASSICAL_GREEK = "Classical Greek (500-323 BCE)"
    HELLENISTIC = "Hellenistic (323-31 BCE)"
    KOINE = "Koine Greek (300 BCE - 300 CE)"
    LATE_ANCIENT = "Late Ancient (300-600 CE)"
    EARLY_BYZANTINE = "Early Byzantine (600-850 CE)"
    MIDDLE_BYZANTINE = "Middle Byzantine (850-1204 CE)"
    LATE_BYZANTINE = "Late Byzantine (1204-1453 CE)"
    OLD_ENGLISH = "Old English (450-1100 CE)"
    MIDDLE_ENGLISH = "Middle English (1100-1500 CE)"
    EARLY_MODERN = "Early Modern (1500-1700 CE)"
    MODERN = "Modern (1700-present)"


@dataclass
class TextInfluence:
    text_id: str
    title: str
    author: str
    language: str
    period: Period
    date_composed: str
    influence_type: InfluenceType
    influence_score: float
    description: str
    key_features: List[str] = field(default_factory=list)
    influenced_works: List[str] = field(default_factory=list)
    linguistic_significance: str = ""
    valency_notes: str = ""
    available_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text_id': self.text_id,
            'title': self.title,
            'author': self.author,
            'language': self.language,
            'period': self.period.value,
            'date_composed': self.date_composed,
            'influence_type': self.influence_type.value,
            'influence_score': self.influence_score,
            'description': self.description,
            'key_features': self.key_features,
            'influenced_works': self.influenced_works,
            'linguistic_significance': self.linguistic_significance,
            'valency_notes': self.valency_notes,
            'available_sources': self.available_sources,
            'metadata': self.metadata,
        }


class InfluentialTextsRegistry:
    
    def __init__(self):
        self.texts: Dict[str, TextInfluence] = {}
        self._load_registry()
    
    def _load_registry(self):
        self._add_archaic_greek()
        self._add_classical_greek()
        self._add_hellenistic_koine()
        self._add_byzantine()
        self._add_latin()
        self._add_old_english()
        self._add_middle_english()
        self._add_early_modern()
    
    def _add_archaic_greek(self):
        self.texts['homer_iliad'] = TextInfluence(
            text_id='homer_iliad',
            title='Iliad',
            author='Homer',
            language='grc',
            period=Period.ARCHAIC_GREEK,
            date_composed='8th century BCE',
            influence_type=InfluenceType.LITERARY,
            influence_score=10.0,
            description='Epic poem about the Trojan War, foundational text of Western literature',
            key_features=['Epic hexameter', 'Oral formulaic composition', 'Divine intervention', 'Heroic code'],
            influenced_works=['Odyssey', 'Aeneid', 'Paradise Lost', 'All Western epic poetry'],
            linguistic_significance='Earliest extensive Greek text, preserves archaic forms, Ionic dialect with Aeolic elements',
            valency_notes='Rich verbal system with complex argument structures, middle voice patterns',
            available_sources=['Perseus', 'First1KGreek', 'Gutenberg'],
        )
        
        self.texts['homer_odyssey'] = TextInfluence(
            text_id='homer_odyssey',
            title='Odyssey',
            author='Homer',
            language='grc',
            period=Period.ARCHAIC_GREEK,
            date_composed='8th century BCE',
            influence_type=InfluenceType.LITERARY,
            influence_score=10.0,
            description='Epic poem about Odysseus return from Troy',
            key_features=['Epic hexameter', 'Nostos theme', 'Xenia (hospitality)', 'Narrative complexity'],
            influenced_works=['Aeneid', 'Ulysses (Joyce)', 'O Brother Where Art Thou'],
            linguistic_significance='Slightly later linguistic features than Iliad, important for Greek verb morphology',
            valency_notes='Motion verbs, verbs of speaking, perception verbs with varied argument structures',
            available_sources=['Perseus', 'First1KGreek', 'Gutenberg'],
        )
        
        self.texts['hesiod_theogony'] = TextInfluence(
            text_id='hesiod_theogony',
            title='Theogony',
            author='Hesiod',
            language='grc',
            period=Period.ARCHAIC_GREEK,
            date_composed='7th century BCE',
            influence_type=InfluenceType.RELIGIOUS,
            influence_score=9.0,
            description='Genealogy of the Greek gods',
            key_features=['Cosmogony', 'Divine genealogy', 'Succession myth'],
            influenced_works=['Greek religion', 'Roman mythology', 'Western cosmological thought'],
            linguistic_significance='Boeotian dialect features, archaic vocabulary',
            available_sources=['Perseus', 'First1KGreek'],
        )
        
        self.texts['hesiod_works_days'] = TextInfluence(
            text_id='hesiod_works_days',
            title='Works and Days',
            author='Hesiod',
            language='grc',
            period=Period.ARCHAIC_GREEK,
            date_composed='7th century BCE',
            influence_type=InfluenceType.PHILOSOPHICAL,
            influence_score=8.5,
            description='Didactic poem on agriculture and moral life',
            key_features=['Didactic poetry', 'Agricultural calendar', 'Five Ages of Man'],
            influenced_works=['Virgil Georgics', 'Didactic poetry tradition'],
            linguistic_significance='Important for understanding archaic Greek vocabulary and syntax',
            available_sources=['Perseus', 'First1KGreek'],
        )
    
    def _add_classical_greek(self):
        self.texts['herodotus_histories'] = TextInfluence(
            text_id='herodotus_histories',
            title='Histories',
            author='Herodotus',
            language='grc',
            period=Period.CLASSICAL_GREEK,
            date_composed='5th century BCE',
            influence_type=InfluenceType.HISTORICAL,
            influence_score=9.5,
            description='First major prose work of Western historiography',
            key_features=['Ionic prose', 'Ethnography', 'Persian Wars narrative'],
            influenced_works=['Thucydides', 'All Western historiography'],
            linguistic_significance='Ionic dialect prose, important for Greek syntax development',
            valency_notes='Verbs of saying, thinking, perceiving with complex complementation',
            available_sources=['Perseus', 'First1KGreek', 'Gutenberg'],
        )
        
        self.texts['thucydides_peloponnesian'] = TextInfluence(
            text_id='thucydides_peloponnesian',
            title='History of the Peloponnesian War',
            author='Thucydides',
            language='grc',
            period=Period.CLASSICAL_GREEK,
            date_composed='5th century BCE',
            influence_type=InfluenceType.HISTORICAL,
            influence_score=9.5,
            description='Analytical history of the war between Athens and Sparta',
            key_features=['Attic prose', 'Political analysis', 'Speeches', 'Scientific method'],
            influenced_works=['Political science', 'International relations theory'],
            linguistic_significance='Complex Attic prose style, abstract vocabulary',
            available_sources=['Perseus', 'First1KGreek', 'Gutenberg'],
        )
        
        self.texts['plato_republic'] = TextInfluence(
            text_id='plato_republic',
            title='Republic',
            author='Plato',
            language='grc',
            period=Period.CLASSICAL_GREEK,
            date_composed='4th century BCE',
            influence_type=InfluenceType.PHILOSOPHICAL,
            influence_score=10.0,
            description='Dialogue on justice and the ideal state',
            key_features=['Socratic dialogue', 'Theory of Forms', 'Allegory of the Cave'],
            influenced_works=['All Western philosophy', 'Political theory'],
            linguistic_significance='Model of Attic prose, philosophical vocabulary',
            available_sources=['Perseus', 'First1KGreek', 'Gutenberg'],
        )
        
        self.texts['aristotle_poetics'] = TextInfluence(
            text_id='aristotle_poetics',
            title='Poetics',
            author='Aristotle',
            language='grc',
            period=Period.CLASSICAL_GREEK,
            date_composed='4th century BCE',
            influence_type=InfluenceType.LITERARY,
            influence_score=9.5,
            description='Treatise on dramatic theory',
            key_features=['Tragedy theory', 'Catharsis', 'Mimesis', 'Plot structure'],
            influenced_works=['All Western literary criticism', 'Drama theory'],
            linguistic_significance='Technical philosophical vocabulary',
            available_sources=['Perseus', 'First1KGreek', 'Gutenberg'],
        )
        
        self.texts['sophocles_oedipus'] = TextInfluence(
            text_id='sophocles_oedipus',
            title='Oedipus Tyrannus',
            author='Sophocles',
            language='grc',
            period=Period.CLASSICAL_GREEK,
            date_composed='5th century BCE',
            influence_type=InfluenceType.LITERARY,
            influence_score=9.5,
            description='Tragedy of Oedipus discovering his fate',
            key_features=['Tragic irony', 'Peripeteia', 'Anagnorisis', 'Choral odes'],
            influenced_works=['Freudian psychology', 'Western drama'],
            linguistic_significance='Attic tragic diction, lyric meters',
            available_sources=['Perseus', 'First1KGreek'],
        )
        
        self.texts['euripides_medea'] = TextInfluence(
            text_id='euripides_medea',
            title='Medea',
            author='Euripides',
            language='grc',
            period=Period.CLASSICAL_GREEK,
            date_composed='5th century BCE',
            influence_type=InfluenceType.LITERARY,
            influence_score=9.0,
            description='Tragedy of Medea revenge on Jason',
            key_features=['Psychological realism', 'Female protagonist', 'Rhetoric'],
            influenced_works=['Roman tragedy', 'Modern drama'],
            linguistic_significance='Colloquial Attic elements, emotional vocabulary',
            available_sources=['Perseus', 'First1KGreek', 'Gutenberg'],
        )
        
        self.texts['aeschylus_oresteia'] = TextInfluence(
            text_id='aeschylus_oresteia',
            title='Oresteia',
            author='Aeschylus',
            language='grc',
            period=Period.CLASSICAL_GREEK,
            date_composed='5th century BCE',
            influence_type=InfluenceType.LITERARY,
            influence_score=9.5,
            description='Trilogy on the House of Atreus',
            key_features=['Trilogy structure', 'Justice theme', 'Divine-human interaction'],
            influenced_works=['Greek tragedy', 'Legal philosophy'],
            linguistic_significance='Archaic tragic diction, complex compound words',
            available_sources=['Perseus', 'First1KGreek'],
        )
    
    def _add_hellenistic_koine(self):
        self.texts['septuagint'] = TextInfluence(
            text_id='septuagint',
            title='Septuagint',
            author='Seventy Translators',
            language='grc',
            period=Period.HELLENISTIC,
            date_composed='3rd-2nd century BCE',
            influence_type=InfluenceType.RELIGIOUS,
            influence_score=10.0,
            description='Greek translation of the Hebrew Bible',
            key_features=['Translation Greek', 'Hebraisms', 'Religious vocabulary'],
            influenced_works=['New Testament', 'Church Fathers', 'Western Christianity'],
            linguistic_significance='Major source for Koine Greek, translation interference patterns',
            valency_notes='Hebrew-influenced argument structures, calques',
            available_sources=['Perseus', 'First1KGreek', 'Gutenberg'],
        )
        
        self.texts['new_testament'] = TextInfluence(
            text_id='new_testament',
            title='New Testament',
            author='Various',
            language='grc',
            period=Period.KOINE,
            date_composed='1st century CE',
            influence_type=InfluenceType.RELIGIOUS,
            influence_score=10.0,
            description='Christian scriptures in Koine Greek',
            key_features=['Koine Greek', 'Semitisms', 'Varied registers', 'Narrative and epistolary'],
            influenced_works=['All Christian literature', 'Western civilization'],
            linguistic_significance='Most studied Koine text, shows vernacular features',
            valency_notes='Verbs of motion, speaking, giving with Semitic influence',
            available_sources=['Perseus', 'First1KGreek', 'Gutenberg', 'PROIEL'],
        )
        
        self.texts['polybius_histories'] = TextInfluence(
            text_id='polybius_histories',
            title='Histories',
            author='Polybius',
            language='grc',
            period=Period.HELLENISTIC,
            date_composed='2nd century BCE',
            influence_type=InfluenceType.HISTORICAL,
            influence_score=8.5,
            description='History of Rome rise to power',
            key_features=['Pragmatic history', 'Political analysis', 'Eyewitness accounts'],
            influenced_works=['Roman historiography', 'Political theory'],
            linguistic_significance='Hellenistic Koine prose, technical vocabulary',
            available_sources=['Perseus', 'First1KGreek'],
        )
        
        self.texts['marcus_aurelius_meditations'] = TextInfluence(
            text_id='marcus_aurelius_meditations',
            title='Meditations',
            author='Marcus Aurelius',
            language='grc',
            period=Period.KOINE,
            date_composed='2nd century CE',
            influence_type=InfluenceType.PHILOSOPHICAL,
            influence_score=9.0,
            description='Stoic philosophical reflections',
            key_features=['Personal philosophy', 'Stoicism', 'Self-examination'],
            influenced_works=['Stoic philosophy', 'Self-help literature'],
            linguistic_significance='Late Koine Greek, philosophical vocabulary',
            available_sources=['Perseus', 'Gutenberg'],
        )
    
    def _add_byzantine(self):
        self.texts['malalas_chronographia'] = TextInfluence(
            text_id='malalas_chronographia',
            title='Chronographia',
            author='John Malalas',
            language='grc',
            period=Period.EARLY_BYZANTINE,
            date_composed='6th century CE',
            influence_type=InfluenceType.HISTORICAL,
            influence_score=8.0,
            description='World chronicle from creation to Justinian',
            key_features=['Chronicle genre', 'Popular style', 'Vernacular elements'],
            influenced_works=['Byzantine chronicles', 'Slavic chronicles'],
            linguistic_significance='Early Medieval Greek, vernacular features, important for Greek language history',
            valency_notes='Shows transition from Koine to Medieval Greek verbal system',
            available_sources=['Internet Archive', 'Byzantine archives'],
        )
        
        self.texts['theophanes_chronographia'] = TextInfluence(
            text_id='theophanes_chronographia',
            title='Chronographia',
            author='Theophanes the Confessor',
            language='grc',
            period=Period.MIDDLE_BYZANTINE,
            date_composed='9th century CE',
            influence_type=InfluenceType.HISTORICAL,
            influence_score=8.5,
            description='Chronicle covering 284-813 CE',
            key_features=['Annalistic structure', 'Iconoclasm coverage', 'Source compilation'],
            influenced_works=['Later Byzantine chronicles'],
            linguistic_significance='Middle Byzantine Greek, shows language development',
            available_sources=['Internet Archive', 'Byzantine archives'],
        )
        
        self.texts['psellos_chronographia'] = TextInfluence(
            text_id='psellos_chronographia',
            title='Chronographia',
            author='Michael Psellos',
            language='grc',
            period=Period.MIDDLE_BYZANTINE,
            date_composed='11th century CE',
            influence_type=InfluenceType.HISTORICAL,
            influence_score=9.0,
            description='History of Byzantine emperors 976-1078',
            key_features=['Psychological portraits', 'Classicizing style', 'Personal observation'],
            influenced_works=['Byzantine historiography', 'Biography genre'],
            linguistic_significance='Learned Byzantine Greek, Atticizing tendencies',
            available_sources=['Internet Archive', 'Byzantine archives'],
        )
        
        self.texts['anna_komnene_alexiad'] = TextInfluence(
            text_id='anna_komnene_alexiad',
            title='Alexiad',
            author='Anna Komnene',
            language='grc',
            period=Period.MIDDLE_BYZANTINE,
            date_composed='12th century CE',
            influence_type=InfluenceType.HISTORICAL,
            influence_score=9.0,
            description='History of Emperor Alexios I Komnenos',
            key_features=['Female author', 'Classicizing style', 'First Crusade account'],
            influenced_works=['Byzantine historiography', 'Crusade studies'],
            linguistic_significance='Learned Byzantine Greek, complex syntax',
            available_sources=['Internet Archive', 'Byzantine archives'],
        )
        
        self.texts['planudes_translations'] = TextInfluence(
            text_id='planudes_translations',
            title='Translations (Ovid, Boethius, Cato, Augustine)',
            author='Maximus Planudes',
            language='grc',
            period=Period.LATE_BYZANTINE,
            date_composed='13th century CE',
            influence_type=InfluenceType.LITERARY,
            influence_score=9.0,
            description='Greek translations of Latin classics',
            key_features=['Latin to Greek translation', 'Interlingual translation', 'Cultural transmission'],
            influenced_works=['Byzantine scholarship', 'Translation studies'],
            linguistic_significance='Important for understanding Byzantine translation techniques and Greek-Latin contact',
            valency_notes='Shows how Latin argument structures were adapted to Greek',
            available_sources=['Internet Archive', 'Byzantine archives'],
        )
        
        self.texts['digenis_akritas'] = TextInfluence(
            text_id='digenis_akritas',
            title='Digenis Akritas',
            author='Anonymous',
            language='grc',
            period=Period.MIDDLE_BYZANTINE,
            date_composed='12th century CE',
            influence_type=InfluenceType.LITERARY,
            influence_score=8.5,
            description='Byzantine epic poem about a border warrior',
            key_features=['Vernacular Greek', 'Epic poetry', 'Frontier culture'],
            influenced_works=['Modern Greek literature', 'Folk poetry'],
            linguistic_significance='Important witness to vernacular Medieval Greek',
            available_sources=['Internet Archive', 'Byzantine archives'],
        )
        
        self.texts['chronicle_morea'] = TextInfluence(
            text_id='chronicle_morea',
            title='Chronicle of Morea',
            author='Anonymous',
            language='grc',
            period=Period.LATE_BYZANTINE,
            date_composed='14th century CE',
            influence_type=InfluenceType.HISTORICAL,
            influence_score=8.0,
            description='Chronicle of Frankish Greece',
            key_features=['Vernacular Greek', 'Crusader period', 'Political verse'],
            influenced_works=['Late Byzantine literature'],
            linguistic_significance='Important for vernacular Medieval Greek, French loanwords',
            available_sources=['Internet Archive', 'Byzantine archives'],
        )
        
        self.texts['john_chrysostom_homilies'] = TextInfluence(
            text_id='john_chrysostom_homilies',
            title='Homilies',
            author='John Chrysostom',
            language='grc',
            period=Period.LATE_ANCIENT,
            date_composed='4th century CE',
            influence_type=InfluenceType.RELIGIOUS,
            influence_score=9.5,
            description='Sermons and biblical commentaries',
            key_features=['Rhetorical skill', 'Biblical exegesis', 'Moral instruction'],
            influenced_works=['Christian preaching', 'Patristic literature'],
            linguistic_significance='Late Koine Greek, rhetorical style',
            available_sources=['Perseus', 'First1KGreek', 'Internet Archive'],
        )
    
    def _add_latin(self):
        self.texts['vulgate'] = TextInfluence(
            text_id='vulgate',
            title='Vulgate Bible',
            author='Jerome',
            language='lat',
            period=Period.LATE_ANCIENT,
            date_composed='4th century CE',
            influence_type=InfluenceType.RELIGIOUS,
            influence_score=10.0,
            description='Latin translation of the Bible',
            key_features=['Translation Latin', 'Christian vocabulary', 'Hebraisms'],
            influenced_works=['Medieval Latin', 'Western Christianity', 'Romance languages'],
            linguistic_significance='Major influence on Medieval Latin, source of Christian vocabulary',
            available_sources=['Perseus', 'Gutenberg'],
        )
        
        self.texts['virgil_aeneid'] = TextInfluence(
            text_id='virgil_aeneid',
            title='Aeneid',
            author='Virgil',
            language='lat',
            period=Period.CLASSICAL_GREEK,
            date_composed='1st century BCE',
            influence_type=InfluenceType.LITERARY,
            influence_score=10.0,
            description='Roman national epic',
            key_features=['Epic hexameter', 'Roman values', 'Augustan ideology'],
            influenced_works=['Dante', 'Milton', 'All Western epic'],
            linguistic_significance='Model of Classical Latin poetry',
            available_sources=['Perseus', 'Gutenberg'],
        )
        
        self.texts['boethius_consolation'] = TextInfluence(
            text_id='boethius_consolation',
            title='Consolation of Philosophy',
            author='Boethius',
            language='lat',
            period=Period.LATE_ANCIENT,
            date_composed='6th century CE',
            influence_type=InfluenceType.PHILOSOPHICAL,
            influence_score=9.5,
            description='Philosophical dialogue on fortune and providence',
            key_features=['Prosimetrum', 'Neoplatonism', 'Fortune theme'],
            influenced_works=['Medieval philosophy', 'Chaucer', 'King Alfred translation'],
            linguistic_significance='Late Latin, important for Medieval philosophy vocabulary',
            available_sources=['Perseus', 'Gutenberg'],
        )
    
    def _add_old_english(self):
        self.texts['beowulf'] = TextInfluence(
            text_id='beowulf',
            title='Beowulf',
            author='Anonymous',
            language='ang',
            period=Period.OLD_ENGLISH,
            date_composed='8th-11th century CE',
            influence_type=InfluenceType.LITERARY,
            influence_score=10.0,
            description='Old English epic poem',
            key_features=['Alliterative verse', 'Heroic code', 'Monster fights', 'Elegiac tone'],
            influenced_works=['Tolkien', 'Modern fantasy', 'English literature'],
            linguistic_significance='Most important Old English literary text, West Saxon dialect',
            valency_notes='Germanic verbal system, prefixed verbs with varied argument structures',
            available_sources=['Gutenberg', 'PROIEL'],
        )
        
        self.texts['alfred_boethius'] = TextInfluence(
            text_id='alfred_boethius',
            title='Old English Boethius',
            author='King Alfred',
            language='ang',
            period=Period.OLD_ENGLISH,
            date_composed='9th century CE',
            influence_type=InfluenceType.PHILOSOPHICAL,
            influence_score=8.5,
            description='Old English translation of Boethius',
            key_features=['Translation', 'Adaptation', 'Christian interpretation'],
            influenced_works=['Old English prose', 'Translation tradition'],
            linguistic_significance='Important for Old English prose style and vocabulary',
            available_sources=['PROIEL'],
        )
    
    def _add_middle_english(self):
        self.texts['chaucer_canterbury'] = TextInfluence(
            text_id='chaucer_canterbury',
            title='Canterbury Tales',
            author='Geoffrey Chaucer',
            language='enm',
            period=Period.MIDDLE_ENGLISH,
            date_composed='14th century CE',
            influence_type=InfluenceType.LITERARY,
            influence_score=10.0,
            description='Collection of stories told by pilgrims',
            key_features=['Frame narrative', 'Social satire', 'Varied genres', 'London dialect'],
            influenced_works=['English literature', 'Frame narrative tradition'],
            linguistic_significance='Foundation of English literary language, London dialect basis for Standard English',
            available_sources=['Gutenberg', 'PROIEL'],
        )
        
        self.texts['wycliffe_bible'] = TextInfluence(
            text_id='wycliffe_bible',
            title='Wycliffe Bible',
            author='John Wycliffe and associates',
            language='enm',
            period=Period.MIDDLE_ENGLISH,
            date_composed='14th century CE',
            influence_type=InfluenceType.RELIGIOUS,
            influence_score=9.0,
            description='First complete English Bible translation',
            key_features=['Bible translation', 'Vernacular religion', 'Lollard movement'],
            influenced_works=['English Bible tradition', 'Reformation'],
            linguistic_significance='Important for Middle English religious vocabulary',
            available_sources=['Gutenberg'],
        )
    
    def _add_early_modern(self):
        self.texts['shakespeare_complete'] = TextInfluence(
            text_id='shakespeare_complete',
            title='Complete Works',
            author='William Shakespeare',
            language='en',
            period=Period.EARLY_MODERN,
            date_composed='16th-17th century CE',
            influence_type=InfluenceType.LITERARY,
            influence_score=10.0,
            description='Plays and sonnets of Shakespeare',
            key_features=['Dramatic innovation', 'Language creativity', 'Universal themes'],
            influenced_works=['All English literature', 'World drama'],
            linguistic_significance='Major influence on English vocabulary and idiom',
            valency_notes='Creative verbal constructions, conversion, new argument structures',
            available_sources=['Gutenberg', 'First Folio'],
        )
        
        self.texts['kjv_bible'] = TextInfluence(
            text_id='kjv_bible',
            title='King James Bible',
            author='King James translators',
            language='en',
            period=Period.EARLY_MODERN,
            date_composed='1611 CE',
            influence_type=InfluenceType.RELIGIOUS,
            influence_score=10.0,
            description='Authorized English Bible translation',
            key_features=['Formal register', 'Rhythmic prose', 'Hebraisms'],
            influenced_works=['English prose style', 'Religious language'],
            linguistic_significance='Major influence on English prose style and religious vocabulary',
            available_sources=['Gutenberg'],
        )
        
        self.texts['milton_paradise_lost'] = TextInfluence(
            text_id='milton_paradise_lost',
            title='Paradise Lost',
            author='John Milton',
            language='en',
            period=Period.EARLY_MODERN,
            date_composed='1667 CE',
            influence_type=InfluenceType.LITERARY,
            influence_score=9.5,
            description='Epic poem on the Fall of Man',
            key_features=['Blank verse', 'Classical allusion', 'Theological themes'],
            influenced_works=['English poetry', 'Romantic poets'],
            linguistic_significance='Latinate syntax, elevated diction',
            available_sources=['Gutenberg'],
        )
    
    def get_text(self, text_id: str) -> Optional[TextInfluence]:
        return self.texts.get(text_id)
    
    def get_all_texts(self) -> List[TextInfluence]:
        return list(self.texts.values())
    
    def get_texts_by_period(self, period: Period) -> List[TextInfluence]:
        return [t for t in self.texts.values() if t.period == period]
    
    def get_texts_by_language(self, language: str) -> List[TextInfluence]:
        return [t for t in self.texts.values() if t.language == language]
    
    def get_texts_by_influence_type(self, influence_type: InfluenceType) -> List[TextInfluence]:
        return [t for t in self.texts.values() if t.influence_type == influence_type]
    
    def get_greek_texts(self) -> List[TextInfluence]:
        return [t for t in self.texts.values() if t.language == 'grc']
    
    def get_byzantine_texts(self) -> List[TextInfluence]:
        byzantine_periods = [Period.EARLY_BYZANTINE, Period.MIDDLE_BYZANTINE, Period.LATE_BYZANTINE]
        return [t for t in self.texts.values() if t.period in byzantine_periods]
    
    def get_planudes_works(self) -> List[TextInfluence]:
        return [t for t in self.texts.values() if 'Planudes' in t.author or 'planudes' in t.text_id]
    
    def get_texts_with_valency_notes(self) -> List[TextInfluence]:
        return [t for t in self.texts.values() if t.valency_notes]
    
    def get_most_influential(self, limit: int = 10) -> List[TextInfluence]:
        sorted_texts = sorted(self.texts.values(), key=lambda t: t.influence_score, reverse=True)
        return sorted_texts[:limit]
    
    def search_texts(self, query: str) -> List[TextInfluence]:
        query_lower = query.lower()
        results = []
        
        for text in self.texts.values():
            if (query_lower in text.title.lower() or
                query_lower in text.author.lower() or
                query_lower in text.description.lower() or
                any(query_lower in f.lower() for f in text.key_features)):
                results.append(text)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_texts': len(self.texts),
            'by_period': {},
            'by_language': {},
            'by_influence_type': {},
            'average_influence_score': 0,
        }
        
        for text in self.texts.values():
            period_name = text.period.value
            stats['by_period'][period_name] = stats['by_period'].get(period_name, 0) + 1
            
            stats['by_language'][text.language] = stats['by_language'].get(text.language, 0) + 1
            
            influence_name = text.influence_type.value
            stats['by_influence_type'][influence_name] = stats['by_influence_type'].get(influence_name, 0) + 1
        
        if self.texts:
            stats['average_influence_score'] = sum(t.influence_score for t in self.texts.values()) / len(self.texts)
        
        return stats
