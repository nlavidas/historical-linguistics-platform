"""
PROIEL XML Processor for Diachronic Linguistics
Parses PROIEL XML format and extracts linguistic data
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PROIELProcessor:
    """Process PROIEL XML files for valency and syntactic analysis"""
    
    def __init__(self):
        self.sentences = []
        self.tokens = []
        self.valency_patterns = []
        
    def parse_proiel_xml(self, xml_path: str) -> Dict:
        """Parse PROIEL XML file and extract linguistic data"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            result = {
                'source': root.find('.//source').get('title') if root.find('.//source') is not None else 'Unknown',
                'sentences': [],
                'tokens': [],
                'valency_patterns': [],
                'statistics': {}
            }
            
            # Parse sentences
            for sentence in root.findall('.//sentence'):
                sent_id = sentence.get('id')
                sent_data = {
                    'id': sent_id,
                    'tokens': []
                }
                
                # Parse tokens
                for token in sentence.findall('.//token'):
                    token_data = {
                        'id': token.get('id'),
                        'form': token.get('form'),
                        'lemma': token.get('lemma'),
                        'pos': token.get('postag', '').split('-')[0] if token.get('postag') else None,
                        'morphology': token.get('postag'),
                        'head_id': token.get('head-id'),
                        'relation': token.get('relation')
                    }
                    sent_data['tokens'].append(token_data)
                    result['tokens'].append(token_data)
                
                result['sentences'].append(sent_data)
            
            # Extract valency patterns
            result['valency_patterns'] = self._extract_valency_patterns(result['sentences'])
            
            # Calculate statistics
            result['statistics'] = {
                'total_sentences': len(result['sentences']),
                'total_tokens': len(result['tokens']),
                'total_valency_patterns': len(result['valency_patterns'])
            }
            
            logger.info(f"✓ Parsed PROIEL file: {result['statistics']}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing PROIEL XML: {e}")
            raise
    
    def _extract_valency_patterns(self, sentences: List[Dict]) -> List[Dict]:
        """Extract verb valency patterns from parsed sentences"""
        patterns = []
        
        for sentence in sentences:
            tokens = sentence['tokens']
            
            # Find verbs
            for i, token in enumerate(tokens):
                if token['pos'] in ['VERB', 'V']:
                    pattern = self._analyze_verb_pattern(token, tokens)
                    if pattern:
                        patterns.append(pattern)
        
        return patterns
    
    def _analyze_verb_pattern(self, verb: Dict, tokens: List[Dict]) -> Optional[Dict]:
        """Analyze valency pattern for a single verb"""
        verb_id = verb['id']
        
        # Find dependents
        dependents = []
        for token in tokens:
            if token['head_id'] == verb_id:
                dependents.append({
                    'form': token['form'],
                    'lemma': token['lemma'],
                    'relation': token['relation'],
                    'pos': token['pos']
                })
        
        if not dependents:
            return None
        
        return {
            'verb_lemma': verb['lemma'],
            'verb_form': verb['form'],
            'dependents': dependents,
            'pattern': ' '.join([d['relation'] for d in dependents if d['relation']]),
            'argument_count': len(dependents)
        }
    
    def annotate_proiel(self, text: str, language: str = 'grc', output_path: Optional[str] = None) -> Dict:
        """Generate PROIEL XML from raw text with full annotation"""
        try:
            # Use Stanza to annotate
            import stanza
            
            # Get pipeline
            nlp = stanza.Pipeline(language, processors='tokenize,pos,lemma,depparse', verbose=False)
            doc = nlp(text)
            
            # Generate PROIEL XML
            proiel_xml = self._generate_proiel_xml(doc, language)
            
            # Count statistics
            tokens_count = sum(len(sent.words) for sent in doc.sentences)
            lemmas_count = sum(1 for sent in doc.sentences for word in sent.words if word.lemma)
            pos_tags_count = sum(1 for sent in doc.sentences for word in sent.words if word.upos)
            dependencies_count = sum(1 for sent in doc.sentences for word in sent.words if word.head is not None)
            
            # Extract valency patterns from doc
            valency_patterns = self._extract_valency_from_stanza(doc)
            
            result = {
                'proiel_xml': proiel_xml,
                'valency_patterns': valency_patterns,
                'statistics': {
                    'sentences': len(doc.sentences),
                    'tokens': tokens_count,
                    'lemmas': lemmas_count,
                    'pos_tags': pos_tags_count,
                    'dependencies': dependencies_count,
                    'verbs': len(valency_patterns)
                }
            }
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(proiel_xml)
                logger.info(f"✓ Saved PROIEL XML to: {output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"PROIEL annotation failed: {e}")
            return {
                'proiel_xml': None,
                'valency_patterns': [],
                'statistics': {
                    'sentences': 0,
                    'tokens': 0,
                    'lemmas': 0,
                    'pos_tags': 0,
                    'dependencies': 0,
                    'verbs': 0
                }
            }
    
    def _generate_proiel_xml(self, doc, language: str) -> str:
        """Generate PROIEL XML from Stanza document"""
        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_lines.append('<proiel>')
        xml_lines.append(f'  <source language="{language}">')
        
        for sent_idx, sent in enumerate(doc.sentences, 1):
            xml_lines.append(f'    <sentence id="{sent_idx}">')
            
            for word in sent.words:
                xml_lines.append(
                    f'      <token id="{word.id}" form="{word.text}" '
                    f'lemma="{word.lemma or ""}" postag="{word.upos or ""}" '
                    f'head-id="{word.head or 0}" relation="{word.deprel or ""}" />'
                )
            
            xml_lines.append('    </sentence>')
        
        xml_lines.append('  </source>')
        xml_lines.append('</proiel>')
        
        return '\n'.join(xml_lines)
    
    def _extract_valency_from_stanza(self, doc) -> List[Dict]:
        """Extract valency patterns from Stanza document"""
        patterns = []
        
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos == 'VERB':
                    # Find dependents
                    dependents = [w for w in sent.words if w.head == word.id]
                    
                    if dependents:
                        patterns.append({
                            'verb': word.text,
                            'lemma': word.lemma,
                            'pattern': ' '.join([w.deprel for w in dependents]),
                            'frequency': 1
                        })
        
        return patterns
    
    def _classify_verbs(self, patterns: List[Dict]) -> Dict:
        """Classify verbs by valency patterns"""
        classification = {
            'intransitive': [],
            'transitive': [],
            'ditransitive': [],
            'complex': []
        }
        
        for pattern in patterns:
            arg_count = pattern['argument_count']
            verb = pattern['verb_lemma']
            
            if arg_count == 0:
                classification['intransitive'].append(verb)
            elif arg_count == 1:
                classification['transitive'].append(verb)
            elif arg_count == 2:
                classification['ditransitive'].append(verb)
            else:
                classification['complex'].append(verb)
        
        return classification
    
    def _analyze_argument_structures(self, patterns: List[Dict]) -> List[Dict]:
        """Analyze and categorize argument structures"""
        structures = []
        
        for pattern in patterns:
            structure = {
                'verb': pattern['verb_lemma'],
                'pattern': pattern['pattern'],
                'args': [d['relation'] for d in pattern['dependents']],
                'complexity': len(pattern['dependents'])
            }
            structures.append(structure)
        
        return structures
    
    def _extract_diachronic_features(self, parsed: Dict) -> Dict:
        """Extract features relevant for diachronic analysis"""
        return {
            'verb_types': len(set(p['verb_lemma'] for p in parsed['valency_patterns'])),
            'pattern_types': len(set(p['pattern'] for p in parsed['valency_patterns'])),
            'avg_arguments': sum(p['argument_count'] for p in parsed['valency_patterns']) / 
                           len(parsed['valency_patterns']) if parsed['valency_patterns'] else 0
        }


def process_proiel_file(file_path: str) -> Dict:
    """Convenience function to process a PROIEL file"""
    processor = PROIELProcessor()
    return processor.parse_proiel_xml(file_path)
