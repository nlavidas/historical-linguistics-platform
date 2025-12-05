"""
HLP IO - Input/Output Module for Linguistic Data Formats

This package provides comprehensive support for reading and writing
linguistic data in various formats including PROIEL XML, CoNLL-U,
and other standard formats.

Modules:
    proiel_xml: PROIEL 3.0 XML format support
    conllu_io: CoNLL-U format support
    format_converters: Conversion between formats
    tei_xml: TEI XML format support (basic)

University of Athens - Nikolaos Lavidas
"""

from hlp_io.proiel_xml import (
    PROIELReader,
    PROIELWriter,
    PROIELValidator,
    parse_proiel_file,
    parse_proiel_string,
    write_proiel_file,
    write_proiel_string,
)

from hlp_io.conllu_io import (
    CoNLLUReader,
    CoNLLUWriter,
    CoNLLUValidator,
    parse_conllu_file,
    parse_conllu_string,
    write_conllu_file,
    write_conllu_string,
)

from hlp_io.format_converters import (
    FormatConverter,
    proiel_to_conllu,
    conllu_to_proiel,
    proiel_to_dict,
    conllu_to_dict,
)

__version__ = "1.0.0"
__author__ = "Nikolaos Lavidas"

__all__ = [
    "PROIELReader",
    "PROIELWriter",
    "PROIELValidator",
    "parse_proiel_file",
    "parse_proiel_string",
    "write_proiel_file",
    "write_proiel_string",
    "CoNLLUReader",
    "CoNLLUWriter",
    "CoNLLUValidator",
    "parse_conllu_file",
    "parse_conllu_string",
    "write_conllu_file",
    "write_conllu_string",
    "FormatConverter",
    "proiel_to_conllu",
    "conllu_to_proiel",
    "proiel_to_dict",
    "conllu_to_dict",
]
