"""
Documentation Viewer Component
Professional interface for platform documentation and help
"""

import streamlit as st
from typing import Dict, List

class DocumentationViewer:
    """Platform documentation and help viewer"""
    
    def __init__(self):
        self.sections = {
            'quick_start': 'Quick Start Guide',
            'corpus_browser': 'Corpus Browser',
            'analysis_studio': 'Analysis Studio',
            'syntactic_tools': 'Syntactic Tools',
            'valency_explorer': 'Valency Explorer',
            'api_reference': 'API Reference',
            'data_formats': 'Data Formats',
            'troubleshooting': 'Troubleshooting',
            'faq': 'FAQ'
        }
    
    def render_quick_start(self):
        """Render quick start guide"""
        st.markdown("""
        ## Quick Start Guide
        
        ### 1. Corpus Browser
        
        The Corpus Browser allows you to search and explore historical linguistic texts.
        
        **Basic Search:**
        - Enter keywords in the search field
        - Filter by language (Ancient Greek, Latin, Sanskrit, etc.)
        - Filter by historical period
        - Click "Search" to find matching texts
        
        **Viewing Results:**
        - Click on any result to expand details
        - Use "Analyze" to send text to Analysis Studio
        - Use "Export" to download in various formats
        
        ### 2. Analysis Studio
        
        Perform comprehensive linguistic analysis on texts.
        
        **Running an Analysis:**
        1. Paste or type text in the input area
        2. Select the language (or use auto-detect)
        3. Choose analysis types (morphology, syntax, valency, etc.)
        4. Click "Analyze Text"
        
        **Viewing Results:**
        - Results are organized in tabs by analysis type
        - Morphological analysis shows full paradigm information
        - Syntactic analysis includes dependency trees
        - Export results in JSON, CoNLL-U, or other formats
        
        ### 3. Syntactic Tools
        
        Work with treebanks and syntactic structures.
        
        **Loading Treebanks:**
        - Select from available treebanks (PROIEL, Syntacticus, etc.)
        - Upload custom treebanks in supported formats
        - Browse sentences and view dependencies
        
        **Querying:**
        - Use simple search for basic queries
        - Use advanced query builder for complex patterns
        - Export query results for further analysis
        
        ### 4. Valency Explorer
        
        Explore verbal valency patterns across languages and periods.
        
        **Searching Patterns:**
        - Enter a verb lemma to find its valency patterns
        - Filter by language and pattern type
        - View frequency and examples
        
        **Diachronic Analysis:**
        - Track how valency patterns change over time
        - Compare patterns across historical periods
        - Visualize changes with charts
        """)
    
    def render_corpus_browser_docs(self):
        """Render corpus browser documentation"""
        st.markdown("""
        ## Corpus Browser Documentation
        
        ### Overview
        
        The Corpus Browser provides access to the platform's collection of historical 
        linguistic texts. It supports multiple languages and historical periods.
        
        ### Supported Languages
        
        | Code | Language | Period Coverage |
        |------|----------|-----------------|
        | grc | Ancient Greek | 800 BCE - 1453 CE |
        | la | Latin | 200 BCE - 1500 CE |
        | sa | Sanskrit | 1500 BCE - 1000 CE |
        | got | Gothic | 350 - 550 CE |
        | cop | Coptic | 200 - 1400 CE |
        | cu | Old Church Slavonic | 850 - 1100 CE |
        | xcl | Classical Armenian | 400 - 1100 CE |
        
        ### Search Syntax
        
        **Basic Search:**
        ```
        word          - Search for exact word
        word*         - Wildcard search
        "exact phrase" - Phrase search
        ```
        
        **Advanced Filters:**
        - Language: Filter by ISO 639-3 code
        - Period: Filter by historical period
        - Author: Filter by author name
        - Genre: Filter by text genre
        
        ### Export Formats
        
        - **CSV**: Spreadsheet-compatible format
        - **JSON**: Structured data format
        - **TEI XML**: Text Encoding Initiative format
        - **Plain Text**: Raw text without markup
        """)
    
    def render_analysis_studio_docs(self):
        """Render analysis studio documentation"""
        st.markdown("""
        ## Analysis Studio Documentation
        
        ### Analysis Types
        
        #### Morphological Analysis
        
        Provides complete morphological parsing including:
        - Part of speech tagging
        - Lemmatization
        - Full morphological features (case, number, gender, tense, mood, voice, etc.)
        
        #### Syntactic Analysis
        
        Dependency parsing following Universal Dependencies or PROIEL standards:
        - Head-dependent relations
        - Dependency labels
        - Tree visualization
        
        #### Valency Analysis
        
        Automatic extraction of verbal valency patterns:
        - Argument structure identification
        - Case frame extraction
        - Pattern classification
        
        #### Etymology Analysis
        
        Tracks etymological information:
        - Cognate identification
        - Loanword detection
        - Semantic field analysis
        
        ### Annotation Standards
        
        The platform supports multiple annotation standards:
        
        | Standard | Description |
        |----------|-------------|
        | UD | Universal Dependencies v2 |
        | PROIEL | PROIEL Treebank format |
        | AGDT | Ancient Greek Dependency Treebank |
        | Perseus | Perseus Digital Library format |
        
        ### Output Formats
        
        - **JSON**: Full structured output
        - **CoNLL-U**: Standard treebank format
        - **PROIEL XML**: PROIEL treebank XML
        - **TEI XML**: Text Encoding Initiative
        """)
    
    def render_syntactic_tools_docs(self):
        """Render syntactic tools documentation"""
        st.markdown("""
        ## Syntactic Tools Documentation
        
        ### Treebank Browser
        
        Browse and explore syntactic treebanks from various sources.
        
        **Available Treebanks:**
        - PROIEL Greek New Testament
        - PROIEL Latin Vulgate
        - PROIEL Gothic Bible
        - PROIEL Armenian NT
        - PROIEL Old Church Slavonic
        - Syntacticus collections
        - AGDT Perseus texts
        
        ### Dependency Relations
        
        #### PROIEL Relations
        
        | Relation | Description |
        |----------|-------------|
        | pred | Predicate |
        | sub | Subject |
        | obj | Object |
        | obl | Oblique |
        | atr | Attribute |
        | adv | Adverbial |
        | ag | Agent |
        | comp | Complement |
        | apos | Apposition |
        | aux | Auxiliary |
        | xobj | External object |
        | xadv | External adverbial |
        
        ### Query Language
        
        The platform supports a query language for finding syntactic patterns:
        
        ```
        [lemma='verb' & pos='V'] >obj [case='acc']
        ```
        
        This finds verbs with accusative objects.
        
        ### Format Conversion
        
        Convert between treebank formats:
        - PROIEL XML to CoNLL-U
        - CoNLL-U to PROIEL XML
        - Custom format mappings
        """)
    
    def render_api_reference(self):
        """Render API reference documentation"""
        st.markdown("""
        ## API Reference
        
        ### Base URL
        
        ```
        http://localhost:5000/api/v1
        ```
        
        ### Authentication
        
        API authentication is optional. If enabled, include the API key in headers:
        
        ```
        Authorization: Bearer YOUR_API_KEY
        ```
        
        ### Endpoints
        
        #### Parse Text
        
        ```
        POST /parse
        Content-Type: application/json
        
        {
            "text": "Your text here",
            "language": "grc",
            "analyses": ["morphology", "syntax", "valency"]
        }
        ```
        
        **Response:**
        ```json
        {
            "language": "grc",
            "sentences": [...],
            "statistics": {...}
        }
        ```
        
        #### Search Corpus
        
        ```
        GET /corpus/search?q=term&lang=grc&period=classical&limit=50
        ```
        
        **Parameters:**
        - `q`: Search query (required)
        - `lang`: Language filter (optional)
        - `period`: Period filter (optional)
        - `limit`: Max results (default: 50)
        
        #### Get Valency Patterns
        
        ```
        GET /valency/patterns?lemma=lego&language=grc
        ```
        
        **Parameters:**
        - `lemma`: Verb lemma (required)
        - `language`: Language code (optional)
        
        #### System Status
        
        ```
        GET /status
        ```
        
        Returns system health and metrics.
        
        ### Rate Limits
        
        - Default: 100 requests per minute
        - Authenticated: 1000 requests per minute
        
        ### Error Codes
        
        | Code | Description |
        |------|-------------|
        | 400 | Bad request |
        | 401 | Unauthorized |
        | 404 | Not found |
        | 429 | Rate limit exceeded |
        | 500 | Server error |
        """)
    
    def render_data_formats(self):
        """Render data formats documentation"""
        st.markdown("""
        ## Data Formats
        
        ### CoNLL-U Format
        
        Standard format for dependency treebanks:
        
        ```
        # text = Example sentence
        1    word    lemma    UPOS    XPOS    Feats    Head    Deprel    Deps    Misc
        2    ...
        ```
        
        **Fields:**
        1. ID: Word index
        2. FORM: Word form
        3. LEMMA: Lemma
        4. UPOS: Universal POS tag
        5. XPOS: Language-specific POS
        6. FEATS: Morphological features
        7. HEAD: Head index
        8. DEPREL: Dependency relation
        9. DEPS: Enhanced dependencies
        10. MISC: Miscellaneous
        
        ### PROIEL XML Format
        
        ```xml
        <proiel>
          <source>
            <div>
              <sentence id="1">
                <token id="1" form="word" lemma="lemma" 
                       part-of-speech="V-" morphology="..." 
                       head-id="0" relation="pred"/>
              </sentence>
            </div>
          </source>
        </proiel>
        ```
        
        ### JSON Export Format
        
        ```json
        {
          "metadata": {
            "source": "...",
            "language": "grc",
            "date_analyzed": "..."
          },
          "sentences": [
            {
              "text": "...",
              "tokens": [
                {
                  "id": 1,
                  "form": "...",
                  "lemma": "...",
                  "pos": "...",
                  "morphology": {...},
                  "head": 0,
                  "deprel": "..."
                }
              ]
            }
          ]
        }
        ```
        """)
    
    def render_troubleshooting(self):
        """Render troubleshooting documentation"""
        st.markdown("""
        ## Troubleshooting
        
        ### Common Issues
        
        #### Analysis Timeout
        
        **Problem:** Analysis takes too long or times out.
        
        **Solutions:**
        - Reduce text length (max 10,000 characters recommended)
        - Select fewer analysis types
        - Check system resources in Monitoring panel
        
        #### Database Connection Error
        
        **Problem:** "Database connection failed" error.
        
        **Solutions:**
        - Check if PostgreSQL service is running
        - Verify database credentials in settings
        - Check disk space availability
        
        #### Model Loading Error
        
        **Problem:** Language models fail to load.
        
        **Solutions:**
        - Ensure sufficient memory (8GB+ recommended)
        - Check if models are downloaded
        - Restart the parser service
        
        #### Export Failure
        
        **Problem:** Export generates empty or corrupted file.
        
        **Solutions:**
        - Verify data exists before export
        - Check disk space
        - Try a different export format
        
        ### Service Management
        
        **Restart Services:**
        ```bash
        systemctl restart corpus_platform.service
        systemctl restart corpus_monitor.service
        ```
        
        **Check Logs:**
        ```bash
        journalctl -u corpus_platform.service -f
        tail -f /root/corpus_platform/corpus_platform.log
        ```
        
        **Clear Cache:**
        ```bash
        rm -rf /root/corpus_platform/__pycache__
        rm -rf /tmp/streamlit_*
        ```
        """)
    
    def render_faq(self):
        """Render FAQ section"""
        st.markdown("""
        ## Frequently Asked Questions
        
        ### General
        
        **Q: What languages are supported?**
        
        A: The platform supports Ancient Greek, Latin, Sanskrit, Gothic, Coptic, 
        Old Church Slavonic, Classical Armenian, and several other historical languages.
        
        **Q: Can I add my own texts?**
        
        A: Yes, use the Import function in the Corpus Browser to add texts in 
        supported formats (TEI XML, plain text, CoNLL-U).
        
        **Q: How accurate is the automatic parsing?**
        
        A: Accuracy varies by language and text type. For well-attested languages 
        like Ancient Greek and Latin, accuracy is typically 90-95% for morphology 
        and 85-90% for syntax.
        
        ### Technical
        
        **Q: What are the system requirements?**
        
        A: Minimum 8GB RAM, 50GB disk space. Recommended: 16GB RAM, SSD storage.
        
        **Q: Can I run the platform offline?**
        
        A: Yes, once models are downloaded, the platform works fully offline.
        
        **Q: How do I backup my data?**
        
        A: Use the Export function or enable automatic backups in Settings.
        
        ### Research
        
        **Q: How should I cite the platform?**
        
        A: Please cite as: "Diachronic Linguistics Research Platform, v2.0"
        
        **Q: Can I use the data for publications?**
        
        A: Yes, please check individual corpus licenses for attribution requirements.
        
        **Q: How do I report errors in annotations?**
        
        A: Use the feedback function in the Analysis Studio or submit an issue 
        on the project repository.
        """)
    
    def render(self):
        """Main render method for documentation viewer"""
        st.header("Documentation")
        
        # Create tabs for documentation sections
        tabs = st.tabs([
            "Quick Start",
            "Corpus Browser",
            "Analysis Studio",
            "Syntactic Tools",
            "API Reference",
            "Data Formats",
            "Troubleshooting",
            "FAQ"
        ])
        
        with tabs[0]:
            self.render_quick_start()
        
        with tabs[1]:
            self.render_corpus_browser_docs()
        
        with tabs[2]:
            self.render_analysis_studio_docs()
        
        with tabs[3]:
            self.render_syntactic_tools_docs()
        
        with tabs[4]:
            self.render_api_reference()
        
        with tabs[5]:
            self.render_data_formats()
        
        with tabs[6]:
            self.render_troubleshooting()
        
        with tabs[7]:
            self.render_faq()
