# Greek Diachronic Corpus Platform
## European Research Infrastructure & Funding Strategy

**Principal Investigator**: Nikolaos Lavidas  
**Institution**: National and Kapodistrian University of Athens  
**Platform**: http://54.37.228.155

---

## 🎯 Target Infrastructures

### 1. EOSC (European Open Science Cloud)
- **URL**: https://eosc-portal.eu/
- **What**: Pan-European federation of research data infrastructures
- **Fit**: FAIR-compliant corpus, open access, reproducible research
- **Action**: Register as EOSC Service Provider

### 2. CLARIN (Common Language Resources and Technology Infrastructure)
- **URL**: https://www.clarin.eu/
- **What**: European research infrastructure for language resources
- **Fit**: Linguistic corpora, NLP tools, UD annotation
- **National Node**: CLARIN:EL (Greece) - https://www.clarin.gr/
- **Action**: Deposit corpus, apply for CLARIN Centre status

### 3. DARIAH (Digital Research Infrastructure for Arts and Humanities)
- **URL**: https://www.dariah.eu/
- **What**: Digital humanities research infrastructure
- **Fit**: Historical linguistics, Byzantine/Medieval studies
- **National Node**: DARIAH-GR
- **Action**: Join as contributing partner

### 4. INESS (Infrastructure for the Exploration of Syntax and Semantics)
- **URL**: http://iness.uib.no/
- **What**: Treebank infrastructure (Norway-based)
- **Fit**: PROIEL-style annotation, dependency parsing
- **Action**: Submit treebank for inclusion

---

## 💰 Funding Opportunities

### ERC (European Research Council)

#### ERC Starting Grant
- **Budget**: Up to €1.5M for 5 years
- **Eligibility**: 2-7 years post-PhD
- **Deadline**: Usually October
- **URL**: https://erc.europa.eu/funding/starting-grants

#### ERC Consolidator Grant
- **Budget**: Up to €2M for 5 years
- **Eligibility**: 7-12 years post-PhD
- **Deadline**: Usually February
- **URL**: https://erc.europa.eu/funding/consolidator-grants

#### ERC Advanced Grant
- **Budget**: Up to €2.5M for 5 years
- **Eligibility**: Established researchers
- **Deadline**: Usually May
- **URL**: https://erc.europa.eu/funding/advanced-grants

### Horizon Europe

#### HORIZON-CL2-2024-HERITAGE
- **Topic**: Cultural heritage and democracy
- **Fit**: Greek linguistic heritage preservation
- **Budget**: €2-4M collaborative projects

#### HORIZON-INFRA-2024-EOSC
- **Topic**: EOSC integration
- **Fit**: Research infrastructure development
- **Budget**: €1-3M

### National Funding (Greece)

#### HFRI (Hellenic Foundation for Research & Innovation)
- **URL**: https://www.elidek.gr/
- **Programs**: Research projects for faculty
- **Budget**: €100K-500K

#### GSRT (General Secretariat for Research and Technology)
- **URL**: http://www.gsrt.gr/
- **Programs**: National research programs

---

## 📋 Project Proposal Template

### Title
**DIACHRONIC-GRC: A FAIR-Compliant Diachronic Corpus Platform for Greek Linguistics**

### Abstract (300 words)
The Greek language presents a unique opportunity for diachronic linguistic research, with an unbroken written tradition spanning over 3,000 years from Mycenaean Greek to Modern Greek. Despite this extraordinary continuity, no comprehensive, FAIR-compliant digital infrastructure exists that covers all periods with consistent annotation standards.

DIACHRONIC-GRC addresses this gap by developing a state-of-the-art corpus platform integrating:
- **Archaic to Modern Greek** texts with full morphosyntactic annotation
- **PROIEL/Universal Dependencies** annotation standards
- **Semantic Role Labeling** following PropBank/FrameNet conventions
- **Machine Learning** tools for automatic annotation
- **OCR pipeline** for digitizing manuscript sources
- **Comparative corpora** (Latin, Romance, Gothic, Old Church Slavonic)

The platform implements FAIR principles (Findable, Accessible, Interoperable, Reusable) and will be integrated with CLARIN, DARIAH, and EOSC infrastructures.

### Work Packages

#### WP1: Corpus Development (M1-M36)
- Expand Ancient Greek coverage (Homer to Late Antiquity)
- Develop Byzantine Greek corpus (600-1453 CE)
- Create Medieval vernacular Greek corpus
- Add Early Modern Greek texts (1453-1830)
- **Deliverables**: 10M+ annotated tokens

#### WP2: Annotation Infrastructure (M1-M24)
- Implement PROIEL-compatible annotation schema
- Develop Penn Treebank conversion layer
- Create semantic role annotation guidelines
- Build valency lexicon
- **Deliverables**: Annotation guidelines, lexicon

#### WP3: NLP Tools (M6-M48)
- Train period-specific POS taggers
- Develop lemmatizers for all periods
- Create dependency parsers
- Implement SRL models
- **Deliverables**: ML models, evaluation reports

#### WP4: OCR & Digitization (M12-M48)
- Develop Greek manuscript OCR
- Process Byzantine manuscripts
- Digitize early printed books
- **Deliverables**: OCR pipeline, digitized texts

#### WP5: Infrastructure Integration (M24-M60)
- CLARIN integration
- DARIAH integration
- EOSC deployment
- **Deliverables**: Federated access, persistent identifiers

#### WP6: Dissemination (M1-M60)
- Publications in peer-reviewed journals
- Conference presentations
- Training workshops
- **Deliverables**: 10+ publications, 5+ workshops

### Budget (5-year ERC Consolidator)

| Category | Amount |
|----------|--------|
| PI Salary | €400,000 |
| Postdocs (2) | €500,000 |
| PhD Students (2) | €300,000 |
| Research Assistants | €200,000 |
| Equipment & Cloud | €150,000 |
| Travel & Conferences | €100,000 |
| Open Access Publishing | €50,000 |
| Workshops & Training | €100,000 |
| Indirect Costs (25%) | €450,000 |
| **TOTAL** | **€2,250,000** |

### Impact

1. **Scientific**: First comprehensive diachronic Greek corpus with consistent annotation
2. **Methodological**: New standards for historical corpus linguistics
3. **Technological**: Open-source NLP tools for historical Greek
4. **Educational**: Training materials for digital humanities
5. **Cultural**: Preservation of Greek linguistic heritage

---

## 🔧 Technical Requirements for Infrastructure Integration

### CLARIN Requirements
- [ ] Persistent Identifiers (PIDs) - Handle.net or DOI
- [ ] Metadata in CMDI format
- [ ] OAI-PMH endpoint for harvesting
- [ ] SAML/Shibboleth authentication
- [ ] License: CC-BY or CC-BY-SA
- [ ] Data deposition agreement

### DARIAH Requirements
- [ ] DARIAH-compliant metadata
- [ ] Integration with DARIAH-DE Repository
- [ ] SSH Open Marketplace registration
- [ ] Contribution agreement

### EOSC Requirements
- [ ] EOSC Portal registration
- [ ] EOSC AAI integration
- [ ] FAIR assessment (RDA indicators)
- [ ] Service Level Agreement
- [ ] EOSC Rules of Participation compliance

### INESS Requirements
- [ ] INESS-compatible treebank format
- [ ] LFG or dependency annotation
- [ ] Documentation in English
- [ ] Submission to INESS portal

---

## 📅 Timeline

### Year 1 (2025-2026)
- Q1: CLARIN:EL contact, corpus preparation
- Q2: DARIAH-GR membership application
- Q3: ERC proposal preparation
- Q4: HFRI application, EOSC registration

### Year 2 (2026-2027)
- Q1: ERC submission (if Consolidator)
- Q2: CLARIN Centre application
- Q3: Horizon Europe consortium building
- Q4: INESS treebank submission

### Year 3+ (2027+)
- Infrastructure integration
- International collaborations
- Corpus expansion
- Tool development

---

## 📞 Key Contacts

### CLARIN
- **CLARIN ERIC**: info@clarin.eu
- **CLARIN:EL**: contact@clarin.gr
- **Knowledge Centre**: https://www.clarin.eu/content/knowledge-centres

### DARIAH
- **DARIAH ERIC**: info@dariah.eu
- **DARIAH-GR**: Contact via Academy of Athens

### EOSC
- **EOSC Portal**: https://eosc-portal.eu/
- **EOSC Association**: https://eosc.eu/

### ERC
- **ERC Executive Agency**: https://erc.europa.eu/
- **National Contact Point Greece**: GSRT

---

## 📚 References & Models

### Successful Similar Projects
1. **PROIEL** (Oslo): https://proiel.github.io/
2. **Perseus Digital Library**: http://www.perseus.tufts.edu/
3. **Syntacticus**: https://syntacticus.org/
4. **INESS**: http://iness.uib.no/
5. **Universal Dependencies**: https://universaldependencies.org/

### Key Publications for Proposal
- Lavidas, N. (2021). Transitivity in Greek. JGL.
- Haug, D. (2015). PROIEL Treebank.
- de Marneffe et al. (2021). Universal Dependencies.

---

## ✅ Action Items

### Immediate (This Week)
- [ ] Register on CLARIN:EL portal
- [ ] Contact DARIAH-GR coordinator
- [ ] Create EOSC account
- [ ] Review ERC 2025 call deadlines

### Short-term (1-3 Months)
- [ ] Prepare corpus metadata in CMDI format
- [ ] Draft ERC proposal abstract
- [ ] Identify potential consortium partners
- [ ] Apply for HFRI funding

### Medium-term (3-12 Months)
- [ ] Submit ERC proposal
- [ ] Complete CLARIN integration
- [ ] Publish corpus documentation
- [ ] Present at DH conferences

---

**Document Version**: 1.0  
**Last Updated**: December 3, 2025  
**Status**: Strategic Planning
