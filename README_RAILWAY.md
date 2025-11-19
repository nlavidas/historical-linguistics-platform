# 🚂 Diachronic Multilingual Corpus Collector

Automated collection of texts across languages, periods, and genres for linguistic research.

## Features

- **Multiple Languages**: Ancient Greek, Latin, English, German, French
- **Diachronic Coverage**: 8th century BCE to 21st century
- **Biblical Texts**: Multiple translations across periods
- **Classical Texts**: Homer, Virgil, etc.
- **Retranslations**: Track language evolution
- **Automatic Collection**: Runs continuously

## Quick Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

1. Click the button above
2. Sign in to Railway
3. Deploy!

## Manual Deployment

```bash
# Clone or upload this repo to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_URL
git push -u origin main

# Then deploy from Railway dashboard
```

## What Gets Collected

- Biblical texts (multiple translations)
- Classical epics (Homer, Virgil)
- Medieval texts (Beowulf, Chaucer)
- Renaissance literature (Shakespeare)
- Modern translations
- Retellings and adaptations

## Database

SQLite database with full metadata:
- Language and diachronic stage
- Author and translator
- Translation year
- Genre and period
- Text type flags

## Requirements

- Python 3.11+
- requests
- pytz

## License

For academic research use.
