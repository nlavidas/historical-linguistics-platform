#!/bin/bash
# Export for Open-Source Note Apps (Trilium/Appflowy compatible)
# This creates a folder structure suitable for import into Trilium or Appflowy

OUTPUT_DIR="research_exports/notes"
mkdir -p "$OUTPUT_DIR"

# Get latest corpus data
python3 << PYEOF
import sqlite3
import json
import os
from datetime import datetime

conn = sqlite3.connect('corpus_platform.db')
cursor = conn.cursor()

# Export texts as JSON for Trilium import
cursor.execute("SELECT id, title, language, content, word_count, date_added FROM corpus_items ORDER BY date_added DESC LIMIT 100")
texts = cursor.fetchall()

notes = []
for text_id, title, language, content, word_count, date_added in texts:
    note = {
        "title": f"{title} ({language})",
        "content": f"# {title}\n\n**Language:** {language}\n**Words:** {word_count}\n**Date:** {date_added}\n\n{content[:2000]}..." if len(content) > 2000 else content,
        "tags": [language, "corpus", "historical"],
        "created": date_added,
        "updated": datetime.now().isoformat()
    }
    notes.append(note)

# Save as JSON for Trilium import
with open('research_exports/notes/corpus_notes.json', 'w') as f:
    json.dump(notes, f, indent=2)

# Export research reports as markdown
import glob
reports = glob.glob('research_exports/agent_reports/*.md') + glob.glob('research_exports/visual_reports/*.html')

for report_path in reports[:10]:  # Limit to 10 latest
    filename = os.path.basename(report_path)
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Convert HTML to markdown if needed
    if report_path.endswith('.html'):
        # Simple HTML to markdown conversion
        content = content.replace('<h1>', '# ').replace('</h1>', '\n\n')
        content = content.replace('<h2>', '## ').replace('</h2>', '\n\n')
        content = content.replace('<p>', '').replace('</p>', '\n\n')
        content = content.replace('<br>', '\n')
        filename = filename.replace('.html', '.md')
    
    with open(f'research_exports/notes/{filename}', 'w') as f:
        f.write(f"# Research Report\n\n{content}")

print("Exported notes for Trilium/Appflowy")
PYEOF

echo "Notes exported to research_exports/notes/"
echo "Import corpus_notes.json into Trilium Notes"
echo "Or copy the entire notes/ folder to your Appflowy workspace"
