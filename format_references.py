import bibtexparser

with open('nzmj_feedback/references.bib') as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)

for i, entry in enumerate(bib_database.entries):
    author = entry['author'].split(',')[0]
    year = entry['year']
    label = entry['ID']
    record_number = i + 1
    print(f"{{ {author}, {year} @{label} #{record_number} }}")

for i, entry in enumerate(bib_database.entries):
    print(f"{i+1}. {entry['author']}. {entry['title']}. {entry.get('journal', '')}. {entry['year']};{entry.get('volume', '')}:{entry.get('pages', '')}.")
