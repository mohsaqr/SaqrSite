"""
Google Scholar Bibliography Sync Tool

A Streamlit app that finds papers on Google Scholar not in your papers.bib,
formats them correctly, and saves verified entries to a new file.
"""

import streamlit as st
import requests
import re
import urllib.parse
import sqlite3
from datetime import datetime
from pathlib import Path
from fuzzywuzzy import fuzz
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bs4 import BeautifulSoup

# Configuration
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
DB_PATH = Path(__file__).parent / "papers.db"

# Configuration Constants
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"
DEFAULT_CROSSREF_EMAIL = "your@email.com"
DEFAULT_BIB_URL = "https://raw.githubusercontent.com/mohsaqr/mohsaqr.github.io/main/_bibliography/papers.bib"
DEFAULT_MATCH_THRESHOLD = 85
DEFAULT_UNCERTAIN_THRESHOLD = 60


# ============ Database Functions ============
def init_db():
    conn = sqlite3.connect(DB_PATH)
    # Papers table
    conn.execute("""CREATE TABLE IF NOT EXISTS papers (
        id INTEGER PRIMARY KEY, title TEXT UNIQUE, authors TEXT, year TEXT,
        venue TEXT, doi TEXT, citations INTEGER, citation_key TEXT, abbr TEXT,
        bibtex TEXT, source TEXT, added_date TEXT)""")
    # Authors table for author management
    conn.execute("""CREATE TABLE IF NOT EXISTS authors (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        canonical_name TEXT,
        paper_count INTEGER DEFAULT 0)""")
    # Paper-Author relationship (many-to-many)
    conn.execute("""CREATE TABLE IF NOT EXISTS paper_authors (
        paper_id INTEGER,
        author_id INTEGER,
        position INTEGER,
        PRIMARY KEY (paper_id, author_id),
        FOREIGN KEY (paper_id) REFERENCES papers(id),
        FOREIGN KEY (author_id) REFERENCES authors(id))""")

    # Add trashed column if not exists (for trash bin feature)
    try:
        conn.execute("ALTER TABLE papers ADD COLUMN trashed INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add trashed_date column if not exists
    try:
        conn.execute("ALTER TABLE papers ADD COLUMN trashed_date TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add number (issue) column if not exists
    try:
        conn.execute("ALTER TABLE papers ADD COLUMN number TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add publisher column if not exists
    try:
        conn.execute("ALTER TABLE papers ADD COLUMN publisher TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add external_id column for Google Scholar ID etc.
    try:
        conn.execute("ALTER TABLE papers ADD COLUMN external_id TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.commit()
    conn.close()


def find_or_create_author(name: str) -> int:
    """Find author by name or create if not exists. Returns author ID."""
    conn = sqlite3.connect(DB_PATH)
    name = name.strip()
    if not name or len(name) < 3:
        conn.close()
        return None

    # Check if author exists
    row = conn.execute("SELECT id FROM authors WHERE name=? OR canonical_name=?", (name, name)).fetchone()
    if row:
        conn.close()
        return row[0]

    # Create new author
    conn.execute("INSERT INTO authors (name, canonical_name, paper_count) VALUES (?, ?, 0)", (name, name))
    conn.commit()
    author_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return author_id


def link_paper_authors(paper_id: int, authors_str: str):
    """Parse authors string and link to paper."""
    if not authors_str:
        return

    conn = sqlite3.connect(DB_PATH)
    # Clean up and split authors
    authors_str = authors_str.replace(" & ", " and ")
    parts = [a.strip() for a in authors_str.split(" and ") if a.strip() and len(a.strip()) > 2]

    for pos, author_name in enumerate(parts):
        author_id = find_or_create_author(author_name)
        if author_id:
            try:
                conn.execute("INSERT OR IGNORE INTO paper_authors (paper_id, author_id, position) VALUES (?, ?, ?)",
                           (paper_id, author_id, pos))
            except sqlite3.IntegrityError:
                pass  # Relationship already exists
    conn.commit()
    conn.close()


def update_author_counts():
    """Update paper counts for all authors."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""UPDATE authors SET paper_count = (
        SELECT COUNT(*) FROM paper_authors WHERE author_id = authors.id)""")
    conn.commit()
    conn.close()


def get_authors_list(sort_by="paper_count", limit=100):
    """Get list of authors with counts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM authors WHERE paper_count > 0 ORDER BY {sort_by} DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def merge_authors(old_author_id: int, new_author_id: int):
    """Merge old author into new author (update all references)."""
    conn = sqlite3.connect(DB_PATH)
    # Update paper_authors references
    conn.execute("UPDATE OR IGNORE paper_authors SET author_id=? WHERE author_id=?", (new_author_id, old_author_id))
    # Delete any duplicate entries
    conn.execute("DELETE FROM paper_authors WHERE author_id=?", (old_author_id,))
    # Delete old author
    conn.execute("DELETE FROM authors WHERE id=?", (old_author_id,))
    conn.commit()
    conn.close()
    update_author_counts()


def set_canonical_name(author_id: int, canonical_name: str):
    """Set canonical/display name for an author."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE authors SET canonical_name=? WHERE id=?", (canonical_name, author_id))
    conn.commit()
    conn.close()


# ============ Venue Management Functions ============
def get_venues_list(limit: int = 200) -> list:
    """Get list of unique venues with paper counts."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT venue, COUNT(*) as paper_count
        FROM papers
        WHERE venue IS NOT NULL AND venue != ''
        AND (trashed=0 OR trashed IS NULL)
        GROUP BY venue
        ORDER BY paper_count DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [{"venue": r[0], "paper_count": r[1]} for r in rows]


def merge_venues(old_venue: str, new_venue: str) -> int:
    """Merge old venue into new venue (rename all papers with old venue).

    Args:
        old_venue: The venue name to replace
        new_venue: The new venue name to use

    Returns:
        Number of papers updated
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "UPDATE papers SET venue = ? WHERE venue = ?",
        (new_venue, old_venue)
    )
    updated = cursor.rowcount
    conn.commit()
    conn.close()
    return updated


def find_similar_venues(threshold: int = 80) -> list:
    """Find venues with similar names that might be duplicates."""
    venues = get_venues_list(limit=500)
    groups = []
    grouped = set()

    for i, v1 in enumerate(venues):
        if v1['venue'] in grouped:
            continue

        group = [v1]
        venue1 = v1['venue'].lower().strip()

        for j, v2 in enumerate(venues):
            if i >= j or v2['venue'] in grouped:
                continue

            venue2 = v2['venue'].lower().strip()
            score = fuzz.ratio(venue1, venue2)
            token_score = fuzz.token_sort_ratio(venue1, venue2)

            if max(score, token_score) >= threshold:
                group.append(v2)
                grouped.add(v2['venue'])

        if len(group) > 1:
            grouped.add(v1['venue'])
            groups.append(group)

    return groups


# ============ Data Quality Functions ============
def find_bad_records() -> dict:
    """Find records with data quality issues.

    Returns dict with categories of problematic records:
    - incomplete_papers: Missing title, authors, or year
    - short_titles: Titles < 10 characters (likely errors)
    - bad_author_names: Malformed author names
    - bad_venues: Empty or very short venue names
    - no_doi_recent: Recent papers (2020+) without DOI
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    issues = {
        "incomplete_papers": [],
        "short_titles": [],
        "bad_author_names": [],
        "bad_venues": [],
        "no_doi_recent": [],
    }

    papers = conn.execute("SELECT * FROM papers WHERE trashed=0 OR trashed IS NULL").fetchall()
    conn.close()

    for p in papers:
        paper = dict(p)
        paper_id = paper['id']
        title = paper.get('title', '') or ''
        authors = paper.get('authors', '') or ''
        year = paper.get('year', '') or ''
        venue = paper.get('venue', '') or ''
        doi = paper.get('doi', '') or ''

        # Check for incomplete papers
        if not title or not authors or not year:
            missing = []
            if not title: missing.append("title")
            if not authors: missing.append("authors")
            if not year: missing.append("year")
            issues["incomplete_papers"].append({
                **paper,
                "issue": f"Missing: {', '.join(missing)}"
            })

        # Check for very short titles (likely errors)
        elif len(title) < 15:
            issues["short_titles"].append({
                **paper,
                "issue": f"Title too short ({len(title)} chars)"
            })

        # Check for truncated titles (cut mid-word or at suspicious lengths)
        if title and len(title) > 20:
            last_char = title[-1]
            # Truncated if: ends with lowercase letter (mid-word), or ends at round numbers
            is_truncated = False
            truncation_reason = ""

            # Check if ends mid-word (lowercase letter, not punctuation)
            if last_char.islower() and last_char not in '.?!':
                is_truncated = True
                truncation_reason = "ends mid-word"
            # Check suspicious lengths (common truncation points)
            elif len(title) in [50, 64, 80, 100, 128, 255, 256]:
                is_truncated = True
                truncation_reason = f"suspicious length ({len(title)} chars)"
            # Check if ends with common cut patterns
            elif title.rstrip().endswith(('...', '…', ' -', ' –')):
                is_truncated = True
                truncation_reason = "ends with ellipsis/dash"

            if is_truncated:
                if "truncated_titles" not in issues:
                    issues["truncated_titles"] = []
                issues["truncated_titles"].append({
                    **paper,
                    "issue": f"Truncated: {truncation_reason}"
                })

        # Check for bad author names
        if authors:
            bad_patterns = []
            # Check for single letter names
            if re.search(r'\b[A-Z]\b(?!\.)|\b[A-Z]\.\s*$', authors):
                bad_patterns.append("single letters")
            # Check for all caps names
            if authors == authors.upper() and len(authors) > 10:
                bad_patterns.append("ALL CAPS")
            # Check for incomplete names (just initials like "M. S.")
            if re.match(r'^[A-Z]\.\s*[A-Z]\.?\s*$', authors.strip()):
                bad_patterns.append("only initials")
            # Check for names with numbers
            if re.search(r'\d', authors):
                bad_patterns.append("contains numbers")
            # Check for very short author string
            if len(authors) < 5:
                bad_patterns.append("too short")
            # Check for "et al" only
            if authors.strip().lower() in ['et al', 'et al.', 'others']:
                bad_patterns.append("incomplete (et al only)")

            if bad_patterns:
                issues["bad_author_names"].append({
                    **paper,
                    "issue": f"Author issues: {', '.join(bad_patterns)}"
                })

        # Check for bad venues
        if venue and len(venue) < 5:
            issues["bad_venues"].append({
                **paper,
                "issue": f"Venue too short: '{venue}'"
            })
        elif not venue:
            issues["bad_venues"].append({
                **paper,
                "issue": "No venue"
            })

        # Check for recent papers without DOI
        try:
            if year and int(year) >= 2020 and not doi:
                issues["no_doi_recent"].append({
                    **paper,
                    "issue": f"No DOI for {year} paper"
                })
        except ValueError:
            pass

    return issues


def find_bad_authors() -> list:
    """Find author names that look malformed or incomplete."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, name, paper_count FROM authors ORDER BY paper_count DESC").fetchall()
    conn.close()

    bad_authors = []
    for row in rows:
        author_id, name, count = row
        issues = []

        if not name:
            issues.append("empty name")
        elif len(name) < 4:
            issues.append("too short")
        elif re.match(r'^[A-Z]\.\s*[A-Z]?\.?\s*$', name.strip()):
            issues.append("only initials")
        elif name == name.upper() and len(name) > 5:
            issues.append("ALL CAPS")
        elif re.search(r'\d', name):
            issues.append("contains numbers")
        elif name.count(',') > 1:
            issues.append("multiple commas")
        elif re.search(r'[^\w\s,.\-\'\u00C0-\u017F]', name):
            issues.append("special characters")

        if issues:
            bad_authors.append({
                "id": author_id,
                "name": name,
                "paper_count": count,
                "issues": issues
            })

    return bad_authors


def find_duplicate_papers(threshold: int = 85, aggressive: bool = False):
    """Find potential duplicate papers using fuzzy matching on titles.

    Args:
        threshold: Fuzzy match threshold (0-100). Higher = stricter matching.
                   Aggressive: 70, Normal: 85, Strict: 95
        aggressive: If True, also considers partial matches and year differences

    Returns:
        List of duplicate groups, each group is a list of similar papers
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    papers = conn.execute("""
        SELECT id, title, authors, year, doi, venue, citations
        FROM papers
        WHERE trashed=0 OR trashed IS NULL
        ORDER BY title
    """).fetchall()
    conn.close()

    papers = [dict(p) for p in papers]
    n = len(papers)

    # Track which papers have been grouped
    grouped = set()
    duplicate_groups = []

    for i in range(n):
        if i in grouped:
            continue

        p1 = papers[i]
        title1 = p1['title'].lower().strip() if p1['title'] else ""

        # Find all papers similar to this one
        group = [p1]

        for j in range(i + 1, n):
            if j in grouped:
                continue

            p2 = papers[j]
            title2 = p2['title'].lower().strip() if p2['title'] else ""

            # Skip if titles have very different lengths (likely different papers)
            len1, len2 = len(title1), len(title2)
            if len1 == 0 or len2 == 0:
                continue  # Skip empty titles

            length_ratio = min(len1, len2) / max(len1, len2)

            # If one title is less than 70% the length of the other, require exact substring
            if length_ratio < 0.7:
                # Only match if short one is exact prefix of long one
                short, long_title = (title1, title2) if len1 < len2 else (title2, title1)
                if not long_title.startswith(short):
                    continue

            # Calculate similarity
            ratio = fuzz.ratio(title1, title2)
            token_sort = fuzz.token_sort_ratio(title1, title2)

            # Only use partial_ratio in aggressive mode AND when lengths are similar
            if aggressive and length_ratio >= 0.7:
                partial_ratio = fuzz.partial_ratio(title1, title2)
                score = max(ratio, partial_ratio, token_sort)
            else:
                score = max(ratio, token_sort)

            if score >= threshold:
                # Additional checks for non-aggressive mode
                if not aggressive:
                    # Same year check (allow 1 year difference)
                    y1 = p1.get('year', '0')
                    y2 = p2.get('year', '0')
                    try:
                        if abs(int(y1 or 0) - int(y2 or 0)) > 1:
                            continue
                    except ValueError:
                        pass

                group.append(p2)
                grouped.add(j)

        if len(group) > 1:
            grouped.add(i)
            duplicate_groups.append(group)

    # Sort groups by size (largest first)
    duplicate_groups.sort(key=lambda g: len(g), reverse=True)

    return duplicate_groups


def remove_duplicate_papers(threshold: int = 85, aggressive: bool = False):
    """Remove duplicate papers keeping the best one from each group.

    Best paper is determined by:
    1. Has DOI
    2. Has more citations
    3. Earlier ID (first added)
    """
    groups = find_duplicate_papers(threshold, aggressive)
    removed = 0
    conn = sqlite3.connect(DB_PATH)

    for group in groups:
        # Sort by quality: DOI > citations > earlier ID
        def quality_score(p):
            doi_score = 2 if p.get('doi') else 0
            citation_score = 1 if p.get('citations', 0) else 0
            return (doi_score, citation_score, -p['id'])

        sorted_group = sorted(group, key=quality_score, reverse=True)
        keep = sorted_group[0]

        for paper in sorted_group[1:]:
            conn.execute("DELETE FROM papers WHERE id=?", (paper['id'],))
            conn.execute("DELETE FROM paper_authors WHERE paper_id=?", (paper['id'],))
            removed += 1

    conn.commit()
    conn.close()
    return removed


def delete_paper(paper_id: int):
    """Move a paper to trash (soft delete)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE papers SET trashed=1, trashed_date=? WHERE id=?",
        (datetime.now().isoformat(), paper_id)
    )
    conn.commit()
    conn.close()


def restore_paper(paper_id: int):
    """Restore a paper from trash."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE papers SET trashed=0, trashed_date=NULL WHERE id=?", (paper_id,))
    conn.commit()
    conn.close()


def permanent_delete_paper(paper_id: int):
    """Permanently delete a paper from database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM papers WHERE id=?", (paper_id,))
    conn.execute("DELETE FROM paper_authors WHERE paper_id=?", (paper_id,))
    conn.commit()
    conn.close()


def get_trashed_papers() -> list:
    """Get all papers in trash."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    papers = conn.execute("""
        SELECT * FROM papers
        WHERE trashed=1
        ORDER BY trashed_date DESC
    """).fetchall()
    conn.close()
    return [dict(p) for p in papers]


def empty_trash():
    """Permanently delete all papers in trash."""
    conn = sqlite3.connect(DB_PATH)
    # Get IDs of trashed papers
    ids = [r[0] for r in conn.execute("SELECT id FROM papers WHERE trashed=1").fetchall()]
    # Delete from papers and paper_authors
    conn.execute("DELETE FROM papers WHERE trashed=1")
    for pid in ids:
        conn.execute("DELETE FROM paper_authors WHERE paper_id=?", (pid,))
    conn.commit()
    conn.close()
    return len(ids)


def restore_all_trash():
    """Restore all papers from trash."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("UPDATE papers SET trashed=0, trashed_date=NULL WHERE trashed=1")
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


def save_to_db(paper: dict, source: str = "sync") -> bool:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""INSERT OR REPLACE INTO papers
            (title,authors,year,venue,doi,citations,citation_key,abbr,bibtex,source,added_date,
             abstract,keywords,pages,volume,url,number,publisher,external_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (paper.get("title"), paper.get("authors"), paper.get("year"), paper.get("venue"),
             paper.get("doi",""), paper.get("citations",0), paper.get("citation_key"),
             paper.get("abbr"), paper.get("bibtex"), source, datetime.now().isoformat(),
             paper.get("abstract",""), paper.get("keywords",""), paper.get("pages",""),
             paper.get("volume",""), paper.get("url",""), paper.get("number",""),
             paper.get("publisher",""), paper.get("external_id","")))
        conn.commit()
        return True
    except: return False
    finally: conn.close()


def import_from_csv(csv_content: str, update_existing: bool = True) -> dict:
    """Import papers from CSV content.

    Supports columns: Authors, Title, Publication/Venue, Volume, Number, Pages, Year, Publisher, Citations, DOI

    Args:
        csv_content: CSV file content as string
        update_existing: If True, update existing papers with new data (matched by title)

    Returns:
        dict with imported, updated, skipped counts
    """
    import csv
    import io

    reader = csv.DictReader(io.StringIO(csv_content))

    imported = 0
    updated = 0
    skipped = 0

    conn = sqlite3.connect(DB_PATH)

    for row in reader:
        # Map CSV columns to our fields (handle various column names)
        title = row.get('Title', row.get('title', '')).strip()
        if not title:
            skipped += 1
            continue

        authors = row.get('Authors', row.get('authors', row.get('Author', ''))).strip()
        # Clean up author format (remove trailing semicolons/commas)
        authors = authors.rstrip(';, ')

        year = row.get('Year', row.get('year', '')).strip()
        venue = row.get('Publication', row.get('Venue', row.get('Journal', row.get('venue', '')))).strip()
        volume = row.get('Volume', row.get('volume', '')).strip()
        number = row.get('Number', row.get('Issue', row.get('number', ''))).strip()
        pages = row.get('Pages', row.get('pages', '')).strip()
        publisher = row.get('Publisher', row.get('publisher', '')).strip()
        doi = row.get('DOI', row.get('doi', '')).strip()
        citations = row.get('Citations', row.get('citations', row.get('Cited by', '0'))).strip()
        external_id = row.get('ID', row.get('Scholar ID', row.get('external_id', ''))).strip()

        # Convert citations to integer
        try:
            citations = int(citations) if citations else 0
        except ValueError:
            citations = 0

        # Check if paper exists
        existing = conn.execute("SELECT id, citations FROM papers WHERE title=?", (title,)).fetchone()

        if existing:
            if update_existing:
                # Update with new data (only non-empty fields)
                updates = []
                params = []

                if authors:
                    updates.append("authors=?")
                    params.append(authors)
                if year:
                    updates.append("year=?")
                    params.append(year)
                if venue:
                    updates.append("venue=?")
                    params.append(venue)
                if volume:
                    updates.append("volume=?")
                    params.append(volume)
                if number:
                    updates.append("number=?")
                    params.append(number)
                if pages:
                    updates.append("pages=?")
                    params.append(pages)
                if publisher:
                    updates.append("publisher=?")
                    params.append(publisher)
                if doi:
                    updates.append("doi=?")
                    params.append(doi)
                if citations > 0:
                    updates.append("citations=?")
                    params.append(citations)
                if external_id:
                    updates.append("external_id=?")
                    params.append(external_id)

                if updates:
                    params.append(existing[0])
                    conn.execute(f"UPDATE papers SET {', '.join(updates)} WHERE id=?", params)
                    updated += 1
            else:
                skipped += 1
        else:
            # Insert new paper
            conn.execute("""INSERT INTO papers
                (title, authors, year, venue, volume, number, pages, publisher, doi, citations, external_id, source, added_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'csv_import', ?)""",
                (title, authors, year, venue, volume, number, pages, publisher, doi, citations, external_id,
                 datetime.now().isoformat()))
            imported += 1

    conn.commit()
    conn.close()

    return {"imported": imported, "updated": updated, "skipped": skipped}

def search_db(query="", author="", year="", sort_by="year", sort_order="DESC", limit=200):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conds, params = ["(trashed=0 OR trashed IS NULL)"], []  # Exclude trashed papers
    if query: conds.append("(title LIKE ? OR venue LIKE ?)"); params += [f"%{query}%"]*2
    if author: conds.append("authors LIKE ?"); params.append(f"%{author}%")
    if year: conds.append("year=?"); params.append(year)
    where = " AND ".join(conds)
    order = f"{sort_by} {sort_order}"
    rows = conn.execute(f"SELECT * FROM papers WHERE {where} ORDER BY {order} LIMIT ?", params+[limit]).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_db_years():
    conn = sqlite3.connect(DB_PATH)
    years = [r[0] for r in conn.execute("SELECT DISTINCT year FROM papers WHERE year!='' AND (trashed=0 OR trashed IS NULL) ORDER BY year DESC")]
    conn.close()
    return years

def delete_from_db(pid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM papers WHERE id=?", (pid,))
    conn.commit()
    conn.close()

def update_db_doi(pid, doi):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE papers SET doi=? WHERE id=?", (doi, pid))
    conn.commit()
    conn.close()


def update_paper(pid, **fields):
    """Update any fields of a paper in the database."""
    conn = sqlite3.connect(DB_PATH)
    sets = ", ".join(f"{k}=?" for k in fields.keys())
    conn.execute(f"UPDATE papers SET {sets} WHERE id=?", list(fields.values()) + [pid])
    conn.commit()
    conn.close()


def format_reference(paper: dict) -> str:
    """Format paper as a readable academic reference."""
    # format_authors_display handles decode_latex internally
    authors = format_authors_display(paper.get('authors', ''))
    title = decode_latex(paper.get('title', ''))
    venue = decode_latex(paper.get('venue', ''))
    year = paper.get('year', '')
    doi = paper.get('doi', '')

    # Format: Authors (Year). Title. *Venue*. DOI
    ref = f"**{authors}** ({year}). {title}."
    if venue:
        ref += f" *{venue}*."
    if doi:
        ref += f" [DOI](https://doi.org/{doi})"
    return ref


def format_citation_badge(citations: int, style: str = "short") -> str:
    """Format citation count as a display badge.

    Args:
        citations: Number of citations
        style: 'short' for '[N]', 'long' for '[N citations]'

    Returns:
        Formatted badge string or empty string if no citations
    """
    if citations and citations > 0:
        return f" [{citations}]" if style == "short" else f" [{citations} citations]"
    return ""


def get_stats():
    """Get database statistics."""
    conn = sqlite3.connect(DB_PATH)
    not_trashed = "(trashed=0 OR trashed IS NULL)"

    stats = {}
    stats['total'] = conn.execute(f"SELECT COUNT(*) FROM papers WHERE {not_trashed}").fetchone()[0]
    stats['with_doi'] = conn.execute(f"SELECT COUNT(*) FROM papers WHERE doi != '' AND doi IS NOT NULL AND {not_trashed}").fetchone()[0]
    stats['years'] = conn.execute(f"SELECT MIN(year), MAX(year) FROM papers WHERE year != '' AND {not_trashed}").fetchone()
    stats['by_year'] = dict(conn.execute(f"SELECT year, COUNT(*) FROM papers WHERE year != '' AND {not_trashed} GROUP BY year ORDER BY year DESC").fetchall())
    stats['top_venues'] = conn.execute(f"SELECT venue, COUNT(*) as cnt FROM papers WHERE venue != '' AND {not_trashed} GROUP BY venue ORDER BY cnt DESC LIMIT 10").fetchall()

    # Citation statistics
    citation_row = conn.execute(f"""
        SELECT SUM(citations), AVG(citations), MAX(citations), COUNT(*)
        FROM papers
        WHERE citations > 0 AND {not_trashed}
    """).fetchone()
    stats['total_citations'] = citation_row[0] or 0
    stats['avg_citations'] = round(citation_row[1] or 0, 1)
    stats['max_citations'] = citation_row[2] or 0
    stats['papers_with_citations'] = citation_row[3] or 0

    # Top cited papers
    stats['top_cited'] = conn.execute(f"""
        SELECT title, authors, year, citations, doi
        FROM papers
        WHERE citations > 0 AND {not_trashed}
        ORDER BY citations DESC
        LIMIT 10
    """).fetchall()

    # Citations by year
    stats['citations_by_year'] = dict(conn.execute(f"""
        SELECT year, SUM(citations)
        FROM papers
        WHERE year != '' AND citations > 0 AND {not_trashed}
        GROUP BY year
        ORDER BY year DESC
    """).fetchall())

    # Get all authors for author stats
    all_authors = conn.execute(f"SELECT authors FROM papers WHERE authors != '' AND {not_trashed}").fetchall()
    conn.close()

    # Parse author counts - handle both BibTeX (" and ") and CSV (";") formats
    author_counts = {}
    for (authors_str,) in all_authors:
        # Normalize separators: replace semicolons with " and "
        normalized = authors_str.replace(';', ' and ')
        for author in normalized.split(" and "):
            author = author.strip()
            if author and len(author) > 2:
                # Normalize to "Firstname Lastname" for consistent counting
                display_name = format_author_name(author)
                author_counts[display_name] = author_counts.get(display_name, 0) + 1
    stats['top_authors'] = sorted(author_counts.items(), key=lambda x: -x[1])[:15]
    stats['author_counts'] = author_counts
    return stats


def get_all_venues():
    """Get all unique venues for standardization."""
    conn = sqlite3.connect(DB_PATH)
    venues = conn.execute("SELECT DISTINCT venue FROM papers WHERE venue != '' ORDER BY venue").fetchall()
    conn.close()
    return [v[0] for v in venues]


def get_all_authors():
    """Get all unique authors for standardization."""
    conn = sqlite3.connect(DB_PATH)
    all_authors = conn.execute("SELECT authors FROM papers WHERE authors != ''").fetchall()
    conn.close()

    author_set = set()
    for (authors_str,) in all_authors:
        for author in authors_str.split(" and "):
            author = author.strip()
            if author:
                author_set.add(author)
    return sorted(author_set)


def standardize_venue(old_venue: str, new_venue: str):
    """Replace all occurrences of old venue with new venue."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE papers SET venue=? WHERE venue=?", (new_venue, old_venue))
    conn.commit()
    affected = conn.total_changes
    conn.close()
    return affected


def standardize_author(old_author: str, new_author: str):
    """Replace author name in all papers."""
    conn = sqlite3.connect(DB_PATH)
    papers = conn.execute("SELECT id, authors FROM papers WHERE authors LIKE ?", (f"%{old_author}%",)).fetchall()
    updated = 0
    for pid, authors in papers:
        new_authors = authors.replace(old_author, new_author)
        if new_authors != authors:
            conn.execute("UPDATE papers SET authors=? WHERE id=?", (new_authors, pid))
            updated += 1
    conn.commit()
    conn.close()
    return updated

def get_db_count():
    conn = sqlite3.connect(DB_PATH)
    cnt = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    conn.close()
    return cnt


def fetch_dois_from_website(url: str) -> dict:
    """
    Fetch DOIs from a publication website (saqr.me or sonsoles.me).
    Returns dict mapping title fragments to DOIs.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        dois = {}

        # Find DOI links (look for doi.org URLs)
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if 'doi.org/' in href:
                # Extract DOI from URL
                doi_match = re.search(r'doi\.org/(.+?)(?:\s|$|")', href)
                if doi_match:
                    doi = doi_match.group(1).strip()
                    # Try to find associated title (look for nearby text)
                    parent = link.find_parent(['li', 'div', 'p', 'tr'])
                    if parent:
                        text = parent.get_text()[:200].lower()
                        dois[text] = doi

        # Also look for DOI text patterns
        text = soup.get_text()
        for match in re.finditer(r'10\.\d{4,}/[^\s<>"]+', text):
            doi = match.group(0).rstrip('.,;)')
            context = text[max(0, match.start()-200):match.start()].lower()
            if context:
                dois[context] = doi

        return dois
    except Exception as e:
        return {"error": str(e)}


def match_doi_from_website(paper_title: str, website_dois: dict) -> str:
    """Try to match a paper title to a DOI from website data."""
    title_lower = paper_title.lower()[:60]
    for context, doi in website_dois.items():
        if "error" in context:
            continue
        # Check if title words appear in context
        title_words = [w for w in title_lower.split() if len(w) > 4]
        matches = sum(1 for w in title_words if w in context)
        if matches >= 3:
            return doi
    return ""


def fetch_metadata_from_doi(doi: str) -> dict:
    """
    Fetch paper metadata from CrossRef using DOI.
    Returns dict with title, authors, year, venue, etc.
    """
    if not doi:
        return {"error": "No DOI provided"}

    try:
        # Clean DOI
        doi = doi.strip()
        if doi.startswith("http"):
            doi = re.search(r'10\.\d{4,}/[^\s]+', doi)
            if doi:
                doi = doi.group(0)
            else:
                return {"error": "Invalid DOI URL"}

        url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
        headers = {'User-Agent': 'SaqrSite/1.0 (mailto:user@example.com)'}

        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return {"error": f"DOI not found (status {response.status_code})"}

        data = response.json()
        item = data.get("message", {})

        # Extract metadata
        result = {
            "doi": doi,
            "title": "",
            "authors": "",
            "year": "",
            "venue": "",
            "volume": "",
            "pages": "",
            "publisher": "",
            "type": item.get("type", ""),
        }

        # Title
        titles = item.get("title", [])
        if titles:
            result["title"] = titles[0]

        # Authors
        authors_list = item.get("author", [])
        author_names = []
        for author in authors_list:
            given = author.get("given", "")
            family = author.get("family", "")
            if given and family:
                author_names.append(f"{family}, {given}")
            elif family:
                author_names.append(family)
        result["authors"] = " and ".join(author_names)

        # Year
        published = item.get("published-print") or item.get("published-online") or item.get("created")
        if published:
            date_parts = published.get("date-parts", [[]])
            if date_parts and date_parts[0]:
                result["year"] = str(date_parts[0][0])

        # Venue (journal or conference)
        container = item.get("container-title", [])
        if container:
            result["venue"] = container[0]
        elif item.get("publisher"):
            result["venue"] = item.get("publisher")

        # Volume and pages
        result["volume"] = item.get("volume", "")
        result["pages"] = item.get("page", "")

        return result

    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}


def update_paper_from_doi(paper_id: int, doi: str = None, overwrite: bool = True) -> dict:
    """Fetch metadata from DOI and update paper in database.

    Args:
        paper_id: ID of paper to update
        doi: DOI to fetch (uses paper's DOI if not provided)
        overwrite: If True (default), replace existing data with CrossRef data
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Get current paper
    paper = conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
    if not paper:
        conn.close()
        return {"error": "Paper not found"}

    paper = dict(paper)
    doi = doi or paper.get("doi", "")

    if not doi:
        conn.close()
        return {"error": "No DOI available"}

    # Fetch metadata
    metadata = fetch_metadata_from_doi(doi)
    if "error" in metadata:
        conn.close()
        return metadata

    # Update paper with fetched data (overwrite existing or only fill empty)
    updates = {}
    if metadata.get("title") and (overwrite or not paper.get("title")):
        updates["title"] = metadata["title"]
    if metadata.get("authors") and (overwrite or not paper.get("authors")):
        updates["authors"] = metadata["authors"]
    if metadata.get("year") and (overwrite or not paper.get("year")):
        updates["year"] = metadata["year"]
    if metadata.get("venue") and (overwrite or not paper.get("venue")):
        updates["venue"] = metadata["venue"]
    if metadata.get("doi"):
        updates["doi"] = metadata["doi"]

    if updates:
        sets = ", ".join(f"{k}=?" for k in updates.keys())
        try:
            conn.execute(f"UPDATE papers SET {sets} WHERE id=?", list(updates.values()) + [paper_id])
            conn.commit()
        except sqlite3.IntegrityError:
            # Title conflict - try updating without title
            if "title" in updates:
                del updates["title"]
                if updates:
                    sets = ", ".join(f"{k}=?" for k in updates.keys())
                    conn.execute(f"UPDATE papers SET {sets} WHERE id=?", list(updates.values()) + [paper_id])
                    conn.commit()

    conn.close()
    return {"status": "updated", "updated": list(updates.keys()), "metadata": metadata}


def batch_update_from_dois(progress_callback=None) -> dict:
    """Update all papers that have DOIs with metadata from CrossRef."""
    conn = sqlite3.connect(DB_PATH)
    papers = conn.execute("SELECT id, doi FROM papers WHERE doi != '' AND doi IS NOT NULL").fetchall()
    conn.close()

    updated = 0
    errors = 0
    total = len(papers)

    for i, (paper_id, doi) in enumerate(papers):
        if progress_callback:
            progress_callback(i + 1, total, doi)

        result = update_paper_from_doi(paper_id, doi)
        if result.get("success"):
            updated += 1
        else:
            errors += 1

        # Rate limiting - be nice to CrossRef
        import time
        time.sleep(0.5)

    return {"updated": updated, "errors": errors, "total": total}


def decode_latex(text: str) -> str:
    """Decode LaTeX special characters for display."""
    if not text:
        return text

    # Use regex for more flexible matching
    import re as regex

    result = text

    # Common LaTeX accent patterns (order matters - more specific first)
    patterns = [
        # Dotless-i with accents (must come first)
        (r"\{\\\'\\i\}", lambda m: 'í'),  # {\'\i}
        (r"\\\'\\i", lambda m: 'í'),      # \'\i
        (r"\{\\`\\i\}", lambda m: 'ì'),   # {\`\i}
        (r"\\`\\i", lambda m: 'ì'),       # \`\i
        # Umlaut (diaeresis)
        (r'\\"\{([aouAOU])\}', lambda m: {'a':'ä','o':'ö','u':'ü','A':'Ä','O':'Ö','U':'Ü'}[m.group(1)]),
        (r'\{\\\"([aouAOU])\}', lambda m: {'a':'ä','o':'ö','u':'ü','A':'Ä','O':'Ö','U':'Ü'}[m.group(1)]),
        (r'\\\"([aouAOU])', lambda m: {'a':'ä','o':'ö','u':'ü','A':'Ä','O':'Ö','U':'Ü'}[m.group(1)]),
        # Acute accent
        (r"\\'\{([aeiouyAEIOUY])\}", lambda m: {'a':'á','e':'é','i':'í','o':'ó','u':'ú','y':'ý','A':'Á','E':'É','I':'Í','O':'Ó','U':'Ú','Y':'Ý'}[m.group(1)]),
        (r"\{\\\'([aeiouyAEIOUY])\}", lambda m: {'a':'á','e':'é','i':'í','o':'ó','u':'ú','y':'ý','A':'Á','E':'É','I':'Í','O':'Ó','U':'Ú','Y':'Ý'}[m.group(1)]),
        (r"\\'([aeiouyAEIOUY])", lambda m: {'a':'á','e':'é','i':'í','o':'ó','u':'ú','y':'ý','A':'Á','E':'É','I':'Í','O':'Ó','U':'Ú','Y':'Ý'}[m.group(1)]),
        # Grave accent
        (r'\\`\{([aeiouAEIOU])\}', lambda m: {'a':'à','e':'è','i':'ì','o':'ò','u':'ù','A':'À','E':'È','I':'Ì','O':'Ò','U':'Ù'}[m.group(1)]),
        (r'\{\\`([aeiouAEIOU])\}', lambda m: {'a':'à','e':'è','i':'ì','o':'ò','u':'ù','A':'À','E':'È','I':'Ì','O':'Ò','U':'Ù'}[m.group(1)]),
        (r'\\`([aeiouAEIOU])', lambda m: {'a':'à','e':'è','i':'ì','o':'ò','u':'ù','A':'À','E':'È','I':'Ì','O':'Ò','U':'Ù'}[m.group(1)]),
        # Tilde
        (r'\\~\{([nN])\}', lambda m: 'ñ' if m.group(1)=='n' else 'Ñ'),
        (r'\{\\~([nN])\}', lambda m: 'ñ' if m.group(1)=='n' else 'Ñ'),
        (r'\\~([nN])', lambda m: 'ñ' if m.group(1)=='n' else 'Ñ'),
        # Cedilla
        (r'\\c\{([cC])\}', lambda m: 'ç' if m.group(1)=='c' else 'Ç'),
        # Other special chars
        (r'\\ss\b', lambda m: 'ß'),
        (r'\\aa\b', lambda m: 'å'),
        (r'\\AA\b', lambda m: 'Å'),
        (r'\\o\b', lambda m: 'ø'),
        (r'\\O\b', lambda m: 'Ø'),
        (r'\\ae\b', lambda m: 'æ'),
        (r'\\AE\b', lambda m: 'Æ'),
    ]

    for pattern, repl in patterns:
        result = regex.sub(pattern, repl, result)

    # Clean up remaining braces
    result = result.replace('{', '').replace('}', '')

    return result


# DOI Lookup and Verification Functions
def lookup_doi(title: str, authors: str, year: str, mailto: str = "user@example.com") -> dict:
    """
    Query CrossRef API to find DOI by title and author.
    Returns: {"doi": "10.xxx/xxx", "status": "found|not_found|error", "confidence": float}
    """
    try:
        first_author = authors.split(",")[0].strip() if authors else ""
        if " and " in first_author:
            first_author = first_author.split(" and ")[0].strip()

        params = {
            "query.title": title,
            "rows": 5,
            "mailto": mailto
        }
        if first_author:
            params["query.author"] = first_author

        response = requests.get(
            "https://api.crossref.org/works",
            params=params,
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        items = data.get("message", {}).get("items", [])

        if not items:
            return {"doi": "", "status": "not_found", "confidence": 0.0}

        # Find best match by title similarity
        best_match = None
        best_score = 0

        for item in items:
            item_title = " ".join(item.get("title", []))
            score = fuzz.ratio(normalize_title(title), normalize_title(item_title))

            # Boost score if year matches
            item_year = str(item.get("published-print", {}).get("date-parts", [[""]])[0][0] or
                          item.get("published-online", {}).get("date-parts", [[""]])[0][0] or "")
            if item_year == year:
                score += 10

            if score > best_score:
                best_score = score
                best_match = item

        if best_match and best_score >= 70:
            confidence = min(best_score / 100.0, 1.0)
            return {
                "doi": best_match.get("DOI", ""),
                "status": "found",
                "confidence": confidence
            }

        return {"doi": "", "status": "not_found", "confidence": 0.0}

    except requests.exceptions.Timeout:
        return {"doi": "", "status": "error", "confidence": 0.0, "error": "Request timed out"}
    except Exception as e:
        return {"doi": "", "status": "error", "confidence": 0.0, "error": str(e)}


def verify_doi_exists(doi: str) -> bool:
    """Check if DOI is valid via HEAD request to CrossRef."""
    if not doi:
        return False
    try:
        encoded_doi = urllib.parse.quote(doi, safe='')
        response = requests.head(
            f"https://api.crossref.org/works/{encoded_doi}",
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False


def lookup_dois_batch(papers: list[dict], mailto: str = "user@example.com",
                      progress_callback=None) -> dict:
    """
    Batch lookup DOIs for multiple papers.
    Returns dict mapping paper index to DOI info.
    """
    results = {}
    total = len(papers)

    for i, paper in enumerate(papers):
        if progress_callback:
            progress_callback(i + 1, total, paper.get("title", "")[:50])

        result = lookup_doi(
            paper.get("title", ""),
            paper.get("authors", ""),
            paper.get("year", ""),
            mailto
        )

        # Auto-verify if found
        if result["status"] == "found" and result["doi"]:
            result["verified"] = verify_doi_exists(result["doi"])
        else:
            result["verified"] = False

        results[i] = result

    return results


def clean_html_for_llm(html_content: str) -> str:
    """Extract just the publication text from HTML to reduce size."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "meta", "link", "noscript"]):
        tag.decompose()

    # Try to extract publications from Google Scholar structure
    publications_text = []

    # Look for Google Scholar publication rows
    pub_rows = soup.select("tr.gsc_a_tr")
    if pub_rows:
        for row in pub_rows:
            # Get title - check for full title in data-href or title attribute first
            title_link = row.select_one("a.gsc_a_at")
            if title_link:
                # Full title might be in title attribute or data attribute
                full_title = title_link.get("title") or title_link.get("data-title") or title_link.get_text(strip=True)
                publications_text.append(f"TITLE: {full_title}")

            # Get authors and venue
            gray_text = row.select("td.gsc_a_t div.gs_gray")
            if gray_text:
                for gt in gray_text:
                    publications_text.append(gt.get_text(strip=True))

            # Get year
            year_cell = row.select_one("td.gsc_a_y span")
            if year_cell:
                publications_text.append(f"YEAR: {year_cell.get_text(strip=True)}")

            # Get citations
            cite_cell = row.select_one("td.gsc_a_c a")
            if cite_cell:
                publications_text.append(f"CITATIONS: {cite_cell.get_text(strip=True)}")

            publications_text.append("---")  # Separator between papers

        return "\n".join(publications_text)

    # Fallback: get all text
    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines[:2000])


def extract_with_llm(html_content: str, lm_studio_url: str = DEFAULT_LM_STUDIO_URL) -> list[dict]:
    """Use LM Studio to extract publication data from HTML."""
    import json

    # Clean HTML first to reduce size
    cleaned_text = clean_html_for_llm(html_content)

    prompt = f"""Extract ALL academic publications from this text. Return a JSON array.

IMPORTANT: Extract the COMPLETE title - never truncate or shorten titles.

For each publication extract:
- title: FULL paper title (do NOT truncate)
- authors: author names
- year: publication year (4 digits)
- venue: journal/conference name
- citations: citation count (number)

Return ONLY valid JSON array:
[{{"title": "Full Complete Paper Title Here", "authors": "A Smith", "year": "2024", "venue": "Nature", "citations": 10}}]

TEXT:
{cleaned_text}

JSON:"""

    try:
        response = requests.post(
            f"{lm_studio_url}/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that extracts structured data from text. You help researchers organize their own publication lists. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 16000
            },
            timeout=300
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()

        # Check for refusal responses
        refusal_phrases = ["i can't", "i cannot", "i'm sorry", "i am sorry", "unable to", "not able to"]
        if any(phrase in content.lower() for phrase in refusal_phrases) and "[" not in content:
            st.error("⚠️ LLM refused the request. Try a different model in LM Studio, or use the manual paste method.")
            st.info("Some models refuse to process HTML. Try: Llama, Mistral, or Qwen models.")
            return []

        # Clean up response - extract JSON if wrapped in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # Find JSON array in response
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            content = content[start:end]

        publications = json.loads(content)

        # Normalize the data
        normalized = []
        for pub in publications:
            normalized.append({
                "title": str(pub.get("title", "")),
                "authors": str(pub.get("authors", "")),
                "year": str(pub.get("year", "")),
                "venue": str(pub.get("venue", "")),
                "citations": int(pub.get("citations", 0)) if pub.get("citations") else 0,
                "pub_type": "article",
            })
        return normalized

    except json.JSONDecodeError as e:
        st.error(f"LLM returned invalid JSON: {e}")
        st.code(content[:500])  # Show what was returned
        return []
    except Exception as e:
        st.error(f"LM Studio error: {e}")
        return []


def fetch_scholar_html(user_id: str, limit: int = 20) -> str:
    """Fetch raw HTML from Google Scholar profile."""
    url = f"https://scholar.google.com/citations?user={user_id}&cstart=0&pagesize={limit}&sortby=pubdate"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text


def parse_manual_publications(text: str) -> list[dict]:
    """Parse manually pasted publications (one per line: Title | Authors | Year | Venue)."""
    publications = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 1:
            publications.append({
                "title": parts[0] if len(parts) > 0 else "",
                "authors": parts[1] if len(parts) > 1 else "",
                "year": parts[2] if len(parts) > 2 else "",
                "venue": parts[3] if len(parts) > 3 else "",
                "citations": 0,
                "pub_type": "article",
            })
    return publications


def parse_bibtex_input(text: str) -> list[dict]:
    """Parse BibTeX entries pasted by user."""
    try:
        parser = BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False
        bib_database = bibtexparser.loads(text, parser=parser)

        publications = []
        for entry in bib_database.entries:
            publications.append({
                "title": entry.get("title", "").strip("{}"),
                "authors": entry.get("author", ""),
                "year": entry.get("year", ""),
                "venue": entry.get("journal", "") or entry.get("booktitle", ""),
                "doi": entry.get("doi", ""),
                "citations": 0,
                "pub_type": entry.get("ENTRYTYPE", "article"),
                "abstract": entry.get("abstract", ""),
                "keywords": entry.get("keywords", ""),
                "pages": entry.get("pages", ""),
                "volume": entry.get("volume", "").replace("NULL", ""),
                "url": entry.get("url", "") or entry.get("website", ""),
                "abbr": entry.get("abbr", ""),
            })
        return publications
    except Exception as e:
        st.error(f"Error parsing BibTeX: {e}")
        return []


def format_author_name(name: str) -> str:
    """Format a single author name: 'Lastname, Firstname' -> 'Firstname Lastname'."""
    name = name.strip()
    if not name:
        return name

    # Check if name is in "Lastname, Firstname" format
    if ',' in name:
        parts = name.split(',', 1)
        if len(parts) == 2:
            lastname = parts[0].strip()
            firstname = parts[1].strip()
            # Return as "Firstname Lastname"
            return f"{firstname} {lastname}"
    return name


def format_authors_display(authors: str) -> str:
    """Format authors for display (replace 'and' with commas, clean up, fix name order)."""
    if not authors:
        return authors

    # First decode any LaTeX in author names
    authors = decode_latex(authors)

    # Clean up stray ampersands and normalize
    import re as rx
    authors = rx.sub(r'\s*&\s*and\s*', ' and ', authors)  # "& and" -> "and"
    authors = rx.sub(r'\s*&\s+', ' and ', authors)  # "& " -> "and"
    authors = rx.sub(r'\s+and\s+and\s+', ' and ', authors)  # "and and" -> "and"

    # Split by " and " and rejoin with commas
    parts = [a.strip() for a in authors.split(" and ") if a.strip()]
    # Filter out single initials, empty parts, and lone symbols
    parts = [p for p in parts if p and len(p) > 2 and p not in ('&', 'and', '&and')]

    # Format each author name (Lastname, Firstname -> Firstname Lastname)
    parts = [format_author_name(p) for p in parts]

    if len(parts) > 5:
        # Show first 2 authors + et al.
        return f"{parts[0]}, {parts[1]} et al."
    elif len(parts) > 1:
        return ", ".join(parts[:-1]) + f" & {parts[-1]}"
    elif parts:
        return parts[0]
    return authors


def fetch_existing_bib(url: str) -> list[dict]:
    """Download and parse papers.bib from a URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        parser = BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False
        bib_database = bibtexparser.loads(response.text, parser=parser)

        entries = []
        for entry in bib_database.entries:
            entries.append({
                "title": entry.get("title", "").strip("{}"),
                "authors": entry.get("author", ""),
                "year": entry.get("year", ""),
                "key": entry.get("ID", ""),
            })
        return entries
    except Exception as e:
        st.error(f"Error fetching BibTeX file: {e}")
        return []


def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = " ".join(title.split())
    return title


def normalize_title_advanced(title: str) -> str:
    """Advanced title normalization for better matching."""
    title = title.lower()
    # Remove common variations
    title = re.sub(r"[^\w\s]", " ", title)  # Replace punctuation with space
    title = re.sub(r"\b(a|an|the|of|in|on|at|to|for|and|or|with)\b", "", title)  # Remove stop words
    title = re.sub(r"\s+", " ", title).strip()
    return title


def get_title_tokens(title: str) -> set:
    """Get significant tokens from title for set-based comparison."""
    normalized = normalize_title_advanced(title)
    # Filter out very short words
    return set(word for word in normalized.split() if len(word) > 2)


def normalize_author_name(name: str) -> str:
    """Normalize author name for comparison."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    # Extract last name (handle both "Last, First" and "First Last")
    parts = name.replace(",", " ").split()
    if parts:
        # Return longest part (likely the last name)
        return max(parts, key=len)
    return name


def ai_disambiguate(pub1: dict, pub2: dict, lm_studio_url: str) -> bool:
    """Use AI to determine if two papers are the same."""
    import json

    prompt = f"""Are these two academic paper entries referring to the SAME paper? Consider:
- Minor title variations (punctuation, word order, abbreviations)
- Author name formats may differ
- Same year is a strong signal

Paper 1:
Title: {pub1.get('title', '')}
Authors: {pub1.get('authors', '')}
Year: {pub1.get('year', '')}

Paper 2:
Title: {pub2.get('title', '')}
Authors: {pub2.get('authors', '')}
Year: {pub2.get('year', '')}

Answer with ONLY "SAME" or "DIFFERENT" (one word, no explanation):"""

    try:
        response = requests.post(
            f"{lm_studio_url}/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 10
            },
            timeout=15
        )
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip().upper()
        return "SAME" in answer
    except Exception:
        # If AI fails, fall back to conservative (not same)
        return False


def compute_similarity_score(pub: dict, existing: dict) -> tuple[float, str]:
    """
    Compute comprehensive similarity score between two papers.
    Returns (score, reason) where score is 0-100.
    """
    scores = []
    reasons = []

    pub_title = pub.get("title", "")
    existing_title = existing.get("title", "")

    # 1. Exact normalized match
    norm_pub = normalize_title(pub_title)
    norm_existing = normalize_title(existing_title)
    if norm_pub == norm_existing:
        return 100, "exact_match"

    # 1b. Truncated title match (one title is prefix of the other)
    # Check if shorter title is a significant prefix of the longer one
    shorter = norm_pub if len(norm_pub) < len(norm_existing) else norm_existing
    longer = norm_existing if len(norm_pub) < len(norm_existing) else norm_pub
    if len(shorter) >= 30 and longer.startswith(shorter):
        # Shorter title is prefix of longer - likely truncated
        return 98, "truncated_match"
    # Also check if titles share a long common prefix (both may be truncated)
    if len(norm_pub) >= 30 and len(norm_existing) >= 30:
        common_prefix_len = 0
        for i, (c1, c2) in enumerate(zip(norm_pub, norm_existing)):
            if c1 == c2:
                common_prefix_len = i + 1
            else:
                break
        # If 80%+ of shorter title matches as prefix, consider it a match
        if common_prefix_len >= min(len(norm_pub), len(norm_existing)) * 0.8:
            return 95, "common_prefix_match"

    # 2. Fuzzy ratio
    ratio = fuzz.ratio(norm_pub, norm_existing)
    scores.append(ratio)

    # 3. Token overlap (handles word reordering)
    pub_tokens = get_title_tokens(pub_title)
    existing_tokens = get_title_tokens(existing_title)
    if pub_tokens and existing_tokens:
        overlap = len(pub_tokens & existing_tokens)
        union = len(pub_tokens | existing_tokens)
        token_score = (overlap / union) * 100 if union > 0 else 0
        scores.append(token_score)

    # 4. Partial ratio (handles substrings/truncated titles)
    partial = fuzz.partial_ratio(norm_pub, norm_existing)
    scores.append(partial * 0.95)  # High weight for partial matches (truncated titles)

    # 5. Token sort ratio (handles reordering)
    token_sort = fuzz.token_sort_ratio(norm_pub, norm_existing)
    scores.append(token_sort)

    base_score = max(scores)

    # Boost score if year matches
    if pub.get("year") and existing.get("year") and pub["year"] == existing["year"]:
        base_score = min(100, base_score + 10)
        reasons.append("year_match")

    # Boost score if first author matches
    pub_author = normalize_author_name(pub.get("authors", "").split(",")[0].split(" and ")[0])
    existing_author = normalize_author_name(existing.get("authors", "").split(",")[0].split(" and ")[0])
    if pub_author and existing_author and (pub_author in existing_author or existing_author in pub_author):
        base_score = min(100, base_score + 10)
        reasons.append("author_match")

    reason = "+".join(reasons) if reasons else "title_similarity"
    return base_score, reason


def find_missing_papers(scholar_pubs: list[dict], existing_entries: list[dict],
                        lm_studio_url: str = DEFAULT_LM_STUDIO_URL,
                        use_ai_disambiguation: bool = False,
                        match_threshold: int = DEFAULT_MATCH_THRESHOLD,
                        uncertain_threshold: int = DEFAULT_UNCERTAIN_THRESHOLD) -> list[dict]:
    """Find papers in Scholar that are not in existing BibTeX using advanced matching."""
    missing = []

    for pub in scholar_pubs:
        best_score = 0

        for existing in existing_entries:
            score, reason = compute_similarity_score(pub, existing)
            if score > best_score:
                best_score = score

        # Only skip if very high confidence match
        if best_score >= match_threshold:
            continue
        else:
            missing.append(pub)

    return missing


def generate_citation_key(authors: str, year: str) -> str:
    """Generate citation key like 'SaqrM2024'."""
    if not authors:
        return f"Unknown{year}"

    # Parse first author's last name
    first_author = authors.split(" and ")[0].strip()

    # Handle "Last, First" format
    if "," in first_author:
        last_name = first_author.split(",")[0].strip()
    else:
        # Handle "First Last" format
        parts = first_author.split()
        last_name = parts[-1] if parts else "Unknown"

    # Get first initial
    first_initial = ""
    if "," in first_author:
        first_part = first_author.split(",")[1].strip() if len(first_author.split(",")) > 1 else ""
        first_initial = first_part[0].upper() if first_part else ""
    else:
        parts = first_author.split()
        first_initial = parts[0][0].upper() if parts else ""

    # Clean last name
    last_name = re.sub(r"[^\w]", "", last_name)

    return f"{last_name}{first_initial}{year}"


def format_authors_bibtex(authors: str) -> str:
    """Format authors for BibTeX: 'LastName FirstInitial.' format.

    Input: "Mohammed Saqr" or "Saqr, Mohammed" or "Mohammed Saqr and Sonsoles López-Pernas"
    Output: "Saqr M. and López-Pernas S."
    """
    if not authors:
        return authors

    # Split by " and " or ";"
    author_list = []
    for sep in [" and ", ";"]:
        if sep in authors:
            author_list = [a.strip() for a in authors.split(sep) if a.strip()]
            break
    if not author_list:
        author_list = [authors.strip()]

    formatted = []
    for author in author_list:
        author = author.strip()
        if not author:
            continue

        # Handle "LastName, FirstName" format
        if "," in author:
            parts = author.split(",", 1)
            last_name = parts[0].strip()
            first_name = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Handle "FirstName LastName" format
            parts = author.split()
            if len(parts) >= 2:
                last_name = parts[-1]
                first_name = " ".join(parts[:-1])
            else:
                last_name = author
                first_name = ""

        # Get first initial
        first_initial = first_name[0].upper() + "." if first_name else ""
        formatted.append(f"{last_name} {first_initial}".strip())

    return " and ".join(formatted)


def format_bibtex_entry(paper: dict, citation_key: str, abbr: str, doi: str = "",
                        format_type: str = "website") -> str:
    """Generate formatted BibTeX string with all fields.

    Args:
        paper: Paper data dictionary
        citation_key: Citation key for the entry
        abbr: Abbreviation for display
        doi: DOI if available
        format_type: "website" for Jekyll/al-folio style or "standard" for classic BibTeX

    Returns:
        Formatted BibTeX string
    """
    # Determine entry type
    entry_type = "ARTICLE"
    venue = paper.get("venue", "") or ""
    venue_lower = venue.lower()
    if any(kw in venue_lower for kw in ["conference", "proceedings", "symposium", "workshop"]):
        entry_type = "INPROCEEDINGS"

    # Format authors in BibTeX style: "LastName F. and LastName F."
    authors = format_authors_bibtex(paper.get("authors", ""))

    # Build entry
    lines = [f"@{entry_type}{{{citation_key},"]
    lines.append(f'  author = {{{authors}}},')
    lines.append(f'  year = {{{paper.get("year", "")}}},')
    lines.append(f'  title = {{{paper.get("title", "")}}},')

    # Add both journal and booktitle (al-folio format uses both)
    if venue:
        lines.append(f'  journal = {{{venue}}},')
        lines.append(f'  booktitle = {{{venue}}},')

    # Volume, number, and pages
    volume = paper.get("volume", "")
    if volume and volume != "NULL":
        lines.append(f'  volume = {{{volume}}},')

    number = paper.get("number", "")
    if number and number != "NULL":
        lines.append(f'  number = {{{number}}},')

    pages = paper.get("pages", "")
    if pages:
        lines.append(f'  pages = {{{pages}}},')

    # DOI, URL and Website (for Jekyll/al-folio compatibility)
    doi = doi or paper.get("doi", "")
    if doi:
        # Clean DOI - remove URL prefix if present
        if doi.startswith("http"):
            doi = doi.split("doi.org/")[-1]
        lines.append(f'  doi = {{{doi}}},')

    # Keywords (before url/website)
    keywords = paper.get("keywords", "")
    if keywords:
        lines.append(f'  keywords = {{{keywords}}},')

    # URL and website after doi
    if doi:
        lines.append(f'  url = {{https://doi.org/{doi}}},')
        lines.append(f'  website = {{https://doi.org/{doi}}},')
    elif paper.get("url"):
        lines.append(f'  url = {{{paper.get("url")}}},')

    # Website-specific metadata (for Jekyll/al-folio themes)
    if format_type == "website":
        lines.append(f'  abbr = {{{abbr}}},')

        # Abstract
        abstract = paper.get("abstract", "")
        if abstract:
            # Escape any special characters in abstract for BibTeX
            abstract = abstract.replace('{', '\\{').replace('}', '\\}')
            lines.append(f'  abstract = {{{abstract}}},')

        lines.append('  bibtex_show = {true},')
        lines.append('  selected = {false},')

    lines.append('}')

    return "\n".join(lines)


def export_all_bibtex(format_type: str = "website", author_filter: str = "") -> str:
    """Export all papers to BibTeX format.

    Args:
        format_type: "website" or "standard"
        author_filter: Only include papers by this author

    Returns:
        Complete BibTeX file content
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if author_filter:
        papers = conn.execute("""
            SELECT * FROM papers
            WHERE authors LIKE ?
            ORDER BY year DESC, title
        """, (f"%{author_filter}%",)).fetchall()
    else:
        papers = conn.execute("SELECT * FROM papers ORDER BY year DESC, title").fetchall()

    conn.close()

    entries = []
    for p in papers:
        paper = dict(p)
        citation_key = paper.get('citation_key') or generate_citation_key(
            paper.get('authors', ''), paper.get('year', '')
        )
        abbr = paper.get('abbr') or generate_abbr(paper.get('venue', ''))

        entry = format_bibtex_entry(
            paper=paper,
            citation_key=citation_key,
            abbr=abbr,
            doi=paper.get('doi', ''),
            format_type=format_type
        )
        entries.append(entry)

    return "\n\n".join(entries)


def render_paper_card(paper: dict, index: int, selections: dict, dois: dict) -> None:
    """Render a nicely formatted paper card."""
    paper_id = f"paper_{index}"

    # Custom CSS for cards
    st.markdown("""
    <style>
    .paper-card {
        background: linear-gradient(135deg, #667eea11 0%, #764ba211 100%);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }
    .paper-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    .paper-authors {
        color: #4a4a6a;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .paper-meta {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .paper-badge {
        background: #667eea22;
        color: #667eea;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .paper-badge-cite {
        background: #f59e0b22;
        color: #d97706;
    }
    .paper-venue {
        color: #6b7280;
        font-style: italic;
        font-size: 0.9rem;
    }
    .doi-found { color: #10b981; font-weight: 500; }
    .doi-unverified { color: #f59e0b; font-weight: 500; }
    .doi-notfound { color: #6b7280; font-weight: 500; }
    .doi-invalid { color: #ef4444; font-weight: 500; }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        # Include checkbox
        include = st.checkbox(
            "Include",
            value=selections[paper_id]["include"],
            key=f"include_{paper_id}",
            label_visibility="collapsed"
        )
        selections[paper_id]["include"] = include

        # Paper card HTML - decode LaTeX for display
        display_title = decode_latex(paper['title'])
        display_authors = decode_latex(paper['authors'])
        display_venue = decode_latex(paper['venue'] or 'Venue not specified')

        citations_badge = f'<span class="paper-badge paper-badge-cite">{paper["citations"]} citations</span>' if paper["citations"] else ''
        year_badge = f'<span class="paper-badge">{paper["year"]}</span>' if paper["year"] else ''

        card_html = f"""
        <div class="paper-card">
            <div class="paper-title">{display_title}</div>
            <div class="paper-authors">{display_authors}</div>
            <div class="paper-venue">{display_venue}</div>
            <div class="paper-meta">
                {year_badge}
                {citations_badge}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # DOI Section
        doi_info = dois.get(index, {"doi": "", "status": "not_found", "verified": False})

        # Initialize DOI in selections if not present
        if "doi" not in selections[paper_id]:
            selections[paper_id]["doi"] = doi_info.get("doi", "")

        # DOI status display
        st.markdown("**DOI:**")
        doi_col1, doi_col2, doi_col3 = st.columns([3, 1, 1])

        with doi_col1:
            # Determine status indicator
            current_doi = selections[paper_id]["doi"]
            if current_doi:
                if doi_info.get("verified"):
                    status_html = '<span class="doi-found">Found</span>'
                elif doi_info.get("status") == "found":
                    status_html = '<span class="doi-unverified">Unverified</span>'
                else:
                    status_html = '<span class="doi-notfound">Manual</span>'
            else:
                if doi_info.get("status") == "not_found":
                    status_html = '<span class="doi-notfound">Not found</span>'
                elif doi_info.get("status") == "error":
                    status_html = '<span class="doi-invalid">Error</span>'
                else:
                    status_html = '<span class="doi-notfound">Not looked up</span>'

            st.markdown(status_html, unsafe_allow_html=True)

            # Editable DOI field
            new_doi = st.text_input(
                "DOI",
                value=selections[paper_id]["doi"],
                key=f"doi_{paper_id}",
                placeholder="10.xxxx/xxxxx",
                label_visibility="collapsed"
            )
            selections[paper_id]["doi"] = new_doi

        with doi_col2:
            # Manual lookup button
            if st.button("Lookup", key=f"lookup_doi_{paper_id}"):
                with st.spinner("Looking up DOI..."):
                    mailto = st.session_state.get("crossref_email", "user@example.com")
                    result = lookup_doi(
                        paper.get("title", ""),
                        paper.get("authors", ""),
                        paper.get("year", ""),
                        mailto
                    )
                    if result["status"] == "found" and result["doi"]:
                        result["verified"] = verify_doi_exists(result["doi"])
                        selections[paper_id]["doi"] = result["doi"]
                        # Also update widget state directly
                        st.session_state[f"doi_{paper_id}"] = result["doi"]
                        dois[index] = result
                        st.rerun()
                    else:
                        st.warning("DOI not found")

        with doi_col3:
            # Verify button
            if st.button("Verify", key=f"verify_doi_{paper_id}"):
                current_doi = selections[paper_id]["doi"]
                if current_doi:
                    with st.spinner("Verifying..."):
                        is_valid = verify_doi_exists(current_doi)
                        dois[index] = {
                            "doi": current_doi,
                            "status": "found" if is_valid else "error",
                            "verified": is_valid,
                            "confidence": 1.0 if is_valid else 0.0
                        }
                        if is_valid:
                            st.success("DOI verified!")
                        else:
                            st.error("DOI invalid!")
                        st.rerun()
                else:
                    st.warning("Enter a DOI first")

        # Editable fields in columns
        col1, col2 = st.columns(2)
        with col1:
            selections[paper_id]["citation_key"] = st.text_input(
                "Citation Key",
                value=selections[paper_id]["citation_key"],
                key=f"key_{paper_id}"
            )
        with col2:
            selections[paper_id]["abbr"] = st.text_input(
                "Abbreviation",
                value=selections[paper_id]["abbr"],
                key=f"abbr_{paper_id}"
            )


def main():
    st.set_page_config(
        page_title="Scholar BibTeX Sync",
        page_icon="📚",
        layout="wide"
    )

    st.title("Scholar BibTeX Sync")
    st.caption("Find papers on Google Scholar missing from your papers.bib")

    # Initialize session state
    SESSION_DEFAULTS = {
        "scholar_pubs": [], "existing_entries": [], "missing_papers": [],
        "step": 1, "dois": {}, "doi_lookup_enabled": True,
        "list_sel": set(), "quality_issues": None, "input_method": "Upload CSV"
    }
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Initialize database
    init_db()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")

        input_method = st.radio(
            "Scholar Input Method",
            ["Google Scholar URL", "Upload CSV", "Paste BibTeX", "LM Studio (AI extraction)", "Paste HTML + AI", "Paste text list"],
            help="Choose how to input your publications",
            index=["Google Scholar URL", "Upload CSV", "Paste BibTeX", "LM Studio (AI extraction)", "Paste HTML + AI", "Paste text list"].index(
                st.session_state.get("input_method", "Upload CSV")
            )
        )
        st.session_state.input_method = input_method

        lm_studio_url = st.text_input(
            "LM Studio URL",
            value=st.session_state.get("lm_studio_url", DEFAULT_LM_STUDIO_URL),
            help="LM Studio API endpoint"
        )
        st.session_state.lm_studio_url = lm_studio_url

        bib_url = st.text_input(
            "papers.bib URL",
            value=st.session_state.get("bib_url", DEFAULT_BIB_URL),
            help="Raw GitHub URL to your existing papers.bib"
        )
        st.session_state.bib_url = bib_url

        st.divider()
        st.subheader("Duplicate Detection")

        match_threshold = st.slider(
            "Match threshold",
            50, 100, 85,
            help="Score ≥ this = confident match (skip paper)"
        )
        st.session_state.match_threshold = match_threshold

        uncertain_threshold = st.slider(
            "Uncertain threshold",
            30, 80, 60,
            help="Score between this and match threshold = ask AI"
        )
        st.session_state.uncertain_threshold = uncertain_threshold

        use_ai_disambiguation = st.toggle(
            "Use AI for uncertain matches",
            value=st.session_state.get("use_ai_disambiguation", True),
            help="Use LLM to disambiguate papers with uncertain similarity"
        )
        st.session_state.use_ai_disambiguation = use_ai_disambiguation

        st.divider()
        st.subheader("DOI Lookup")

        doi_lookup_enabled = st.toggle(
            "Lookup DOIs from CrossRef",
            value=st.session_state.doi_lookup_enabled,
            help="Automatically find DOIs for papers using CrossRef API"
        )
        st.session_state.doi_lookup_enabled = doi_lookup_enabled

        crossref_email = st.text_input(
            "CrossRef Email (optional)",
            value=st.session_state.get("crossref_email", ""),
            help="Email for CrossRef polite pool (faster API access)",
            placeholder="your@email.com"
        )
        st.session_state.crossref_email = crossref_email if crossref_email else "user@example.com"

    # ============ MAIN TABS ============
    tab_papers, tab_sync = st.tabs(["📂 My Papers", "🔄 Sync Papers"])

    # ============ MY PAPERS TAB ============
    with tab_papers:
        # Sub-tabs for My Papers
        refs_tab, stats_tab, table_tab, list_tab, authors_tab, venues_tab, quality_tab, dupes_tab, export_tab, import_tab, trash_tab = st.tabs([
            "📚 References", "📊 Statistics", "📋 Table View", "📄 List View",
            "👥 Authors", "📰 Venues", "⚠️ Quality", "🔍 Duplicates", "📤 Export", "📥 Import", "🗑️ Trash"
        ])

        # ===== CLEAN REFERENCES TAB =====
        with refs_tab:
            st.subheader("📚 Publication References")

            # Search and Sort - clean layout
            r1, r2, r3, r4 = st.columns([3, 2, 1, 1])
            with r1:
                ref_search = st.text_input("🔍 Search", key="ref_search", placeholder="Search title, venue...")
            with r2:
                ref_author = st.text_input("👤 Author", key="ref_author", placeholder="Author name...")
            with r3:
                ref_year = st.selectbox("Year", ["All"] + get_db_years(), key="ref_year")
            with r4:
                ref_sort = st.selectbox("Sort", ["year ↓", "year ↑", "citations ↓", "citations ↑", "title ↑", "title ↓"], key="ref_sort")

            # Parse sort
            sort_field = ref_sort.split()[0]
            sort_order = "DESC" if "↓" in ref_sort else "ASC"

            # Get papers
            ref_results = search_db(
                ref_search,
                ref_author,
                ref_year if ref_year != "All" else "",
                sort_field,
                sort_order,
                limit=500
            )

            st.caption(f"{len(ref_results)} papers")
            st.divider()

            # Clean formatted references with rank numbers
            for rank, paper in enumerate(ref_results, 1):
                citations = paper.get('citations', 0) or 0
                citation_badge = f" **{format_citation_badge(citations, 'long')}**" if citations > 0 else ""
                st.markdown(f"**{rank}.** {format_reference(paper)}{citation_badge}")
                st.markdown("")  # spacing

        # ===== STATISTICS TAB =====
        with stats_tab:
            stats = get_stats()
            st.subheader("📊 Publication Statistics")

            # Key metrics - Row 1
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Total Papers", stats['total'])
            with m2: st.metric("With DOI", stats['with_doi'])
            with m3: st.metric("DOI Coverage", f"{100*stats['with_doi']//max(stats['total'],1)}%")
            with m4:
                if stats['years'][0]:
                    st.metric("Year Range", f"{stats['years'][0]}-{stats['years'][1]}")

            # Citation metrics - Row 2
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("📈 Total Citations", f"{stats['total_citations']:,}")
            with c2: st.metric("📊 Avg Citations", stats['avg_citations'])
            with c3: st.metric("🏆 Max Citations", stats['max_citations'])
            with c4: st.metric("📝 With Citations", stats['papers_with_citations'])

            st.divider()

            # Charts
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Papers by Year")
                if stats['by_year']:
                    import pandas as pd
                    df_years = pd.DataFrame(list(stats['by_year'].items()), columns=['Year', 'Count'])
                    df_years = df_years.sort_values('Year')
                    st.bar_chart(df_years.set_index('Year'))

            with col2:
                st.subheader("Citations by Year")
                if stats.get('citations_by_year'):
                    import pandas as pd
                    df_cites = pd.DataFrame(list(stats['citations_by_year'].items()), columns=['Year', 'Citations'])
                    df_cites = df_cites.sort_values('Year')
                    st.bar_chart(df_cites.set_index('Year'))

            # Top Cited Papers
            st.divider()
            st.subheader("🏆 Top Cited Papers")
            if stats.get('top_cited'):
                for i, (title, authors, year, citations, doi) in enumerate(stats['top_cited'][:10], 1):
                    authors_display = format_author_name(decode_latex(authors.split(' and ')[0])) if authors else ""
                    if ' and ' in (authors or ''):
                        authors_display += " et al."
                    st.markdown(f"**{i}.{format_citation_badge(citations, 'long')}** {title[:80]}{'...' if len(title) > 80 else ''}")
                    st.caption(f"{authors_display} ({year})" + (f" | DOI: {doi}" if doi else ""))
            else:
                st.info("No citation data available. Import a CSV with citation counts.")

            # Top Venues
            st.divider()
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                st.subheader("📰 Top Venues")
                if stats['top_venues']:
                    for venue, count in stats['top_venues'][:10]:
                        st.write(f"**{count}** - {decode_latex(venue)}")

            # Author Statistics
            with tcol2:
                st.subheader("📝 Top Co-Authors")
                if stats.get('top_authors'):
                    authors_list = stats['top_authors']
                    for author, count in authors_list[:10]:
                        # Names already normalized in get_stats()
                        st.write(f"**{count}** - {decode_latex(author)}")

            # Standardization Section
            st.divider()
            st.subheader("🔧 Standardization")
            std_col1, std_col2 = st.columns(2)

            with std_col1:
                st.markdown("**Combine Venues**")
                st.caption("Replace one venue name with another")
                all_venues = get_all_venues()
                if all_venues:
                    old_venue = st.selectbox("Old venue name", [""] + all_venues, key="old_venue")
                    new_venue = st.text_input("New venue name", key="new_venue", placeholder="Enter standardized name")
                    if st.button("Apply Venue Change", disabled=not (old_venue and new_venue)):
                        affected = standardize_venue(old_venue, new_venue)
                        st.success(f"Updated {affected} papers")
                        st.rerun()

            with std_col2:
                st.markdown("**Combine Authors**")
                st.caption("Replace one author name with another")
                all_authors = get_all_authors()
                if all_authors:
                    old_author = st.selectbox("Old author name", [""] + all_authors, key="old_author")
                    new_author = st.text_input("New author name", key="new_author", placeholder="Enter standardized name")
                    if st.button("Apply Author Change", disabled=not (old_author and new_author)):
                        affected = standardize_author(old_author, new_author)
                        st.success(f"Updated {affected} papers")
                        st.rerun()

            # DOI Batch Fetch Section
            st.divider()
            st.subheader("🔗 Batch DOI Lookup")
            doi_method = st.radio("DOI source", ["CrossRef API", "From Website"], horizontal=True, key="doi_batch_method")

            if doi_method == "CrossRef API":
                st.caption("Lookup DOIs from CrossRef for papers without DOIs")
                if st.button("🔍 Lookup Missing DOIs from CrossRef"):
                    # Get papers without DOI
                    conn = sqlite3.connect(DB_PATH)
                    papers_no_doi = conn.execute(
                        "SELECT id, title, authors, year FROM papers WHERE doi = '' OR doi IS NULL"
                    ).fetchall()
                    conn.close()

                    if not papers_no_doi:
                        st.success("All papers already have DOIs!")
                    else:
                        progress = st.progress(0)
                        status = st.empty()
                        found = 0
                        for i, (pid, title, authors, year) in enumerate(papers_no_doi):
                            progress.progress((i + 1) / len(papers_no_doi))
                            status.text(f"Checking: {title[:50]}...")
                            res = lookup_doi(title, authors or "", year or "")
                            if res.get("doi"):
                                update_db_doi(pid, res["doi"])
                                found += 1
                        progress.empty()
                        status.empty()
                        st.success(f"Found {found} new DOIs out of {len(papers_no_doi)} papers checked")
                        st.rerun()

            else:  # From Website
                st.caption("Fetch DOIs from publication websites")
                website_urls = st.text_area(
                    "Website URLs (one per line)",
                    value="https://saqr.me/publications/\nhttps://sonsoles.me/publications.html",
                    height=80
                )
                if st.button("🌐 Fetch DOIs from Websites"):
                    urls = [u.strip() for u in website_urls.split('\n') if u.strip()]
                    all_website_dois = {}

                    progress = st.progress(0)
                    for i, url in enumerate(urls):
                        progress.progress((i + 0.5) / len(urls))
                        st.info(f"Fetching {url}...")
                        dois = fetch_dois_from_website(url)
                        if "error" not in dois:
                            all_website_dois.update(dois)
                        progress.progress((i + 1) / len(urls))

                    progress.empty()
                    st.success(f"Found {len(all_website_dois)} DOI references from websites")

                    # Match to papers
                    conn = sqlite3.connect(DB_PATH)
                    papers_no_doi = conn.execute(
                        "SELECT id, title FROM papers WHERE doi = '' OR doi IS NULL"
                    ).fetchall()
                    conn.close()

                    matched = 0
                    for pid, title in papers_no_doi:
                        doi = match_doi_from_website(title, all_website_dois)
                        if doi:
                            update_db_doi(pid, doi)
                            matched += 1

                    st.success(f"Matched {matched} papers with DOIs from websites")
                    if matched > 0:
                        st.rerun()

            # Update metadata from DOIs section
            st.divider()
            st.subheader("📥 Update Metadata from DOIs")
            st.caption("Fetch paper details (title, authors, year, venue) from CrossRef using existing DOIs")

            ucol1, ucol2 = st.columns(2)
            with ucol1:
                if st.button("📥 Update All Papers from DOIs"):
                    conn = sqlite3.connect(DB_PATH)
                    papers_with_doi = conn.execute("SELECT id, doi, title FROM papers WHERE doi != '' AND doi IS NOT NULL").fetchall()
                    conn.close()

                    if not papers_with_doi:
                        st.warning("No papers with DOIs found")
                    else:
                        progress = st.progress(0)
                        status = st.empty()
                        updated = 0
                        for i, (pid, doi, title) in enumerate(papers_with_doi):
                            progress.progress((i + 1) / len(papers_with_doi))
                            status.text(f"Updating: {title[:40]}...")

                            result = update_paper_from_doi(pid, doi)
                            if result.get("success") and result.get("updated"):
                                updated += 1

                            import time
                            time.sleep(0.3)  # Rate limiting

                        progress.empty()
                        status.empty()
                        st.success(f"Updated {updated} papers with metadata from CrossRef")
                        st.rerun()

            with ucol2:
                st.markdown("**Test single DOI:**")
                test_doi = st.text_input("Enter DOI to test", placeholder="10.1000/xyz123", key="test_doi")
                if st.button("🔍 Fetch Metadata", disabled=not test_doi):
                    with st.spinner("Fetching..."):
                        metadata = fetch_metadata_from_doi(test_doi)
                        if "error" in metadata:
                            st.error(metadata["error"])
                        else:
                            st.success("Metadata found!")
                            st.json(metadata)

        # ===== TABLE TAB =====
        with table_tab:
            st.subheader("📋 Papers Table")

            # Filters
            fc1, fc2, fc3, fc4 = st.columns([2, 2, 1, 1])
            with fc1: tbl_search = st.text_input("🔍 Search", key="tbl_search", placeholder="Title or venue...")
            with fc2: tbl_author = st.text_input("👤 Author", key="tbl_author")
            with fc3: tbl_year = st.selectbox("📅 Year", ["All"] + get_db_years(), key="tbl_year")
            with fc4: tbl_sort = st.selectbox("Sort", ["year ↓", "citations ↓", "title ↑"], key="tbl_sort")

            # Parse sort
            tbl_sort_field = tbl_sort.split()[0]
            tbl_sort_order = "DESC" if "↓" in tbl_sort else "ASC"

            results = search_db(tbl_search, tbl_author, tbl_year if tbl_year != "All" else "",
                              tbl_sort_field, tbl_sort_order)

            if results:
                import pandas as pd
                # Create DataFrame for table display with rank
                table_data = []
                for rank, p in enumerate(results, 1):
                    table_data.append({
                        '#': rank,
                        'Year': p['year'],
                        'Title': decode_latex(p['title'])[:70],
                        'Authors': decode_latex(p['authors'])[:40],
                        'Venue': decode_latex(p['venue'] or '')[:25],
                        'Cites': p.get('citations', 0) or 0,
                        'DOI': '✓' if p.get('doi') else '',
                        'ID': p['id']
                    })
                df = pd.DataFrame(table_data)

                # Selection
                if "tbl_selected" not in st.session_state:
                    st.session_state.tbl_selected = []

                st.dataframe(
                    df[['#', 'Year', 'Title', 'Authors', 'Venue', 'Cites', 'DOI']],
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(f"Showing {len(results)} papers | Total citations: {sum(p.get('citations', 0) or 0 for p in results)}")

                # Export all filtered
                if st.button("📥 Export All Filtered to BibTeX"):
                    bib_out = []
                    for p in results:
                        if p.get("bibtex"): bib_out.append(p["bibtex"])
                        else:
                            key = p.get("citation_key") or generate_citation_key(p.get("authors",""), p.get("year",""))
                            bib_out.append(format_bibtex_entry(p, key, p.get("abbr","Article"), p.get("doi","")))
                    st.download_button("💾 Download", "\n\n".join(bib_out), f"papers_{datetime.now().strftime('%Y%m%d')}.bib")
            else:
                st.info("No papers found")

        # ===== LIST TAB =====
        with list_tab:
            st.subheader("📄 Papers List")

            # Search and Sort
            lc1, lc2, lc3, lc4, lc5 = st.columns([3, 2, 1, 1, 1])
            with lc1: list_search = st.text_input("🔍 Search", key="list_search")
            with lc2: list_author = st.text_input("👤 Author", key="list_author")
            with lc3: list_year = st.selectbox("Year", ["All"] + get_db_years(), key="list_year")
            with lc4: list_sort = st.selectbox("Sort", ["year", "citations", "title", "authors", "venue"], key="list_sort")
            with lc5: list_order = st.selectbox("Order", ["DESC", "ASC"], key="list_order", format_func=lambda x: "↓ Z-A" if x == "DESC" else "↑ A-Z")

            results = search_db(list_search, list_author, list_year if list_year != "All" else "", list_sort, list_order)
            total_cites = sum(p.get('citations', 0) or 0 for p in results)
            st.caption(f"Found {len(results)} papers • Total citations: {total_cites} • Sort by {list_sort} {'↑' if list_order == 'ASC' else '↓'}")

            if "list_sel" not in st.session_state: st.session_state.list_sel = set()

            # Actions
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                if st.button("☑️ Select All", key="list_sel_all"):
                    st.session_state.list_sel = {r["id"] for r in results}
                    st.rerun()
            with ac2:
                if st.button("⬜ Clear", key="list_clear"):
                    st.session_state.list_sel.clear()
                    st.rerun()
            with ac3:
                if st.session_state.list_sel:
                    sel_papers = [r for r in results if r["id"] in st.session_state.list_sel]
                    bib = "\n\n".join([p.get("bibtex") or format_bibtex_entry(p, generate_citation_key(p["authors"], p["year"]), "Article", p.get("doi","")) for p in sel_papers])
                    st.download_button(f"📥 Export {len(st.session_state.list_sel)} Selected", bib, "selected.bib")

            st.divider()

            # View mode toggle
            view_mode = st.radio("View", ["📄 Clean References", "✏️ Edit Mode"], horizontal=True, key="list_view_mode")

            if view_mode == "📄 Clean References":
                # Clean formatted references - just the references, no editing
                for rank, paper in enumerate(results, 1):
                    pid = paper["id"]
                    sel = pid in st.session_state.list_sel
                    citations = paper.get('citations', 0) or 0
                    cite_badge = f" **{format_citation_badge(citations)}**" if citations > 0 else ""

                    col_sel, col_ref = st.columns([1, 30])
                    with col_sel:
                        if st.checkbox("", value=sel, key=f"lsel_{pid}", label_visibility="collapsed"):
                            st.session_state.list_sel.add(pid)
                        else:
                            st.session_state.list_sel.discard(pid)
                    with col_ref:
                        st.markdown(f"**{rank}.** {format_reference(paper)}{cite_badge}")

            else:  # Edit Mode - detailed cards with all fields
                for rank, paper in enumerate(results, 1):
                    pid = paper["id"]
                    sel = pid in st.session_state.list_sel

                    cites = paper.get('citations', 0) or 0
                    cite_str = format_citation_badge(cites, 'long').replace('citations', 'cites')
                    with st.expander(f"{rank}. {'☑️' if sel else '⬜'} {decode_latex(paper['title'])} ({paper['year']}){cite_str}"):
                        # Checkbox for selection
                        col1, col2 = st.columns([1, 20])
                        with col1:
                            if st.checkbox("Select", value=sel, key=f"lsel2_{pid}", label_visibility="collapsed"):
                                st.session_state.list_sel.add(pid)
                            else:
                                st.session_state.list_sel.discard(pid)
                        with col2:
                            st.markdown(format_reference(paper))

                        st.markdown("---")
                        st.markdown("**Edit Fields:**")

                        # Title
                        new_title = st.text_input("Title", value=paper.get('title', ''), key=f"title_{pid}")

                        # Authors and Year
                        ec1, ec2 = st.columns([3, 1])
                        with ec1:
                            new_authors = st.text_input("Authors", value=paper.get('authors', ''), key=f"authors_{pid}")
                        with ec2:
                            new_year = st.text_input("Year", value=paper.get('year', ''), key=f"year_{pid}")

                        # Venue and DOI
                        ec3, ec4 = st.columns(2)
                        with ec3:
                            new_venue = st.text_input("Venue", value=paper.get('venue', ''), key=f"venue_{pid}")
                        with ec4:
                            new_doi = st.text_input("DOI", value=paper.get('doi', ''), key=f"doi_{pid}")

                        # Action buttons
                        st.markdown("---")
                        b1, b2, b3, b4, b5 = st.columns(5)
                        with b1:
                            if st.button("💾 Save", key=f"save_{pid}"):
                                update_paper(pid, title=new_title, authors=new_authors, year=new_year, venue=new_venue, doi=new_doi)
                                st.success("Saved!")
                                st.rerun()
                        with b2:
                            if st.button("🔍 Find DOI", key=f"ldoi_{pid}"):
                                res = lookup_doi(paper["title"], paper["authors"], paper["year"])
                                if res["status"] == "found" and res["doi"]:
                                    update_db_doi(pid, res["doi"])
                                    st.success(f"Found: {res['doi']}")
                                    st.rerun()
                                else:
                                    st.warning("DOI not found")
                        with b3:
                            if st.button("📥 From DOI", key=f"lfetch_{pid}", help="Fetch metadata from DOI", disabled=not paper.get("doi")):
                                result = update_paper_from_doi(pid, paper.get("doi"))
                                if result.get("success"):
                                    st.success(f"Updated: {', '.join(result.get('updated', []))}")
                                    st.rerun()
                                else:
                                    st.error(result.get("error", "Failed"))
                        with b4:
                            if st.button("📋 BibTeX", key=f"lbib_{pid}"):
                                bib = paper.get("bibtex") or format_bibtex_entry(paper, generate_citation_key(paper["authors"], paper["year"]), "Article", paper.get("doi", ""))
                                st.code(bib, language="bibtex")
                        with b5:
                            if st.button("🗑️ Delete", key=f"ldel_{pid}"):
                                delete_from_db(pid)
                                st.session_state.list_sel.discard(pid)
                                st.rerun()

        # ===== AUTHORS TAB =====
        with authors_tab:
            st.subheader("👥 Author Management")

            # Actions row
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                if st.button("🔄 Index Authors from Papers"):
                    with st.spinner("Indexing authors..."):
                        conn = sqlite3.connect(DB_PATH)
                        papers = conn.execute("SELECT id, authors FROM papers").fetchall()
                        conn.close()
                        for paper_id, authors in papers:
                            if authors:
                                link_paper_authors(paper_id, authors)
                        update_author_counts()
                    st.success("Authors indexed!")
                    st.rerun()
            with ac2:
                if st.button("🧹 Find Duplicate Papers"):
                    dupes = find_duplicate_papers()
                    if dupes:
                        st.warning(f"Found {len(dupes)} potential duplicates")
                        for p1, p2 in dupes[:10]:
                            st.write(f"• **{p1['title']}** vs **{p2['title']}**")
                    else:
                        st.success("No duplicates found!")
            with ac3:
                if st.button("🗑️ Remove Duplicates"):
                    removed = remove_duplicate_papers()
                    st.success(f"Removed {removed} duplicate papers")
                    st.rerun()

            st.divider()

            # Authors list
            st.markdown("### All Authors")
            authors_list = get_authors_list(limit=200)
            if authors_list:
                st.caption(f"Found {len(authors_list)} authors")

                # Merge authors section
                st.markdown("**Merge Authors:**")
                mc1, mc2, mc3 = st.columns([2, 2, 1])
                author_names = [(a['id'], a['name'], a['paper_count']) for a in authors_list]
                author_options = {f"{name} ({cnt} papers)": aid for aid, name, cnt in author_names}

                with mc1:
                    merge_from = st.selectbox("Merge FROM (will be deleted)", [""] + list(author_options.keys()), key="merge_from")
                with mc2:
                    merge_to = st.selectbox("Merge INTO (will keep)", [""] + list(author_options.keys()), key="merge_to")
                with mc3:
                    if st.button("🔗 Merge", disabled=not (merge_from and merge_to and merge_from != merge_to)):
                        from_id = author_options[merge_from]
                        to_id = author_options[merge_to]
                        merge_authors(from_id, to_id)
                        st.success("Authors merged!")
                        st.rerun()

                st.divider()

                # Display authors in columns
                st.markdown("**Author List:**")
                cols = st.columns(3)
                for i, author in enumerate(authors_list[:60]):
                    col = cols[i % 3]
                    with col:
                        st.write(f"**{author['paper_count']}** - {decode_latex(author['name'])}")
            else:
                st.info("No authors indexed yet. Click 'Index Authors from Papers' to start.")

        # ===== VENUES TAB =====
        with venues_tab:
            st.subheader("📰 Venue/Journal Management")
            st.caption("Standardize and merge journal/conference names")

            venues_list = get_venues_list(limit=300)

            if venues_list:
                st.info(f"Found **{len(venues_list)}** unique venues")

                # Find similar venues button
                if st.button("🔍 Find Similar Venues", type="secondary"):
                    with st.spinner("Searching for similar venue names..."):
                        st.session_state.similar_venues = find_similar_venues(threshold=75)

                # Show similar venue groups if found
                if "similar_venues" in st.session_state and st.session_state.similar_venues:
                    st.success(f"Found **{len(st.session_state.similar_venues)}** groups of similar venues")
                    for gi, group in enumerate(st.session_state.similar_venues[:10]):
                        with st.expander(f"Group {gi+1}: {len(group)} similar venues"):
                            for v in group:
                                st.write(f"• **{v['venue']}** ({v['paper_count']} papers)")

                st.divider()

                # Merge venues section
                st.markdown("### 🔗 Merge Venues")
                st.caption("Rename all papers from one venue to another")

                vc1, vc2 = st.columns(2)

                # Build venue options with counts
                venue_options = {f"{v['venue']} ({v['paper_count']} papers)": v['venue'] for v in venues_list}

                with vc1:
                    old_venue_select = st.selectbox(
                        "Old venue name (will be replaced)",
                        [""] + list(venue_options.keys()),
                        key="old_venue_select",
                        help="Select the venue name you want to change"
                    )

                with vc2:
                    merge_mode = st.radio(
                        "New venue name",
                        ["Select from list", "Type custom name"],
                        horizontal=True,
                        key="venue_merge_mode"
                    )

                if merge_mode == "Select from list":
                    new_venue_select = st.selectbox(
                        "Select new venue name",
                        [""] + list(venue_options.keys()),
                        key="new_venue_select",
                        help="Select the venue name to use"
                    )
                    new_venue_value = venue_options.get(new_venue_select, "") if new_venue_select else ""
                else:
                    new_venue_value = st.text_input(
                        "Type new venue name",
                        key="new_venue_custom",
                        placeholder="e.g., Computers & Education"
                    )

                # Get actual old venue value
                old_venue_value = venue_options.get(old_venue_select, "") if old_venue_select else ""

                # Merge button
                can_merge = old_venue_value and new_venue_value and old_venue_value != new_venue_value
                if st.button("🔗 Merge Venues", disabled=not can_merge, type="primary"):
                    with st.spinner(f"Merging '{old_venue_value}' → '{new_venue_value}'..."):
                        updated = merge_venues(old_venue_value, new_venue_value)
                        st.success(f"Updated {updated} papers!")
                        # Clear similar venues cache
                        if "similar_venues" in st.session_state:
                            del st.session_state.similar_venues
                        st.rerun()

                st.divider()

                # All venues list
                st.markdown("### 📋 All Venues")

                # Search filter
                venue_search = st.text_input("🔍 Filter venues", key="venue_search", placeholder="Search...")

                # Filter and display
                filtered_venues = venues_list
                if venue_search:
                    filtered_venues = [v for v in venues_list if venue_search.lower() in v['venue'].lower()]

                st.caption(f"Showing {len(filtered_venues)} venues")

                # Display in columns
                vcols = st.columns(2)
                for i, v in enumerate(filtered_venues[:100]):
                    col = vcols[i % 2]
                    with col:
                        st.write(f"**{v['paper_count']}** - {decode_latex(v['venue'])}")
            else:
                st.info("No venues found. Import some papers first.")

        # ===== DATA QUALITY TAB =====
        with quality_tab:
            st.subheader("⚠️ Data Quality Check")
            st.caption("Find and fix problematic records in your database")

            # Action buttons row
            qb1, qb2, qb3 = st.columns(3)
            with qb1:
                if st.button("🔍 Scan for Issues", type="primary", use_container_width=True):
                    with st.spinner("Scanning database for quality issues..."):
                        st.session_state.quality_issues = find_bad_records()
                        st.session_state.bad_authors = find_bad_authors()
                    st.rerun()

            with qb2:
                if st.button("🔧 Fix All with DOI", use_container_width=True, help="Update all papers that have DOI with CrossRef data"):
                    conn = sqlite3.connect(DB_PATH)
                    conn.row_factory = sqlite3.Row
                    papers_with_doi = conn.execute("""
                        SELECT id, doi FROM papers
                        WHERE doi IS NOT NULL AND doi != ''
                        AND (authors IS NULL OR authors = '' OR venue IS NULL OR venue = ''
                             OR year IS NULL OR year = '')
                    """).fetchall()
                    conn.close()

                    if papers_with_doi:
                        progress = st.progress(0)
                        fixed = 0
                        for i, p in enumerate(papers_with_doi):
                            result = update_paper_from_doi(p['id'], p['doi'], overwrite=True)
                            if result.get("status") == "updated":
                                fixed += 1
                            progress.progress((i + 1) / len(papers_with_doi))
                        st.success(f"Fixed {fixed} of {len(papers_with_doi)} papers!")
                        st.rerun()
                    else:
                        st.info("No papers with DOI need fixing")

            with qb3:
                if st.button("🔎 Lookup Missing DOIs", use_container_width=True, help="Find DOIs for papers without them"):
                    conn = sqlite3.connect(DB_PATH)
                    conn.row_factory = sqlite3.Row
                    papers_no_doi = conn.execute("""
                        SELECT id, title, authors, year FROM papers
                        WHERE (doi IS NULL OR doi = '')
                        AND title IS NOT NULL AND title != ''
                        LIMIT 50
                    """).fetchall()
                    conn.close()

                    if papers_no_doi:
                        progress = st.progress(0)
                        found = 0
                        for i, p in enumerate(papers_no_doi):
                            result = lookup_doi(p['title'], p['authors'] or '', p['year'] or '')
                            if result.get("status") == "found" and result.get("doi"):
                                conn = sqlite3.connect(DB_PATH)
                                conn.execute("UPDATE papers SET doi=? WHERE id=?", (result["doi"], p['id']))
                                conn.commit()
                                conn.close()
                                # Also fetch full metadata
                                update_paper_from_doi(p['id'], result["doi"], overwrite=False)
                                found += 1
                            progress.progress((i + 1) / len(papers_no_doi))
                        st.success(f"Found DOIs for {found} of {len(papers_no_doi)} papers!")
                        st.rerun()
                    else:
                        st.info("All papers have DOIs")

            if st.session_state.get("quality_issues"):
                issues = st.session_state.quality_issues
                bad_authors = st.session_state.get("bad_authors", [])

                # Summary metrics
                truncated_count = len(issues.get("truncated_titles", []))
                total_issues = sum(len(v) for v in issues.values()) + len(bad_authors)
                qm1, qm2, qm3, qm4, qm5, qm6 = st.columns(6)
                with qm1:
                    st.metric("Incomplete", len(issues["incomplete_papers"]))
                with qm2:
                    st.metric("Short Titles", len(issues["short_titles"]))
                with qm3:
                    st.metric("Truncated", truncated_count)
                with qm4:
                    st.metric("Bad Authors", len(issues["bad_author_names"]))
                with qm5:
                    st.metric("Bad Venues", len(issues["bad_venues"]))
                with qm6:
                    st.metric("No DOI (2020+)", len(issues["no_doi_recent"]))

                if total_issues == 0:
                    st.success("✅ No quality issues found! Your database looks clean.")
                else:
                    st.warning(f"Found **{total_issues}** potential issues to review")

                st.divider()

                # Incomplete Papers
                if issues["incomplete_papers"]:
                    with st.expander(f"🚫 Incomplete Papers ({len(issues['incomplete_papers'])})", expanded=True):
                        st.caption("Papers missing title, authors, or year")
                        for p in issues["incomplete_papers"][:20]:
                            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                            with col1:
                                title = p.get('title', '[No title]') or '[No title]'
                                st.markdown(f"**{title[:70]}**")
                                st.caption(f"⚠️ {p['issue']} | DOI: {p.get('doi', 'None') or 'None'}")
                            with col2:
                                if p.get('doi'):
                                    if st.button("🔧 Fix", key=f"fix_inc_{p['id']}", help="Fix with DOI data"):
                                        result = update_paper_from_doi(p['id'], p['doi'], overwrite=True)
                                        if result.get("status") == "updated":
                                            st.success("Fixed!")
                                        st.rerun()
                                else:
                                    if st.button("🔎 DOI", key=f"lookup_inc_{p['id']}", help="Lookup DOI"):
                                        result = lookup_doi(p.get('title',''), p.get('authors',''), p.get('year',''))
                                        if result.get("doi"):
                                            conn = sqlite3.connect(DB_PATH)
                                            conn.execute("UPDATE papers SET doi=? WHERE id=?", (result["doi"], p['id']))
                                            conn.commit()
                                            conn.close()
                                            update_paper_from_doi(p['id'], result["doi"], overwrite=True)
                                            st.success(f"Found: {result['doi']}")
                                        else:
                                            st.warning("Not found")
                                        st.rerun()
                            with col3:
                                pass  # spacer
                            with col4:
                                if st.button("🗑️", key=f"del_inc_{p['id']}", help="Delete"):
                                    delete_paper(p['id'])
                                    st.rerun()

                # Short Titles
                if issues["short_titles"]:
                    with st.expander(f"📝 Short Titles ({len(issues['short_titles'])})"):
                        st.caption("Titles under 15 characters - may be errors")
                        for p in issues["short_titles"][:20]:
                            col1, col2, col3 = st.columns([4, 1, 1])
                            with col1:
                                st.markdown(f"**{p.get('title', '')}**")
                                st.caption(f"⚠️ {p['issue']} | DOI: {p.get('doi', 'None') or 'None'}")
                            with col2:
                                if p.get('doi'):
                                    if st.button("🔧 Fix", key=f"fix_short_{p['id']}", help="Fix with DOI"):
                                        update_paper_from_doi(p['id'], p['doi'], overwrite=True)
                                        st.rerun()
                            with col3:
                                if st.button("🗑️", key=f"del_short_{p['id']}", help="Delete"):
                                    delete_paper(p['id'])
                                    st.rerun()

                # Truncated Titles
                if issues.get("truncated_titles"):
                    with st.expander(f"✂️ Truncated Titles ({len(issues['truncated_titles'])})", expanded=True):
                        st.caption("Titles that appear to be cut off - fix with DOI lookup")

                        # Bulk fix button
                        if st.button("🔧 Fix All with DOI Lookup", key="fix_all_truncated"):
                            fixed = 0
                            progress = st.progress(0)
                            status = st.empty()
                            for i, p in enumerate(issues["truncated_titles"]):
                                progress.progress((i + 1) / len(issues["truncated_titles"]))
                                status.text(f"Processing {i+1}/{len(issues['truncated_titles'])}: {p.get('title', '')[:40]}...")

                                # Try to fix with existing DOI
                                if p.get('doi'):
                                    result = update_paper_from_doi(p['id'], p['doi'], overwrite=True)
                                    if result.get("status") == "updated":
                                        fixed += 1
                                        continue

                                # Try to lookup DOI
                                result = lookup_doi(p.get('title',''), p.get('authors',''), p.get('year',''))
                                if result.get("doi"):
                                    conn = sqlite3.connect(DB_PATH)
                                    conn.execute("UPDATE papers SET doi=? WHERE id=?", (result["doi"], p['id']))
                                    conn.commit()
                                    conn.close()
                                    update_result = update_paper_from_doi(p['id'], result["doi"], overwrite=True)
                                    if update_result.get("status") == "updated":
                                        fixed += 1

                            status.empty()
                            progress.empty()
                            st.success(f"Fixed {fixed} of {len(issues['truncated_titles'])} truncated titles")
                            st.rerun()

                        st.divider()
                        for p in issues["truncated_titles"][:30]:
                            col1, col2, col3 = st.columns([4, 1, 1])
                            with col1:
                                st.markdown(f"**{p.get('title', '')}**")
                                st.caption(f"⚠️ {p['issue']} | DOI: {p.get('doi', 'None') or 'None'}")
                            with col2:
                                if p.get('doi'):
                                    if st.button("🔧 Fix", key=f"fix_trunc_{p['id']}", help="Fix with DOI"):
                                        update_paper_from_doi(p['id'], p['doi'], overwrite=True)
                                        st.rerun()
                                else:
                                    if st.button("🔎 DOI", key=f"lookup_trunc_{p['id']}", help="Lookup DOI"):
                                        result = lookup_doi(p.get('title',''), p.get('authors',''), p.get('year',''))
                                        if result.get("doi"):
                                            conn = sqlite3.connect(DB_PATH)
                                            conn.execute("UPDATE papers SET doi=? WHERE id=?", (result["doi"], p['id']))
                                            conn.commit()
                                            conn.close()
                                            update_paper_from_doi(p['id'], result["doi"], overwrite=True)
                                            st.success(f"Found & fixed: {result['doi']}")
                                        else:
                                            st.warning("DOI not found")
                                        st.rerun()
                            with col3:
                                if st.button("🗑️", key=f"del_trunc_{p['id']}", help="Delete"):
                                    delete_paper(p['id'])
                                    st.rerun()

                # Bad Author Names in Papers
                if issues["bad_author_names"]:
                    with st.expander(f"👤 Bad Author Names ({len(issues['bad_author_names'])})"):
                        st.caption("Papers with malformed author fields")
                        for p in issues["bad_author_names"][:20]:
                            col1, col2, col3 = st.columns([4, 1, 1])
                            with col1:
                                st.markdown(f"**{p.get('title', '')[:60]}**")
                                st.caption(f"⚠️ {p['issue']}")
                                st.caption(f"Authors: `{p.get('authors', '')}`")
                            with col2:
                                if p.get('doi'):
                                    if st.button("🔧 Fix", key=f"fix_auth_{p['id']}", help="Fix with DOI"):
                                        update_paper_from_doi(p['id'], p['doi'], overwrite=True)
                                        st.rerun()
                                else:
                                    if st.button("🔎 DOI", key=f"lookup_auth_{p['id']}", help="Lookup DOI"):
                                        result = lookup_doi(p.get('title',''), p.get('authors',''), p.get('year',''))
                                        if result.get("doi"):
                                            conn = sqlite3.connect(DB_PATH)
                                            conn.execute("UPDATE papers SET doi=? WHERE id=?", (result["doi"], p['id']))
                                            conn.commit()
                                            conn.close()
                                            update_paper_from_doi(p['id'], result["doi"], overwrite=True)
                                            st.success(f"Found & fixed!")
                                        else:
                                            st.warning("DOI not found")
                                        st.rerun()
                            with col3:
                                if st.button("🗑️", key=f"del_auth_{p['id']}", help="Delete"):
                                    delete_paper(p['id'])
                                    st.rerun()

                # Bad Venues
                if issues["bad_venues"]:
                    with st.expander(f"📰 Bad Venues ({len(issues['bad_venues'])})"):
                        st.caption("Papers with missing or very short venue names")
                        for p in issues["bad_venues"][:20]:
                            col1, col2, col3 = st.columns([4, 1, 1])
                            with col1:
                                st.markdown(f"**{p.get('title', '')[:60]}**")
                                st.caption(f"⚠️ {p['issue']} | DOI: {p.get('doi', 'None') or 'None'}")
                            with col2:
                                if p.get('doi'):
                                    if st.button("🔧 Fix", key=f"fix_venue_{p['id']}", help="Fix with DOI"):
                                        update_paper_from_doi(p['id'], p['doi'], overwrite=True)
                                        st.rerun()
                                else:
                                    if st.button("🔎 DOI", key=f"lookup_venue_{p['id']}", help="Lookup DOI"):
                                        result = lookup_doi(p.get('title',''), p.get('authors',''), p.get('year',''))
                                        if result.get("doi"):
                                            conn = sqlite3.connect(DB_PATH)
                                            conn.execute("UPDATE papers SET doi=? WHERE id=?", (result["doi"], p['id']))
                                            conn.commit()
                                            conn.close()
                                            update_paper_from_doi(p['id'], result["doi"], overwrite=True)
                                            st.success(f"Found & fixed!")
                                        else:
                                            st.warning("DOI not found")
                                        st.rerun()
                            with col3:
                                if st.button("🗑️", key=f"del_venue_{p['id']}", help="Delete"):
                                    delete_paper(p['id'])
                                    st.rerun()

                # Recent papers without DOI
                if issues["no_doi_recent"]:
                    with st.expander(f"🔗 Recent Papers Without DOI ({len(issues['no_doi_recent'])})"):
                        st.caption("Papers from 2020+ without DOI - click to lookup")

                        # Batch lookup button
                        if st.button("🔎 Lookup All DOIs", key="lookup_all_recent"):
                            progress = st.progress(0)
                            found = 0
                            for i, p in enumerate(issues["no_doi_recent"]):
                                result = lookup_doi(p.get('title',''), p.get('authors',''), p.get('year',''))
                                if result.get("doi"):
                                    conn = sqlite3.connect(DB_PATH)
                                    conn.execute("UPDATE papers SET doi=? WHERE id=?", (result["doi"], p['id']))
                                    conn.commit()
                                    conn.close()
                                    update_paper_from_doi(p['id'], result["doi"], overwrite=False)
                                    found += 1
                                progress.progress((i + 1) / len(issues["no_doi_recent"]))
                            st.success(f"Found {found} DOIs!")
                            st.rerun()

                        for p in issues["no_doi_recent"][:20]:
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"**{p.get('title', '')[:70]}** ({p.get('year', '')})")
                            with col2:
                                if st.button("🔎", key=f"lookup_nodoi_{p['id']}", help="Lookup DOI"):
                                    result = lookup_doi(p.get('title',''), p.get('authors',''), p.get('year',''))
                                    if result.get("doi"):
                                        conn = sqlite3.connect(DB_PATH)
                                        conn.execute("UPDATE papers SET doi=? WHERE id=?", (result["doi"], p['id']))
                                        conn.commit()
                                        conn.close()
                                        update_paper_from_doi(p['id'], result["doi"], overwrite=False)
                                        st.success(f"Found: {result['doi']}")
                                    else:
                                        st.warning("Not found")
                                    st.rerun()

                # Bad Authors in Author Table
                if bad_authors:
                    with st.expander(f"👥 Malformed Author Names ({len(bad_authors)})"):
                        st.caption("Author entries that look incomplete or malformed")
                        for a in bad_authors[:30]:
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"**{a['name']}** ({a['paper_count']} papers)")
                                st.caption(f"⚠️ Issues: {', '.join(a['issues'])}")
                            with col2:
                                if st.button("🗑️", key=f"del_author_{a['id']}", help="Delete this author"):
                                    conn = sqlite3.connect(DB_PATH)
                                    conn.execute("DELETE FROM authors WHERE id=?", (a['id'],))
                                    conn.execute("DELETE FROM paper_authors WHERE author_id=?", (a['id'],))
                                    conn.commit()
                                    conn.close()
                                    st.rerun()

        # ===== DUPLICATES TAB =====
        with dupes_tab:
            st.subheader("🔍 Find & Remove Duplicates")
            st.caption("Detect and manage duplicate papers in your database")

            # Helper function to calculate completeness score
            def calc_completeness(paper):
                """Calculate completeness score for a paper (0-100)."""
                score = 0
                weights = {
                    'title': 15, 'authors': 15, 'year': 10, 'venue': 10,
                    'doi': 20, 'abstract': 10, 'volume': 5, 'pages': 5,
                    'citations': 5, 'keywords': 5
                }
                for field, weight in weights.items():
                    val = paper.get(field)
                    if val and str(val).strip() and str(val) != '0':
                        score += weight
                return score

            # Settings
            d1, d2, d3 = st.columns([1, 1, 2])
            with d1:
                match_mode = st.selectbox(
                    "Matching Mode",
                    ["Normal (85%)", "Aggressive (70%)", "Strict (95%)", "Custom"],
                    help="How strictly to match titles"
                )
            with d2:
                if match_mode == "Custom":
                    custom_threshold = st.slider("Custom Threshold", 50, 100, 85)
                else:
                    custom_threshold = {"Normal (85%)": 85, "Aggressive (70%)": 70, "Strict (95%)": 95}.get(match_mode, 85)
                    st.metric("Threshold", f"{custom_threshold}%")

            with d3:
                aggressive_mode = st.checkbox(
                    "Include partial matches",
                    value=match_mode == "Aggressive (70%)",
                    help="Also consider partial title matches and ignore year differences"
                )

            # Find duplicates button
            if st.button("🔍 Find Duplicates", type="primary", use_container_width=True):
                with st.spinner("Searching for duplicates..."):
                    # Get full paper data for completeness check
                    conn = sqlite3.connect(DB_PATH)
                    conn.row_factory = sqlite3.Row
                    all_papers = {p['id']: dict(p) for p in conn.execute("SELECT * FROM papers").fetchall()}
                    conn.close()

                    groups = find_duplicate_papers(threshold=custom_threshold, aggressive=aggressive_mode)

                    # Enrich groups with full paper data
                    enriched_groups = []
                    for group in groups:
                        enriched = []
                        for p in group:
                            full_paper = all_papers.get(p['id'], p)
                            full_paper['completeness'] = calc_completeness(full_paper)
                            enriched.append(full_paper)
                        # Sort by completeness (best first)
                        enriched.sort(key=lambda x: x['completeness'], reverse=True)
                        enriched_groups.append(enriched)

                    st.session_state.duplicate_groups = enriched_groups

            # Display results
            if "duplicate_groups" in st.session_state and st.session_state.duplicate_groups:
                groups = st.session_state.duplicate_groups

                st.success(f"Found **{len(groups)} groups** of potential duplicates ({sum(len(g) for g in groups)} papers total)")

                # Quick actions
                qc1, qc2, qc3 = st.columns([1, 1, 2])
                with qc1:
                    if st.button("🗑️ Auto-Remove All Duplicates", type="secondary"):
                        removed = remove_duplicate_papers(custom_threshold, aggressive_mode)
                        st.success(f"Removed {removed} duplicate papers!")
                        del st.session_state.duplicate_groups
                        st.rerun()
                with qc2:
                    if st.button("🗑️ Delete All Suggested", type="secondary", help="Delete all papers marked for deletion"):
                        deleted = 0
                        for group in groups:
                            # Keep best (first), delete rest
                            for paper in group[1:]:
                                delete_paper(paper['id'])
                                deleted += 1
                        st.success(f"Deleted {deleted} papers!")
                        del st.session_state.duplicate_groups
                        st.rerun()
                with qc3:
                    st.info(f"Will keep {len(groups)} best papers, delete {sum(len(g)-1 for g in groups)} duplicates")

                st.divider()

                # Show each group
                for gi, group in enumerate(groups):
                    best_paper = group[0]
                    with st.expander(f"**Group {gi+1}**: {len(group)} papers — {best_paper['title']}", expanded=gi < 5):

                        # Show papers in group (sorted by completeness, best first)
                        for pi, paper in enumerate(group):
                            is_best = pi == 0
                            completeness = paper.get('completeness', calc_completeness(paper))

                            # Status indicators
                            has_doi = "✅" if paper.get('doi') else "❌"
                            has_abstract = "✅" if paper.get('abstract') else "❌"
                            has_venue = "✅" if paper.get('venue') else "❌"
                            citations = paper.get('citations', 0) or 0

                            if is_best:
                                st.markdown(f"### ✅ KEEP: {paper['title']}")
                                st.caption(f"**Completeness: {completeness}%** | DOI: {has_doi} | Abstract: {has_abstract} | Venue: {has_venue} | Citations: {citations}")
                                if paper.get('authors'):
                                    st.caption(f"Authors: {paper['authors'][:100]}")
                                if paper.get('doi'):
                                    st.caption(f"DOI: `{paper['doi']}`")
                            else:
                                col1, col2, col3 = st.columns([3, 1, 1])
                                with col1:
                                    st.markdown(f"### ❌ DELETE: {paper['title']}")
                                    st.caption(f"Completeness: {completeness}% | DOI: {has_doi} | Abstract: {has_abstract} | Venue: {has_venue}")
                                    if paper.get('authors'):
                                        st.caption(f"Authors: {paper['authors'][:80]}")

                                with col2:
                                    # Option to keep this one instead
                                    if st.button("✅ Keep This", key=f"keep_{gi}_{paper['id']}", help="Keep this and delete others"):
                                        # Delete all others in group
                                        for other in group:
                                            if other['id'] != paper['id']:
                                                delete_paper(other['id'])
                                        # Remove this group from session state (it's resolved)
                                        st.session_state.duplicate_groups = [
                                            g for i, g in enumerate(st.session_state.duplicate_groups) if i != gi
                                        ]
                                        st.toast(f"✅ Kept 1 paper, deleted {len(group)-1} duplicates")
                                        st.rerun()

                                with col3:
                                    if st.button("🗑️ Delete", key=f"del_{gi}_{paper['id']}", help="Delete this paper"):
                                        delete_paper(paper['id'])
                                        # Remove paper from this group in session state
                                        new_groups = []
                                        for i, g in enumerate(st.session_state.duplicate_groups):
                                            if i == gi:
                                                # Remove deleted paper from this group
                                                new_group = [p for p in g if p['id'] != paper['id']]
                                                # Only keep group if still has duplicates
                                                if len(new_group) > 1:
                                                    new_groups.append(new_group)
                                            else:
                                                new_groups.append(g)
                                        st.session_state.duplicate_groups = new_groups
                                        st.toast(f"🗑️ Deleted paper")
                                        st.rerun()

                            st.divider()

            elif "duplicate_groups" in st.session_state:
                st.success("✅ No duplicate papers found!")

        # ===== EXPORT TAB =====
        with export_tab:
            st.subheader("📤 Export BibTeX")
            st.caption("Export your papers to BibTeX format")

            e1, e2 = st.columns(2)
            with e1:
                export_format = st.radio(
                    "BibTeX Format",
                    ["Website (Jekyll/al-folio)", "Standard BibTeX"],
                    help="Website format includes abbr, bibtex_show, selected fields"
                )
                format_type = "website" if "Website" in export_format else "standard"

            with e2:
                export_author_filter = st.text_input(
                    "Filter by Author (optional)",
                    placeholder="e.g., Saqr",
                    help="Only export papers by this author"
                )

            # Preview stats
            conn = sqlite3.connect(DB_PATH)
            if export_author_filter:
                count = conn.execute("SELECT COUNT(*) FROM papers WHERE authors LIKE ?",
                                   (f"%{export_author_filter}%",)).fetchone()[0]
            else:
                count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            conn.close()

            st.info(f"📊 Will export **{count}** papers")

            if st.button("📤 Generate BibTeX", type="primary", use_container_width=True):
                with st.spinner("Generating BibTeX..."):
                    bibtex_content = export_all_bibtex(format_type, export_author_filter)
                    st.session_state.export_bibtex = bibtex_content

            if "export_bibtex" in st.session_state:
                st.text_area("BibTeX Output", st.session_state.export_bibtex, height=400)

                # Download button
                st.download_button(
                    "⬇️ Download papers.bib",
                    st.session_state.export_bibtex,
                    file_name="papers.bib",
                    mime="text/plain",
                    type="primary"
                )

        # ===== IMPORT TAB =====
        with import_tab:
            st.subheader("Import Papers to Database")
            import_method = st.radio("Import from", ["Paste BibTeX", "Upload CSV", "Paste Text List"], horizontal=True)

            if import_method == "Paste BibTeX":
                bibtex_import = st.text_area("Paste BibTeX entries", height=300,
                    placeholder="@article{key,\n  author = {...},\n  title = {...},\n  ...\n}")
                if st.button("📥 Import BibTeX to Database", disabled=not bibtex_import):
                    with st.spinner("Importing..."):
                        papers = parse_bibtex_input(bibtex_import)
                        imported = 0
                        for p in papers:
                            key = generate_citation_key(p.get("authors",""), p.get("year",""))
                            p["citation_key"] = key
                            p["bibtex"] = format_bibtex_entry(p, key, "Article", p.get("doi",""))
                            if save_to_db(p, "bibtex_import"): imported += 1
                        st.success(f"Imported {imported} of {len(papers)} papers!")
                        st.rerun()

            elif import_method == "Upload CSV":
                st.markdown("""
                **Supported CSV columns:** Authors, Title, Publication/Venue, Volume, Number, Pages, Year, Publisher, Citations, DOI
                """)

                csv_file = st.file_uploader("Upload CSV file", type=['csv'])

                update_existing = st.checkbox("Update existing papers with CSV data", value=True,
                    help="If a paper with the same title exists, update it with data from CSV")

                if csv_file:
                    try:
                        # Read CSV content
                        csv_content = csv_file.read().decode('utf-8-sig')
                        import csv as csv_module
                        import io
                        reader = csv_module.DictReader(io.StringIO(csv_content))
                        rows = list(reader)

                        st.info(f"Found **{len(rows)}** rows with columns: {', '.join(reader.fieldnames or [])}")

                        # Preview first 5 rows - show FULL titles to verify CSV has complete data
                        st.markdown("**Preview (full titles):**")
                        for i, row in enumerate(rows[:5], 1):
                            title = row.get('Title', row.get('title', '')) or ''
                            authors = row.get('Authors', row.get('authors', '')) or ''
                            year = row.get('Year', row.get('year', ''))
                            st.markdown(f"**{i}.** {title}")
                            st.caption(f"   {authors} ({year}) - {len(title)} chars")

                        if len(rows) > 5:
                            st.caption(f"... and {len(rows) - 5} more rows")

                        ic1, ic2 = st.columns(2)
                        with ic1:
                            if st.button("📥 Import CSV to Database", type="primary", use_container_width=True):
                                csv_file.seek(0)
                                csv_content = csv_file.read().decode('utf-8-sig')
                                result = import_from_csv(csv_content, update_existing=update_existing)
                                st.success(f"✅ Imported: {result['imported']} | Updated: {result['updated']} | Skipped: {result['skipped']}")
                                st.rerun()

                        with ic2:
                            if st.button("📊 Update Citations Only", use_container_width=True,
                                help="Only update citation counts for existing papers"):
                                updated = 0
                                conn = sqlite3.connect(DB_PATH)
                                for row in rows:
                                    title = row.get('Title', row.get('title', '')).strip()
                                    citations = row.get('Citations', row.get('citations', row.get('Cited by', '0')))
                                    try:
                                        citations = int(citations) if citations else 0
                                    except (ValueError, TypeError):
                                        citations = 0
                                    if title and citations > 0:
                                        cursor = conn.execute("UPDATE papers SET citations=? WHERE title=?", (citations, title))
                                        if cursor.rowcount > 0:
                                            updated += 1
                                conn.commit()
                                conn.close()
                                st.success(f"Updated citations for {updated} papers!")
                                st.rerun()

                        # Fix truncated titles button
                        st.divider()
                        st.markdown("**🔧 Fix Truncated Titles**")
                        st.caption("Match truncated titles in database with full titles from CSV")

                        if st.button("✂️ Fix Truncated Titles from CSV", use_container_width=True):
                            fixed = 0
                            conn = sqlite3.connect(DB_PATH)
                            db_papers = conn.execute("SELECT id, title FROM papers WHERE (trashed=0 OR trashed IS NULL)").fetchall()

                            for db_id, db_title in db_papers:
                                if not db_title:
                                    continue
                                db_title_lower = db_title.lower().strip()

                                # Check if title looks truncated (ends mid-word or suspicious length)
                                is_truncated = (
                                    (db_title[-1].islower() and db_title[-1] not in '.?!') or
                                    len(db_title) in [50, 64, 80, 100, 128, 255, 256]
                                )

                                if is_truncated:
                                    # Find matching full title in CSV
                                    for row in rows:
                                        csv_title = row.get('Title', row.get('title', '')).strip()
                                        if not csv_title or len(csv_title) <= len(db_title):
                                            continue

                                        # Check if CSV title starts with DB title (truncated match)
                                        if csv_title.lower().startswith(db_title_lower[:min(40, len(db_title_lower))]):
                                            # Update with full title
                                            conn.execute("UPDATE papers SET title=? WHERE id=?", (csv_title, db_id))
                                            fixed += 1
                                            break

                            conn.commit()
                            conn.close()
                            if fixed > 0:
                                st.success(f"Fixed {fixed} truncated titles!")
                                st.rerun()
                            else:
                                st.info("No truncated titles found that match CSV data")

                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")

            else:  # Text list
                st.markdown("**Format:** `Title | Authors | Year | Venue`")
                text_import = st.text_area("Paste papers (one per line)", height=300,
                    placeholder="Paper Title | Smith J, Doe A | 2024 | Nature")
                if st.button("📥 Import List to Database", disabled=not text_import):
                    with st.spinner("Importing..."):
                        papers = parse_manual_publications(text_import)
                        imported = 0
                        for p in papers:
                            key = generate_citation_key(p.get("authors",""), p.get("year",""))
                            p["citation_key"] = key
                            p["bibtex"] = format_bibtex_entry(p, key, "Article", "")
                            if save_to_db(p, "text_import"): imported += 1
                        st.success(f"Imported {imported} of {len(papers)} papers!")
                        st.rerun()

        # ===== TRASH TAB =====
        with trash_tab:
            st.subheader("🗑️ Trash Bin")
            st.caption("Papers moved to trash can be restored or permanently deleted")

            trashed_papers = get_trashed_papers()

            if trashed_papers:
                st.warning(f"**{len(trashed_papers)}** papers in trash")

                # Bulk actions
                tc1, tc2, tc3 = st.columns([1, 1, 2])
                with tc1:
                    if st.button("♻️ Restore All", type="secondary", use_container_width=True):
                        count = restore_all_trash()
                        st.success(f"Restored {count} papers!")
                        st.rerun()
                with tc2:
                    if st.button("🗑️ Empty Trash", type="secondary", use_container_width=True):
                        count = empty_trash()
                        st.success(f"Permanently deleted {count} papers!")
                        st.rerun()

                st.divider()

                # List trashed papers
                for paper in trashed_papers:
                    col1, col2, col3 = st.columns([4, 1, 1])

                    with col1:
                        st.markdown(f"**{paper.get('title', '[No title]')[:80]}**")
                        trashed_date = paper.get('trashed_date', '')
                        if trashed_date:
                            try:
                                dt = datetime.fromisoformat(trashed_date)
                                trashed_str = dt.strftime("%Y-%m-%d %H:%M")
                            except ValueError:
                                trashed_str = trashed_date[:16]
                        else:
                            trashed_str = "Unknown"
                        st.caption(f"Trashed: {trashed_str} | Year: {paper.get('year', 'N/A')} | Authors: {(paper.get('authors') or '')[:50]}")

                    with col2:
                        if st.button("♻️ Restore", key=f"restore_{paper['id']}", help="Restore this paper"):
                            restore_paper(paper['id'])
                            st.toast("Paper restored!")
                            st.rerun()

                    with col3:
                        if st.button("❌ Delete", key=f"perm_del_{paper['id']}", help="Permanently delete"):
                            permanent_delete_paper(paper['id'])
                            st.toast("Permanently deleted!")
                            st.rerun()

                    st.divider()
            else:
                st.success("🎉 Trash is empty!")

            # ===== DANGER ZONE =====
            st.divider()
            st.subheader("⚠️ Danger Zone")
            st.caption("Database cleanup and reset options - use with caution!")

            with st.expander("🔴 Database Cleanup Options", expanded=False):
                st.warning("**These actions cannot be undone!** Make sure to export your data first.")

                # Initialize confirmation states
                if "confirm_purge_papers" not in st.session_state:
                    st.session_state.confirm_purge_papers = False
                if "confirm_purge_all" not in st.session_state:
                    st.session_state.confirm_purge_all = False
                if "confirm_purge_authors" not in st.session_state:
                    st.session_state.confirm_purge_authors = False

                st.markdown("---")

                # Option 1: Purge papers only (keep authors)
                st.markdown("### 📄 Purge Papers Only")
                st.caption("Delete all papers but keep the authors database for disambiguation")
                if not st.session_state.confirm_purge_papers:
                    if st.button("🗑️ Purge All Papers", key="purge_papers_btn", type="secondary"):
                        st.session_state.confirm_purge_papers = True
                        st.rerun()
                else:
                    st.error("⚠️ Are you sure? This will delete ALL papers permanently!")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Yes, Delete All Papers", key="confirm_purge_papers_btn", type="primary"):
                            conn = sqlite3.connect(DB_PATH)
                            conn.execute("DELETE FROM papers")
                            conn.execute("DELETE FROM paper_authors")
                            conn.commit()
                            conn.close()
                            st.session_state.confirm_purge_papers = False
                            st.success("All papers deleted! Authors database preserved.")
                            st.rerun()
                    with col2:
                        if st.button("❌ Cancel", key="cancel_purge_papers_btn"):
                            st.session_state.confirm_purge_papers = False
                            st.rerun()

                st.markdown("---")

                # Option 2: Purge authors only
                st.markdown("### 👥 Purge Authors Only")
                st.caption("Delete authors database but keep papers")
                if not st.session_state.confirm_purge_authors:
                    if st.button("🗑️ Purge Authors Database", key="purge_authors_btn", type="secondary"):
                        st.session_state.confirm_purge_authors = True
                        st.rerun()
                else:
                    st.error("⚠️ Are you sure? This will delete all author disambiguation data!")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Yes, Delete Authors", key="confirm_purge_authors_btn", type="primary"):
                            conn = sqlite3.connect(DB_PATH)
                            conn.execute("DELETE FROM authors")
                            conn.execute("DELETE FROM paper_authors")
                            conn.commit()
                            conn.close()
                            st.session_state.confirm_purge_authors = False
                            st.success("Authors database cleared! Papers preserved.")
                            st.rerun()
                    with col2:
                        if st.button("❌ Cancel", key="cancel_purge_authors_btn"):
                            st.session_state.confirm_purge_authors = False
                            st.rerun()

                st.markdown("---")

                # Option 3: Purge everything (factory reset)
                st.markdown("### 💣 Factory Reset")
                st.caption("Delete ALL data - papers, authors, everything")
                if not st.session_state.confirm_purge_all:
                    if st.button("🔴 FACTORY RESET", key="purge_all_btn", type="secondary"):
                        st.session_state.confirm_purge_all = True
                        st.rerun()
                else:
                    st.error("🚨 **FINAL WARNING!** This will permanently delete your ENTIRE database!")
                    confirm_text = st.text_input("Type 'DELETE ALL' to confirm:", key="confirm_delete_text")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💣 CONFIRM FACTORY RESET", key="confirm_purge_all_btn", type="primary", disabled=(confirm_text != "DELETE ALL")):
                            conn = sqlite3.connect(DB_PATH)
                            conn.execute("DELETE FROM papers")
                            conn.execute("DELETE FROM authors")
                            conn.execute("DELETE FROM paper_authors")
                            conn.commit()
                            conn.close()
                            st.session_state.confirm_purge_all = False
                            st.success("Database reset complete. All data deleted.")
                            st.rerun()
                    with col2:
                        if st.button("❌ Cancel", key="cancel_purge_all_btn"):
                            st.session_state.confirm_purge_all = False
                            st.rerun()

    # ============ SYNC PAPERS TAB ============
    # This tab syncs papers against the GitHub website bib file to find missing papers
    with tab_sync:
        st.caption("Find papers missing from your GitHub website bibliography")

        if st.session_state.step == 1:
            input_method = st.session_state.get("input_method", "Upload CSV")
            bib_url = st.session_state.get("bib_url", DEFAULT_BIB_URL)
            lm_studio_url = st.session_state.get("lm_studio_url", DEFAULT_LM_STUDIO_URL)

            # Show current input method
            st.info(f"📥 Input method: **{input_method}** (change in sidebar)")

            if input_method == "Google Scholar URL":
                st.info("Enter a Google Scholar profile URL and paste the page HTML")

                scholar_url = st.text_input(
                    "Google Scholar Profile URL",
                    value="https://scholar.google.com/citations?hl=en&user=U-O6R7YAAAAJ&view_op=list_works&sortby=pubdate",
                    help="Your Google Scholar profile URL"
                )

                st.warning("⚠️ Auto-fetch may be blocked by Google. Use the manual paste method below.")

                # Quick auto-fetch option
                user_id = ""
                if scholar_url and "user=" in scholar_url:
                    match = re.search(r'user=([^&]+)', scholar_url)
                    if match:
                        user_id = match.group(1)

                auto_col1, auto_col2, auto_col3 = st.columns([2, 1, 2])
                with auto_col1:
                    num_papers = st.number_input("Papers to fetch", 10, 100, 50, 10, key="autofetch_num")
                with auto_col2:
                    if st.button("🚀 Try Auto-Fetch", disabled=not (user_id and bib_url), help="May be blocked by Google", type="primary"):
                        with st.spinner("Fetching Google Scholar profile..."):
                            try:
                                html_content = fetch_scholar_html(user_id, num_papers)
                                st.success("Fetched Scholar page successfully!")
                            except Exception as e:
                                st.error(f"Blocked or failed: {e}")
                                html_content = None

                        if html_content:
                            with st.spinner("AI is extracting publications..."):
                                st.session_state.scholar_pubs = extract_with_llm(html_content, lm_studio_url)

                            with st.spinner("Fetching existing BibTeX entries..."):
                                st.session_state.existing_entries = fetch_existing_bib(bib_url)

                            if st.session_state.scholar_pubs:
                                st.success(f"Extracted {len(st.session_state.scholar_pubs)} publications")
                                if st.session_state.existing_entries:
                                    with st.spinner("Finding missing papers..."):
                                        st.session_state.missing_papers = find_missing_papers(
                                            st.session_state.scholar_pubs,
                                            st.session_state.existing_entries,
                                            lm_studio_url,
                                            st.session_state.get("use_ai_disambiguation", True),
                                            st.session_state.get("match_threshold", DEFAULT_MATCH_THRESHOLD),
                                            st.session_state.get("uncertain_threshold", DEFAULT_UNCERTAIN_THRESHOLD)
                                        )
                                    st.session_state.step = 2
                                    st.rerun()

                st.divider()

                with st.expander("📋 Manual: Copy your Scholar HTML", expanded=False):
                    st.markdown(f"""
                    1. Open your [Google Scholar profile]({scholar_url}) in a browser
                    2. Scroll down to load all papers you want
                    3. Right-click → **View Page Source** (or Ctrl+U / Cmd+Option+U)
                    4. Select all (Ctrl+A / Cmd+A) and copy
                    5. Paste below
                    """)

                html_input = st.text_area(
                    "Paste Google Scholar HTML here",
                    height=200,
                    placeholder="<!DOCTYPE html>...",
                    help="Paste the full page source HTML"
                )

                if st.button("📋 Extract from Pasted HTML", disabled=not (html_input and bib_url)):
                    with st.spinner("AI is extracting publications (this may take a minute)..."):
                        st.session_state.scholar_pubs = extract_with_llm(html_input, lm_studio_url)

                    with st.spinner("Fetching existing BibTeX entries..."):
                        st.session_state.existing_entries = fetch_existing_bib(bib_url)

                    if st.session_state.scholar_pubs:
                        st.success(f"Extracted {len(st.session_state.scholar_pubs)} publications")
                        if st.session_state.existing_entries:
                            with st.spinner("Finding missing papers..."):
                                st.session_state.missing_papers = find_missing_papers(
                                    st.session_state.scholar_pubs,
                                    st.session_state.existing_entries,
                                    lm_studio_url,
                                    st.session_state.get("use_ai_disambiguation", True),
                                    st.session_state.get("match_threshold", DEFAULT_MATCH_THRESHOLD),
                                    st.session_state.get("uncertain_threshold", DEFAULT_UNCERTAIN_THRESHOLD)
                                )
                            st.session_state.step = 2
                            st.rerun()

            elif input_method == "LM Studio (AI extraction)":
                st.info("Paste your Google Scholar profile HTML and let AI extract publications")

                with st.expander("How to get your Scholar HTML"):
                    st.markdown("""
                    1. Go to your Google Scholar profile
                    2. Right-click → "View Page Source" (or Ctrl+U / Cmd+Option+U)
                    3. Select all (Ctrl+A / Cmd+A) and copy
                    4. Paste below
                    """)

                html_input = st.text_area(
                    "Paste Google Scholar HTML here",
                    height=300,
                    placeholder="<!DOCTYPE html>..."
                )

                if st.button("Extract with AI", type="primary", disabled=not (html_input and bib_url)):
                    with st.spinner("AI is extracting publications (this may take a minute)..."):
                        st.session_state.scholar_pubs = extract_with_llm(html_input, lm_studio_url)

                    with st.spinner("Fetching existing BibTeX entries..."):
                        st.session_state.existing_entries = fetch_existing_bib(bib_url)

                    if st.session_state.scholar_pubs:
                        st.success(f"Extracted {len(st.session_state.scholar_pubs)} publications")
                        if st.session_state.existing_entries:
                            with st.spinner("Finding missing papers..."):
                                st.session_state.missing_papers = find_missing_papers(
                                    st.session_state.scholar_pubs,
                                    st.session_state.existing_entries,
                                    lm_studio_url,
                                    st.session_state.get("use_ai_disambiguation", True),
                                    st.session_state.get("match_threshold", DEFAULT_MATCH_THRESHOLD),
                                    st.session_state.get("uncertain_threshold", DEFAULT_UNCERTAIN_THRESHOLD)
                                )
                            st.session_state.step = 2
                            st.rerun()

            elif input_method == "Paste HTML + AI":
                st.info("Same as above - paste HTML for AI extraction")

                html_input = st.text_area(
                    "Paste any HTML containing publications",
                    height=300,
                    placeholder="<html>..."
                )

                if st.button("Extract with AI", type="primary", disabled=not (html_input and bib_url)):
                    with st.spinner("AI is extracting publications..."):
                        st.session_state.scholar_pubs = extract_with_llm(html_input, lm_studio_url)

                    with st.spinner("Fetching existing BibTeX entries..."):
                        st.session_state.existing_entries = fetch_existing_bib(bib_url)

                    if st.session_state.scholar_pubs and st.session_state.existing_entries:
                        with st.spinner("Finding missing papers..."):
                            st.session_state.missing_papers = find_missing_papers(
                                st.session_state.scholar_pubs,
                                st.session_state.existing_entries,
                                lm_studio_url,
                                st.session_state.get("use_ai_disambiguation", True),
                                st.session_state.get("match_threshold", DEFAULT_MATCH_THRESHOLD),
                                st.session_state.get("uncertain_threshold", DEFAULT_UNCERTAIN_THRESHOLD)
                            )
                        st.session_state.step = 2
                        st.rerun()

            elif input_method == "Upload CSV":
                st.info("Upload a CSV file with your publications")
                st.markdown("**Expected columns:** Title, Authors, Year, Publication/Venue, Citations (optional)")

                csv_file = st.file_uploader("Upload CSV", type=["csv"], key="sync_csv_upload")

                if csv_file is not None:
                    csv_content = csv_file.read().decode("utf-8")

                    # Preview CSV
                    import csv as csv_module
                    import io
                    reader = csv_module.DictReader(io.StringIO(csv_content))
                    rows = list(reader)
                    if rows:
                        st.caption(f"Found {len(rows)} rows in CSV")
                        # Show first 3 rows as preview
                        preview_df = pd.DataFrame(rows[:3])
                        st.dataframe(preview_df, use_container_width=True, height=150)

                    if st.button("Process CSV", type="primary", disabled=not bib_url):
                        with st.spinner("Parsing CSV..."):
                            # Parse CSV to publication list
                            publications = []
                            for row in rows:
                                title = row.get('Title', row.get('title', '')).strip()
                                if not title:
                                    continue
                                publications.append({
                                    "title": title,
                                    "authors": row.get('Authors', row.get('authors', row.get('Author', ''))).strip(),
                                    "year": row.get('Year', row.get('year', '')).strip(),
                                    "venue": row.get('Publication', row.get('Venue', row.get('Journal', row.get('venue', '')))).strip(),
                                    "citations": int(row.get('Citations', row.get('citations', row.get('Cited by', '0'))) or 0),
                                    "doi": row.get('DOI', row.get('doi', '')).strip(),
                                    "pub_type": "article",
                                })
                            st.session_state.scholar_pubs = publications

                        with st.spinner("Fetching existing BibTeX entries..."):
                            st.session_state.existing_entries = fetch_existing_bib(bib_url)

                        if st.session_state.scholar_pubs:
                            st.success(f"Parsed {len(st.session_state.scholar_pubs)} publications from CSV")
                            if st.session_state.existing_entries:
                                with st.spinner("Finding missing papers..."):
                                    st.session_state.missing_papers = find_missing_papers(
                                        st.session_state.scholar_pubs,
                                        st.session_state.existing_entries,
                                        lm_studio_url,
                                        st.session_state.get("use_ai_disambiguation", True),
                                        st.session_state.get("match_threshold", DEFAULT_MATCH_THRESHOLD),
                                        st.session_state.get("uncertain_threshold", DEFAULT_UNCERTAIN_THRESHOLD)
                                    )
                                st.session_state.step = 2
                                st.rerun()

            elif input_method == "Paste BibTeX":
                st.info("Export BibTeX from Google Scholar and paste below")
                st.markdown("**How to export:** Go to Scholar profile → Select articles → Export → BibTeX")

                bibtex_input = st.text_area(
                    "Paste BibTeX entries here",
                    height=300,
                    placeholder="@article{...\n}\n@inproceedings{...\n}"
                )

                if st.button("Process BibTeX", type="primary", disabled=not (bibtex_input and bib_url)):
                    with st.spinner("Parsing BibTeX..."):
                        st.session_state.scholar_pubs = parse_bibtex_input(bibtex_input)

                    with st.spinner("Fetching existing BibTeX entries..."):
                        st.session_state.existing_entries = fetch_existing_bib(bib_url)

                    if st.session_state.scholar_pubs and st.session_state.existing_entries:
                        with st.spinner("Finding missing papers..."):
                            st.session_state.missing_papers = find_missing_papers(
                                st.session_state.scholar_pubs,
                                st.session_state.existing_entries,
                                lm_studio_url,
                                st.session_state.get("use_ai_disambiguation", True),
                                st.session_state.get("match_threshold", DEFAULT_MATCH_THRESHOLD),
                                st.session_state.get("uncertain_threshold", DEFAULT_UNCERTAIN_THRESHOLD)
                            )
                        st.session_state.step = 2
                        st.rerun()

            elif input_method == "Paste text list":
                st.info("Paste publications as text (one per line)")
                st.markdown("**Format:** `Title | Authors | Year | Venue`")

                text_input = st.text_area(
                    "Paste publications here",
                    height=300,
                    placeholder="My Paper Title | Smith J, Doe A | 2024 | Nature\nAnother Paper | Jones B | 2023 | Science"
                )

                if st.button("Process List", type="primary", disabled=not (text_input and bib_url)):
                    with st.spinner("Parsing publications..."):
                        st.session_state.scholar_pubs = parse_manual_publications(text_input)

                    with st.spinner("Fetching existing BibTeX entries..."):
                        st.session_state.existing_entries = fetch_existing_bib(bib_url)

                    if st.session_state.scholar_pubs and st.session_state.existing_entries:
                        with st.spinner("Finding missing papers..."):
                            st.session_state.missing_papers = find_missing_papers(
                                st.session_state.scholar_pubs,
                                st.session_state.existing_entries,
                                lm_studio_url,
                                st.session_state.get("use_ai_disambiguation", True),
                                st.session_state.get("match_threshold", DEFAULT_MATCH_THRESHOLD),
                                st.session_state.get("uncertain_threshold", DEFAULT_UNCERTAIN_THRESHOLD)
                            )
                        st.session_state.step = 2
                        st.rerun()

            else:
                st.error(f"Unknown input method: '{input_method}'. Please select a valid method in the sidebar.")

        elif st.session_state.step >= 2:
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Scholar Publications", len(st.session_state.scholar_pubs))
            with col2:
                st.metric("Existing in BibTeX", len(st.session_state.existing_entries))
            with col3:
                st.metric("Missing Papers", len(st.session_state.missing_papers))

            st.divider()

            if not st.session_state.missing_papers:
                st.success("All your Google Scholar papers are already in papers.bib!")
            else:
                # ===== CLEANUP TOOLS FOR MISSING PAPERS =====
                with st.expander("🧹 Clean & Check Missing Papers", expanded=False):
                    st.caption("Find duplicates and issues in the missing papers before adding to website")

                    clean_col1, clean_col2 = st.columns(2)

                    with clean_col1:
                        st.markdown("**🔍 Find Duplicates Among Missing**")
                        if st.button("Check for Duplicates", key="sync_check_dupes"):
                            missing = st.session_state.missing_papers
                            dupe_groups = []
                            seen = set()
                            for i, p1 in enumerate(missing):
                                if i in seen:
                                    continue
                                group = [{"index": i, **p1}]
                                title1 = p1.get("title", "").lower().strip()
                                for j, p2 in enumerate(missing[i+1:], i+1):
                                    if j in seen:
                                        continue
                                    title2 = p2.get("title", "").lower().strip()
                                    if fuzz.ratio(title1, title2) > 80:
                                        group.append({"index": j, **p2})
                                        seen.add(j)
                                if len(group) > 1:
                                    seen.add(i)
                                    dupe_groups.append(group)
                            st.session_state.sync_missing_dupes = dupe_groups

                        if "sync_missing_dupes" in st.session_state:
                            dupes = st.session_state.sync_missing_dupes
                            if dupes:
                                st.warning(f"Found {len(dupes)} duplicate groups!")
                                for gi, group in enumerate(dupes[:5]):
                                    st.markdown(f"**Group {gi+1}:**")
                                    for p in group:
                                        idx = p["index"]
                                        st.write(f"  • [{idx}] {p.get('title', '')[:50]}...")
                            else:
                                st.success("No duplicates found!")

                    with clean_col2:
                        st.markdown("**⚠️ Check for Issues**")
                        if st.button("Find Quality Issues", key="sync_check_quality"):
                            issues = []
                            for i, p in enumerate(st.session_state.missing_papers):
                                title = p.get("title", "")
                                authors = p.get("authors", "")
                                year = p.get("year", "")
                                problems = []
                                if not title or len(title) < 10:
                                    problems.append("short/missing title")
                                if not authors or len(authors) < 3:
                                    problems.append("missing authors")
                                if not year:
                                    problems.append("missing year")
                                if problems:
                                    issues.append({"index": i, "title": title[:40], "issues": problems})
                            st.session_state.sync_missing_issues = issues

                        if "sync_missing_issues" in st.session_state:
                            issues = st.session_state.sync_missing_issues
                            if issues:
                                st.warning(f"Found {len(issues)} papers with issues!")
                                for p in issues[:10]:
                                    idx = p["index"]
                                    st.write(f"• [{idx}] {p['title']}... - {', '.join(p['issues'])}")
                            else:
                                st.success("No quality issues found!")

                st.divider()
                st.subheader("Review Missing Papers")

                # DOI Batch Lookup Section
                if st.session_state.doi_lookup_enabled:
                    doi_col1, doi_col2 = st.columns([3, 1])
                    with doi_col1:
                        found_count = sum(1 for d in st.session_state.dois.values()
                                         if d.get("status") == "found")
                        total_papers = len(st.session_state.missing_papers)
                        st.caption(f"DOIs found: {found_count}/{total_papers}")

                    with doi_col2:
                        if st.button("Lookup All DOIs", type="secondary"):
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            def update_progress(current, total, title):
                                progress_bar.progress(current / total)
                                status_text.text(f"Looking up: {title}...")

                            mailto = st.session_state.get("crossref_email", "user@example.com")
                            results = lookup_dois_batch(
                                st.session_state.missing_papers,
                                mailto,
                                update_progress
                            )
                            st.session_state.dois = results

                            # Update selections with found DOIs and clear BibTeX cache
                            for i, doi_info in results.items():
                                paper_id = f"paper_{i}"
                                if doi_info.get("doi"):
                                    doi_value = doi_info["doi"]
                                    # Update selections
                                    if paper_id in st.session_state.selections:
                                        st.session_state.selections[paper_id]["doi"] = doi_value
                                    # CRITICAL: Also update the text_input widget state directly
                                    # (Streamlit widgets cache their values by key)
                                    st.session_state[f"doi_{paper_id}"] = doi_value
                                    # Clear cached BibTeX so it regenerates with DOI
                                    if f"bibtex_{paper_id}" in st.session_state:
                                        del st.session_state[f"bibtex_{paper_id}"]

                            progress_bar.empty()
                            status_text.empty()
                            found = sum(1 for d in results.values() if d.get("status") == "found")
                            st.success(f"Found {found} DOIs out of {len(results)} papers")
                            st.rerun()

                # Store selections in session state
                if "selections" not in st.session_state:
                    st.session_state.selections = {}

                # Initialize all selections first
                for i, paper in enumerate(st.session_state.missing_papers):
                    paper_id = f"paper_{i}"
                    if paper_id not in st.session_state.selections:
                        default_key = generate_citation_key(paper["authors"], paper["year"])
                        default_abbr = "Article"
                        venue = paper["venue"].lower() if paper["venue"] else ""
                        if "conference" in venue or "proceedings" in venue:
                            default_abbr = "Conf"
                        elif "journal" in venue:
                            default_abbr = "Journal"

                        # Get DOI if already looked up
                        doi_info = st.session_state.dois.get(i, {})
                        default_doi = doi_info.get("doi", "")

                        st.session_state.selections[paper_id] = {
                            "include": True,
                            "citation_key": default_key,
                            "abbr": default_abbr,
                            "doi": default_doi,
                        }
                    else:
                        # Update DOI if newly discovered and not already set
                        doi_info = st.session_state.dois.get(i, {})
                        if doi_info.get("doi") and not st.session_state.selections[paper_id].get("doi"):
                            st.session_state.selections[paper_id]["doi"] = doi_info["doi"]
                            # Also update widget state
                            st.session_state[f"doi_{paper_id}"] = doi_info["doi"]

                # Render papers
                for i, paper in enumerate(st.session_state.missing_papers):
                    paper_id = f"paper_{i}"

                    # Add DOI status indicator to expander title
                    doi_info = st.session_state.dois.get(i, {})
                    doi_status_icon = ""
                    if doi_info.get("verified"):
                        doi_status_icon = " [DOI]"
                    elif doi_info.get("status") == "found":
                        doi_status_icon = " [DOI?]"

                    display_title = decode_latex(paper['title'])
                    with st.expander(
                        f"{'✅' if st.session_state.selections[paper_id]['include'] else '⬜'} **{display_title}** ({paper['year']}){doi_status_icon}",
                        expanded=False
                    ):
                        render_paper_card(paper, i, st.session_state.selections, st.session_state.dois)

                        # Editable BibTeX
                        if st.session_state.selections[paper_id]["include"]:
                            st.markdown("##### Generated BibTeX")
                            paper_doi = st.session_state.selections[paper_id].get("doi", "")
                            default_bibtex = format_bibtex_entry(
                                paper,
                                st.session_state.selections[paper_id]["citation_key"],
                                st.session_state.selections[paper_id]["abbr"],
                                paper_doi
                            )

                            # Store edited bibtex in session state
                            # Regenerate if DOI was added/changed since last generation
                            cached_doi_key = f"bibtex_doi_{paper_id}"
                            cached_doi = st.session_state.get(cached_doi_key, "")
                            if f"bibtex_{paper_id}" not in st.session_state or (paper_doi and paper_doi != cached_doi):
                                st.session_state[f"bibtex_{paper_id}"] = default_bibtex
                                st.session_state[cached_doi_key] = paper_doi

                            col_edit, col_copy = st.columns([5, 1])
                            with col_edit:
                                edited_bibtex = st.text_area(
                                    "Edit BibTeX",
                                    value=st.session_state[f"bibtex_{paper_id}"],
                                    height=200,
                                    key=f"edit_bibtex_{paper_id}",
                                    label_visibility="collapsed"
                                )
                                st.session_state[f"bibtex_{paper_id}"] = edited_bibtex

                            with col_copy:
                                # Copy button using JS
                                st.markdown(f"""
                                <button onclick="navigator.clipboard.writeText(document.getElementById('bibtex_content_{i}').value); this.innerHTML='✓ Copied!';"
                                        style="background:#667eea; color:white; border:none; padding:8px 16px; border-radius:6px; cursor:pointer; margin-top:5px;">
                                    📋 Copy
                                </button>
                                <textarea id="bibtex_content_{i}" style="display:none;">{edited_bibtex}</textarea>
                                """, unsafe_allow_html=True)

                            # Reset to default button
                            if st.button("↻ Reset to default", key=f"reset_{paper_id}"):
                                st.session_state[f"bibtex_{paper_id}"] = default_bibtex
                                st.rerun()

                st.divider()

                # Re-run disambiguation button
                if st.button("🤖 Re-run AI Disambiguation"):
                    with st.spinner("Re-checking with AI..."):
                        lm_url = st.session_state.get("lm_studio_url", "http://localhost:1234/v1")
                        st.session_state.missing_papers = find_missing_papers(
                            st.session_state.scholar_pubs, st.session_state.existing_entries, lm_url,
                            st.session_state.get("use_ai_disambiguation", True),
                            st.session_state.get("match_threshold", DEFAULT_MATCH_THRESHOLD),
                            st.session_state.get("uncertain_threshold", DEFAULT_UNCERTAIN_THRESHOLD))
                        st.session_state.selections = {}
                        st.session_state.dois = {}
                    st.success(f"Found {len(st.session_state.missing_papers)} missing papers")
                    st.rerun()

                st.divider()

                # Generate output
                selected_count = sum(1 for s in st.session_state.selections.values() if s["include"])
                st.info(f"Selected {selected_count} papers")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📥 Generate BibTeX", type="primary", disabled=selected_count == 0):
                        # Build output - always regenerate with current DOI
                        output_entries = []
                        for i, paper in enumerate(st.session_state.missing_papers):
                            paper_id = f"paper_{i}"
                            if st.session_state.selections[paper_id]["include"]:
                                # Always use current DOI from selections
                                paper_doi = st.session_state.selections[paper_id].get("doi", "")
                                # Regenerate BibTeX to ensure DOI is included
                                bibtex = format_bibtex_entry(
                                    paper,
                                    st.session_state.selections[paper_id]["citation_key"],
                                    st.session_state.selections[paper_id]["abbr"],
                                    paper_doi
                                )
                                output_entries.append(bibtex)

                        output_content = "\n\n".join(output_entries)

                        # Save to file
                        date_str = datetime.now().strftime("%Y-%m-%d")
                        output_file = OUTPUT_DIR / f"new_papers_{date_str}.bib"
                        output_file.write_text(output_content)

                        st.session_state.output_content = output_content
                        st.session_state.output_file = str(output_file)
                        st.session_state.step = 3
                        st.rerun()

                with col2:
                    if st.button("💾 Save to Database", disabled=selected_count == 0):
                        saved = 0
                        for i, paper in enumerate(st.session_state.missing_papers):
                            paper_id = f"paper_{i}"
                            if st.session_state.selections[paper_id]["include"]:
                                paper_doi = st.session_state.selections[paper_id].get("doi", "")
                                paper_data = {**paper, "doi": paper_doi,
                                    "citation_key": st.session_state.selections[paper_id]["citation_key"],
                                    "abbr": st.session_state.selections[paper_id]["abbr"],
                                    "bibtex": st.session_state.get(f"bibtex_{paper_id}",
                                        format_bibtex_entry(paper, st.session_state.selections[paper_id]["citation_key"],
                                            st.session_state.selections[paper_id]["abbr"], paper_doi))}
                                if save_to_db(paper_data, "sync"): saved += 1
                        st.success(f"Saved {saved} papers to database!")

                with col3:
                    if st.button("🔄 Start Over"):
                        for key in ["scholar_pubs", "existing_entries", "missing_papers",
                                   "selections", "output_content", "output_file", "dois"]:
                            if key in st.session_state: del st.session_state[key]
                        st.session_state.step = 1
                        st.rerun()

        if st.session_state.step == 3:
            st.success(f"Saved to: {st.session_state.output_file}")

            st.download_button(
                label="Download BibTeX File",
                data=st.session_state.output_content,
                file_name=Path(st.session_state.output_file).name,
                mime="text/plain"
            )

            with st.expander("Preview Generated BibTeX", expanded=True):
                st.code(st.session_state.output_content, language="bibtex")

            if st.button("Start Over"):
                for key in ["scholar_pubs", "existing_entries", "missing_papers",
                           "selections", "output_content", "output_file", "step", "dois"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()


if __name__ == "__main__":
    main()
