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


# ============ Database Functions ============
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS papers (
        id INTEGER PRIMARY KEY, title TEXT UNIQUE, authors TEXT, year TEXT,
        venue TEXT, doi TEXT, citations INTEGER, citation_key TEXT, abbr TEXT,
        bibtex TEXT, source TEXT, added_date TEXT)""")
    conn.commit()
    conn.close()

def save_to_db(paper: dict, source: str = "sync") -> bool:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""INSERT OR REPLACE INTO papers
            (title,authors,year,venue,doi,citations,citation_key,abbr,bibtex,source,added_date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (paper.get("title"), paper.get("authors"), paper.get("year"), paper.get("venue"),
             paper.get("doi",""), paper.get("citations",0), paper.get("citation_key"),
             paper.get("abbr"), paper.get("bibtex"), source, datetime.now().isoformat()))
        conn.commit()
        return True
    except: return False
    finally: conn.close()

def search_db(query="", author="", year="", sort_by="year", sort_order="DESC", limit=200):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conds, params = [], []
    if query: conds.append("(title LIKE ? OR venue LIKE ?)"); params += [f"%{query}%"]*2
    if author: conds.append("authors LIKE ?"); params.append(f"%{author}%")
    if year: conds.append("year=?"); params.append(year)
    where = " AND ".join(conds) if conds else "1=1"
    order = f"{sort_by} {sort_order}"
    rows = conn.execute(f"SELECT * FROM papers WHERE {where} ORDER BY {order} LIMIT ?", params+[limit]).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_db_years():
    conn = sqlite3.connect(DB_PATH)
    years = [r[0] for r in conn.execute("SELECT DISTINCT year FROM papers WHERE year!='' ORDER BY year DESC")]
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
    authors = decode_latex(paper.get('authors', ''))
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


def get_stats():
    """Get database statistics."""
    conn = sqlite3.connect(DB_PATH)
    stats = {}
    stats['total'] = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    stats['with_doi'] = conn.execute("SELECT COUNT(*) FROM papers WHERE doi != ''").fetchone()[0]
    stats['years'] = conn.execute("SELECT MIN(year), MAX(year) FROM papers WHERE year != ''").fetchone()
    stats['by_year'] = dict(conn.execute("SELECT year, COUNT(*) FROM papers WHERE year != '' GROUP BY year ORDER BY year DESC").fetchall())
    stats['top_venues'] = conn.execute("SELECT venue, COUNT(*) as cnt FROM papers WHERE venue != '' GROUP BY venue ORDER BY cnt DESC LIMIT 10").fetchall()
    conn.close()
    return stats

def get_db_count():
    conn = sqlite3.connect(DB_PATH)
    cnt = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    conn.close()
    return cnt


def decode_latex(text: str) -> str:
    """Decode LaTeX special characters for display."""
    if not text:
        return text
    replacements = [
        (r"{\"a}", "ä"), (r"{\"o}", "ö"), (r"{\"u}", "ü"),
        (r"{\"A}", "Ä"), (r"{\"O}", "Ö"), (r"{\"U}", "Ü"),
        (r"{\\'a}", "á"), (r"{\'a}", "á"), (r"\\'a", "á"),
        (r"{\\'e}", "é"), (r"{\'e}", "é"), (r"\\'e", "é"),
        (r"{\\'i}", "í"), (r"{\'i}", "í"), (r"\\'i", "í"),
        (r"{\\'o}", "ó"), (r"{\'o}", "ó"), (r"\\'o", "ó"),
        (r"{\\'u}", "ú"), (r"{\'u}", "ú"), (r"\\'u", "ú"),
        (r"{\\`a}", "à"), (r"{`a}", "à"),
        (r"{\\`e}", "è"), (r"{`e}", "è"),
        (r"{\\~n}", "ñ"), (r"{~n}", "ñ"), (r"\\~n", "ñ"),
        (r"{\\c{c}}", "ç"), (r"{\\c c}", "ç"),
        (r"{\ss}", "ß"), (r"\\ss", "ß"),
        (r"{\\aa}", "å"), (r"{\\AA}", "Å"),
        (r"{\\o}", "ø"), (r"{\\O}", "Ø"),
        (r"{\\ae}", "æ"), (r"{\\AE}", "Æ"),
        (r"{\\'y}", "ý"), (r"{\'y}", "ý"),
        (r"{\\\"u}", "ü"), (r"\\\"u", "ü"),
        (r"{\\\"o}", "ö"), (r"\\\"o", "ö"),
        (r"{\\\"a}", "ä"), (r"\\\"a", "ä"),
        (r"{\\'", ""), (r"{\'", ""), (r"'}", ""),  # cleanup
        (r"{", ""), (r"}", ""),  # remove remaining braces
    ]
    result = text
    for latex, char in replacements:
        result = result.replace(latex, char)
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

    # Try to find the publications table specifically
    pub_table = soup.select_one("#gsc_a_t")  # Google Scholar publications table
    if pub_table:
        text = pub_table.get_text(separator="\n", strip=True)
    else:
        # Fallback: get all text
        text = soup.get_text(separator="\n", strip=True)

    # Remove excessive whitespace
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines[:500])  # Limit lines


def extract_with_llm(html_content: str, lm_studio_url: str = "http://localhost:1234/v1") -> list[dict]:
    """Use LM Studio to extract publication data from HTML."""
    import json

    # Clean HTML first to reduce size
    cleaned_text = clean_html_for_llm(html_content)

    prompt = f"""Extract academic publications from this text. Return a JSON array only.

For each publication extract:
- title: paper title
- authors: author names
- year: publication year (4 digits)
- venue: journal/conference name
- citations: citation count (number)

Return ONLY valid JSON like:
[{{"title": "Paper", "authors": "A Smith", "year": "2024", "venue": "Nature", "citations": 10}}]

TEXT:
{cleaned_text}

JSON:"""

    try:
        response = requests.post(
            f"{lm_studio_url}/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 4000
            },
            timeout=180
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()

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
                "citations": 0,
                "pub_type": entry.get("ENTRYTYPE", "article"),
            })
        return publications
    except Exception as e:
        st.error(f"Error parsing BibTeX: {e}")
        return []


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
    if normalize_title(pub_title) == normalize_title(existing_title):
        return 100, "exact_match"

    # 2. Fuzzy ratio
    ratio = fuzz.ratio(normalize_title(pub_title), normalize_title(existing_title))
    scores.append(ratio)

    # 3. Token overlap (handles word reordering)
    pub_tokens = get_title_tokens(pub_title)
    existing_tokens = get_title_tokens(existing_title)
    if pub_tokens and existing_tokens:
        overlap = len(pub_tokens & existing_tokens)
        union = len(pub_tokens | existing_tokens)
        token_score = (overlap / union) * 100 if union > 0 else 0
        scores.append(token_score)

    # 4. Partial ratio (handles substrings)
    partial = fuzz.partial_ratio(normalize_title(pub_title), normalize_title(existing_title))
    scores.append(partial * 0.9)  # Slightly discount partial matches

    # 5. Token sort ratio (handles reordering)
    token_sort = fuzz.token_sort_ratio(normalize_title(pub_title), normalize_title(existing_title))
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
                        lm_studio_url: str = "http://localhost:1234/v1",
                        use_ai_disambiguation: bool = True,
                        match_threshold: int = 85,
                        uncertain_threshold: int = 60) -> list[dict]:
    """Find papers in Scholar that are not in existing BibTeX using advanced matching."""
    missing = []
    uncertain = []  # Papers that need AI disambiguation

    for pub in scholar_pubs:
        best_score = 0
        best_match = None
        best_reason = ""

        for existing in existing_entries:
            score, reason = compute_similarity_score(pub, existing)
            if score > best_score:
                best_score = score
                best_match = existing
                best_reason = reason

        # Decision thresholds (configurable)
        if best_score >= match_threshold:
            # High confidence match - skip this paper
            continue
        elif best_score >= uncertain_threshold and use_ai_disambiguation:
            # Uncertain - queue for AI disambiguation
            uncertain.append((pub, best_match, best_score, best_reason))
        else:
            # Low similarity - definitely missing
            missing.append(pub)

    # Process uncertain cases with AI
    if uncertain and use_ai_disambiguation:
        for pub, best_match, score, reason in uncertain:
            try:
                is_same = ai_disambiguate(pub, best_match, lm_studio_url)
                if not is_same:
                    missing.append(pub)
            except Exception:
                # If AI fails, be conservative and include as missing
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


def format_bibtex_entry(paper: dict, citation_key: str, abbr: str, doi: str = "") -> str:
    """Generate formatted BibTeX string."""
    # Determine entry type
    entry_type = "ARTICLE"
    venue_lower = paper["venue"].lower() if paper["venue"] else ""
    if any(kw in venue_lower for kw in ["conference", "proceedings", "symposium", "workshop"]):
        entry_type = "INPROCEEDINGS"

    # Format authors
    authors = paper["authors"]

    # Build entry
    lines = [f"@{entry_type}{{{citation_key},"]
    lines.append(f'  author = {{{authors}}},')
    lines.append(f'  year = {{{paper["year"]}}},')
    lines.append(f'  title = {{{paper["title"]}}},')

    if entry_type == "INPROCEEDINGS":
        lines.append(f'  booktitle = {{{paper["venue"]}}},')
    else:
        lines.append(f'  journal = {{{paper["venue"]}}},')

    # Add DOI if available
    if doi:
        lines.append(f'  doi = {{{doi}}},')

    lines.append('  volume = {NULL},')
    lines.append(f'  abbr = {{{abbr}}},')
    lines.append('  bibtex_show = {true},')
    lines.append('  selected = {false},')
    lines.append('}')

    return "\n".join(lines)


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
    if "scholar_pubs" not in st.session_state:
        st.session_state.scholar_pubs = []
    if "existing_entries" not in st.session_state:
        st.session_state.existing_entries = []
    if "missing_papers" not in st.session_state:
        st.session_state.missing_papers = []
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "dois" not in st.session_state:
        st.session_state.dois = {}
    if "doi_lookup_enabled" not in st.session_state:
        st.session_state.doi_lookup_enabled = True

    # Initialize database
    init_db()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")

        input_method = st.radio(
            "Scholar Input Method",
            ["Google Scholar URL", "LM Studio (AI extraction)", "Paste BibTeX", "Paste HTML + AI", "Paste text list"],
            help="Choose how to input your publications"
        )

        lm_studio_url = st.text_input(
            "LM Studio URL",
            value="http://localhost:1234/v1",
            help="LM Studio API endpoint"
        )

        bib_url = st.text_input(
            "papers.bib URL",
            value="https://raw.githubusercontent.com/mohsaqr/mohsaqr.github.io/main/_bibliography/papers.bib",
            help="Raw GitHub URL to your existing papers.bib"
        )

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

        st.session_state.input_method = input_method
        st.session_state.bib_url = bib_url
        st.session_state.lm_studio_url = lm_studio_url

    # ============ MAIN TABS ============
    tab_papers, tab_sync = st.tabs(["📂 My Papers", "🔄 Sync Papers"])

    # ============ MY PAPERS TAB ============
    with tab_papers:
        # Sub-tabs for My Papers
        stats_tab, table_tab, list_tab, import_tab = st.tabs(["📊 Statistics", "📋 Table View", "📄 List View", "📥 Import"])

        # ===== STATISTICS TAB =====
        with stats_tab:
            stats = get_stats()
            st.subheader("📊 Publication Statistics")

            # Key metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Total Papers", stats['total'])
            with m2: st.metric("With DOI", stats['with_doi'])
            with m3: st.metric("DOI Coverage", f"{100*stats['with_doi']//max(stats['total'],1)}%")
            with m4:
                if stats['years'][0]:
                    st.metric("Year Range", f"{stats['years'][0]}-{stats['years'][1]}")

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
                st.subheader("Top Venues")
                if stats['top_venues']:
                    for venue, count in stats['top_venues'][:10]:
                        st.write(f"**{count}** - {decode_latex(venue)[:50]}")

        # ===== TABLE TAB =====
        with table_tab:
            st.subheader("📋 Papers Table")

            # Filters
            fc1, fc2, fc3 = st.columns(3)
            with fc1: tbl_search = st.text_input("🔍 Search", key="tbl_search", placeholder="Title or venue...")
            with fc2: tbl_author = st.text_input("👤 Author", key="tbl_author")
            with fc3: tbl_year = st.selectbox("📅 Year", ["All"] + get_db_years(), key="tbl_year")

            results = search_db(tbl_search, tbl_author, tbl_year if tbl_year != "All" else "")

            if results:
                import pandas as pd
                # Create DataFrame for table display
                table_data = []
                for p in results:
                    table_data.append({
                        'Year': p['year'],
                        'Title': decode_latex(p['title'])[:80],
                        'Authors': decode_latex(p['authors'])[:50],
                        'Venue': decode_latex(p['venue'] or '')[:30],
                        'DOI': '✓' if p.get('doi') else '',
                        'ID': p['id']
                    })
                df = pd.DataFrame(table_data)

                # Selection
                if "tbl_selected" not in st.session_state:
                    st.session_state.tbl_selected = []

                st.dataframe(
                    df[['Year', 'Title', 'Authors', 'Venue', 'DOI']],
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(f"Showing {len(results)} papers")

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
            lc1, lc2, lc3, lc4 = st.columns([3,2,1,1])
            with lc1: list_search = st.text_input("🔍 Search", key="list_search")
            with lc2: list_author = st.text_input("👤 Author", key="list_author")
            with lc3: list_year = st.selectbox("Year", ["All"] + get_db_years(), key="list_year")
            with lc4: list_sort = st.selectbox("Sort", ["year", "title", "authors"], key="list_sort")

            results = search_db(list_search, list_author, list_year if list_year != "All" else "", list_sort, "DESC")
            st.caption(f"Found {len(results)} papers")

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

            # Paper list with formatted references
            for paper in results:
                pid = paper["id"]
                sel = pid in st.session_state.list_sel

                with st.expander(f"{'☑️' if sel else '⬜'} {decode_latex(paper['title'][:70])}{'...' if len(paper['title'])>70 else ''} ({paper['year']})"):
                    # Checkbox and formatted reference
                    col1, col2 = st.columns([1, 20])
                    with col1:
                        if st.checkbox("", value=sel, key=f"lsel_{pid}", label_visibility="collapsed"):
                            st.session_state.list_sel.add(pid)
                        else:
                            st.session_state.list_sel.discard(pid)

                    with col2:
                        # Formatted reference
                        st.markdown(format_reference(paper))

                    # Editable fields
                    st.markdown("---")
                    st.markdown("**Edit:**")
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        new_doi = st.text_input("DOI", value=paper.get('doi', ''), key=f"doi_{pid}")
                    with ec2:
                        new_venue = st.text_input("Venue", value=paper.get('venue', ''), key=f"venue_{pid}")

                    if st.button("💾 Save Changes", key=f"save_{pid}"):
                        update_paper(pid, doi=new_doi, venue=new_venue)
                        st.success("Saved!")
                        st.rerun()

                    # Actions row
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        if st.button("🔍 Lookup DOI", key=f"ldoi_{pid}"):
                            res = lookup_doi(paper["title"], paper["authors"], paper["year"])
                            if res["status"]=="found" and res["doi"]:
                                update_db_doi(pid, res["doi"])
                                st.success(f"Found: {res['doi']}")
                                st.rerun()
                            else:
                                st.warning("Not found")
                    with b2:
                        if st.button("📋 Copy BibTeX", key=f"lbib_{pid}"):
                            bib = paper.get("bibtex") or format_bibtex_entry(paper, generate_citation_key(paper["authors"], paper["year"]), "Article", paper.get("doi",""))
                            st.code(bib, language="bibtex")
                    with b3:
                        if st.button("🗑️ Delete", key=f"ldel_{pid}"):
                            delete_from_db(pid)
                            st.session_state.list_sel.discard(pid)
                            st.rerun()

        # ===== IMPORT TAB =====
        with import_tab:
            st.subheader("Import Papers to Database")
            import_method = st.radio("Import from", ["Paste BibTeX", "Paste Text List"], horizontal=True)

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

    # ============ SYNC PAPERS TAB ============
    with tab_sync:
        if st.session_state.step == 1:
            input_method = st.session_state.get("input_method", "LM Studio (AI extraction)")
            bib_url = st.session_state.get("bib_url", "")
            lm_studio_url = st.session_state.get("lm_studio_url", "http://localhost:1234/v1")

            if input_method == "Google Scholar URL":
                st.info("Enter a Google Scholar profile URL and paste the page HTML")

                scholar_url = st.text_input(
                    "Google Scholar Profile URL",
                    value="https://scholar.google.com/citations?hl=en&user=U-O6R7YAAAAJ&view_op=list_works&sortby=pubdate",
                    help="Your Google Scholar profile URL"
                )

                st.warning("⚠️ Auto-fetch may be blocked by Google. Use the manual paste method below.")

                with st.expander("📋 How to copy your Scholar HTML (recommended)", expanded=True):
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

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Extract from Pasted HTML", type="primary", disabled=not (html_input and bib_url)):
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
                                        st.session_state.get("match_threshold", 85),
                                        st.session_state.get("uncertain_threshold", 60)
                                    )
                                st.session_state.step = 2
                                st.rerun()

                with col2:
                    # Extract user ID for auto-fetch attempt
                    user_id = ""
                    if scholar_url and "user=" in scholar_url:
                        match = re.search(r'user=([^&]+)', scholar_url)
                        if match:
                            user_id = match.group(1)

                    num_papers = st.number_input("Papers to fetch", 10, 100, 50, 10)

                    if st.button("Try Auto-Fetch", disabled=not (user_id and bib_url), help="May be blocked by Google"):
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
                                            st.session_state.get("match_threshold", 85),
                                            st.session_state.get("uncertain_threshold", 60)
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
                                    st.session_state.get("match_threshold", 85),
                                    st.session_state.get("uncertain_threshold", 60)
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
                                st.session_state.get("match_threshold", 85),
                                st.session_state.get("uncertain_threshold", 60)
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
                                st.session_state.get("match_threshold", 85),
                                st.session_state.get("uncertain_threshold", 60)
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
                                st.session_state.get("match_threshold", 85),
                                st.session_state.get("uncertain_threshold", 60)
                            )
                        st.session_state.step = 2
                        st.rerun()

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

                            # Update selections with found DOIs
                            for i, doi_info in results.items():
                                paper_id = f"paper_{i}"
                                if paper_id in st.session_state.selections:
                                    if doi_info.get("doi"):
                                        st.session_state.selections[paper_id]["doi"] = doi_info["doi"]

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

                    with st.expander(
                        f"{'✅' if st.session_state.selections[paper_id]['include'] else '⬜'} **{paper['title'][:70]}{'...' if len(paper['title']) > 70 else ''}** ({paper['year']}){doi_status_icon}",
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
                            if f"bibtex_{paper_id}" not in st.session_state:
                                st.session_state[f"bibtex_{paper_id}"] = default_bibtex

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
                            st.session_state.get("match_threshold", 85),
                            st.session_state.get("uncertain_threshold", 60))
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
                        # Build output using edited bibtex
                        output_entries = []
                        for i, paper in enumerate(st.session_state.missing_papers):
                            paper_id = f"paper_{i}"
                            if st.session_state.selections[paper_id]["include"]:
                                # Use edited version if available
                                paper_doi = st.session_state.selections[paper_id].get("doi", "")
                                bibtex = st.session_state.get(
                                    f"bibtex_{paper_id}",
                                    format_bibtex_entry(
                                        paper,
                                        st.session_state.selections[paper_id]["citation_key"],
                                        st.session_state.selections[paper_id]["abbr"],
                                        paper_doi
                                    )
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
