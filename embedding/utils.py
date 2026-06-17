import csv
import os
import pdfplumber
import nltk
from html.parser import HTMLParser
from typing import Optional


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def get_file_paths(root_dir: str, file_extensions: list[str]) -> list[str]:
    """
    Retrieves a list of paths to all files with specified extensions in the given root directory and its subdirectories.

    Args:
        root_dir (str): The root directory to search for files.
        file_extensions (list[str]): A list of file extensions to retrieve. For example, ["txt", "pdf"]

    Returns:
        List[str]: A list of file paths to all matching files found within the root directory and its subdirectories.
    """
    file_paths = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.endswith(f".{ext}") for ext in file_extensions):
                file_paths.append(os.path.join(dirpath, filename))
    
    return file_paths



def read_text_file(file_path: str) -> str:
    """
    Reads the content of a text file and returns it as a single string.

    Args:
        file_path (str): The path to the .txt file to read.

    Returns:
        str: The content of the file as a single string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return content


def read_pdf_file(file_path: str) -> str:
    """
    Reads the content of a PDF file and returns it as a single string.
    
    Args:
        file_path (str): The path to the PDF file to read.
    
    Returns:
        str: The content of the PDF as a single string.
    """
    text_content = []
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract text from each page
            page_text = page.extract_text()
            if page_text:  # Ensure the page has text
                text_content.append(page_text)
    
    # Join all pages' text into a single string
    return "\n".join(text_content)


def split_text_into_sentences(text: str, language: str) -> list[str]:
    """
    Splits the given text into a list of sentences using NLTK's sentence tokenizer.

    Args:
        text (str): The input text to split into sentences.
        language (str): The language of the text for the sentence tokenizer

    Returns:
        list[str]: A list of sentences.
    """
    sentences = nltk.sent_tokenize(text, language=language)
    return sentences


def chunk_sentences(sentences: list[str], chunk_size: int, overlap_size: int) -> list[str]:
    """
    Groups a list of sentences into overlapping chunks.

    Args:
        sentences (list[str]): A list of sentences to be chunked.
        chunk_size (int): The number of sentences in each chunk.
        overlap_size (int): The number of sentences to overlap between consecutive chunks.

    Returns:
        list[str]: A list of text chunks with the specified overlap.
        
    Raises:
        ValueError: If overlap_size is greater than or equal to chunk_size.
    """
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be smaller than chunk_size.")

    chunks = []
    # The step of the range is the distance between the start of consecutive chunks.
    step = chunk_size - overlap_size
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# HTML helper (used by both read_html_file and read_markdown_file)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Minimal HTMLParser subclass that collects visible text nodes."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_tags = {"script", "style", "head"}
        self._skip = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._skip_tags:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags:
            self._skip = False
        # Insert a space after block-level tags to prevent words merging.
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                   "tr", "br"}:
            self._parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join("".join(self._parts).split())


def _html_to_text(html: str) -> str:
    """Strip HTML tags and return clean visible text.

    Args:
        html: Raw HTML string.

    Returns:
        Plain text with whitespace normalised.
    """
    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.get_text()


# ---------------------------------------------------------------------------
# New file-type readers
# ---------------------------------------------------------------------------

def read_docx_file(file_path: str) -> str:
    """Read a Word (.docx) document and return its text as a single string.

    Paragraphs are joined with newlines.  Empty paragraphs (used in Word
    for visual spacing) are omitted.

    Args:
        file_path: Path to the .docx file.

    Returns:
        The document's text content.
    """
    import docx  # python-docx; imported here to keep the module importable
                  # even when the package is not installed.

    document = docx.Document(file_path)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def read_html_file(file_path: str, encoding: str = "utf-8") -> str:
    """Read an HTML file and return its visible text content.

    Tags, scripts, and style blocks are stripped.  Whitespace is normalised
    to single spaces.

    Args:
        file_path: Path to the .html (or .htm) file.
        encoding: File encoding (default ``utf-8``).

    Returns:
        Plain text extracted from the HTML.
    """
    with open(file_path, "r", encoding=encoding, errors="replace") as fh:
        html = fh.read()
    return _html_to_text(html)


def read_markdown_file(file_path: str, encoding: str = "utf-8") -> str:
    """Read a Markdown file and return its plain-text content.

    The file is rendered to HTML via ``markdown-it-py`` (already a project
    dependency) and then stripped of tags so the text passed to the
    sentence tokeniser is clean prose rather than raw Markdown syntax.

    Args:
        file_path: Path to the .md file.
        encoding: File encoding (default ``utf-8``).

    Returns:
        Plain text extracted from the Markdown source.
    """
    from markdown_it import MarkdownIt  # markdown-it-py is already installed.

    with open(file_path, "r", encoding=encoding, errors="replace") as fh:
        source = fh.read()

    md = MarkdownIt()
    html = md.render(source)
    return _html_to_text(html)


def read_csv_file(
    file_path: str,
    encoding: str = "utf-8",
    delimiter: str = ",",
    has_header: bool = True,
    max_rows: Optional[int] = None,
) -> str:
    """Read a CSV file and return its contents as human-readable prose.

    Each data row is serialised as ``"column: value, column: value."`` so
    that the sentence tokeniser can treat individual rows as discrete units.

    Args:
        file_path: Path to the .csv file.
        encoding: File encoding (default ``utf-8``).
        delimiter: Column delimiter (default ``,``).
        has_header: When ``True`` (default) the first row is treated as
            column names.  When ``False`` columns are labelled ``col_0``,
            ``col_1``, etc.
        max_rows: Optional cap on the number of data rows read.  ``None``
            reads all rows.

    Returns:
        A single string with one sentence per row, ready for sentence
        tokenisation.
    """
    rows: list[str] = []

    with open(file_path, "r", encoding=encoding, errors="replace",
              newline="") as fh:
        if has_header:
            reader = csv.DictReader(fh, delimiter=delimiter)
            for idx, row in enumerate(reader):
                if max_rows is not None and idx >= max_rows:
                    break
                pairs = ", ".join(
                    f"{key}: {val.strip()}"
                    for key, val in row.items()
                    if val and val.strip()
                )
                rows.append(pairs + ".")
        else:
            plain = csv.reader(fh, delimiter=delimiter)
            for idx, row in enumerate(plain):
                if max_rows is not None and idx >= max_rows:
                    break
                pairs = ", ".join(
                    f"col_{i}: {val.strip()}" for i, val in enumerate(row)
                )
                rows.append(pairs + ".")

    return "\n".join(rows)
