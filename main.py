"""
Tax Document Phrase Extractor
A FastAPI web application for extracting and categorizing key phrases from tax and legal documents.

Requirements:
pip install fastapi uvicorn python-multipart PyMuPDF python-docx spacy
python -m spacy download en_core_web_sm

Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import fitz  # PyMuPDF
from docx import Document
import spacy
import re
from typing import Dict, List, Tuple
import io
from datetime import datetime

app = FastAPI(title="Tax Document Phrase Extractor")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


class DocumentAnalyzer:
    """Analyzes tax and legal documents to extract key phrases."""
    
    def __init__(self):
        self.case_keywords = [
            'case no', 'case number', 'docket', 'plaintiff', 'defendant',
            'petitioner', 'respondent', 'court', 'judge', 'jurisdiction'
        ]
        self.tax_keywords = [
            'tax', 'penalty', 'interest', 'assessment', 'liability',
            'payment', 'amount due', 'balance', 'deficiency', 'refund'
        ]
        self.date_keywords = [
            'deadline', 'due date', 'filing date', 'hearing', 'trial date',
            'effective date', 'tax year', 'period'
        ]
        self.legal_keywords = [
            'section', 'code', 'statute', 'regulation', 'act', 'usc',
            'cfr', 'irc', 'title', 'chapter', 'subsection'
        ]
    
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF file."""
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += str(page.get_text())
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")
    
    def extract_text_from_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(io.BytesIO(file_bytes))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            raise Exception(f"Error extracting DOCX: {str(e)}")
    
    def extract_case_info(self, text: str, doc) -> List[Tuple[str, str]]:
        """Extract case-related information with headings."""
        results = []
        lines = text.split('\n')
        
        # Pattern matching for case numbers
        case_patterns = [
            r'Case\s+No\.?\s*[:.]?\s*([A-Z0-9-]+)',
            r'Docket\s+No\.?\s*[:.]?\s*([A-Z0-9-]+)',
            r'Case\s+Number\s*[:.]?\s*([A-Z0-9-]+)'
        ]
        
        for pattern in case_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                results.append(("Case Number", match.group(1)))
        
        # Extract parties
        for line in lines[:50]:
            line_lower = line.lower()
            if 'plaintiff' in line_lower and len(line.strip()) < 100 and len(line.strip()) > 5:
                results.append(("Plaintiff", line.strip().split(':', 1)[-1].strip() if ':' in line else line.strip()))
            elif 'defendant' in line_lower and len(line.strip()) < 100 and len(line.strip()) > 5:
                results.append(("Defendant", line.strip().split(':', 1)[-1].strip() if ':' in line else line.strip()))
            elif 'petitioner' in line_lower and len(line.strip()) < 100 and len(line.strip()) > 5:
                results.append(("Petitioner", line.strip().split(':', 1)[-1].strip() if ':' in line else line.strip()))
            elif 'respondent' in line_lower and len(line.strip()) < 100 and len(line.strip()) > 5:
                results.append(("Respondent", line.strip().split(':', 1)[-1].strip() if ':' in line else line.strip()))
        
        # Extract court information
        for ent in doc.ents:
            if ent.label_ == "ORG" and any(word in ent.text.lower() for word in ['court', 'tribunal']):
                results.append(("Court", ent.text))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for heading, content in results:
            key = (heading, content)
            if key not in seen:
                seen.add(key)
                unique_results.append((heading, content))
        
        return unique_results[:15]
    
    def extract_tax_amounts(self, text: str, doc) -> List[Tuple[str, str]]:
        """Extract tax amounts, penalties, and financial information with headings."""
        results = []
        
        money_patterns = [
            r'\$\s*[\d,]+\.?\d*',
            r'USD\s*[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*dollars?'
        ]
        
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            line_stripped = line.strip()
            
            if len(line_stripped) < 10 or len(line_stripped) > 200:
                continue
            
            # Detect specific tax categories
            heading = "Amount"
            if 'penalty' in line_lower or 'penalties' in line_lower:
                heading = "Penalty"
            elif 'interest' in line_lower:
                heading = "Interest"
            elif 'unpaid' in line_lower or 'owed' in line_lower or 'due' in line_lower:
                heading = "Amount Due"
            elif 'refund' in line_lower:
                heading = "Refund"
            elif 'assessment' in line_lower:
                heading = "Assessment"
            elif 'payment' in line_lower:
                heading = "Payment"
            elif 'liability' in line_lower:
                heading = "Liability"
            elif 'deficiency' in line_lower:
                heading = "Deficiency"
            
            if any(kw in line_lower for kw in self.tax_keywords):
                for pattern in money_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        results.append((heading, line_stripped))
                        break
        
        # Extract using NER for MONEY entities
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                start = max(0, ent.start - 10)
                end = min(len(doc), ent.end + 10)
                context = doc[start:end].text.strip()
                context_lower = context.lower()
                
                if any(kw in context_lower for kw in self.tax_keywords) and len(context) < 200:
                    heading = "Amount"
                    if 'penalty' in context_lower:
                        heading = "Penalty"
                    elif 'interest' in context_lower:
                        heading = "Interest"
                    elif 'due' in context_lower or 'owed' in context_lower:
                        heading = "Amount Due"
                    
                    results.append((heading, context))
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for heading, content in results:
            if content not in seen:
                seen.add(content)
                unique_results.append((heading, content))
        
        return unique_results[:20]
    
    def extract_dates(self, text: str, doc) -> List[Tuple[str, str]]:
        """Extract important dates and deadlines with headings."""
        results = []
        
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            line_stripped = line.strip()
            
            if len(line_stripped) < 10 or len(line_stripped) > 200:
                continue
            
            heading = "Date"
            if 'deadline' in line_lower:
                heading = "Deadline"
            elif 'due date' in line_lower or 'filing date' in line_lower:
                heading = "Filing Date"
            elif 'hearing' in line_lower:
                heading = "Hearing Date"
            elif 'trial' in line_lower:
                heading = "Trial Date"
            elif 'effective' in line_lower:
                heading = "Effective Date"
            elif 'tax year' in line_lower or 'taxable year' in line_lower:
                heading = "Tax Year"
            
            if any(kw in line_lower for kw in self.date_keywords):
                date_patterns = [
                    r'\d{1,2}/\d{1,2}/\d{2,4}',
                    r'\d{1,2}-\d{1,2}-\d{2,4}',
                    r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
                    r'\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
                ]
                
                for pattern in date_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        results.append((heading, line_stripped))
                        break
        
        # Extract using NER
        for ent in doc.ents:
            if ent.label_ == "DATE":
                start = max(0, ent.start - 8)
                end = min(len(doc), ent.end + 8)
                context = doc[start:end].text.strip()
                context_lower = context.lower()
                
                if any(kw in context_lower for kw in self.date_keywords) and len(context) < 200:
                    heading = "Date"
                    if 'deadline' in context_lower:
                        heading = "Deadline"
                    elif 'filing' in context_lower:
                        heading = "Filing Date"
                    
                    results.append((heading, context))
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for heading, content in results:
            if content not in seen:
                seen.add(content)
                unique_results.append((heading, content))
        
        return unique_results[:20]
    
    def extract_legal_references(self, text: str) -> List[Tuple[str, str]]:
        """Extract legal references with headings."""
        results = []
        
        legal_patterns = [
            (r'(?:Section|Sec\.|§)\s*\d+(?:\.\d+)*(?:\([a-z0-9]+\))?', "Section"),
            (r'\d+\s+U\.?S\.?C\.?\s*§?\s*\d+', "USC Reference"),
            (r'IRC\s*§?\s*\d+(?:\([a-z0-9]+\))?', "IRC Section"),
            (r'\d+\s+CFR\s+\d+(?:\.\d+)*', "CFR Reference"),
            (r'Title\s+\d+(?:,\s*(?:Section|§)\s*\d+)?', "Title Reference"),
            (r'(?:Pub\.|Public)\s*L\.\s*(?:No\.\s*)?\d+-\d+', "Public Law")
        ]
        
        lines = text.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if len(line_stripped) < 10 or len(line_stripped) > 200:
                continue
                
            for pattern, heading in legal_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    results.append((heading, line_stripped))
                    break
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for heading, content in results:
            if content not in seen:
                seen.add(content)
                unique_results.append((heading, content))
        
        return unique_results[:25]
    
    def extract_other_details(self, text: str, doc) -> List[Tuple[str, str]]:
        """Extract other important details with headings."""
        results = []
        
        lines = text.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if len(line_stripped) > 20 and len(line_stripped) < 150:
                if re.match(r'^(?:\d+\.|[A-Z]\.|•|\*)', line_stripped):
                    results.append(("Key Point", line_stripped))
        
        # Extract entities
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(results) < 30:
                results.append(("Person", ent.text))
            elif ent.label_ == "ORG" and len(results) < 30:
                if 'court' not in ent.text.lower():
                    results.append(("Organization", ent.text))
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for heading, content in results:
            if content not in seen:
                seen.add(content)
                unique_results.append((heading, content))
        
        return unique_results[:15]
    
    def analyze_document(self, text: str) -> Dict[str, List[Tuple[str, str]]]:
        """Main analysis function."""
        if not text or len(text.strip()) < 100:
            raise ValueError("Document text is too short or empty")
        
        max_length = 1000000
        if len(text) > max_length:
            text = text[:max_length]
        
        doc = nlp(text)
        
        results = {
            'case_info': self.extract_case_info(text, doc),
            'tax_amounts': self.extract_tax_amounts(text, doc),
            'dates': self.extract_dates(text, doc),
            'legal_refs': self.extract_legal_references(text),
            'other': self.extract_other_details(text, doc)
        }
        
        return results


analyzer = DocumentAnalyzer()


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML page."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tax Document Phrase Extractor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .upload-section {
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        
        .drop-zone {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9ff;
        }
        
        .drop-zone:hover {
            background: #eef2ff;
            border-color: #764ba2;
        }
        
        .drop-zone.dragging {
            background: #e0e7ff;
            border-color: #4f46e5;
        }
        
        .drop-zone-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .drop-zone-text {
            font-size: 1.2em;
            color: #333;
            margin-bottom: 10px;
        }
        
        .drop-zone-subtext {
            color: #666;
            font-size: 0.9em;
        }
        
        #fileInput {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            margin-top: 15px;
            transition: transform 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .spinner {
            display: none;
            margin: 40px auto;
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results {
            display: none;
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }
        
        .results-title {
            font-size: 1.8em;
            color: #333;
        }
        
        .category {
            margin-bottom: 40px;
        }
        
        .category-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            font-size: 1.4em;
            color: #333;
        }
        
        .category-icon {
            margin-right: 10px;
            font-size: 1.2em;
        }
        
        .category-content {
            background: #f8f9ff;
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid #667eea;
        }
        
        /* Flexbox grid for related items */
        .items-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        
        .item-card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            flex: 1 1 calc(50% - 6px);
            min-width: 280px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .item-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        }
        
        .item-heading {
            font-weight: 700;
            color: #667eea;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 2px solid #e0e7ff;
        }
        
        .item-content {
            color: #333;
            line-height: 1.6;
            font-size: 0.95em;
        }
        
        .empty-state {
            color: #999;
            font-style: italic;
            padding: 20px;
            text-align: center;
        }
        
        .error {
            background: #fee;
            color: #c33;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }
        
        .copy-btn {
            background: #10b981;
            color: white;
            border: none;
            padding: 8px 20px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            margin-right: 10px;
        }
        
        .copy-btn:hover {
            background: #059669;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            .upload-section, .results {
                padding: 20px;
            }
            
            .results-header {
                flex-direction: column;
                gap: 15px;
            }
            
            .item-card {
                flex: 1 1 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📄 Tax Document Phrase Extractor</h1>
            <p>Upload tax or legal documents to automatically extract and categorize key information</p>
        </div>
        
        <div class="upload-section">
            <div class="drop-zone" id="dropZone">
                <div class="drop-zone-icon">📁</div>
                <div class="drop-zone-text">Drag & drop your document here</div>
                <div class="drop-zone-subtext">or click to browse (PDF or DOCX, max 50MB)</div>
                <button class="btn" onclick="document.getElementById('fileInput').click()">
                    Choose File
                </button>
            </div>
            <input type="file" id="fileInput" accept=".pdf,.docx" />
            
            <div class="spinner" id="spinner"></div>
            <div class="error" id="error"></div>
        </div>
        
        <div class="results" id="results">
            <div class="results-header">
                <h2 class="results-title">Extracted Information</h2>
                <div>
                    <button class="copy-btn" onclick="copyResults()">📋 Copy All</button>
                    <button class="btn" onclick="resetUpload()">🔄 Analyze Another</button>
                </div>
            </div>
            
            <div id="resultsContent"></div>
        </div>
    </div>
    
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const spinner = document.getElementById('spinner');
        const error = document.getElementById('error');
        const results = document.getElementById('results');
        const resultsContent = document.getElementById('resultsContent');
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragging');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragging');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragging');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        async function handleFile(file) {
            if (!file.name.match(/\.(pdf|docx)$/i)) {
                showError('Please upload a PDF or DOCX file');
                return;
            }
            
            if (file.size > 50 * 1024 * 1024) {
                showError('File size must be less than 50MB');
                return;
            }
            
            error.style.display = 'none';
            results.style.display = 'none';
            spinner.style.display = 'block';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Analysis failed');
                }
                
                const data = await response.text();
                displayResults(data);
            } catch (err) {
                showError(err.message);
            } finally {
                spinner.style.display = 'none';
            }
        }
        
        function displayResults(text) {
            resultsContent.innerHTML = text;
            results.style.display = 'block';
            results.scrollIntoView({ behavior: 'smooth' });
        }
        
        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
        }
        
        function resetUpload() {
            fileInput.value = '';
            results.style.display = 'none';
            error.style.display = 'none';
            document.querySelector('.upload-section').scrollIntoView({ behavior: 'smooth' });
        }
        
        function copyResults() {
            const text = resultsContent.innerText;
            navigator.clipboard.writeText(text).then(() => {
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = '✓ Copied!';
                setTimeout(() => {
                    btn.textContent = originalText;
                }, 2000);
            });
        }
    </script>
</body>
</html>
    """


@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """Analyze uploaded document and return formatted results."""
    try:
        if not file.filename or not file.filename.lower().endswith(('.pdf', '.docx')):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
        
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if file.filename.lower().endswith('.pdf'):
            text = analyzer.extract_text_from_pdf(file_bytes)
        else:
            text = analyzer.extract_text_from_docx(file_bytes)
        
        results = analyzer.analyze_document(text)
        html = format_results_html(results)
        
        return HTMLResponse(content=html)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


def format_results_html(results: Dict[str, List[Tuple[str, str]]]) -> str:
    """Format analysis results as HTML with highlighted headings in flexbox cards."""
    categories = [
        ('case_info', '🏛️ Case Information', 'No case information found'),
        ('tax_amounts', '💰 Tax Amounts / Penalties', 'No tax amounts found'),
        ('dates', '📅 Important Dates / Deadlines', 'No dates found'),
        ('legal_refs', '⚖️ Legal References', 'No legal references found'),
        ('other', '🧾 Other Key Details', 'No additional details found')
    ]
    
    html_parts = []
    
    for key, title, empty_msg in categories:
        items = results.get(key, [])
        
        html_parts.append(f'<div class="category">')
        html_parts.append(f'<div class="category-header">')
        html_parts.append(f'<span class="category-icon">{title.split()[0]}</span>')
        html_parts.append(f'<span>{" ".join(title.split()[1:])}</span>')
        html_parts.append(f'</div>')
        html_parts.append(f'<div class="category-content">')
        
        if items:
            html_parts.append(f'<div class="items-grid">')
            for heading, content in items:
                html_parts.append(f'<div class="item-card">')
                html_parts.append(f'<div class="item-heading">{heading}</div>')
                html_parts.append(f'<div class="item-content">{content}</div>')
                html_parts.append(f'</div>')
            html_parts.append(f'</div>')
        else:
            html_parts.append(f'<div class="empty-state">{empty_msg}</div>')
        
        html_parts.append(f'</div>')
        html_parts.append(f'</div>')
    
    return '\n'.join(html_parts)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)