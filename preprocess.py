import os
import sys
import json
import re
import logging
import io
import time
from datetime import datetime, timedelta, timedelta
from filelock import FileLock, Timeout
from PIL import Image
import pytesseract
import pandas as pd
import glob
import hashlib
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY not set in .env file")
    raise ValueError("GOOGLE_API_KEY not set in .env file")
genai.configure(api_key=GOOGLE_API_KEY)

# File Management
METADATA_FILE = os.path.join(os.getcwd(), "pdf_metadata.json")
ALL_INVOICES_FILE = os.path.join(os.getcwd(), "all_invoices.json")

def init_metadata_file():
    """Initialize pdf_metadata.json if it doesn't exist."""
    lock = FileLock(METADATA_FILE + ".lock", timeout=10)
    try:
        with lock:
            if not os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, 'w') as f:
                    json.dump({"files": []}, f, indent=4)
                logging.debug("Initialized metadata file: %s", METADATA_FILE)
    except Timeout:
        logging.error("Could not acquire lock for %s", METADATA_FILE)
        raise
    except (PermissionError, IOError) as e:
        logging.error("Error creating metadata file %s: %s", METADATA_FILE, str(e))
        raise

def load_metadata():
    """Load metadata from pdf_metadata.json."""
    init_metadata_file()
    lock = FileLock(METADATA_FILE + ".lock", timeout=10)
    try:
        with lock:
            if not os.path.exists(METADATA_FILE):
                return []
            with open(METADATA_FILE, 'r') as f:
                return json.load(f).get("files", [])
    except Timeout:
        logging.error("Could not acquire lock for %s", METADATA_FILE)
        return []
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        logging.error("Error loading metadata: %s", str(e))
        return []

def save_metadata(metadata):
    """Save metadata to pdf_metadata.json."""
    lock = FileLock(METADATA_FILE + ".lock", timeout=10)
    try:
        with lock:
            with open(METADATA_FILE, 'w') as f:
                json.dump({"files": metadata}, f, indent=4)
            logging.debug("Saved metadata to %s", METADATA_FILE)
    except Timeout:
        logging.error("Could not acquire lock for %s", METADATA_FILE)
        raise
    except (PermissionError, IOError) as e:
        logging.error("Error saving metadata: %s", str(e))
        raise

def init_all_invoices_file():
    """Initialize all_invoices.json if it doesn't exist."""
    lock = FileLock(ALL_INVOICES_FILE + ".lock", timeout=10)
    try:
        with lock:
            if not os.path.exists(ALL_INVOICES_FILE):
                with open(ALL_INVOICES_FILE, 'w') as f:
                    json.dump({"invoices": []}, f, indent=4)
                logging.debug("Initialized all invoices file: %s", ALL_INVOICES_FILE)
    except Timeout:
        logging.error("Could not acquire lock for %s", ALL_INVOICES_FILE)
        raise
    except (PermissionError, IOError) as e:
        logging.error("Error creating all invoices file %s: %s", ALL_INVOICES_FILE, str(e))
        raise

def load_all_invoices():
    """Load all invoices from JSON file."""
    init_all_invoices_file()
    lock = FileLock(ALL_INVOICES_FILE + ".lock", timeout=10)
    try:
        with lock:
            with open(ALL_INVOICES_FILE, 'r') as f:
                return json.load(f).get("invoices", [])
    except Timeout:
        logging.error("Could not acquire lock for %s", ALL_INVOICES_FILE)
        return []
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        logging.error("Error loading all invoices: %s", str(e))
        return []

def save_all_invoices(invoices):
    """Save all invoices to JSON file."""
    lock = FileLock(ALL_INVOICES_FILE + ".lock", timeout=10)
    try:
        with lock:
            with open(ALL_INVOICES_FILE, 'w') as f:
                json.dump({"invoices": invoices}, f, indent=4)
            logging.debug("Saved all invoices")
    except Timeout:
        logging.error("Could not acquire lock for %s", ALL_INVOICES_FILE)
        raise
    except (PermissionError, IOError) as e:
        logging.error("Error saving all invoices: %s", str(e))
        raise

def clean_text(text):
    """Clean extracted text by removing extra spaces and non-ASCII characters."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def clean_name(name):
    """Clean name by removing titles."""
    if not name or not isinstance(name, str):
        return ""
    titles = ["mr.", "ms.", "mrs.", "dr.", "miss", "shri", "smt", "m/s"]
    name = name.lower().strip()
    for title in titles:
        if name.startswith(title):
            name = name[len(title):].strip()
    return name.title()

def extract_text_from_image(image):
    """Extract text from an image file using OCR with enhanced settings."""
    try:
        gray = image.convert("L")
        custom_config = r'--oem 3 --psm 6'  # Default OCR engine mode 3, page segmentation mode 6
        text = pytesseract.image_to_string(gray, config=custom_config)
        return clean_text(text)
    except Exception as e:
        logging.error("Error extracting text from image: %s", str(e))
        return ""

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF with improved page handling."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # Increase resolution for better OCR
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            full_text += extract_text_from_image(img) + "\n\n"
        return clean_text(full_text)
    except Exception as e:
        logging.error("Error extracting text from PDF %s: %s", pdf_path, str(e))
        return ""

def extract_csv_text(file_path):
    """Extract text from a CSV file with error handling for malformed data."""
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')  # Skip malformed lines
        return clean_text(df.to_string(index=False))
    except (FileNotFoundError, PermissionError, pd.errors.ParserError) as e:
        logging.error("Error extracting CSV text from %s: %s", file_path, str(e))
        return ""

def clean_gemini_response(text):
    """Clean Gemini response by removing markdown code blocks and ensuring valid JSON."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE)
    text = text.strip()
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_invoice_data_with_gemini(cleaned_text, retries=5, backoff_factor=2):
    """Use Gemini model to extract invoice-specific data with retry logic."""
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""Extract invoice-related data from the following text in a structured JSON format. Include fields: invoice_number, date, due_date, issuer, recipient, items (list with description, quantity, price), subtotal, tax, total, and bill_to_name (name under 'Bill To' or 'Issued To'). If due_date is missing, infer it as 30 days after date if date is present, otherwise set to null. Convert dates to YYYY-MM-DD format. If any field is not found, set it to null. Ignore non-invoice-related information. If no invoice data is found, return {{'error': 'No invoice data found'}}.

Text:
{cleaned_text}
"""
            response = model.generate_content(prompt)
            logging.debug("Gemini raw response: %s", response.text)
            cleaned_response = clean_gemini_response(response.text)
            logging.debug("Cleaned Gemini response: %s", cleaned_response)
            try:
                data = json.loads(cleaned_response)
                if data.get("date") and not data.get("due_date"):
                    try:
                        invoice_date = datetime.strptime(data["date"], "%Y-%m-%d")
                        due_date = invoice_date + timedelta(days=30)
                        data["due_date"] = due_date.strftime("%Y-%m-%d")
                    except ValueError:
                        logging.warning("Could not compute due_date for date: %s", data["date"])
                return data
            except json.JSONDecodeError as e:
                logging.error("Gemini response is not valid JSON: %s", cleaned_response)
                if attempt < retries - 1:
                    retry_delay = backoff_factor ** attempt
                    logging.warning("Retrying JSON parsing in %d seconds (attempt %d/%d)", retry_delay, attempt + 1, retries)
                    time.sleep(retry_delay)
                    continue
                return {"error": f"Invalid JSON response after cleaning: {cleaned_response}"}
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                retry_delay = backoff_factor ** attempt
                logging.warning("Rate limit hit, retrying in %d seconds (attempt %d/%d)", retry_delay, attempt + 1, retries)
                time.sleep(retry_delay)
                continue
            logging.error("Error processing with Gemini: %s", str(e))
            return {"error": f"Error processing with Gemini: {str(e)}"}
    return {"error": "Max retries reached due to rate limits or invalid responses"}

def get_file_hash(file_path):
    """Compute MD5 hash of a file."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except (FileNotFoundError, PermissionError, IOError) as e:
        logging.error("Error computing hash for %s: %s", file_path, str(e))
        return None

def preprocess_files(username):
    """Process files in Uploads/{username}/* and save metadata to pdf_metadata.json and all_invoices.json."""
    if not username or not re.match(r'^[a-zA-Z0-9_-]+$', username):
        logging.error("Invalid username: %s", username)
        return {"success": False, "message": "Invalid username"}

    upload_dir = os.path.join(os.getcwd(), 'Uploads', username)
    os.makedirs(upload_dir, exist_ok=True)
    logging.debug("Processing upload directory: %s", upload_dir)

    file_patterns = [
        os.path.join(upload_dir, "*.pdf"),
        os.path.join(upload_dir, "*.png"),
        os.path.join(upload_dir, "*.jpg"),
        os.path.join(upload_dir, "*.jpeg"),
        os.path.join(upload_dir, "*.csv")
    ]
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern))
        logging.debug("Found files for pattern %s: %s", pattern, glob.glob(pattern))

    current_metadata = load_metadata()
    all_invoices = load_all_invoices()
    new_metadata = []
    existing_hashes = {m["file_hash"] for m in current_metadata if "file_hash" in m}
    processed_count = 0

    for file_path in files:
        filename = os.path.basename(file_path)
        file_type = filename.lower().split(".")[-1]
        logging.debug("Processing file: %s", filename)

        file_hash = get_file_hash(file_path)
        if not file_hash or file_hash in existing_hashes:
            logging.debug("Skipping already processed file: %s", filename)
            continue

        try:
            full_content = ""
            if file_type == "pdf":
                full_content = extract_text_from_pdf(file_path)
            elif file_type in ["png", "jpg", "jpeg"]:
                with Image.open(file_path) as img:
                    full_content = extract_text_from_image(img)
            elif file_type == "csv":
                full_content = extract_csv_text(file_path)
            else:
                logging.warning("Unsupported file type: %s", file_type)
                continue

            if not full_content:
                logging.warning("No text extracted from %s", filename)
                continue

            extracted_data = extract_invoice_data_with_gemini(full_content)
            if "error" in extracted_data:
                logging.error("Failed to extract invoice data from %s: %s", filename, extracted_data["error"])
                continue

            if clean_name(extracted_data.get("bill_to_name", "")) != clean_name(username):
                logging.warning("Invoice %s not addressed to %s", filename, username)
                continue

            metadata_entry = {
                "username": username,
                "filename": filename,
                "file_hash": file_hash,
                "invoice_data": extracted_data,
                "timestamp": datetime.now().isoformat()
            }
            new_metadata.append(metadata_entry)
            all_invoices.append({
                "username": username,
                "filename": filename,
                "invoice_number": extracted_data.get("invoice_number"),
                "data": extracted_data,
                "file_hash": file_hash,
                "timestamp": datetime.now().isoformat()
            })
            processed_count += 1
            logging.debug("Extracted metadata for %s", filename)
        except Exception as e:
            logging.error("Error processing %s: %s", filename, str(e))

    if new_metadata:
        current_metadata.extend(new_metadata)
        save_metadata(current_metadata)
        save_all_invoices(all_invoices)
        logging.info("Processed %d new files for %s", processed_count, username)
    else:
        logging.info("No new files to process for %s", username)

    return {"success": True, "message": f"Processed {processed_count} files for {username}"}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: python preprocess.py <username>")
        sys.exit(1)
    username = sys.argv[1]
    result = preprocess_files(username)
    print(json.dumps(result))