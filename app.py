import io
import os
import json
import re
import logging
import subprocess
import shlex
import bcrypt
import uuid
import glob
import hashlib
import time
import pandas as pd
from datetime import datetime, date, timedelta
from filelock import FileLock, Timeout
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv


# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
if app.secret_key == "your-secret-key":
    logging.warning("Using default FLASK_SECRET_KEY. Set a secure key in .env")

# Ensure template folder exists
os.makedirs(os.path.join(app.root_path, 'templates'), exist_ok=True)

# Debug template folder
logging.debug("Template folder: %s", app.template_folder)
try:
    logging.debug("Templates found: %s", os.listdir(os.path.join(app.root_path, 'templates')))
except OSError as e:
    logging.error("Error listing templates: %s", str(e))

# Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY not set in .env file")
    raise ValueError("GOOGLE_API_KEY not set in .env file")
genai.configure(api_key=GOOGLE_API_KEY)

# User Credentials with Roles
USERS = {
    "Raju": {"password": bcrypt.hashpw("123".encode(), bcrypt.gensalt()).decode(), "role": "user"},
    "divya": {"password": bcrypt.hashpw("divya".encode(), bcrypt.gensalt()).decode(), "role": "manager"},
}

# File Management
APPROVAL_REQUESTS_FILE = os.path.join(os.getcwd(), "approval_requests.json")
ALL_INVOICES_FILE = os.path.join(os.getcwd(), "all_invoices.json")

def init_approval_requests_file():
    """Initialize approval requests file if it doesn't exist."""
    lock = FileLock(APPROVAL_REQUESTS_FILE + ".lock", timeout=10)
    try:
        with lock:
            if not os.path.exists(APPROVAL_REQUESTS_FILE):
                with open(APPROVAL_REQUESTS_FILE, 'w') as f:
                    json.dump({"requests": []}, f, indent=4)
                logging.debug("Initialized approval requests file: %s", APPROVAL_REQUESTS_FILE)
    except Timeout:
        logging.error("Could not acquire lock for %s", APPROVAL_REQUESTS_FILE)
        raise
    except (PermissionError, IOError) as e:
        logging.error("Error creating approval requests file %s: %s", APPROVAL_REQUESTS_FILE, str(e))
        raise

def load_approval_requests():
    """Load approval requests from JSON file."""
    init_approval_requests_file()
    lock = FileLock(APPROVAL_REQUESTS_FILE + ".lock", timeout=10)
    try:
        with lock:
            with open(APPROVAL_REQUESTS_FILE, 'r') as f:
                return json.load(f).get("requests", [])
    except Timeout:
        logging.error("Could not acquire lock for %s", APPROVAL_REQUESTS_FILE)
        return []
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        logging.error("Error loading approval requests: %s", str(e))
        return []

def save_approval_requests(requests):
    """Save approval requests to JSON file."""
    lock = FileLock(APPROVAL_REQUESTS_FILE + ".lock", timeout=10)
    try:
        with lock:
            with open(APPROVAL_REQUESTS_FILE, 'w') as f:
                json.dump({"requests": requests}, f, indent=4)
            logging.debug("Saved approval requests")
    except Timeout:
        logging.error("Could not acquire lock for %s", APPROVAL_REQUESTS_FILE)
        raise
    except (PermissionError, IOError) as e:
        logging.error("Error saving approval requests: %s", str(e))
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

def extract_invoice_total(invoice_data):
    """Extract total amount from invoice data."""
    try:
        return float(invoice_data.get("total", 0))
    except (ValueError, TypeError) as e:
        logging.error("Invalid total format: %s", str(e))
        return None

def extract_invoice_number(invoice_data):
    """Extract invoice number from invoice data."""
    try:
        return invoice_data.get("invoice_number")
    except Exception as e:
        logging.error("Error extracting invoice number: %s", str(e))
        return None

# Invoice Data Cache Management
def get_invoice_data_file(username):
    """Get path to user's invoice data JSON file."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        logging.error("Invalid username format: %s", username)
        raise ValueError("Invalid username format")
    return os.path.join(os.getcwd(), f"invoices_data_{username}.json")

def init_invoice_data_file(username):
    """Initialize invoice data file if it doesn't exist."""
    if session.get('role') == 'manager':
        return  # Skip creating invoice data file for managers
    invoice_data_file = get_invoice_data_file(username)
    lock = FileLock(invoice_data_file + ".lock", timeout=10)
    try:
        with lock:
            if not os.path.exists(invoice_data_file):
                with open(invoice_data_file, 'w') as f:
                    json.dump({"files": {}}, f, indent=4)
                logging.debug("Initialized invoice data file: %s", invoice_data_file)
    except Timeout:
        logging.error("Could not acquire lock for %s", invoice_data_file)
        raise
    except (PermissionError, IOError) as e:
        logging.error("Error creating invoice data file %s: %s", invoice_data_file, str(e))
        raise

def load_invoice_data(username):
    """Load invoice data from JSON file."""
    if session.get('role') == 'manager':
        return {}  # No invoice data file for managers
    invoice_data_file = get_invoice_data_file(username)
    init_invoice_data_file(username)
    lock = FileLock(invoice_data_file + ".lock", timeout=10)
    try:
        with lock:
            with open(invoice_data_file, 'r') as f:
                return json.load(f).get("files", {})
    except Timeout:
        logging.error("Could not acquire lock for %s", invoice_data_file)
        return {}
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        logging.error("Error loading invoice data: %s", str(e))
        return {}

def save_invoice_data(username, invoice_data):
    """Save invoice data to JSON file."""
    if session.get('role') == 'manager':
        return  # Skip saving invoice data for managers
    invoice_data_file = get_invoice_data_file(username)
    lock = FileLock(invoice_data_file + ".lock", timeout=10)
    try:
        with lock:
            with open(invoice_data_file, 'w') as f:
                json.dump({"files": invoice_data}, f, indent=4)
            logging.debug("Saved invoice data for %s", username)
    except Timeout:
        logging.error("Could not acquire lock for %s", invoice_data_file)
        raise
    except (PermissionError, IOError) as e:
        logging.error("Error saving invoice data for %s: %s", username, str(e))
        raise

# Chat History File Management
def get_chat_history_file(username):
    """Get path to user's chat history JSON file."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        logging.error("Invalid username format: %s", username)
        raise ValueError("Invalid username format")
    return os.path.join(os.getcwd(), f"chat_history_{username}.json")

def init_chat_history_file(username):
    """Initialize chat history file if it doesn't exist."""
    history_file = get_chat_history_file(username)
    lock = FileLock(history_file + ".lock", timeout=10)
    try:
        with lock:
            if not os.path.exists(history_file):
                with open(history_file, 'w') as f:
                    json.dump({"username": username, "sessions": []}, f, indent=4)
                logging.debug("Initialized chat history file: %s", history_file)
    except Timeout:
        logging.error("Could not acquire lock for %s", history_file)
        raise
    except (PermissionError, IOError) as e:
        logging.error("Error creating chat history file %s: %s", history_file, str(e))
        raise

def load_chat_history(username):
    """Load chat history from JSON file."""
    history_file = get_chat_history_file(username)
    init_chat_history_file(username)
    lock = FileLock(history_file + ".lock", timeout=10)
    try:
        with lock:
            with open(history_file, 'r') as f:
                data = json.load(f)
                sessions = {}
                for idx, session_data in enumerate(data.get("sessions", [])):
                    session_id = session_data.get("session_id", str(uuid.uuid4()))
                    sessions[session_id] = session_data.get("messages", [])
                return sessions
    except Timeout:
        logging.error("Could not acquire lock for %s", history_file)
        return {}
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        logging.error("Error loading chat history for %s: %s", username, str(e))
        return {}

def save_chat_history(username, sessions):
    """Save chat history to JSON file."""
    history_file = get_chat_history_file(username)
    lock = FileLock(history_file + ".lock", timeout=10)
    try:
        with lock:
            formatted_sessions = [
                {
                    "session_id": session_id,
                    "session_label": f"Session {idx + 1} ({messages[0]['timestamp'][:10] if messages else datetime.now().strftime('%Y-%m-%d')})",
                    "messages": messages
                }
                for idx, (session_id, messages) in enumerate(sessions.items())
            ]
            with open(history_file, 'w') as f:
                json.dump({"username": username, "sessions": formatted_sessions}, f, indent=4)
            logging.debug("Saved chat history for %s", username)
    except Timeout:
        logging.error("Could not acquire lock for %s", history_file)
        raise
    except (PermissionError, IOError) as e:
        logging.error("Error saving chat history for %s: %s", username, str(e))
        raise

# Utility Functions
def clean_text(text):
    """Clean extracted text by removing extra spaces and non-ASCII characters."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def clean_name(name):
    """Clean name by removing titles."""
    if not name:
        return ""
    titles = ["mr.", "ms.", "mrs.", "dr.", "miss", "shri", "smt", "m/s"]
    name = name.lower().strip()
    for title in titles:
        if name.startswith(title):
            name = name[len(title):].strip()
    return name.title()

def get_file_hash(file_path):
    """Compute MD5 hash of a file."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except (FileNotFoundError, PermissionError, IOError) as e:
        logging.error("Error computing hash for %s: %s", file_path, str(e))
        return None

def extract_pdf_text(file_path):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text += pytesseract.image_to_string(img.convert("L")) + "\n\n"
        return clean_text(text)
    except Exception as e:
        logging.error("Error extracting PDF text from %s: %s", file_path, str(e))
        return ""

def extract_image_text(file_path):
    """Extract text from an image file using OCR."""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image.convert("L"))
        return clean_text(text)
    except (FileNotFoundError, PermissionError, IOError) as e:
        logging.error("Error extracting image text from %s: %s", file_path, str(e))
        return ""

def extract_csv_text(file_path):
    """Extract text from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return clean_text(df.to_string(index=False))
    except (FileNotFoundError, PermissionError, pd.errors.ParserError) as e:
        logging.error("Error extracting CSV text from %s: %s", file_path, str(e))
        return ""

def clean_gemini_response(text):
    """Clean Gemini response by removing markdown code blocks."""
    if not text:
        return ""
    text = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE)
    text = text.strip()
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_invoice_data_with_gemini(cleaned_text, retries=3, backoff_factor=2):
    """Use Gemini model to extract invoice-specific data with retry logic."""
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""Extract invoice-related data from the following text in a structured JSON format. Include fields: invoice_number, date, due_date, issuer, recipient, items (list with description, quantity, price), subtotal, tax, total, and bill_to_name (name under 'Bill To' or 'Issued To'). If due_date is not found, set it to 30 days after the invoice date in YYYY-MM-DD format. Convert any date formats to YYYY-MM-DD. If any field is not found, set it to null. Ignore non-invoice-related information. If no invoice data is found, return {{'error': 'No invoice data found'}}.

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
                        logging.debug("Set default due_date: %s", data["due_date"])
                    except ValueError:
                        logging.warning("Could not compute default due_date for date: %s", data["date"])
                return data
            except json.JSONDecodeError as e:
                logging.error("Gemini response is not valid JSON: %s", cleaned_response)
                return {"error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                retry_delay = backoff_factor ** attempt
                logging.warning("Rate limit hit, retrying in %d seconds (attempt %d/%d)", retry_delay, attempt + 1, retries)
                time.sleep(retry_delay)
                continue
            logging.error("Error processing with Gemini: %s", str(e))
            return {"error": f"Error processing with Gemini: {str(e)}"}
    return {"error": "Max retries reached due to rate limits"}

def ask_question_about_invoice(question, invoice_data):
    """Use Gemini model to answer invoice-related questions."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        current_date = datetime.strptime("2025-06-18", "%Y-%m-%d").strftime("%Y-%m-%d")
        prompt = f"""You are a smart assistant specializing in invoice data. Answer the following question only if it is directly related to the provided invoice data. If the question is unrelated to invoices, respond with: 'Please ask a question related to invoice data.' If no invoice data is available, respond with: 'No invoice data available to answer this question.' For due date questions, calculate days remaining using the current date: {current_date}.

Invoice Data:
{invoice_data if invoice_data else 'No invoice data available.'}

Question:
{question}
"""
        response = model.generate_content(prompt)
        return response.text if response.text else "No answer provided"
    except Exception as e:
        logging.error("Error answering question: %s", str(e))
        return f"Error answering question: {str(e)}"

def check_invoice_status(username, invoice_id=None):
    """Check the status of invoices for a user from approval_requests.json."""
    try:
        requests = load_approval_requests()
        user_requests = [req for req in requests if req["user"] == username]
        if not user_requests:
            return "You have no approval requests."
        if invoice_id:
            for req in user_requests:
                if req["invoice_id"] == invoice_id:
                    status = req["status"].capitalize()
                    if req["status"] == "pending":
                        return f"Invoice {invoice_id} status: {status} (Raised to {req['manager']} on {req['timestamp'][:10]})"
                    return f"Invoice {invoice_id} status: {status} by {req['manager']} on {req['timestamp'][:10]}"
            return f"No approval request found for invoice {invoice_id}."
        else:
            status_summary = ["Your invoice approval statuses:"]
            for req in user_requests:
                status = req["status"].capitalize()
                if req["status"] == "pending":
                    status_summary.append(f"Invoice {req['invoice_id']}: {status} (Raised to {req['manager']} on {req['timestamp'][:10]})")
                else:
                    status_summary.append(f"Invoice {req['invoice_id']}: {status} by {req['manager']} on {req['timestamp'][:10]}")
            return "\n".join(status_summary)
    except Exception as e:
        logging.error("Error checking invoice status for %s: %s", username, str(e))
        return f"Error checking invoice status: {str(e)}"

def load_metadata():
    """Load metadata from pdf_metadata.json."""
    metadata_file = os.path.join(os.getcwd(), "pdf_metadata.json")
    lock = FileLock(metadata_file + ".lock", timeout=10)
    try:
        with lock:
            if not os.path.exists(metadata_file):
                return {"files": []}
            with open(metadata_file, 'r') as f:
                return json.load(f)
    except Timeout:
        logging.error("Could not acquire lock for %s", metadata_file)
        return {"files": []}
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        logging.error("Error loading metadata: %s", str(e))
        return {"files": []}

def calculate_days_remaining(invoice_id, username, current_date=None):
    """Calculate days remaining for an invoice's due date."""
    if current_date is None:
        current_date = datetime.now().date()
    logging.debug("Calculating days remaining for invoice %s, user %s, current date %s", invoice_id, username, current_date)
    
    try:
        all_invoices = load_all_invoices()
        invoice = next((inv for inv in all_invoices if inv["username"] == username and inv["invoice_number"] == invoice_id), None)
        if not invoice:
            logging.warning("Invoice %s not found for user %s", invoice_id, username)
            return {"success": False, "message": f"Invoice {invoice_id} not found"}
        
        due_date = invoice.get("data", {}).get("due_date")
        if not due_date:
            logging.warning("No due date found for invoice %s", invoice_id)
            return {"success": False, "message": f"No due date found for invoice {invoice_id}"}
        
        date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]
        due_date_obj = None
        for fmt in date_formats:
            try:
                due_date_obj = datetime.strptime(due_date, fmt).date()
                break
            except ValueError:
                continue
        
        if not due_date_obj:
            logging.error("Invalid due date format for %s: %s", invoice_id, due_date)
            return {"success": False, "message": f"Invalid due date format for {invoice_id}: {due_date}"}
        
        days_remaining = (due_date_obj - current_date).days
        overdue = days_remaining < 0
        logging.info("Invoice %s due on %s, %s days remaining, overdue: %s", invoice_id, due_date_obj, days_remaining, overdue)
        return {
            "success": True,
            "message": f"Invoice {invoice_id} is due on {due_date_obj.strftime('%Y-%m-%d')}. "
                       f"{'Overdue' if overdue else f'{days_remaining} days remaining'}."
        }
    except Exception as e:
        logging.error("Error calculating days remaining for %s: %s", invoice_id, str(e))
        return {"success": False, "message": f"Error calculating days remaining: {str(e)}"}

def generate_due_date_message(username, current_date=None):
    """Generate a message about the invoice with the closest due date."""
    if current_date is None:
        current_date = datetime.now().date()
    
    try:
        all_invoices = load_all_invoices()
        user_invoices = [inv for inv in all_invoices if inv["username"].lower() == username.lower()]
        logging.debug("Found %d invoices for user %s", len(user_invoices), username)
        
        if not user_invoices:
            logging.info("No invoices found for user %s", username)
            return "You have no invoices with due dates available."

        closest_invoice = None
        min_days_diff = float('inf')
        date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%d/%m/%Y"]

        for inv in user_invoices:
            due_date = inv.get("data", {}).get("due_date")
            invoice_id = inv.get("invoice_number", "Unknown")
            if not due_date:
                logging.debug("No due date for invoice %s (user %s)", invoice_id, username)
                continue
            
            due_date_obj = None
            for fmt in date_formats:
                try:
                    due_date_obj = datetime.strptime(due_date, fmt).date()
                    break
                except ValueError:
                    continue
            
            if not due_date_obj:
                logging.warning("Invalid due date format for invoice %s: %s", invoice_id, due_date)
                continue
            
            days_diff = abs((due_date_obj - current_date).days)
            # Update if this invoice has a smaller day difference, or same difference but future date
            if days_diff < min_days_diff or (days_diff == min_days_diff and due_date_obj >= current_date):
                min_days_diff = days_diff
                closest_invoice = inv
                logging.debug("Updated closest invoice: %s, due %s, days diff %d", invoice_id, due_date_obj, days_diff)

        if closest_invoice:
            invoice_id = closest_invoice.get("invoice_number", "Unknown")
            due_date = closest_invoice.get("data", {}).get("due_date", "Unknown")
            try:
                due_date_obj = datetime.strptime(due_date, "%Y-%m-%d").date()
            except ValueError:
                for fmt in date_formats:
                    try:
                        due_date_obj = datetime.strptime(due_date, fmt).date()
                        break
                    except ValueError:
                        continue
                else:
                    logging.error("Failed to parse due date %s for invoice %s", due_date, invoice_id)
                    return "Error processing due date information."
            
            days_remaining = (due_date_obj - current_date).days
            overdue = days_remaining < 0
            logging.info("Closest invoice %s for %s: due %s, %s", invoice_id, username, due_date, 
                         "overdue" if overdue else f"{days_remaining} days remaining")
            
            if overdue:
                return f"Reminder: Invoice {invoice_id} was due on {due_date} and is overdue."
            else:
                return f"Reminder: Invoice {invoice_id} is due on {due_date} ({days_remaining} days remaining)."
        else:
            logging.info("No valid due dates found for user %s invoices", username)
            return "No valid due dates found for your invoices."
    
    except Exception as e:
        logging.error("Error generating due date message for %s: %s", username, str(e))
        return "Unable to retrieve due date information at this time."

def process_uploaded_files(username, role):
    """Process files in the Uploads folder and integrate with pdf_metadata.json and all_invoices.json."""
    if not username or not re.match(r'^[a-zA-Z0-9_-]+$', username):
        logging.warning("Invalid or empty username provided: %s", username)
        return []

    # Only create upload directory for users, not managers
    upload_dir = os.path.join(os.getcwd(), 'Uploads', username) if role == 'user' else os.path.join(os.getcwd(), 'Uploads')
    if role == 'user':
        os.makedirs(upload_dir, exist_ok=True)
    logging.debug("Checking upload directory: %s", upload_dir)

    session.setdefault('invoices_data', {})
    cached_invoice_data = load_invoice_data(username) if role == 'user' else {}
    metadata = load_metadata()
    user_metadata = [m for m in metadata["files"] if clean_name(m["username"]) == clean_name(username)]
    all_invoices = load_all_invoices()

    file_patterns = [
        os.path.join(upload_dir, f"*.pdf"),
        os.path.join(upload_dir, f"*.png"),
        os.path.join(upload_dir, f"*.jpg"),
        os.path.join(upload_dir, f"*.jpeg"),
        os.path.join(upload_dir, f"*.csv")
    ]
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern))
        logging.debug("Found files for pattern %s: %s", pattern, glob.glob(pattern))

    results = []
    for file_path in files:
        filename = os.path.basename(file_path)
        file_type = filename.split(".")[-1].lower()
        logging.debug("Processing file: %s", filename)

        current_hash = get_file_hash(file_path)
        if not current_hash:
            results.append({"filename": filename, "success": False, "message": f"Failed to compute hash for {filename}"})
            continue

        metadata_entry = next((m for m in user_metadata if m["filename"] == filename and m["file_hash"] == current_hash), None)
        if metadata_entry:
            extracted_data = metadata_entry["invoice_data"]
            if clean_name(extracted_data.get("bill_to_name", "")) != clean_name(username):
                results.append({"filename": filename, "success": False, "message": f"Invoice {filename} not addressed to {username}"})
                continue
            if role == 'user':
                cached_invoice_data[filename] = {
                    "data": extracted_data,
                    "file_hash": current_hash,
                    "timestamp": metadata_entry["timestamp"]
                }
                session['invoices_data'][filename] = extracted_data
            if not any(inv["filename"] == filename and inv["username"] == username for inv in all_invoices):
                all_invoices.append({
                    "username": username,
                    "filename": filename,
                    "invoice_number": extract_invoice_number(extracted_data),
                    "data": extracted_data,
                    "file_hash": current_hash,
                    "timestamp": metadata_entry["timestamp"]
                })
            results.append({"filename": filename, "success": True, "message": f"Used cached metadata for {filename}"})
            continue

        try:
            if file_type == "pdf":
                text = extract_pdf_text(file_path)
            elif file_type in ["png", "jpg", "jpeg"]:
                text = extract_image_text(file_path)
            elif file_type == "csv":
                text = extract_csv_text(file_path)
            else:
                results.append({"filename": filename, "success": False, "message": f"Unsupported file type: {file_type}"})
                continue

            extracted_data = extract_invoice_data_with_gemini(text)
            if "error" in extracted_data:
                results.append({"filename": filename, "success": False, "message": extracted_data["error"]})
                continue

            if clean_name(extracted_data.get("bill_to_name", "")) != clean_name(username):
                results.append({"filename": filename, "success": False, "message": f"Invoice {filename} not addressed to {username}"})
                continue

            if role == "user":
                total = extract_invoice_total(extracted_data)
                if total is not None and total > 1000:
                    results.append({"filename": filename, "success": False, "message": f"Invoice total (${total}) exceeds user limit ($1000)"})
                    continue

            if role == 'user':
                cached_invoice_data[filename] = {
                    "data": extracted_data,
                    "file_hash": current_hash,
                    "timestamp": datetime.now().isoformat()
                }
                session['invoices_data'][filename] = extracted_data
            all_invoices.append({
                "username": username,
                "filename": filename,
                "invoice_number": extract_invoice_number(extracted_data),
                "data": extracted_data,
                "file_hash": current_hash,
                "timestamp": datetime.now().isoformat()
            })
            results.append({"filename": filename, "success": True, "message": f"Extracted data from {filename}"})
        except Exception as e:
            logging.error("Error processing %s: %s", filename, str(e))
            results.append({"filename": filename, "success": False, "message": f"Error processing {filename}: {str(e)}"})

    try:
        if role == 'user':
            save_invoice_data(username, cached_invoice_data)
        save_all_invoices(all_invoices)
        session['combined_data'] = "\n\n".join([f"--- Data from {inv['filename']} ---\n{json.dumps(inv['data'], indent=2)}"
                                               for inv in all_invoices if inv["username"] == username])
        session.modified = True
        logging.debug("Saved session data for %s: %s", username, json.dumps(session['invoices_data'], indent=2))
    except Exception as e:
        logging.error("Error saving invoice data: %s", str(e))
        results.append({"success": False, "message": f"Error saving invoice data: {str(e)}"})

    if files and role == 'user':
        try:
            safe_username = shlex.quote(username)
            result = subprocess.run(['python', 'preprocess.py', safe_username], check=True, capture_output=True, text=True)
            logging.debug("preprocess.py output: %s", result.stdout)
        except subprocess.CalledProcessError as e:
            logging.error("preprocess.py failed: %s\nStderr: %s", str(e), e.stderr)
            results.append({"success": False, "message": f"Error running preprocess.py: {str(e)}"})
        except FileNotFoundError:
            logging.error("preprocess.py script not found")
            results.append({"success": False, "message": "preprocess.py script not found"})

    return results

# Routes
@app.route('/')
def index():
    if 'authenticated' in session and session['authenticated']:
        return redirect(url_for('chat'))
    return render_template('index.html')

@app.route('/login/<role>')
def login_page(role):
    if role not in ['user', 'manager']:
        return "Invalid role", 400
    return render_template('login.html', role=role)

@app.route('/chat')
def chat():
    if 'authenticated' not in session or not session['authenticated']:
        return redirect(url_for('index'))
    
    pending_requests = []
    approval_requests = []
    if session.get('role') == "manager":
        try:
            requests = load_approval_requests()
            pending_requests = [r for r in requests if r["status"] == "pending"]
        except Exception as e:
            logging.error("Error loading pending requests: %s", str(e))
    else:
        try:
            approval_requests = [r for r in load_approval_requests() if r["user"] == session['username']]
        except Exception as e:
            logging.error("Error loading approval requests: %s", str(e))

    invoice_list = []
    due_dates = []
    current_date = datetime.strptime("2025-06-18", "%Y-%m-%d").date()  # Use provided date
    try:
        all_invoices = load_all_invoices()
        logging.debug("Total invoices loaded: %d", len(all_invoices))
        user_invoices = [inv for inv in all_invoices if inv["username"] == session['username']]
        logging.debug("User %s invoices: %d", session['username'], len(user_invoices))
        for inv in user_invoices:
            invoice_number = inv.get("invoice_number")
            logging.debug("Processing invoice: %s", invoice_number)
            if invoice_number:
                invoice_list.append(invoice_number)
                result = calculate_days_remaining(invoice_number, session['username'], current_date)
                if result["success"]:
                    due_date = inv.get("data", {}).get("due_date")
                    due_date_obj = datetime.strptime(due_date, "%Y-%m-%d").date() if due_date else None
                    if due_date_obj:
                        overdue = due_date_obj < current_date
                        days_remaining = (due_date_obj - current_date).days if not overdue else 0
                        due_dates.append({
                            "invoice": invoice_number,
                            "date": due_date,
                            "overdue": overdue,
                            "days_remaining": days_remaining
                        })
                        logging.debug("Added due date for %s: %s", invoice_number, due_date)
                else:
                    logging.error("Error processing due date for %s: %s", invoice_number, result["message"])
    except Exception as e:
        logging.error("Error loading invoices for chat: %s", str(e))

    due_date_message = []
    for item in due_dates:
        if item["overdue"]:
            due_date_message.append(f"Invoice {item['invoice']} is overdue (Due: {item['date']}).")
        else:
            due_date_message.append(f"Invoice {item['invoice']} is due on {item['date']} ({item['days_remaining']} days remaining).")
    if due_date_message:
        due_date_notification = "\n".join(due_date_message)
    else:
        due_date_notification = "No due date information available."

    logging.debug("Rendering chat.html with invoice_list: %s, due_dates: %s", invoice_list, due_dates)
    return render_template('chat.html',
                         username=session.get('username', 'User'),
                         role=session.get('role', 'user'),
                         chat_history=session.get('chat_sessions', {}),
                         current_session_id=session.get('current_session_id', ''),
                         current_chat=session.get('chat_sessions', {}).get(session.get('current_session_id', ''), []),
                         pending_requests=pending_requests,
                         invoice_list=invoice_list,
                         due_dates=due_dates,
                         approval_requests=approval_requests,
                         due_date_notification=due_date_notification)

@app.route('/login', methods=['POST'])
def login():
    logging.debug("Received login request")
    try:
        data = request.get_json(silent=True)
        if not data:
            logging.error("Invalid JSON in login request")
            return jsonify({"success": False, "message": "Invalid JSON format"}), 400

        username = data.get('username')
        password = data.get('password')
        expected_role = data.get('role')
        logging.debug("Login attempt for username: %s as %s", username, expected_role)

        if not username or not password or not expected_role:
            logging.warning("Missing username, password, or role")
            return jsonify({"success": False, "message": "Username, password, and role required"}), 400

        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            logging.warning("Invalid username format: %s", username)
            return jsonify({"success": False, "message": "Invalid username format"}), 400

        if username not in USERS:
            logging.warning("Username not found: %s", username)
            return jsonify({"success": False, "message": "Username not found"}), 404

        if not bcrypt.checkpw(password.encode(), USERS[username]["password"].encode()):
            logging.warning("Incorrect password for %s", username)
            return jsonify({"success": False, "message": "Incorrect password"}), 401

        if USERS[username]["role"] != expected_role:
            logging.warning("Role mismatch for %s: expected %s, got %s", username, expected_role, USERS[username]["role"])
            return jsonify({"success": False, "message": f"Access denied: {username} is not a {expected_role}"}), 403

        # Initialize session
        session['authenticated'] = True
        session['username'] = username
        session['role'] = USERS[username]["role"]
        session['context'] = {}
        session['invoices_data'] = {}
        session['combined_data'] = ""

        session['chat_sessions'] = load_chat_history(username)
        if not session['chat_sessions']:
            new_session_id = str(uuid.uuid4())
            session['chat_sessions'] = {new_session_id: []}
            session['current_session_id'] = new_session_id
        else:
            session['current_session_id'] = list(session['chat_sessions'].keys())[0]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Process uploaded files first to ensure all_invoices.json is updated
        results = process_uploaded_files(username, session['role'])

        # Static greeting
        greeting_message = f"Welcome, {username}! How can I assist you today?"
        session['chat_sessions'][session['current_session_id']].append({
            "answer": greeting_message,
            "timestamp": timestamp
        })

        # Dynamic due date message (only for users)
        if session['role'] == 'user':
            due_date_message = generate_due_date_message(username, datetime.strptime("2025-06-18", "%Y-%m-%d").date())
            session['chat_sessions'][session['current_session_id']].append({
                "answer": due_date_message,
                "timestamp": timestamp
            })

        session.modified = True
        save_chat_history(username, session['chat_sessions'])
        logging.info("Login successful for %s (%s)", username, session['role'])
        return jsonify({"success": True, "message": "Login successful", "results": results})

    except Exception as e:
        logging.error("Login error: %s", str(e))
        return jsonify({"success": False, "message": "An unexpected error occurred. Please try again later."}), 500
    
@app.route('/logout', methods=['POST'])
def logout():
    logging.debug("Logout request")
    try:
        username = session.get('username')
        if username:
            save_chat_history(username, session.get('chat_sessions', {}))
        logging.info("Session cleared for user: %s", username)
        session.clear()
        return jsonify({"success": True, "message": "Logged out successfully"})
    except Exception as e:
        logging.error("Error during logout: %s", str(e))
        return jsonify({"success": False, "message": "An unexpected error occurred. Please try again later."}), 500

@app.route('/ask', methods=['POST'])
def ask():
    logging.debug("Received ask question request")
    if 'authenticated' not in session or not session['authenticated']:
        logging.warning("Unauthorized ask request")
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        data = request.get_json()
        if not data:
            logging.error("Invalid JSON in ask request")
            return jsonify({"success": False, "message": "Invalid JSON format"}), 400
        question = data.get('question', '')
        if not question:
            logging.warning("No question provided")
            return jsonify({"success": False, "message": "No question provided"}), 400
    except Exception as e:
        logging.error("Error extracting question: %s", str(e))
        return jsonify({"success": False, "message": "An unexpected error occurred. Please try again later."}), 400

    username = session['username']
    role = session['role']
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_date = datetime.strptime("2025-06-18", "%Y-%m-%d").date()  # Use provided date

    try:
        all_invoices = load_all_invoices()
        user_invoices = [inv for inv in all_invoices if inv["username"] == username]
        invoice_list = [inv["invoice_number"] for inv in user_invoices if inv["invoice_number"]]
        logging.debug("Available invoices for %s: %s", username, invoice_list)
    except Exception as e:
        logging.error("Error loading invoice list: %s", str(e))
        return jsonify({"success": False, "message": "An unexpected error occurred. Please try again later."}), 500

    approval_keywords = r'(approve|approval|request\s+approval|submit\s+for\s+approval)'
    status_keywords = r'(status|check\s+status|approval\s+status)'
    due_date_keywords = r'(due\s+date|days\s+remaining|due\s+on|left\s+to\s+pay|days\s+left|remaining\s+to\s+pay)'

    if role == 'user' and re.search(approval_keywords, question, re.IGNORECASE):
        try:
            invoice_match = re.search(r'\b(INV\d+|invoice\s+\S+)', question, re.IGNORECASE)
            if invoice_match:
                invoice_id = invoice_match.group(0)
                if invoice_id in invoice_list:
                    return handle_approval_request(invoice_id, username, timestamp)
                else:
                    answer = f"Invoice {invoice_id} not found. Available invoices: {', '.join(invoice_list) if invoice_list else 'None'}."
                    logging.warning("Invalid invoice %s requested by %s", invoice_id, username)
            elif len(invoice_list) > 1:
                session['context'] = {'awaiting_invoice_number': True}
                session.modified = True
                answer = f"Please specify the invoice number you want to request approval for. Available invoices: {', '.join(invoice_list)}."
                logging.info("Prompting %s for invoice number selection", username)
            elif len(invoice_list) == 1:
                return handle_approval_request(invoice_list[0], username, timestamp)
            else:
                answer = "You have no invoices to request approval for."
                logging.warning("%s has no invoices available for approval", username)
        except Exception as e:
            logging.error("Error handling approval request: %s", str(e))
            answer = "An unexpected error occurred. Please try again later."

    elif role == 'user' and re.search(status_keywords, question, re.IGNORECASE):
        try:
            invoice_match = re.search(r'\b(INV\d+|invoice\s+\S+)', question, re.IGNORECASE)
            invoice_id = invoice_match.group(0) if invoice_match else None
            answer = check_invoice_status(username, invoice_id)
            logging.info("Status query by %s for invoice %s", username, invoice_id if invoice_id else 'all')
        except Exception as e:
            logging.error("Error checking invoice status: %s", str(e))
            answer = "An unexpected error occurred. Please try again later."

    elif re.search(due_date_keywords, question, re.IGNORECASE):
        try:
            invoice_match = re.search(r'\b(INV\d+|invoice\s+\S+)', question, re.IGNORECASE)
            logging.debug("Due date query detected, invoice match: %s", invoice_match)
            if invoice_match:
                invoice_id = invoice_match.group(0)
                if invoice_id in invoice_list:
                    result = calculate_days_remaining(invoice_id, username, current_date)
                    answer = result["message"]
                    if not result["success"]:
                        logging.warning("Failed due date query for %s: %s", invoice_id, answer)
                else:
                    answer = f"Invoice {invoice_id} not found. Available invoices: {', '.join(invoice_list) if invoice_list else 'None'}."
                    logging.warning("Invalid invoice %s for due date query by %s", invoice_id, username)
            else:
                answer = f"Please specify an invoice number for the due date query. Available invoices: {', '.join(invoice_list) if invoice_list else 'None'}."
                logging.warning("No invoice ID provided for due date query by %s", username)
        except Exception as e:
            logging.error("Error handling due date query: %s", str(e))
            answer = "An unexpected error occurred. Please try again later."

    elif session.get('context', {}).get('awaiting_invoice_number'):
        try:
            invoice_match = re.search(r'\b(INV\d+|invoice\s+\S+)', question, re.IGNORECASE)
            if invoice_match:
                invoice_id = invoice_match.group(0)
                if invoice_id in invoice_list:
                    session['context'] = {}
                    session.modified = True
                    return handle_approval_request(invoice_id, username, timestamp)
                else:
                    answer = f"Invoice {invoice_id} not found. Available invoices: {', '.join(invoice_list) if invoice_list else 'None'}."
                    logging.warning("Invalid invoice %s provided by %s", invoice_id, username)
            else:
                answer = f"Please provide a valid invoice number. Available invoices: {', '.join(invoice_list) if invoice_list else 'None'}."
                logging.warning("%s failed to provide valid invoice number", username)
        except Exception as e:
            logging.error("Error handling follow-up invoice number: %s", str(e))
            answer = "An unexpected error occurred. Please try again later."

    else:
        try:
            invoice_data = json.dumps([inv["data"] for inv in user_invoices], indent=2)
            answer = ask_question_about_invoice(question, invoice_data)
            logging.debug("Answered invoice question for %s: %s", username, question)
        except Exception as e:
            logging.error("Error answering invoice question: %s", str(e))
            answer = "An unexpected error occurred. Please try again later."

    try:
        session['chat_sessions'][session['current_session_id']].append({
            "question": question,
            "answer": answer,
            "timestamp": timestamp
        })
        session.modified = True
        save_chat_history(username, session['chat_sessions'])
        logging.debug("Question answered and saved: %s", question)
        return jsonify({"success": True, "answer": answer, "chat_history": session['chat_sessions'][session['current_session_id']]})
    except Exception as e:
        logging.error("Error saving chat history for %s: %s", username, str(e))
        return jsonify({"success": False, "message": "An unexpected error occurred. Please try again later."}), 500

def notify_managers(invoice_id, user, total, timestamp, manager_name):
    """Notify all managers with a formatted message about a new approval request."""
    try:
        parsed_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        formatted_date = parsed_time.strftime("%Y-%m-%d")
    except ValueError as e:
        logging.error("Invalid timestamp format: %s", str(e))
        formatted_date = timestamp[:10]

    formatted_message = (
        f"Invoice: {invoice_id}\n"
        f"Requested By: {user}\n"
        f"Amount: ₹{total:,.2f}\n"
        f"Date of Request: {formatted_date}\n"
        f"Status: ⏳ Pending Manager Approval"
    )
    if manager_name:
        formatted_message = (
            f"Invoice: {invoice_id}\n"
            f"Requested By: {user}\n"
            f"Amount: ₹{total:,.2f}\n"
            f"Date of Request: {formatted_date}\n"
            f"Status: Approved by {manager_name} on {formatted_date}"
        )

    managers = [username for username, info in USERS.items() if info['role'] == 'manager']
    for manager in managers:
        try:
            manager_history = load_chat_history(manager)
            current_session = list(manager_history.keys())[0] if manager_history else str(uuid.uuid4())
            if current_session not in manager_history:
                manager_history[current_session] = []

            manager_history[current_session].append({
                "answer": formatted_message,
                "timestamp": timestamp
            })
            save_chat_history(manager, manager_history)
            logging.info("Notified manager %s about approval request for %s", manager, invoice_id)
        except Exception as e:
            logging.error("Error notifying manager %s for invoice %s: %s", manager, invoice_id, str(e))

def handle_approval_request(invoice_id, username, timestamp):
    """Handle invoice approval request logic."""
    try:
        invoice_data = None
        all_invoices = load_all_invoices()
        for inv in all_invoices:
            if inv["username"] == username and inv["invoice_number"] == invoice_id:
                invoice_data = inv["data"]
                break
        if not invoice_data:
            logging.warning("Invoice %s not found for user %s", invoice_id, username)
            return jsonify({"success": False, "message": f"Invoice {invoice_id} not found"}), 404

        total = extract_invoice_total(invoice_data)
        if total is None:
            logging.warning("Could not extract total for invoice %s", invoice_id)
            return jsonify({"success": False, "message": "Invalid invoice data"}), 400

        requests = load_approval_requests()
        if any(r["invoice_id"] == invoice_id and r["user"] == username and r["status"] == "pending" for r in requests):
            logging.warning("Approval request for %s already pending", invoice_id)
            answer = f"Approval request for invoice {invoice_id} is already pending."
        else:
            # Assign to a manager (for simplicity, assign to the first manager)
            managers = [username for username, info in USERS.items() if info['role'] == 'manager']
            manager_name = managers[0] if managers else "Unknown Manager"
            requests.append({
                "invoice_id": invoice_id,
                "user": username,
                "total": total,
                "status": "pending",
                "manager": manager_name,
                "timestamp": datetime.now().isoformat()
            })
            save_approval_requests(requests)
            answer = f"Approval request for invoice {invoice_id} submitted to {manager_name} on {timestamp[:10]}."
            notify_managers(invoice_id, username, total, timestamp, None)

        session['chat_sessions'][session['current_session_id']].append({
            "answer": answer,
            "timestamp": timestamp
        })
        session.modified = True
        save_chat_history(username, session['chat_sessions'])
        logging.info("Approval request submitted for %s by %s", invoice_id, username)
        return jsonify({"success": True, "answer": answer, "chat_history": session['chat_sessions'][session['current_session_id']]})
    except Exception as e:
        logging.error("Error handling approval request for %s by %s: %s", invoice_id, username, str(e))
        return jsonify({"success": False, "message": "An unexpected error occurred. Please try again later."}), 500

@app.route('/approve', methods=['POST'])
def approve():
    logging.debug("Received approve request")
    if 'authenticated' not in session or not session['authenticated'] or session['role'] != 'manager':
        logging.warning("Unauthorized approve request")
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        data = request.get_json()
        invoice_id = data.get('invoice_id')
        user = data.get('user')
        if not invoice_id or not user or not re.match(r'^[a-zA-Z0-9_-]+$', user):
            logging.warning("Missing or invalid invoice ID or user")
            return jsonify({"success": False, "message": "Valid invoice ID and user required"}), 400

        requests = load_approval_requests()
        for req in requests:
            if req["invoice_id"] == invoice_id and req["user"] == user and req["status"] == "pending":
                req["status"] = "approved"
                req["manager"] = session['username']
                req["timestamp"] = datetime.now().isoformat()
                save_approval_requests(requests)
                user_history = load_chat_history(user)
                current_session = list(user_history.keys())[0] if user_history else str(uuid.uuid4())
                if current_session not in user_history:
                    user_history[current_session] = []
                user_history[current_session].append({
                    "answer": f"Your invoice {invoice_id} has been approved by {session['username']} on {req['timestamp'][:10]}.",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                save_chat_history(user, user_history)
                logging.info("Invoice %s approved by %s for %s", invoice_id, session['username'], user)
                return jsonify({"success": True, "message": f"Invoice {invoice_id} approved for {user}"})
        logging.warning("No pending request for %s by %s", invoice_id, user)
        return jsonify({"success": False, "message": f"No pending request found for {invoice_id}"})
    except Exception as e:
        logging.error("Error approving invoice: %s", str(e))
        return jsonify({"success": False, "message": "An unexpected error occurred. Please try again later."}), 500

@app.route('/reject', methods=['POST'])
def reject():
    logging.debug("Received reject request")
    if 'authenticated' not in session or not session['authenticated'] or session['role'] != 'manager':
        logging.warning("Unauthorized reject request")
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        data = request.get_json()
        invoice_id = data.get('invoice_id')
        user = data.get('user')
        if not invoice_id or not user or not re.match(r'^[a-zA-Z0-9_-]+$', user):
            logging.warning("Missing or invalid invoice ID or user")
            return jsonify({"success": False, "message": "Valid invoice ID and user required"}), 400

        requests = load_approval_requests()
        for req in requests:
            if req["invoice_id"] == invoice_id and req["user"] == user and req["status"] == "pending":
                req["status"] = "rejected"
                req["manager"] = session['username']
                req["timestamp"] = datetime.now().isoformat()
                save_approval_requests(requests)
                user_history = load_chat_history(user)
                current_session = list(user_history.keys())[0] if user_history else str(uuid.uuid4())
                if current_session not in user_history:
                    user_history[current_session] = []
                user_history[current_session].append({
                    "answer": f"Your invoice {invoice_id} has been rejected by {session['username']} on {req['timestamp'][:10]}.",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                save_chat_history(user, user_history)
                logging.info("Invoice %s rejected by %s for %s", invoice_id, session['username'], user)
                return jsonify({"success": True, "message": f"Invoice {invoice_id} rejected for {user}"})
        logging.warning("No pending request for %s by %s", invoice_id, user)
        return jsonify({"success": False, "message": f"No pending request found for {invoice_id}"})
    except Exception as e:
        logging.error("Error rejecting invoice: %s", str(e))
        return jsonify({"success": False, "message": "An unexpected error occurred. Please try again later."}), 500

@app.route('/new_chat', methods=['POST'])
def new_chat():
    logging.debug("New chat session request")
    if 'authenticated' not in session or not session['authenticated']:
        logging.warning("Unauthorized new chat request")
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        new_session_id = str(uuid.uuid4())
        session['chat_sessions'][new_session_id] = []
        session['current_session_id'] = new_session_id
        session['context'] = {}
        session.modified = True
        save_chat_history(session['username'], session['chat_sessions'])
        logging.info("New chat session created: %s", new_session_id)
        return jsonify({"success": True, "message": "New chat session created"})
    except Exception as e:
        logging.error("Error creating new chat session: %s", str(e))
        return jsonify({"success": False, "message": "An unexpected error occurred. Please try again later."}), 500

@app.route('/switch_chat/<session_id>', methods=['POST'])
def switch_chat(session_id):
    logging.debug("Switch chat request for session: %s", session_id)
    if 'authenticated' not in session or not session['authenticated']:
        logging.warning("Unauthorized switch chat request")
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        if session_id in session['chat_sessions']:
            session['current_session_id'] = session_id
            session['context'] = {}
            session.modified = True
            save_chat_history(session['username'], session['chat_sessions'])
            logging.info("Switched to chat session: %s", session_id)
            return jsonify({"success": True, "message": "Switched to chat session"})
        logging.warning("Chat session not found: %s", session_id)
        return jsonify({"success": False, "message": "Chat session not found"}), 404
    except Exception as e:
        logging.error("Error switching chat session: %s", str(e))
        return jsonify({"success": False, "message": "An unexpected error occurred. Please try again later."}), 500

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)
