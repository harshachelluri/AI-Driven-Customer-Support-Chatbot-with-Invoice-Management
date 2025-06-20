# AI-Driven Customer Support Chatbot with Invoice Management

# Project Overview

The AI-Driven Customer Support Chatbot with Invoice Management is a Flask-based web application designed to streamline customer support and invoice processing for small to medium-sized businesses. This project combines a conversational AI chatbot for handling customer queries with robust invoice management features, including data extraction, approval workflows, and real-time tracking. It leverages OCR (Tesseract), Google's Gemini AI, and a Hugging Face model to extract structured data from invoices (PDFs, images, CSVs) and supports role-based access for customers, support agents, and managers. The system addresses real-world business needs for efficient customer service and invoice handling, ensuring transparency, scalability, and compliance.


# Business Case

The chatbot addresses challenges in customer support and invoice processing, such as delayed query resolution, manual invoice handling, and lack of centralized tracking. Key benefits include:

- Efficiency: Automates customer query resolution and invoice data extraction, reducing manual effort and errors.
- Transparency: Provides real-time tracking of customer tickets and invoice statuses, improving visibility for all stakeholders.
- Scalability: Supports multiple users, secure file handling, and concurrent sessions for growing businesses.
- Cost Savings: Minimizes time spent on repetitive tasks, allowing staff to focus on high-value activities.
- Compliance: Enforces invoice approval workflows and ensures secure handling of financial data.

This tool is ideal for businesses seeking to enhance customer support while modernizing invoice management through an AI-powered platform.


# Client Perspective: Problems and Challenges

# Client Problems

- Manual Invoice Processing: Reliance on manual data entry for invoices leads to errors (e.g., incorrect amounts, missed due dates) and delays.
- Delayed Customer Query Resolution: Lack of an automated system for customer inquiries results in slow response times, impacting satisfaction.
- Approval Bottlenecks: Invoice approvals are delayed due to poor visibility, causing late payments and strained vendor relationships.
- Tracking Inefficiencies: Difficulty tracking invoice statuses and customer tickets creates confusion and operational inefficiencies.
- Compliance Risks: Unauthorized invoice payments or incorrect recipient processing pose financial risks.
- Scalability Issues: Manual systems struggle with increasing invoice and query volumes as the business grows.


# Challenges During Development

- Diverse Invoice Formats: Invoices in PDFs, images, or CSVs with varying layouts require robust OCR and AI model tuning for accurate data extraction.
- Query Understanding: The chatbot must accurately interpret diverse customer queries (e.g., invoice status, product issues) using NLP.
- Real-Time Processing: Immediate invoice processing and query responses demand low-latency optimization for large files or high volumes.
- User Adoption: An intuitive chat interface is critical to ensure adoption by customers and staff accustomed to manual processes.
- Security: Secure file handling and role-based access are essential to protect sensitive financial and customer data.
- Integration: The system must integrate with existing customer support and file storage systems for seamless operation.


# Features

# Role-Based Access

Customers: Submit invoices, query statuses, and raise support tickets via the chat interface.
Support Agents: Respond to customer queries, process invoices, and escalate issues.
Managers: Approve or reject invoice requests and oversee support ticket resolutions.


# Invoice Data Extraction

Uses OCR (Tesseract), Gemini AI, and a Hugging Face model to extract data (e.g., invoice number, total, due date) from PDFs, images, and CSVs.


# Customer Support Chatbot

Handles queries about invoices (e.g., status, total) and general support issues (e.g., product inquiries) using NLP.


# Approval Workflow

Customers or agents submit invoices for approval, with managers notified to take action.


# Due Date Notifications

Alerts customers and agents about upcoming or overdue invoice due dates.


# Secure File Management

Stores invoices in user-specific directories with metadata for efficient processing.


# Session Management

Supports multiple chat sessions with history persistence for seamless interactions.

# Customer Workflow

# Login:

Access the app at http://localhost:5000 and select "Login as Customer."
Enter credentials (e.g., Raju/123).
System initializes a chat session and processes uploaded invoices.


# Upload Invoices:

Place invoice files (PDF, PNG, JPG, JPEG, CSV) in Uploads/<username>/ (e.g., Uploads/Raju/).
System processes files using preprocess.py, extracting data with OCR and AI.
Invoices must be addressed to the customer, with totals ≤ $1000.


# Query Invoices:

Ask via chat: "What is the status of INV001?" or "When is INV001 due?"
System retrieves data from all_invoices.json or approval_requests.json.


# Request Approval:

Type "Request approval for INV001" to submit an invoice.
Request logged in approval_requests.json, notifying managers.


# Raise Support Tickets:

Submit queries like "Issue with product X" via chat.
Tickets logged in support_tickets.json and assigned to agents.


# Support Agent Workflow

# Login:

Select "Login as Agent" and use credentials (e.g., Priya/priya).
System loads pending tickets and invoice requests.


# Respond to Queries:

View and reply to customer tickets in the chat interface or sidebar.
Process customer-uploaded invoices if needed.


# Escalate Issues:

Escalate complex tickets or invoice issues to managers via the system.


# Manager Workflow

# Login:

Select "Login as Manager" and use credentials (e.g., Divya/divya).
System loads pending approvals and escalated tickets.


# Review Approvals:

View invoice requests in the sidebar (e.g., from Raju, INV001, $500).
Receive chat notifications for new requests.


# Approve/Reject Invoices:

Click "Approve" or "Reject" in the sidebar.
Updates approval_requests.json and notifies customers/agents.


# Resolve Escalated Tickets:

Address escalated support tickets and provide resolutions.


# System Flow

# File Processing:

Invoices in Uploads/<username>/ are processed by preprocess.py, updating pdf_metadata.json and all_invoices.json.


# Data Storage:

Metadata in pdf_metadata.json, invoices in all_invoices.json, approvals in approval_requests.json, tickets in support_tickets.json, chat histories in chat_history_<username>.json.


# Approval Process:

Customer/agent submits invoice → System logs and notifies managers → Manager approves/rejects → Customer/agent notified.

# Chat Interaction:

Users interact via chat, with the system parsing queries for invoices, tickets, or approvals.


# Key Notes

# Security:

Passwords hashed with bcrypt, file access restricted to user-specific directories.


# Real-Time Processing:

Immediate file and query processing ensures quick feedback.


# Error Handling:

Logging and retries handle API limits, file errors, and invalid inputs.


# AI Models:

Gemini AI and Hugging Face model enhance invoice extraction and query understanding.


# Usage

# Login:

Access at http://localhost:5000.
Log in as Customer or Manager 
Modify USERS in app.py to add users.


# Customer Actions

Upload invoices to Uploads/<username>/.
Use chat to:
- Request approvals (e.g., "Request approval for INV001").
- Check status (e.g., "What is the status of INV001?").
- Query due dates (e.g., "When is INV001 due?").
- Raise tickets (e.g., "Issue with product X").


# Agent Actions

Respond to customer tickets and process invoices.
Escalate issues to managers.


# Manager Actions

Approve/reject invoice requests in the sidebar.
Resolve escalated tickets.


# File Processing

preprocess.py processes files, updating pdf_metadata.json and all_invoices.json.
Run manually: python preprocess.py <username>.


# Project Structure    
customer-support-invoice-chatbot/    
├── app.py                     # Main Flask application      
├── preprocess.py              # Script to process uploaded files    
├── templates/                 # HTML templates    
│   ├── index.html             # Landing page    
│   ├── login.html             # Login page    
│   ├── chat.html              # Chat interface    
├── Uploads/                   # Directory for uploaded invoices    
├── pdf_metadata.json          # Metadata for processed files    
├── all_invoices.json          # Consolidated invoice data    
├── approval_requests.json     # Approval request tracking    
├── support_tickets.json       # Customer support tickets    
├── requirements.txt           # Python dependencies    
├── .env                       # Environment variables    
└── README.md                  # Project documentation    
