# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from werkzeug.utils import secure_filename
import tempfile
import traceback

from extract import extract_text_from_pdf_fileobj, extract_text_from_txt_fileobj
from summarize import generate_summary
from highlight import top_k_sentences

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
SUPPORTED_LENGTHS = ['short', 'medium', 'long']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_length_params(length):
    """Get summary length parameters based on user selection"""
    length_configs = {
        'short': {'max_len': 80, 'min_len': 30},
        'medium': {'max_len': 180, 'min_len': 60},
        'long': {'max_len': 300, 'min_len': 120}
    }
    return length_configs.get(length, length_configs['medium'])

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Document Summarization API",
        "version": "1.0.0"
    })

@app.route('/api/supported_formats', methods=['GET'])
def supported_formats():
    """Return supported file formats"""
    return jsonify({
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size": "16MB"
    })

@app.route('/api/summarize', methods=['POST'])
def summarize_document():
    """Main endpoint for document summarization"""
    try:
        # Check if file part exists
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '' or not file:
            logger.warning("No file selected")
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            logger.warning(f"Unsupported file type: {file.filename}")
            return jsonify({
                "error": f"Unsupported file type. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Get summary length preference
        length = request.form.get('length', 'medium')
        if length not in SUPPORTED_LENGTHS:
            length = 'medium'
        
        length_params = get_length_params(length)
        logger.info(f"Processing {file.filename} with {length} summary length")
        
        # Extract text based on file type
        filename = file.filename.lower()
        text = ""
        
        try:
            if filename.endswith('.pdf'):
                logger.info("Extracting text from PDF")
                text = extract_text_from_pdf_fileobj(file)
            elif filename.endswith('.txt'):
                logger.info("Extracting text from TXT")
                # Reset file pointer for text extraction
                file.seek(0)
                text = extract_text_from_txt_fileobj(file)
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return jsonify({"error": f"Failed to extract text from file: {str(e)}"}), 400
        
        # Validate extracted text
        if not text or text.strip() == "":
            logger.warning("No text content extracted from file")
            return jsonify({"error": "No readable text content found in the document"}), 400
        
        logger.info(f"Extracted {len(text)} characters from document")
        
        # Generate summary
        try:
            logger.info("Generating summary...")
            summary = generate_summary(
                text, 
                max_chunk_chars=1200, 
                summary_max_length=length_params['max_len'],
                summary_min_length=length_params['min_len']
            )
            logger.info("Summary generated successfully")
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return jsonify({"error": "Failed to generate summary. Please try again."}), 500
        
        # Generate highlights
        highlights = []
        try:
            logger.info("Generating highlights...")
            highlights = top_k_sentences(text, summary, k=5)
            logger.info(f"Generated {len(highlights)} highlights")
        except Exception as e:
            logger.warning(f"Highlight generation failed, continuing without highlights: {str(e)}")
            highlights = []
        
        # Prepare response
        response_data = {
            "summary": summary,
            "highlights": highlights,
            "original_excerpt": text[:2000], 
            "metadata": {
                "original_length": len(text),
                "summary_length": len(summary),
                "highlight_count": len(highlights),
                "file_name": secure_filename(file.filename),
                "summary_type": length
            }
        }
        
        logger.info("Request completed successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "An unexpected error occurred. Please try again later."
        }), 500

@app.route('/api/batch_summarize', methods=['POST'])
def batch_summarize_documents():
    """Endpoint for batch processing multiple documents (premium feature placeholder)"""
    return jsonify({
        "error": "Batch processing not implemented yet",
        "message": "This feature is under development"
    }), 501

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({"error": "File size too large. Maximum size is 16MB."}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting Document Summarization API...")
    app.run(host='0.0.0.0', port=5000, debug=True)