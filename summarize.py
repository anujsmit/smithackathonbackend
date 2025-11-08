from transformers import pipeline
import math

# Use a smaller, faster model for development
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
except Exception as e:
    print(f"BART model failed, trying smaller model: {e}")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

def chunk_text_by_tokens(text, approx_chars=800):  # Smaller chunks
    """
    Split text into chunks for processing large documents
    """
    chunks = []
    start = 0
    n = len(text)
    
    while start < n:
        end = min(n, start + approx_chars)
        
        # Try to split at natural boundaries
        if end < n:
            # Look for newline first
            split_at = text.rfind("\n", start, end)
            if split_at == -1:
                # Look for sentence end
                split_at = text.rfind(". ", start, end)
            if split_at == -1:
                # Look for space
                split_at = text.rfind(" ", start, end)
            if split_at == -1:
                split_at = end
            else:
                split_at += 1  # Include the boundary character
            end = split_at
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        start = end
        if start == end:  # Prevent infinite loop
            start += 1
    
    return chunks

def generate_summary(text, max_chunk_chars=800, summary_max_length=150, summary_min_length=40):
    """
    Generate summary for the given text - optimized for speed
    """
    # For very short texts, return as is
    if len(text) < 150:
        return text[:summary_max_length]
    
    # For medium texts, summarize directly
    if len(text) < 2000:
        try:
            out = summarizer(
                text, 
                max_length=summary_max_length, 
                min_length=summary_min_length, 
                do_sample=False
            )
            return out[0]['summary_text']
        except Exception as e:
            print(f"Direct summarization failed: {e}, using fallback")
            return text[:summary_max_length]
    
    # Split into manageable chunks only for large texts
    chunks = chunk_text_by_tokens(text, approx_chars=max_chunk_chars)
    
    if not chunks:
        return "No content to summarize."
    
    summaries = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Skip very short chunks
            if len(chunk.split()) < 10:
                summaries.append(chunk)
                continue
                
            out = summarizer(
                chunk, 
                max_length=min(100, summary_max_length),  # Smaller chunk summaries
                min_length=min(20, summary_min_length), 
                do_sample=False
            )
            summaries.append(out[0]['summary_text'])
        except Exception as e:
            # Fallback for problematic chunks
            print(f"Chunk {i} summarization failed: {str(e)}")
            summaries.append(chunk[:100])  # Smaller fallback
    
    # If only one chunk, return its summary
    if len(summaries) == 1:
        return summaries[0]
    
    # Combine and return without second summarization to save time
    combined = " ".join(summaries)
    return combined[:summary_max_length]