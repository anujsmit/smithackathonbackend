from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import nltk

def split_into_sentences(text):
    """
    Split text into sentences using NLTK with regex fallback
    """
    try:
        # Try using NLTK for better sentence splitting
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        # Fallback to regex-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

def top_k_sentences(original_text, summary_text, k=5):
    """
    Find top k sentences from original text most relevant to summary
    """
    sentences = split_into_sentences(original_text)
    
    if len(sentences) == 0:
        return []
    
    # If fewer sentences than k, return all
    if len(sentences) <= k:
        return [{"index": i, "sentence": sent, "score": 1.0} 
                for i, sent in enumerate(sentences)]
    
    try:
        # Build TF-IDF vectors
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 2),
            max_features=5000
        ).fit(sentences + [summary_text])
        
        sent_vectors = vectorizer.transform(sentences)
        summary_vector = vectorizer.transform([summary_text])
        
        # Calculate similarity scores
        sim_scores = cosine_similarity(sent_vectors, summary_vector).flatten()
        
        # Apply position weighting (earlier sentences often more important)
        position_weights = np.linspace(1.0, 0.7, len(sentences))
        combined_scores = sim_scores * position_weights
        
        # Get top k indices
        k = min(k, len(sentences))
        top_indices = np.argsort(combined_scores)[-k:][::-1]  # Descending order
        
        return [{
            "index": int(idx),
            "sentence": sentences[idx],
            "score": float(combined_scores[idx])
        } for idx in top_indices]
        
    except Exception as e:
        print(f"Highlight generation failed: {str(e)}")
        # Fallback: return first k sentences
        return [{"index": i, "sentence": sentences[i], "score": 0.5} 
                for i in range(min(k, len(sentences)))]