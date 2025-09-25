"""
Query Expansion Module for Unified Embedder
===========================================

This module provides query expansion capabilities to improve search recall
by adding relevant terms, synonyms, and semantically related concepts to queries.
"""

import logging
import re
from typing import List, Dict, Any, Set, Optional, Tuple
import json
from pathlib import Path
import traceback

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError as e:
    print(f"Required libraries not installed: {e}")
    print("Please run: uv add sentence-transformers scikit-learn nltk")

# Configure logging
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Could not download NLTK data")

class QueryExpander:
    """Advanced query expansion with multiple strategies."""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 use_wordnet: bool = True,
                 use_semantic: bool = True,
                 max_expansions: int = 5):
        """
        Initialize the query expander.
        
        Args:
            embedding_model: Model for semantic similarity
            use_wordnet: Whether to use WordNet for synonyms
            use_semantic: Whether to use semantic expansion
            max_expansions: Maximum number of expansion terms per strategy
        """
        self.embedding_model_name = embedding_model
        self.use_wordnet = use_wordnet
        self.use_semantic = use_semantic
        self.max_expansions = max_expansions
        
        # Lazy-loaded components
        self.embedding_model = None
        self.lemmatizer = None
        self.stop_words = None
        
        # Domain-specific expansions
        self.domain_expansions = self._load_domain_expansions()
        
        # Query processing statistics
        self.stats = {
            "queries_processed": 0,
            "total_expansions_added": 0,
            "wordnet_expansions": 0,
            "semantic_expansions": 0,
            "domain_expansions": 0
        }
    
    def _initialize_components(self):
        """Lazy initialization of components."""
        if self.embedding_model is None and self.use_semantic:
            try:
                logger.info(f"Loading embedding model for query expansion: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device="cpu")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.use_semantic = False
        
        if self.lemmatizer is None and self.use_wordnet:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK components: {e}")
                self.use_wordnet = False
    
    def _load_domain_expansions(self) -> Dict[str, List[str]]:
        """Load domain-specific expansion mappings."""
        # Academic/educational domain expansions
        return {
            # Psychology terms
            "memory": ["recall", "retention", "remembering", "memorization", "cognitive"],
            "learning": ["education", "training", "acquisition", "development", "pedagogy"],
            "behavior": ["behaviour", "conduct", "action", "response", "reaction"],
            "cognition": ["thinking", "mental", "cognitive", "mind", "brain"],
            "perception": ["sensing", "awareness", "recognition", "observation"],
            
            # Technology terms
            "ai": ["artificial intelligence", "machine learning", "ml", "deep learning"],
            "algorithm": ["method", "procedure", "technique", "approach", "process"],
            "data": ["information", "dataset", "statistics", "analytics"],
            "model": ["framework", "system", "structure", "design"],
            
            # Research terms
            "study": ["research", "investigation", "analysis", "examination", "survey"],
            "experiment": ["test", "trial", "investigation", "research"],
            "hypothesis": ["theory", "proposition", "assumption", "prediction"],
            "results": ["findings", "outcomes", "conclusions", "data"],
            
            # General academic terms
            "analysis": ["examination", "evaluation", "assessment", "review"],
            "theory": ["framework", "model", "concept", "principle"],
            "method": ["approach", "technique", "procedure", "methodology"],
            "concept": ["idea", "notion", "principle", "theory"]
        }
    
    def _get_wordnet_synonyms(self, word: str) -> List[str]:
        """Get synonyms from WordNet."""
        if not self.use_wordnet:
            return []
            
        self._initialize_components()
        synonyms = set()
        
        try:
            # Get synsets for the word
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    # Filter out the original word and very short synonyms
                    if synonym.lower() != word.lower() and len(synonym) > 2:
                        synonyms.add(synonym)
            
            return list(synonyms)[:self.max_expansions]
            
        except Exception as e:
            logger.warning(f"Error getting WordNet synonyms for '{word}': {e}")
            return []
    
    def _get_domain_expansions(self, query: str) -> List[str]:
        """Get domain-specific expansions."""
        query_lower = query.lower()
        expansions = set()
        
        for key, expansion_list in self.domain_expansions.items():
            if key in query_lower:
                expansions.update(expansion_list)
                
        # Remove terms already in query
        query_terms = set(query_lower.split())
        expansions = expansions - query_terms
        
        return list(expansions)[:self.max_expansions]
    
    def _get_semantic_expansions(self, query: str, corpus_embeddings: Optional[np.ndarray] = None,
                               corpus_terms: Optional[List[str]] = None) -> List[str]:
        """Get semantically similar terms using embeddings."""
        if not self.use_semantic or self.embedding_model is None:
            return []
            
        try:
            # If no corpus provided, use a default set of academic terms
            if corpus_terms is None:
                corpus_terms = [
                    "research", "study", "analysis", "theory", "method", "concept",
                    "psychology", "cognitive", "behavior", "learning", "memory",
                    "artificial intelligence", "machine learning", "algorithm", "data",
                    "experiment", "hypothesis", "results", "findings", "evidence"
                ]
            
            # Generate embeddings
            query_embedding = self.embedding_model.encode([query])
            
            if corpus_embeddings is None:
                corpus_embeddings = self.embedding_model.encode(corpus_terms)
            
            # Compute similarities
            similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
            
            # Get top similar terms
            top_indices = np.argsort(similarities)[::-1][:self.max_expansions]
            expansions = [corpus_terms[i] for i in top_indices if similarities[i] > 0.3]  # Threshold
            
            # Remove terms already in query
            query_lower = query.lower()
            expansions = [term for term in expansions if term.lower() not in query_lower]
            
            return expansions
            
        except Exception as e:
            logger.warning(f"Error getting semantic expansions: {e}")
            return []
    
    def expand_query(self, query: str, 
                    expansion_types: List[str] = None,
                    corpus_terms: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Expand a query using multiple strategies.
        
        Args:
            query: Original query string
            expansion_types: Types of expansion to use ['wordnet', 'domain', 'semantic']
            corpus_terms: Optional corpus terms for semantic expansion
            
        Returns:
            Dictionary with expanded query and metadata
        """
        if expansion_types is None:
            expansion_types = ['wordnet', 'domain', 'semantic']
        
        self.stats["queries_processed"] += 1
        
        # Clean and tokenize query
        query_clean = re.sub(r'[^\w\s]', ' ', query).strip()
        query_tokens = [token.lower() for token in query_clean.split() if len(token) > 2]
        
        all_expansions = []
        expansion_details = {}
        
        # 1. WordNet synonyms
        if 'wordnet' in expansion_types and self.use_wordnet:
            wordnet_expansions = []
            for token in query_tokens:
                synonyms = self._get_wordnet_synonyms(token)
                wordnet_expansions.extend(synonyms)
            
            wordnet_expansions = list(set(wordnet_expansions))[:self.max_expansions]
            all_expansions.extend(wordnet_expansions)
            expansion_details['wordnet'] = wordnet_expansions
            self.stats["wordnet_expansions"] += len(wordnet_expansions)
        
        # 2. Domain-specific expansions
        if 'domain' in expansion_types:
            domain_expansions = self._get_domain_expansions(query)
            all_expansions.extend(domain_expansions)
            expansion_details['domain'] = domain_expansions
            self.stats["domain_expansions"] += len(domain_expansions)
        
        # 3. Semantic expansions
        if 'semantic' in expansion_types and self.use_semantic:
            semantic_expansions = self._get_semantic_expansions(query, corpus_terms=corpus_terms)
            all_expansions.extend(semantic_expansions)
            expansion_details['semantic'] = semantic_expansions
            self.stats["semantic_expansions"] += len(semantic_expansions)
        
        # Remove duplicates and create expanded query
        unique_expansions = list(set(all_expansions))
        self.stats["total_expansions_added"] += len(unique_expansions)
        
        # Create multiple query variants
        expanded_queries = {
            'original': query,
            'expanded_simple': f"{query} {' '.join(unique_expansions)}",
            'expanded_weighted': self._create_weighted_query(query, unique_expansions),
            'expansion_terms': unique_expansions,
            'expansion_details': expansion_details
        }
        
        logger.debug(f"Query expansion: '{query}' -> {len(unique_expansions)} expansions")
        
        return expanded_queries
    
    def _create_weighted_query(self, original: str, expansions: List[str]) -> str:
        """Create a weighted query with original terms having higher priority."""
        # Simple weighting: repeat original terms
        weighted_query = f"{original} {original} {' '.join(expansions)}"
        return weighted_query
    
    def expand_multiple_queries(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Expand multiple queries efficiently."""
        logger.info(f"Expanding {len(queries)} queries")
        
        expanded_queries = []
        for query in queries:
            expanded = self.expand_query(query, **kwargs)
            expanded_queries.append(expanded)
        
        return expanded_queries
    
    def create_expansion_variants(self, query: str) -> List[str]:
        """Create multiple expansion variants for A/B testing."""
        expanded = self.expand_query(query)
        
        variants = [
            expanded['original'],
            expanded['expanded_simple'],
            expanded['expanded_weighted']
        ]
        
        # Create domain-only and semantic-only variants
        if expanded['expansion_details'].get('domain'):
            domain_query = f"{query} {' '.join(expanded['expansion_details']['domain'])}"
            variants.append(domain_query)
        
        if expanded['expansion_details'].get('semantic'):
            semantic_query = f"{query} {' '.join(expanded['expansion_details']['semantic'])}"
            variants.append(semantic_query)
        
        return variants
    
    def analyze_query_coverage(self, query: str, document_texts: List[str]) -> Dict[str, Any]:
        """Analyze how well query terms are covered in a document corpus."""
        expanded = self.expand_query(query)
        
        # Tokenize all documents
        all_terms = set()
        for doc_text in document_texts:
            doc_terms = set(re.findall(r'\b\w+\b', doc_text.lower()))
            all_terms.update(doc_terms)
        
        # Check coverage
        original_terms = set(query.lower().split())
        expansion_terms = set(expanded['expansion_terms'])
        
        original_coverage = len(original_terms.intersection(all_terms)) / max(len(original_terms), 1)
        expansion_coverage = len(expansion_terms.intersection(all_terms)) / max(len(expansion_terms), 1)
        
        return {
            'original_coverage': original_coverage,
            'expansion_coverage': expansion_coverage,
            'corpus_size': len(all_terms),
            'original_terms_found': list(original_terms.intersection(all_terms)),
            'expansion_terms_found': list(expansion_terms.intersection(all_terms))
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query expansion statistics."""
        stats = self.stats.copy()
        if stats["queries_processed"] > 0:
            stats["avg_expansions_per_query"] = stats["total_expansions_added"] / stats["queries_processed"]
        return stats

class AdaptiveQueryExpander(QueryExpander):
    """Query expander that adapts based on search performance."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.performance_history = []
        self.adaptive_weights = {
            'wordnet': 1.0,
            'domain': 1.0,
            'semantic': 1.0
        }
    
    def record_search_performance(self, query: str, expansion_type: str, 
                                performance_score: float):
        """Record search performance for adaptive learning."""
        self.performance_history.append({
            'query': query,
            'expansion_type': expansion_type,
            'performance': performance_score
        })
        
        # Update adaptive weights (simple moving average)
        if len(self.performance_history) >= 10:
            recent_performance = self.performance_history[-10:]
            for exp_type in self.adaptive_weights:
                type_performances = [p['performance'] for p in recent_performance 
                                  if p['expansion_type'] == exp_type]
                if type_performances:
                    avg_performance = np.mean(type_performances)
                    # Adjust weight based on performance (simple linear scaling)
                    self.adaptive_weights[exp_type] = max(0.1, min(2.0, avg_performance))
    
    def expand_query_adaptive(self, query: str, **kwargs) -> Dict[str, Any]:
        """Expand query using adaptive weights."""
        expanded = self.expand_query(query, **kwargs)
        
        # Apply adaptive weighting to expansion terms
        weighted_expansions = []
        
        for exp_type, terms in expanded['expansion_details'].items():
            weight = self.adaptive_weights.get(exp_type, 1.0)
            # Repeat terms based on weight (simple approach)
            repetitions = max(1, int(weight))
            weighted_expansions.extend(terms * repetitions)
        
        expanded['adaptive_expanded'] = f"{query} {' '.join(weighted_expansions)}"
        expanded['adaptive_weights'] = self.adaptive_weights.copy()
        
        return expanded

def integrate_query_expansion_into_retrieval(query: str, 
                                           expansion_config: Dict[str, Any] = None) -> List[str]:
    """
    Integrate query expansion into retrieval pipeline.
    
    Args:
        query: Original search query
        expansion_config: Configuration for query expansion
        
    Returns:
        List of query variants to try
    """
    if expansion_config is None:
        expansion_config = {
            'use_wordnet': True,
            'use_semantic': True,
            'max_expansions': 3
        }
    
    # Initialize expander
    expander = QueryExpander(
        use_wordnet=expansion_config.get('use_wordnet', True),
        use_semantic=expansion_config.get('use_semantic', True),
        max_expansions=expansion_config.get('max_expansions', 3)
    )
    
    # Generate query variants
    variants = expander.create_expansion_variants(query)
    
    logger.info(f"Generated {len(variants)} query variants for: '{query}'")
    for i, variant in enumerate(variants):
        logger.debug(f"  Variant {i+1}: {variant}")
    
    return variants

def test_query_expansion():
    """Test function for query expansion capabilities."""
    logging.basicConfig(level=logging.INFO)
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "memory and cognition",
        "artificial intelligence research",
        "psychological behavior analysis"
    ]
    
    print("Testing Query Expansion")
    print("=" * 50)
    
    expander = QueryExpander()
    
    for query in test_queries:
        print(f"\nOriginal query: '{query}'")
        expanded = expander.expand_query(query)
        
        print(f"Expanded query: '{expanded['expanded_simple']}'")
        print(f"Expansion details:")
        for exp_type, terms in expanded['expansion_details'].items():
            if terms:
                print(f"  {exp_type}: {terms}")
    
    print(f"\nExpansion statistics: {expander.get_stats()}")

if __name__ == "__main__":
    test_query_expansion()