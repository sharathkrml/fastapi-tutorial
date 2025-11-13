import logging
from lancedb import connect
from typing import Optional
from pathlib import Path

# Get the absolute path to the database directory
# This file is in app/utils/, so we go up one level to app/, then to utils/lancedb_data
_current_dir = Path(__file__).parent
db_path = str(_current_dir / "lancedb_data")

# Set up logger
logger = logging.getLogger(__name__)

# Lazy-loaded model cache - only loaded on first use
_model: Optional[object] = None


def _get_model():
    """Lazy load the embedding model only when needed (not on module import)."""
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2' (first use)")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("SentenceTransformer model loaded successfully")
    else:
        logger.debug("Using cached SentenceTransformer model")
    return _model


def fetch_vocab_from_vector_db(query: str, level: str = "A1", n: int = 10) -> list[str]:
    """
    Fetch vocabulary words from the vector database filtered by 'level'.

    Args:
        query (str): Search query for vocab.
        level (str): Language level to filter by ("A1", "A2", "B1", or "B2").
        n (int): Maximum number of vocab items to return.

    Returns:
        list[str]: List of vocabulary words matching the query and level.
    """
    logger.info(f"Fetching vocabulary from vector DB: query='{query}', level={level}, n={n}")
    
    if level not in {"A1", "A2", "B1", "B2"}:
        logger.error(f"Invalid level provided: {level}")
        raise ValueError("Invalid level. Must be 'A1', 'A2', 'B1', or 'B2'.")
    
    dbName = level + "_MINIMAL_vocabulary"
    logger.debug(f"Connecting to LanceDB at path: {db_path}")
    db = connect(db_path)
    
    # LanceDB table names match directory names, which include .lance extension
    table_name = dbName + ".lance"
    logger.debug(f"Attempting to open table: {table_name}")
    try:
        table = db.open_table(table_name)
        logger.debug(f"Successfully opened table: {table_name}")
    except ValueError as e:
        # Fallback: try without .lance extension in case LanceDB handles it automatically
        logger.warning(f"Failed to open table '{table_name}', trying without .lance extension: {e}")
        table = db.open_table(dbName)
        logger.debug(f"Successfully opened table: {dbName}")
    
    # Lazy load model and encode query (model is cached after first use)
    logger.debug("Encoding query using embedding model")
    model = _get_model()
    # Use convert_to_numpy=True for faster encoding and direct compatibility with LanceDB
    query_vector = model.encode(query, convert_to_numpy=True)
    logger.debug(f"Query encoded, vector shape: {query_vector.shape}")
    
    logger.debug(f"Searching table with limit={n}")
    results = table.search(query_vector).limit(n).to_pandas()
    logger.info(f"Search returned {len(results)} results")

    # only take german_term & english_translation
    results = results[['german_term', 'english_translation']]
    logger.debug(f"Results columns: {results.columns.tolist()}")
    logger.debug(f"Results preview:\n{results}")
    
    # Extract vocabulary words from results (adjust column name based on your schema)
    # Assuming results have a column with vocabulary words
    if not results.empty:
        # Try common column names for vocabulary
        vocab_column = None
        for col in ['word', 'vocab', 'text', 'term', 'vocabulary']:
            if col in results.columns:
                vocab_column = col
                break
        
        if vocab_column:
            vocab_list = results[vocab_column].head(n).tolist()
            logger.info(f"Extracted {len(vocab_list)} vocabulary items from column '{vocab_column}'")
            return vocab_list
        else:
            # Return first column if no standard name found
            vocab_list = results.iloc[:, 0].head(n).tolist()
            logger.warning(f"No standard vocab column found, using first column. Extracted {len(vocab_list)} items")
            return vocab_list
    
    logger.warning("No results returned from vector database search")
    return []


# if __name__ == "__main__":
#     results = fetch_vocab_from_vector_db("food", "B2")
#     for result in results:
#         print(result)