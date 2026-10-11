from sentence_transformers import SentenceTransformer, util
import warnings

# Suppress warnings from sentence-transformers if any
warnings.filterwarnings("ignore")
 
class LoopDetector:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")

    def detect_loop(self, history: list[str], current: str, threshold: float = 0.95) -> float:
        """
        Checks if the current prompt is semantically similar to any in the history.
        Returns the maximum similarity score found.
        """
        if not history:
            return 0.0

        # Encode current prompt and history
        current_embedding = self.model.encode(current, convert_to_tensor=True)
        history_embeddings = self.model.encode(history, convert_to_tensor=True)

        # Calculate cosine similarities
        cosine_scores = util.cos_sim(current_embedding, history_embeddings)
        
        # Find maximum similarity
        max_similarity = float(cosine_scores.max())
        
        return max_similarity
