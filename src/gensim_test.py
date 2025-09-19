from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import logging

# Set up logging to see progress
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def test_word2vec():
    # Sample corpus - list of tokenized sentences
    sentences = [
        simple_preprocess("the quick brown fox jumps over the lazy dog"),
        simple_preprocess("i like to watch movies and tv shows"),
        simple_preprocess("the movie was great"),
        simple_preprocess("artificial intelligence is transforming technology"),
        simple_preprocess("machine learning models require training data"),
        simple_preprocess("deep learning is a subset of machine learning")
    ]
    
    print(f"Sample sentences: {sentences[:2]}")
    
    # Train Word2Vec model
    print("Training Word2Vec model...")
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    # Test model functionality
    print("\nTesting model functionality:")
    
    # Get vector for a word
    word = "movie"
    if word in model.wv:
        print(f"Vector for '{word}' (first 5 elements): {model.wv[word][:5]}")
    
    # Find similar words
    words = ["learning", "artificial", "movie"]
    for word in words:
        if word in model.wv:
            print(f"\nWords most similar to '{word}':")
            similar_words = model.wv.most_similar(word, topn=3)
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
    
    print("\nGensim Word2Vec is working correctly!")
    
    return model

if __name__ == "__main__":
    model = test_word2vec()