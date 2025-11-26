"""
Test script to verify summarization module integration.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from summarization.summarization import Summarization


def test_summarization_module():
    """Test the Summarization module independently."""
    print("=" * 80)
    print("Testing Summarization Module")
    print("=" * 80)
    
    # Initialize summarizer
    summarizer = Summarization()
    
    # Test text
    test_text = """
    This is a beautiful tourist attraction with amazing scenery. 
    The location is perfect for families and the staff is very friendly. 
    The facilities are well-maintained and clean. However, the price is a bit expensive. 
    Overall, it's a great place to visit and I highly recommend it to anyone looking for 
    a peaceful getaway. The view from the top is absolutely stunning and worth the climb.
    """
    
    print(f"\nTest Text:\n{test_text.strip()}\n")
    
    # Check available models
    print("\nChecking available models...")
    available = summarizer.get_available_models()
    print(f"Extractive methods: {available['extractive']}")
    print(f"BART models available:")
    for variant, status in available['bart'].items():
        print(f"  - {variant}: {'✓' if status else '✗'}")
    print(f"PEGASUS models available:")
    for variant, status in available['pegasus'].items():
        print(f"  - {variant}: {'✓' if status else '✗'}")
    
    # Test extractive methods (lightweight, no GPU needed)
    print("\n" + "=" * 80)
    print("Testing Extractive Methods")
    print("=" * 80)
    
    print("\n[1] First Sentence Method:")
    summary = summarizer.summarize(test_text, method='first_sentence')
    print(f"Summary: {summary}")
    
    print("\n[2] TextRank Method:")
    summary = summarizer.summarize(test_text, method='textrank')
    print(f"Summary: {summary}")
    
    # Test transformer models (if available)
    print("\n" + "=" * 80)
    print("Testing Transformer Methods (if models available)")
    print("=" * 80)
    
    if available['bart']['sampled']:
        print("\n[3] BART Sampled (Fine-tuned):")
        try:
            summary = summarizer.summarize(test_text, method='bart_sampled')
            print(f"Summary: {summary}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("\n[3] BART Sampled model not found. Skipping.")
        
    if available['bart']['full']:
        print("\n[4] BART Full (Fine-tuned):")
        try:
            summary = summarizer.summarize(test_text, method='bart_full')
            print(f"Summary: {summary}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("\n[4] BART Full model not found. Skipping.")
    
    print("\n[4] Best Model (default):")
    try:
        summary = summarizer.summarize(test_text, method='best')
        print(f"Summary: {summary}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_summarization_module()
