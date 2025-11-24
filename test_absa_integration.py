#!/usr/bin/env python3
"""
Test script for ABSA integration
"""

import sys
import os

# Add src to path
sys.path.append('src')

try:
    from main import IndoTripSight
    dependencies_available = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Testing import structure only...")
    dependencies_available = False

def test_absa_integration():
    """Test ABSA functionality with different models"""

    if not dependencies_available:
        print("Cannot test ABSA functionality due to missing dependencies.")
        return

    # Initialize
    indo_trip_sight = IndoTripSight()

    # Add a test review
    test_review = "The food was delicious but the service was slow. The atmosphere was nice though."
    indo_trip_sight.add_review(test_review)

    print("Testing ABSA Integration")
    print("=" * 50)
    print(f"Test Review: {test_review}")
    print()

    # Test different models
    models_to_test = ['electra', 'tfidf_lr']

    for model in models_to_test:
        try:
            print(f"Testing {model.upper()} model:")
            print("-" * 30)
            result = indo_trip_sight.absa(model)
            print(result)
            print()
        except Exception as e:
            print(f"Error with {model}: {e}")
            print()

def test_imports():
    """Test that all required imports are accessible"""

    print("Testing imports and constants...")
    print("-" * 30)

    try:
        from main import LIST_MODEL_ABSA, VALID_ASPECTS
        print("✓ Constants imported successfully")
        print(f"  Available ABSA models: {LIST_MODEL_ABSA}")
        print(f"  Number of valid aspects: {len(VALID_ASPECTS)}")
    except Exception as e:
        print(f"✗ Import error: {e}")

    try:
        # Test IndoTripSight class structure
        indo_trip_sight = IndoTripSight()
        print("✓ IndoTripSight class instantiated")
        print(f"  Has valid_aspects: {hasattr(indo_trip_sight, 'valid_aspects')}")
        print(f"  Has inference methods: {hasattr(indo_trip_sight, 'perform_absa_inference')}")
    except Exception as e:
        print(f"✗ Class instantiation error: {e}")

if __name__ == "__main__":
    test_imports()
    print()
    test_absa_integration()
