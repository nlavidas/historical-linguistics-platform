"""
Test LightSide integration with proper sample data
"""

import pandas as pd
from pathlib import Path
from lightside_integration import LightSidePlatformIntegration

# Create test data with more samples
test_data = [
    # Ancient Greek texts
    ("Ἐν ἀρχῇ ἦν ὁ λόγος", "ancient_greek"),
    ("καὶ ὁ λόγος ἦν πρὸς τὸν θεόν", "ancient_greek"),
    ("πάντα δι' αὐτοῦ ἐγένετο", "ancient_greek"),
    ("ἐν αὐτῷ ζωὴ ἦν", "ancient_greek"),
    ("καὶ ἡ ζωὴ ἦν τὸ φῶς τῶν ἀνθρώπων", "ancient_greek"),
    
    # Latin texts
    ("In principio erat Verbum", "latin"),
    ("et Verbum erat apud Deum", "latin"),
    ("omnia per ipsum facta sunt", "latin"),
    ("in ipso vita erat", "latin"),
    ("et vita erat lux hominum", "latin"),
    
    # English texts
    ("In the beginning was the Word", "english"),
    ("and the Word was with God", "english"),
    ("all things were made through him", "english"),
    ("in him was life", "english"),
    ("and the life was the light of men", "english"),
    
    # French texts
    ("Au commencement était la Parole", "french"),
    ("et la Parole était avec Dieu", "french"),
    ("toutes choses ont été faites par elle", "french"),
    ("en elle était la vie", "french"),
    ("et la vie était la lumière des hommes", "french"),
]

# Save as CSV
csv_path = Path("Z:/models/lightside/test_data.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(test_data, columns=["text", "label"])
df.to_csv(csv_path, index=False)
print(f"✓ Created test data: {csv_path}")
print(f"  {len(df)} samples, {len(df['label'].unique())} categories")

# Test the integration
print("\n" + "="*70)
print("TESTING LIGHTSIDE INTEGRATION")
print("="*70)

# Initialize
lightside = LightSidePlatformIntegration()

# Train a classifier
print("\n1. Training classifier...")
from lightside_integration import LightSideConfig
config = LightSideConfig(
    algorithm="naive_bayes",
    cross_validation_folds=4  # Use fewer folds for small dataset
)
result = lightside.train_from_csv(
    name="language_detector",
    csv_path=str(csv_path),
    config=config
)

if result:
    print(f"✓ Training successful!")
    if 'cv_accuracy' in result:
        print(f"  Cross-validation accuracy: {result['cv_accuracy']:.3f}")
    if 'test_accuracy' in result:
        print(f"  Test accuracy: {result['test_accuracy']:.3f}")
    print(f"  Model saved successfully!")
else:
    print(f"✗ Training failed")

# Test prediction
print("\n2. Testing predictions...")
test_texts = [
    "τὸ φῶς ἐν τῇ σκοτίᾳ φαίνει",  # Greek
    "lux in tenebris lucet",  # Latin
    "the light shines in darkness",  # English
    "la lumière luit dans les ténèbres",  # French
]

for text in test_texts:
    try:
        result = lightside.predict("language_detector", text)
        if result and 'prediction' in result:
            print(f"  Text: '{text[:30]}...'")
            print(f"  → Predicted: {result['prediction']}")
            if 'confidence' in result:
                print(f"    Confidence: {result['confidence']:.3f}")
        else:
            print(f"  ✗ Prediction failed for '{text[:30]}...'")
    except Exception as e:
        print(f"  ✗ Error predicting '{text[:30]}...': {e}")

print("\n" + "="*70)
print("✓ LightSide integration test complete!")
print("="*70)
