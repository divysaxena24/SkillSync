"""
Verification Script for SkillSync Serialized NLP Model Artifact
Loads models/skillsync_model.pkl and validates its components, metadata,
fitted TF-IDF vocabulary, and taxonomy mappings.
"""

import sys
import pickle
from pathlib import Path


def verify_pickle(model_path="models/skillsync_model.pkl"):
    path = Path(model_path)
    print(f"[*] Checking model artifact at: {path.resolve()}")

    if not path.exists():
        print(f"[!] Error: Model artifact not found at {path}")
        return False

    size_kb = path.stat().st_size / 1024
    print(f"[+] Artifact found! File size: {size_kb:.2f} KB")

    try:
        with open(path, "rb") as f:
            artifact = pickle.load(f)
    except Exception as e:
        print(f"[!] Failed to deserialize model with pickle: {e}")
        return False

    # Check top-level keys
    required_keys = ["metadata", "vectorizer", "vocabulary_size", "taxonomy"]
    for k in required_keys:
        if k not in artifact:
            print(f"[!] Missing expected key in artifact: {k}")
            return False
        print(f"[+] Validated key: '{k}'")

    metadata = artifact["metadata"]
    print(f"[*] Model Name: {metadata.get('model_name')}")
    print(f"[*] Version: {metadata.get('version')}")
    print(f"[*] Python Version: {metadata.get('python_version')}")
    print(f"[*] Scikit-Learn Version: {metadata.get('sklearn_version')}")
    print(f"[*] Formula: {metadata.get('formula')}")

    # Validate vectorizer
    vec = artifact["vectorizer"]
    vocab_len = len(vec.vocabulary_)
    print(f"[+] Fitted TfidfVectorizer loaded successfully with {vocab_len:,} vocabulary terms.")

    # Validate transform capability
    test_terms = ["python developer fastapi docker", "machine learning scikit-learn"]
    vectors = vec.transform(test_terms)
    print(f"[+] Test transform shape: {vectors.shape} (Sparse matrix verification passed)")

    # Validate taxonomy
    tax = artifact["taxonomy"]
    skills_len = len(tax.get("canonical_skills", []))
    aliases_len = len(tax.get("skill_aliases", {}))
    roles_len = len(tax.get("role_taxonomy", {}))
    print(f"[+] Taxonomy loaded: {skills_len} canonical skills, {aliases_len} aliases, {roles_len} role domains.")

    print("\n[SUCCESS] Model artifact verification completed successfully without errors!")
    return True


if __name__ == "__main__":
    success = verify_pickle()
    sys.exit(0 if success else 1)

