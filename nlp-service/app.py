from fastapi import FastAPI

app = FastAPI(title="NLP Translation Service")

@app.post("/correct")
def correct_grammar(data: dict):
    gloss = data["gloss"]
    # Confidence comes from the vision model's predicted class probability;
    # this service only reformats the gloss, so it passes confidence through.
    confidence = float(data.get("confidence", 0.0))
    if not gloss:
        return {"sentence": "", "confidence": 0.0}
    sentence = gloss.replace("_", " ").capitalize()
    return {"sentence": sentence, "confidence": confidence}


@app.post("/compose")
def compose_sentence(data: dict):
    """Fingerspelled words -> one sentence. Deliberately naive: joins words in
    signing order, capitalizes, adds a period."""
    words = [str(w).strip().lower() for w in data.get("words", []) if str(w).strip()]
    if not words:
        return {"sentence": ""}
    return {"sentence": " ".join(words).capitalize() + "."}
