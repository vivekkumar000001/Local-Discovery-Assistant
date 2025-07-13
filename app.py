from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import spacy
import re
from mock_data import MOCK_DATA

app = Flask(__name__)

# Load ML models
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
nlp = spacy.load("en_core_web_sm")

def parse_query(query):
    """Extract intent and entities from user query"""
    # Intent classification
    intent_labels = ["restaurant", "dentist", "event", "food_delivery"]
    intent_result = classifier(query, intent_labels)
    intent = intent_result['labels'][0]
    
    # Entity extraction
    doc = nlp(query)
    locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC"]]
    location = locations[0] if locations else None
    
    # Extract distance
    distance = None
    distance_match = re.search(r"within (\d+) (km|m|kilometer)", query, re.IGNORECASE)
    if distance_match:
        distance = int(distance_match.group(1))
    
    # Extract constraints
    constraints = []
    if "vegetarian" in query.lower():
        constraints.append("vegetarian")
    if "late-night" in query.lower():
        constraints.append("late-night")
    if "top-rated" in query.lower() or "top rated" in query.lower():
        constraints.append("top-rated")
    if "trending" in query.lower():
        constraints.append("trending")
    
    return {
        "intent": intent,
        "location": location,
        "distance": distance,
        "constraints": constraints
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def process_query():
    user_query = request.json["query"]
    
    # Parse query
    params = parse_query(user_query)
    
    # Filter mock data
    results = []
    category_data = MOCK_DATA.get(params["intent"], [])
    
    for item in category_data:
        # Location filter
        if params["location"] and params["location"].lower() not in item["location"].lower():
            continue
        
        # Constraint filter
        if params["constraints"]:
            if not all(constraint in item["tags"] for constraint in params["constraints"]):
                continue
                
        results.append(item)
        if len(results) >= 3:
            break
    
    # Format response
    if results:
        response = "Here are my recommendations:\n"
        for i, item in enumerate(results, 1):
            response += f"\n{i}. {item['name']} ({item['rating']}⭐)\n   📍 {item['address']}"
            if "opening_hours" in item:
                response += f"\n   ⏰ {item['opening_hours']}"
    else:
        response = "I couldn't find any matching results. Could you try different parameters?"
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)