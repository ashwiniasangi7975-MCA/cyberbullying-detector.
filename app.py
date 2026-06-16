import math
import re
from flask import Flask, render_template, request

app = Flask(__name__)

# --- 🤖 MINIMALIST BUILT-IN MACHINE LEARNING BRAIN (NAIVE BAYES + CATEGORIES) ---
# A curated dictionary dataset representing trained classification categories
TRAINED_CATEGORIES = {
    "Cyberstalking/Harassment": [
        "track", "watching", "know where you live", "follow", "find you", 
        "always watching", "behind you", "obsessed", "never safe", "creep"
    ],
    "Trolling/Hate Speech": [
        "hate", "loser", "ugly", "stupid", "worthless", "dumb", "idiot", 
        "destroy", "cancel", "clown", "trash", "garbage", "die", "kill"
    ],
    "Safe/Clean Conversation": [
        "hello", "good morning", "nice to meet you", "thank you", "awesome", 
        "beautiful", "help", "project", "code", "working", "happy", "great"
    ]
}

def clean_and_tokenize(text):
    """Cleans punctuation out of text and splits it into individual words."""
    return re.sub(r'[^\w\s]', '', text.lower()).split()

def predict_cyberbullying_ai(user_input):
    """
    An algorithmic Text Classifier that calculates probability weights 
    for text categories, simulating a Multinomial Naive Bayes model.
    """
    tokens = clean_and_tokenize(user_input)
    if not tokens:
        return "Safe/Clean Conversation", "No text provided to evaluate."

    best_category = "Safe/Clean Conversation"
    highest_probability = -float('inf')
    
    # Calculate text probability loops for each category
    for category, words_list in TRAINED_CATEGORIES.items():
        category_score = 0.0
        for token in tokens:
            # Frequency count with Laplacian Smoothing (+1) to prevent zero-multiplication errors
            word_frequency = words_list.count(token) + 1
            total_category_words = len(words_list) + 100 
            category_score += math.log(word_frequency / total_category_words)
        
        if category_score > highest_probability:
            highest_probability = category_score
            best_category = category

    # Fallback verification switch if vocabulary match density is non-existent
    has_bad_words = any(t in TRAINED_CATEGORIES["Cyberstalking/Harassment"] or 
                        t in TRAINED_CATEGORIES["Trolling/Hate Speech"] for t in tokens)
    
    if best_category == "Safe/Clean Conversation" and has_bad_words:
        best_category = "Trolling/Hate Speech"

    # Context Message Selection Generator
    if best_category == "Cyberstalking/Harassment":
        desc = "Flagged by AI: Text displays intimidation patterns, location tracing threats, or unwanted obsessive behavior."
    elif best_category == "Trolling/Hate Speech":
        desc = "Flagged by AI: Text contains intentional targeted insults, heavy toxic language, or aggressive speech."
    else:
        desc = "Verified Clean: The text doesn't match any harassment vector metrics and is perfectly safe for public feeds."

    return best_category, desc

# --- 🌐 FLASK WEB ROUTING ROUTINES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.form['text']
    category, analysis_description = predict_cyberbullying_ai(user_text)
    
    # Color matrix selector mapping states to UI colors
    if category == "Safe/Clean Conversation":
        ui_color = "#00f0ff"  # Cyber Neon Cyan
        status_label = "CLEAN & SAFE"
    elif category == "Trolling/Hate Speech":
        ui_color = "#ff0055"  # Neon Crimson Pink
        status_label = "CYBERBULLYING DETECTED"
    else:
        ui_color = "#ffaa00"  # Neon warning orange
        status_label = "CYBERSTALKING / HARASSMENT DETECTED"

    return render_template('index.html', 
                           evaluated=True, 
                           text=user_text, 
                           result=status_label, 
                           explanation=analysis_description, 
                           theme_color=ui_color)

if __name__ == '__main__':
    app.run(debug=True)
