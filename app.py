from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import json
import cohere

load_dotenv()

app = Flask(__name__)

co = cohere.Client(os.getenv("COHERE_API_KEY"))

@app.route('/')
def landing():
    return render_template("cover.html")

@app.route('/main')
def main_page():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    options = data.get('options', [])
    criteria = data.get('criteria', [])

    prompt = f"""
You are a helpful assistant helping the user decide between: {', '.join(options)}.
Consider the criteria: {', '.join(criteria)}.

Respond ONLY in valid JSON format with 3 pros and 3 cons for each option like:
{{
  "Option 1": {{
    "pros": ["..."],
    "cons": ["..."]
  }},
  ...
}}
"""

    try:
        response = co.generate(
            model='command-r-plus',
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )

        text = response.generations[0].text.strip()

        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return jsonify({"error": "Cohere didn't return valid JSON."}), 500

        json_str = match.group(0)
        pros_cons_data = json.loads(json_str)

        scores = {opt: len(v["pros"]) - len(v["cons"]) for opt, v in pros_cons_data.items()}
        best_option = max(scores, key=scores.get)

        return jsonify({
            "result_json": json.dumps(pros_cons_data),
            "scores": scores,
            "best_option": best_option
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
