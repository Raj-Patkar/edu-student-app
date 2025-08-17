from flask import Flask, request, jsonify
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
file_path = os.path.join(os.path.dirname(__file__), "courses.csv")
courses = pd.read_csv(file_path)

# Handle missing values
courses["description"] = courses["description"].fillna("")
courses["title"] = courses["title"].fillna("Untitled")

# Validate required columns
required_columns = {"id", "title", "description"}
if not required_columns.issubset(courses.columns):
    raise ValueError(f"courses.csv must contain columns: {required_columns}")

# Create TF-IDF and similarity matrix
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(courses["description"])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route("/")
def home():
    return {"message": "Edu Analytics ML API is running!"}

@app.route("/recommendations", methods=["GET"])
def recommendations():
    try:
        course_id = int(request.args.get("course_id", 1))
        course_index = courses[courses["id"] == course_id].index

        if course_index.empty:
            return jsonify({"error": "Course ID not found"}), 404

        idx = course_index[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1: min(len(sim_scores), 3)]  # top 2 recommendations

        recs = [{
            "id": int(courses.iloc[i[0]]["id"]),
            "title": courses.iloc[i[0]]["title"],
            "description": courses.iloc[i[0]]["description"]
        } for i in sim_scores]

        return jsonify({"course_id": course_id, "recommendations": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analytics")
def analytics():
    try:
        total_courses = len(courses)
        avg_desc_length = courses["description"].str.len().mean()
        return jsonify({
            "total_courses": total_courses,
            "avg_description_length": round(avg_desc_length, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Azure dynamic port support
    app.run(host="0.0.0.0", port=port)
