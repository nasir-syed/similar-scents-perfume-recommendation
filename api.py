from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import re

app = Flask(__name__)
perfume_types = ['eau de parfum', 'extrait de parfum', 'parfum', 'extrait', 'perfume oil', 'fragrance oil', 
                 'eau de toilette', 'parfum intense', 'parfum extrait', 'eau de cologne', 'roll-on perfume oil', 
                 'pure oud oil', 'eau fraiche', 'cologne absolute']

pattern = '|'.join(perfume_types)
pattern += '|[./\'-]'


with open('similarity_df.pkl', 'rb') as f:
    similarity_df = pickle.load(f)


perfume_data = pd.read_csv('modified_perfume_data.csv')

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        liked_perfumes = request.json['liked_perfumes']
        num = request.json.get('num', 5)  

        entered_perfume_name = liked_perfumes[0]
        entered_perfume_details = get_perfume_details(entered_perfume_name)
        
        recommendations = recommend_perfumes(liked_perfumes, similarity_df, num)
        
        recommendations_with_details = [entered_perfume_details]
        for perfume in recommendations:
            perfume_info = get_perfume_details(perfume)
            recommendations_with_details.append(perfume_info)

        return jsonify(recommendations_with_details)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def clean_perfume_name(name, pattern):
    cleaned_name = re.sub(pattern, '', name, flags=re.IGNORECASE).strip()
    return cleaned_name

def recommend_perfumes(liked_perfumes, similarity_df, num=5):
    liked_perfumes = [clean_perfume_name(perfume.lower(), pattern) for perfume in liked_perfumes]
    if not all(perfume in similarity_df.index for perfume in liked_perfumes):
        raise ValueError("One or more liked perfumes are not in the similarity matrix.")

    agg_scores = similarity_df.loc[liked_perfumes].sum(axis=0)
    agg_scores = agg_scores.drop(labels=liked_perfumes)
    recomms = agg_scores.sort_values(ascending=False).head(num)
    
    return recomms.index.tolist()

def get_perfume_details(perfume_name):
    
    perfume_row = perfume_data[perfume_data['Name'].str.lower() == perfume_name.lower()]
    if perfume_row.empty:
        return {
            "Name": perfume_name,
            "Brand": "Unknown",
            "Notes": "Unknown",
            "Image URL": "Unknown"
        }
    
    perfume_info = perfume_row.iloc[0]
    return {
        "Name": perfume_info['Name'],
        "Brand": perfume_info['Brand'],
        "Notes": perfume_info['Notes'],
        "Image URL": perfume_info['Image URL']
    }

if __name__ == '__main__':
    app.run(debug=True)
