import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("courses_clean.csv")

# Define 3 First-Year Profiles
profiles = [
    {
        "name": "Profile 1: CS & AI Enthusiast",
        "interests": "programming, artificial intelligence, python, computers",
        "completed": ["GE1401"],
        "relevant_courses": ["CS1302", "CS2310", "GE2340", "CS2115"] # Ground truth for evaluation
    },
    {
        "name": "Profile 2: Business & Finance Student",
        "interests": "accounting, finance, business, management, financial statements",
        "completed": ["GE1401"],
        "relevant_courses": ["CB2100"]
    },
    {
        "name": "Profile 3: Engineering & Math Student",
        "interests": "math, calculus, linear algebra, matrices, engineering models",
        "completed": ["GE1401"],
        "relevant_courses": ["MA1503", "MA1508"]
    }
]

def evaluate_profile(profile):
    # 1. Filter
    eligible_df = df.copy()
    # (Assuming all our 10 courses are eligible for first-years, so no filter needed here, 
    # but in a larger dataset, you would apply the same filter logic from recommender.py)
    
    # 2. TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(eligible_df['ai_text'])
    student_vector = tfidf.transform([profile['interests']])
    cosine_sim = cosine_similarity(student_vector, tfidf_matrix).flatten()
    
    eligible_df['score'] = cosine_sim
    ranked = eligible_df.sort_values(by='score', ascending=False)
    
    # 3. Calculate Metrics
    top_5 = ranked.head(5)['course_code'].tolist()
    
    # Precision@5: How many of the top 5 are actually relevant?
    relevant_in_top_5 = len(set(top_5) & set(profile['relevant_courses']))
    precision_at_5 = relevant_in_top_5 / 5
    
    # MRR (Mean Reciprocal Rank): What is the rank of the *first* relevant course?
    mrr = 0
    for i, course in enumerate(top_5):
        if course in profile['relevant_courses']:
            mrr = 1 / (i + 1)
            break
            
    print(f"--- {profile['name']} ---")
    print(f"Top 5: {top_5}")
    print(f"Precision@5: {precision_at_5:.2f} | MRR: {mrr:.2f}\n")

print("EVALUATION RESULTS:\n")
for p in profiles:
    evaluate_profile(p)
