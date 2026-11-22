import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the clean dataset
df = pd.read_csv("courses_clean.csv")

# 2. Define a First-Year Student Profile
# This mimics a first-year CS student interested in AI, with 15 credits completed.
student_profile = {
    "interests": "I want to learn about programming, artificial intelligence, python, and how computers work.",
    "completed_courses": ["GE1401"], # They already took University English
    "current_credits": 15
}

print(f"--- STUDENT PROFILE ---")
print(f"Interests: {student_profile['interests']}")
print(f"Completed: {student_profile['completed_courses']}\n")

# 3. HARD CONSTRAINT FILTERING
eligible_courses = []
rejected_courses = []

for index, row in df.iterrows():
    prereq = str(row['prerequisites'])
    
    # Simple rule: If prerequisite is not "Nil", check if the student completed it
    if prereq != "Nil" and prereq != "nan":
        # We check if any of the student's completed courses are mentioned in the prereq string
        has_prereq = any(comp_course in prereq for comp_course in student_profile['completed_courses'])
        
        # If the prereq is a high school requirement (HKAL/HKDSE), we assume the student met it upon admission
        is_hs_req = "HKDSE" in prereq or "HKAL" in prereq
        
        if not has_prereq and not is_hs_req:
            rejected_courses.append(row['course_code'])
            continue # Skip this course, they are not eligible
            
    eligible_courses.append(row)

# Convert back to DataFrame
eligible_df = pd.DataFrame(eligible_courses)
print(f"Hard Constraint Filter: Removed {len(rejected_courses)} ineligible courses.")
print(f"Remaining eligible courses: {len(eligible_df)}\n")

# 4. TF-IDF & COSINE SIMILARITY
# Combine student interests with course text to build the vocabulary
tfidf = TfidfVectorizer(stop_words='english')

# Fit on course text
tfidf_matrix = tfidf.fit_transform(eligible_df['ai_text'])

# Transform the student profile using the same vocabulary
student_vector = tfidf.transform([student_profile['interests']])

# Calculate Cosine Similarity between student and all eligible courses
cosine_sim = cosine_similarity(student_vector, tfidf_matrix).flatten()

# 5. RANK AND OUTPUT
# Add the similarity scores to the dataframe
eligible_df = eligible_df.copy() # Avoid warning
eligible_df['similarity_score'] = cosine_sim

# Sort by highest similarity
ranked_courses = eligible_df.sort_values(by='similarity_score', ascending=False)

print("--- TOP 5 RECOMMENDED COURSES ---")
for i, (index, row) in enumerate(ranked_courses.head(5).iterrows(), 1):
    print(f"{i}. {row['course_code']} - {row['title']} (Score: {row['similarity_score']:.4f})")