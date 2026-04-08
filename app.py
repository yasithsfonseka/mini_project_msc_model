import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Personality Prediction System", layout="wide")

# -----------------------------
# Load Models
# -----------------------------
buddhist_model = pickle.load(open("buddhist_model.pkl", "rb"))
western_model = pickle.load(open("western_model.pkl", "rb"))

# -----------------------------
# Load Datasets
# -----------------------------
buddhist_data = pd.read_csv("buddhist_psychology_training_dataset.csv")
western_data = pd.read_csv("western_psychology_training_dataset.csv")

# -----------------------------
# Title
# -----------------------------
st.title("🧠 AI Personality Prediction & Recommendation System")

st.write(
"This system predicts personality using **Buddhist psychology** and **Western psychology** models "
"and provides personalized recommendations."
)

# -----------------------------
# Scale Explanation
# -----------------------------
with st.expander("📊 Questionnaire Scale Explanation"):

    st.write("""
    **Score Meaning**

    | Score | Meaning |
    |------|--------|
    | 1 | Very Low |
    | 2 | Low |
    | 3 | Moderate |
    | 4 | High |
    | 5 | Very High |
    """)

# -----------------------------
# Tabs for Inputs
# -----------------------------
tab1, tab2 = st.tabs(["🧘 Buddhist Psychology", "🌎 Western Psychology"])

# -----------------------------
# Buddhist Tab
# -----------------------------
with tab1:

    st.subheader("Buddhist Personality Factors")

    col1, col2 = st.columns(2)

    with col1:
        childhood_support = st.slider("Childhood Support", 1, 5, 3)
        stress_level = st.slider("Stress Level", 1, 5, 3)
        work_pressure = st.slider("Work Pressure", 1, 5, 3)

    with col2:
        emotional_attachment = st.slider("Emotional Attachment", 1, 5, 3)
        mental_clarity = st.slider("Mental Clarity", 1, 5, 3)
        coping_skill = st.slider("Coping Skill", 1, 5, 3)

# -----------------------------
# Western Tab
# -----------------------------
with tab2:

    st.subheader("Western Personality Factors")

    col3, col4 = st.columns(2)

    with col3:
        openness = st.slider("Openness", 1, 5, 3)
        conscientiousness = st.slider("Conscientiousness", 1, 5, 3)
        extraversion = st.slider("Extraversion", 1, 5, 3)

    with col4:
        agreeableness = st.slider("Agreeableness", 1, 5, 3)
        neuroticism = st.slider("Neuroticism", 1, 5, 3)

# -----------------------------
# Predict Button
# -----------------------------
st.divider()

if st.button("🔍 Predict Personality", use_container_width=True):

    # Buddhist input
    buddhist_input = pd.DataFrame(
        [[childhood_support, stress_level, work_pressure,
          emotional_attachment, mental_clarity, coping_skill]],
        columns=[
            "childhood_support",
            "stress_level",
            "work_pressure",
            "emotional_attachment",
            "mental_clarity",
            "coping_skill"
        ]
    )

    # Western input
    western_input = pd.DataFrame(
        [[openness, conscientiousness, extraversion,
          agreeableness, neuroticism, stress_level, coping_skill]],
        columns=[
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
            "stress_level",
            "coping_skill"
        ]
    )

    # Predictions
    bud_pred = buddhist_model.predict(buddhist_input)[0]
    west_pred = western_model.predict(western_input)[0]

    # Get recommendations
    bud_rec_row = buddhist_data[buddhist_data["personality_type"] == bud_pred]
    west_rec_row = western_data[western_data["personality_type"] == west_pred]

    bud_rec = bud_rec_row["buddhist_recommendation"].iloc[0] if not bud_rec_row.empty else "No recommendation available"
    west_rec = west_rec_row["recommendation"].iloc[0] if not west_rec_row.empty else "No recommendation available"

    # -----------------------------
    # Results Section
    # -----------------------------
    st.subheader("📊 Prediction Results")

    col5, col6 = st.columns(2)

    with col5:
        st.success("🧘 Buddhist Psychology")
        st.write("**Personality Type:**", bud_pred)
        st.write("**Meditation Recommendation:**")
        st.info(bud_rec)

    with col6:
        st.success("🌎 Western Psychology")
        st.write("**Personality Type:**", west_pred)
        st.write("**Recommendation:**")
        st.info(west_rec)