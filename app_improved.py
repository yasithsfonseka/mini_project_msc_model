"""
Personality & Mental Health Profiler — Improved Streamlit App
MSc Research: The Influence of Childhood Experiences on Adult Mental Health,
              Personality Traits, and Occupational Performance

Integrates:
  • Buddhist Psychology (Kilesa-based typology)
  • Western Psychology (Big Five / Neo-Freudian)
  • Numerology (Life Path + Personal Number)
  • 7-Day Personalized Recovery Schedule
  • Mental Health Risk Scoring
"""

# ====== IMPORT LIBRARIES ======
import streamlit as st  # Web app framework for creating interactive UI
import pandas as pd  # Data manipulation and CSV file handling
import numpy as np  # Numerical computing and array operations
import pickle  # Load pre-trained machine learning models
import os  # Operating system file operations
import glob  # Find files matching patterns
import datetime  # Date/time operations for Date of Birth calculations
import warnings  # Suppress warning messages
import plotly.graph_objects as go  # Create interactive charts (radar, gauge, bars)
from plotly.subplots import make_subplots  # Create multi-panel chart layouts
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # ML classifiers
from sklearn.preprocessing import LabelEncoder  # Convert text labels to numbers for ML

# Suppress warning messages to keep output clean
warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG - Configure Streamlit web app display settings
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Personality & Mental Health Profiler",  # Browser tab title
    page_icon="🧠",  # Emoji icon shown in browser tab
    layout="wide",  # Use full width (not centered)
    initial_sidebar_state="collapsed",  # Hide left sidebar by default
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0e27; color: #e0e0e0; }

    /* Header banner */
    .app-header {
        background: linear-gradient(135deg, #1a3a52, #0a0e27, #16213e);
        color: white;
        text-align: center;
        padding: 36px 20px 28px 20px;
        border-radius: 16px;
        margin-bottom: 28px;
        border: 1px solid #1e3a5f;
        box-shadow: 0 8px 32px rgba(0, 150, 255, 0.15);
    }
    .app-header h1 { font-size: 2.2em; font-weight: 800; margin: 0; }
    .app-header p  { font-size: 1em; opacity: 0.88; margin: 6px 0 0 0; }

    /* Section labels */
    .section-label {
        background: linear-gradient(90deg, #0096ff, #005a96);
        color: white;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1em;
        margin: 20px 0 12px 0;
        display: inline-block;
    }

    /* Personality result card */
    .result-card {
        border-radius: 14px;
        padding: 22px;
        color: white;
        text-align: center;
        margin-bottom: 14px;
        border: 1px solid rgba(0, 150, 255, 0.3);
    }
    .card-buddhist  { background: linear-gradient(135deg, #1a3f5f, #0d2847); }
    .card-western   { background: linear-gradient(135deg, #0d3a66, #051f3a); }
    .card-risk-low  { background: linear-gradient(135deg, #0d5a3a, #1a8f5e); }
    .card-risk-mod  { background: linear-gradient(135deg, #665f0a, #8b7700); color: #f0f0f0; }
    .card-risk-high { background: linear-gradient(135deg, #5a1a1a, #8a2a2a); }
    .card-numerology{ background: linear-gradient(135deg, #4a1a3f, #1a0a2f); }

    /* Schedule day card */
    .day-card {
        background: #1a1f3a;
        border-left: 5px solid #0096ff;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0, 150, 255, 0.1);
        color: #e0e0e0;
    }
    .day-card strong { font-size: 1.05em; color: #0096ff; }
    .day-slot { font-size: 0.88em; margin-top: 5px; color: #b0b0b0; }

    /* Score metric override */
    div[data-testid="metric-container"] {
        background: #1a1f3a;
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 2px 8px rgba(0, 150, 255, 0.1);
        color: #e0e0e0;
        border: 1px solid #0096ff33;
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(135deg, #0096ff, #0066cc);
        color: white;
        border: none;
        padding: 14px 0;
        border-radius: 30px;
        font-size: 1.1em;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #0066cc, #0096ff);
        color: white;
        box-shadow: 0 0 20px rgba(0, 150, 255, 0.4);
    }

    /* Tab styling */
    button[data-baseweb="tab"] { font-weight: 600; }

    /* Guidance box */
    .guidance-box {
        background: white;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        margin-bottom: 14px;
        border-top: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# DATA & KNOWLEDGE BASES - Personality descriptions and guidance
# ═════════════════════════════════════════════════════════════════════════════

# Dictionary containing descriptions of 4 Buddhist psychology personality types
# Each type represents different mental formations (Kilesa) and behaviors
BUDDHIST_INFO = {
    "Balanced Type": {  # Archetype: emotionally stable and wise
        "emoji": "☯️",
        "core": "Clarity, equanimity, and wisdom",
        "childhood": "Supportive, nurturing home environment",
        "mental_impact": "Low mental health risk; emotionally resilient",
        "work_behavior": "High performer; collaborative and calm under pressure",
        "practice": "Mindfulness & analytical meditation (20 min morning)",
    },
    "Stress-Prone (Dosa)": {  # Archetype: anger/aversion type - reactive under pressure
        "emoji": "🌊",
        "core": "Anger, aversion, and reactivity",
        "childhood": "Experienced conflict or instability in early life",
        "mental_impact": "Moderate–high anxiety; prone to irritability and burnout",
        "work_behavior": "Reactive under pressure; elevated burnout risk",
        "practice": "Metta loving-kindness meditation (30 min)",
    },
    "Greedy Type (Raga)": {  # Archetype: attachment/craving type - driven by desire
        "emoji": "🔥",
        "core": "Attachment, desire, and craving",
        "childhood": "Achievement-focused environment; moderate support",
        "mental_impact": "Obsessive thinking patterns; unfulfillment despite success",
        "work_behavior": "Overachiever with strong attachment to outcomes",
        "practice": "Impermanence contemplation & non-attachment practice (25 min)",
    },
    "Deluded Type (Moha)": {  # Archetype: confusion/delusion type - lacks clarity
        "emoji": "🌫️",
        "core": "Confusion, delusion, and lack of clarity",
        "childhood": "Adverse or unsupportive early environments",
        "mental_impact": "Low mental clarity; poor decision-making under stress",
        "work_behavior": "Low focus; avoidance behaviours; needs structured support",
        "practice": "Anapanasati breathing meditation & grounding (25 min)",
    },
}

# Dictionary containing descriptions of 6 Western psychology personality types
# Based on Big Five traits and Neo-Freudian psychology
WESTERN_INFO = {
    "Balanced Personality": {  # Well-rounded across all traits
        "emoji": "⚖️",
        "core": "Emotional balance across Big Five traits",
        "strengths": "Adaptable, stable, well-rounded",
        "risks": "May lack strong identity in any one domain",
        "intervention": "Mindfulness + emotional awareness practices",
        "activity": "Reflective journaling + light social connection",
    },
    "Highly Disciplined": {  # High conscientiousness
        "emoji": "🏆",
        "core": "High conscientiousness; structured and goal-driven",
        "strengths": "Reliable, organised, high-achieving",
        "risks": "Perfectionism; difficulty with unstructured environments",
        "intervention": "Flexibility training + self-compassion practices",
        "activity": "Structured goal review + priority planning",
    },
    "Anxiety Prone": {  # High neuroticism
        "emoji": "⚡",
        "core": "High neuroticism; sensitive to stress and threat",
        "strengths": "Empathetic, detail-oriented, conscientious",
        "risks": "Chronic stress, rumination, burnout",
        "intervention": "CBT techniques + stress management protocols",
        "activity": "CBT thought journaling + deep relaxation techniques",
    },
    "Creative Explorer": {  # High openness
        "emoji": "🎨",
        "core": "High openness; curious and imaginative",
        "strengths": "Innovative, open-minded, versatile",
        "risks": "Scattered focus; difficulty with routine",
        "intervention": "Creative channelling + grounding exercises",
        "activity": "Creative expression (art / music / writing)",
    },
    "Social Leader": {  # High extraversion and agreeableness
        "emoji": "👥",
        "core": "High extraversion and agreeableness; people-centred",
        "strengths": "Influential, collaborative, emotionally intelligent",
        "risks": "Boundary issues; emotional depletion",
        "intervention": "Leadership development + healthy boundaries training",
        "activity": "Community engagement or mentoring session",
    },
    "Emotionally Stable": {  # Low neuroticism
        "emoji": "🌿",
        "core": "Low neuroticism; calm and resilient",
        "strengths": "Consistent, composed, reliable under pressure",
        "risks": "May overlook others' emotional needs",
        "intervention": "Empathy enhancement + preventive wellness",
        "activity": "Gratitude practice + healthy routine check",
    },
}

# Dictionary with recommended interventions based on mental health risk level
RISK_INTERVENTIONS = {
    "High Risk": [  # Urgent intervention needed
        "Professional counselling session (strongly recommended)",
        "Crisis support awareness — save a helpline number",
        "Daily mood tracking journal",
        "Sleep hygiene protocol (bed by 10 pm, screens off 1 hr before)",
        "Immediate workload reduction strategy",
    ],
    "Moderate Risk": [  # Proactive steps recommended
        "Self-reflection journalling (15 min)",
        "Aerobic exercise or yoga (30 min)",
        "Social connection — call or meet someone you trust",
        "Mindfulness app session (10 min)",
        "Weekly emotional check-in",
    ],
    "Low Risk": [  # Maintain current habits
        "Monthly wellness self-assessment",
        "Maintain current healthy habits",
        "Continue meaningful social activities",
        "Preventive learning (read / podcast on well-being)",
        "Nature walk or outdoor activity",
    ],
}

NUMEROLOGY_DATA = {
    1:  {"archetype": "The Leader",          "traits": "Independent, ambitious, pioneering",         "theme": "Leadership & Self-Expression",      "challenge": "Avoid ego; learn to collaborate",             "affirmation": "I lead with purpose and inspire others",                          "guidance": "Channel drive into meaningful goals. Practice humility meditation."},
    2:  {"archetype": "The Mediator",         "traits": "Empathetic, cooperative, diplomatic",        "theme": "Partnership & Balance",             "challenge": "Overcome indecision and people-pleasing",     "affirmation": "I find strength in balance and meaningful connections",           "guidance": "Nurture emotional intelligence. Metta meditation for self-compassion."},
    3:  {"archetype": "The Communicator",     "traits": "Creative, expressive, optimistic",           "theme": "Creative Expression & Joy",         "challenge": "Focus scattered energy; avoid superficiality","affirmation": "I express my authentic self with joy and creativity",             "guidance": "Use creative outlets to process emotions. Journalling suits you."},
    4:  {"archetype": "The Builder",          "traits": "Practical, disciplined, reliable",           "theme": "Stability & Hard Work",             "challenge": "Avoid rigidity; embrace change",              "affirmation": "I build a secure foundation through disciplined effort",          "guidance": "Structured routines support well-being. Schedule rest intentionally."},
    5:  {"archetype": "The Explorer",         "traits": "Adventurous, freedom-loving, curious",       "theme": "Freedom & Adventure",               "challenge": "Develop commitment; avoid impulsiveness",     "affirmation": "I embrace change as an opportunity for growth",                   "guidance": "Channel restless energy into exploration. Mindfulness grounds focus."},
    6:  {"archetype": "The Nurturer",         "traits": "Caring, responsible, harmonious",            "theme": "Service & Nurturing",               "challenge": "Set healthy boundaries; avoid over-responsibility","affirmation": "I give generously while honouring my own needs",                 "guidance": "Practice self-care alongside caregiving. Loving-kindness meditation is ideal."},
    7:  {"archetype": "The Seeker",           "traits": "Analytical, introspective, spiritual",       "theme": "Wisdom & Inner Truth",              "challenge": "Avoid isolation; share insights with others", "affirmation": "I trust the wisdom within and seek deeper truths",               "guidance": "Deep reflection and insight meditation align with your nature."},
    8:  {"archetype": "The Achiever",         "traits": "Ambitious, authoritative, goal-oriented",    "theme": "Material & Personal Power",         "challenge": "Balance material pursuits with spiritual growth","affirmation": "I use my power responsibly to create positive outcomes",         "guidance": "Reflect on attachment to outcomes. Impermanence meditation keeps you grounded."},
    9:  {"archetype": "The Humanitarian",     "traits": "Compassionate, idealistic, generous",        "theme": "Compassion & Completion",           "challenge": "Avoid martyrdom; accept limitations",         "affirmation": "I serve the world while honouring my own boundaries",            "guidance": "Your empathy is your strength. Equanimity practice prevents depletion."},
    11: {"archetype": "The Intuitive ✨",     "traits": "Highly intuitive, inspiring, sensitive",     "theme": "Spiritual Enlightenment",           "challenge": "Manage sensitivity; stay grounded",           "affirmation": "I channel spiritual insight to inspire and uplift others",       "guidance": "Sensitivity is a gift. Anapanasati breathing helps manage overwhelm."},
    22: {"archetype": "The Master Builder ✨","traits": "Visionary, practical, powerful",             "theme": "Building a Legacy",                 "challenge": "Avoid perfectionism and self-doubt",          "affirmation": "I build lasting structures that benefit humanity",               "guidance": "Ground grand visions with step-by-step planning. Mindfulness prevents burnout."},
    33: {"archetype": "The Master Teacher ✨","traits": "Selfless, nurturing, compassionate",         "theme": "Teaching & Healing",                "challenge": "Maintain personal well-being; avoid self-sacrifice","affirmation": "I teach through compassion and illuminate paths for others",    "guidance": "Your calling is to guide others. Regular meditation and self-care are essential."},
}

# ═════════════════════════════════════════════════════════════════════════════
# SURVEY QUESTIONS - Likert scale questionnaire for assessment
# ═════════════════════════════════════════════════════════════════════════════
# Each tuple contains: (question_text, is_reverse_scored)
# is_reverse_scored=True means: "Disagree" is actually GOOD (high score = good)
# is_reverse_scored=False means: "Agree" is actually GOOD (high score = good)

# SECTION A: Questions about childhood experiences and early environment
CHILDHOOD_QS = [
    ("My parents/guardians provided strong emotional support during my childhood.", False),  # Direct scoring
    ("I felt safe and secure in my home environment growing up.", False),
    ("There were frequent conflicts in my family when I was growing up.", True),  # Reverse: disagreeing is better
    ("I experienced bullying or negative treatment during childhood.", True),  # Reverse: disagreeing is better
    ("I had positive relationships with teachers or mentors at school.", False),
    ("My childhood environment encouraged learning and personal growth.", False),
]

# SECTION B: Questions about current mental health and emotional wellbeing
MENTAL_QS = [
    ("I often feel stressed in my daily life.", True),  # Reverse: disagreeing is better
    ("I have difficulty relaxing after work or study.", True),  # Reverse: disagreeing is better
    ("I sometimes feel lonely even when I am around other people.", True),  # Reverse: disagreeing is better
    ("I feel emotionally balanced most of the time.", False),
    ("I can manage my emotions effectively.", False),
    ("I maintain a positive outlook on life generally.", False),
]

# SECTION C: Questions about work/occupational life and performance
WORK_QS = [
    ("I feel overwhelmed by my workload.", True),  # Reverse: disagreeing is better
    ("I find my work meaningful and fulfilling.", False),
    ("I maintain good working relationships with my colleagues.", False),
    ("I feel productive and effective in my role.", False),
    ("I can manage work-related stress reasonably well.", False),
    ("My overall work-life balance is satisfactory.", False),
]

# SECTION D: Questions about coping strategies and resilience
COPING_QS = [
    ("I sometimes avoid dealing with stressful situations.", True),  # Reverse: disagreeing is better
    ("I seek support from others when I face difficulties.", False),
    ("I engage in healthy activities (exercise, hobbies, rest) to manage stress.", False),
    ("I can bounce back from setbacks relatively quickly.", False),
    ("I reflect constructively on problems before acting.", False),
    ("I maintain healthy daily routines even during stressful periods.", False),
]

# Response options for Likert scale (5-point)
LIKERT_OPTS = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]

# Map response text to numeric values (1-5 scale)
LIKERT_VAL  = {"Strongly Disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly Agree": 5}


# ═════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING - Load/train ML classifiers (cached for performance)
# ═════════════════════════════════════════════════════════════════════════════
def find_csv(patterns):
    """
    Find first existing file matching any of the glob patterns provided.
    Used to search for CSV files with flexible naming.
    - patterns: list of glob pattern strings
    Returns: path to first matching file, or None if none found
    """
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None


@st.cache_resource(show_spinner="Training models — please wait a moment…")
def load_models():
    """
    Load or train machine learning models for personality classification.
    CACHED: Runs only once per session to improve performance.
    
    Trains 2 classifiers:
    1. RandomForestClassifier for Buddhist personality (4 types)
    2. GradientBoostingClassifier for Western personality (6 types)
    
    Returns: tuple of (buddhist_classifier, buddhist_label_encoder, 
                       western_classifier, western_label_encoder)
    """
    # Find data CSV files (try multiple naming patterns)
    b_path = find_csv(["buddhist_psychology_training_dataset*.csv",
                        "buddhist_psychology_training_dataset.csv"])
    w_path = find_csv(["western_psychology_training_dataset*.csv",
                        "western_psychology_training_dataset.csv"])

    # Error if data files not found
    if not b_path or not w_path:
        st.error("Dataset CSV files not found. Place them in the same folder as app_improved.py.")
        st.stop()

    # Load datasets from CSV files
    buddhist_df = pd.read_csv(b_path)
    western_df  = pd.read_csv(w_path)

    # ── BUDDHIST MODEL TRAINING (Data Augmentation) ────────────────────────
    # Define features (inputs) and target (output) for Buddhist model
    b_feats  = ["childhood_support","stress_level","work_pressure",
                "emotional_attachment","mental_clarity","coping_skill"]
    b_target = "personality_type"

    # Data augmentation: Create synthetic variations to prevent overfitting
    # Original ~33 rows → ~300 rows by adding small random variations
    rng = np.random.default_rng(42)  # Seed for reproducibility
    aug_rows = []
    for _, row in buddhist_df.iterrows():
        for _ in range(8):  # Create 8 variations per original row
            nr = row.copy()
            for f in b_feats:
                # Add small noise (-1 to +1) and keep within 1-5 range
                nr[f] = int(np.clip(row[f] + rng.integers(-1, 2), 1, 5))
            aug_rows.append(nr)
    
    # Combine original and augmented data
    bdf = pd.concat([buddhist_df, pd.DataFrame(aug_rows)], ignore_index=True)

    # Train RandomForest classifier for Buddhist types
    le_b = LabelEncoder()  # Convert personality type names to numbers (0,1,2,3)
    bdf["_enc"] = le_b.fit_transform(bdf[b_target])
    clf_b = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf_b.fit(bdf[b_feats], bdf["_enc"])  # Fit model on features → encoded targets

    # ── WESTERN MODEL TRAINING (GradientBoosting) ──────────────────────────
    # Define features for Western model (7 Big Five related features)
    w_feats = ["openness","conscientiousness","extraversion","agreeableness",
               "neuroticism","stress_level","coping_skill"]

    # Train GradientBoosting classifier for Western types
    le_w = LabelEncoder()  # Convert personality type names to numbers
    western_df["_enc"] = le_w.fit_transform(western_df[b_target])
    clf_w = GradientBoostingClassifier(n_estimators=150, random_state=42)
    clf_w.fit(western_df[w_feats], western_df["_enc"])  # Fit model

    # Return all 4 objects needed for predictions
    return clf_b, le_b, clf_w, le_w


# ═════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS - Create interactive visualizations with Plotly
# ═════════════════════════════════════════════════════════════════════════════

# Color constants for radar charts
RADAR_PURPLE = "rgba(102,126,234,0.85)"  # Color for Buddhist radar line
RADAR_FILL   = "rgba(102,126,234,0.25)"  # Transparent purple fill
RADAR_TEAL   = "rgba(5,117,230,0.85)"    # Color for Western radar line
RADAR_FILL2  = "rgba(5,117,230,0.22)"    # Transparent teal fill
IDEAL_COLOR  = "rgba(39,174,96,0.35)"    # Green for ideal profile line
IDEAL_FILL   = "rgba(39,174,96,0.08)"    # Transparent green fill


def buddhist_radar(b_vals: list, b_type: str, b_conf: float) -> go.Figure:
    """
    Create a radar chart (spider chart) showing Buddhist personality profile.
    
    Parameters:
    - b_vals: list of 6 scores (1-5 each) for Buddhist personality features
    - b_type: personality type name (e.g., "Balanced Type")
    - b_conf: confidence percentage for the prediction
    
    Returns: Plotly Figure object with interactive radar chart
    
    The chart shows:
    - Green background area: Ideal profile (all 5s)
    - Purple area: User's actual profile
    - Larger overlap = closer to ideal
    """
    labels = [
        "Childhood Support", "Stress Resilience",
        "Work Capacity",     "Emotional Attachment",
        "Mental Clarity",    "Coping Skill",
    ]
    # Close the polygon by adding first point at end
    r_user  = b_vals + [b_vals[0]]
    r_ideal = [5] * 6 + [5]
    theta   = labels + [labels[0]]

    fig = go.Figure()
    
    # Add ideal profile trace (green background)
    fig.add_trace(go.Scatterpolar(
        r=r_ideal, theta=theta, fill="toself",
        name="Ideal", line=dict(color=IDEAL_COLOR, width=1),
        fillcolor=IDEAL_FILL, opacity=0.6,
    ))
    
    # Add user's profile trace (purple)
    fig.add_trace(go.Scatterpolar(
        r=r_user, theta=theta, fill="toself",
        name=b_type,
        line=dict(color=RADAR_PURPLE, width=2),
        fillcolor=RADAR_FILL,
        marker=dict(size=7, color=RADAR_PURPLE),
    ))
    
    # Configure chart layout and styling
    fig.update_layout(
        title=dict(text=f"🪷 Buddhist Profile<br><sup>{b_type} — {b_conf:.1f}% confidence</sup>",
                   font=dict(size=14), x=0.5),
        polar=dict(
            bgcolor="rgba(248,249,255,0.9)",
            radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=9),
                            gridcolor="#dde", linecolor="#dde"),
            angularaxis=dict(tickfont=dict(size=10), linecolor="#ccc"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=80, b=60, l=60, r=60),
        height=380,
    )
    return fig


def western_radar(w_vals: list, w_type: str, w_conf: float) -> go.Figure:
    """
    Create a radar chart for Western (Big Five) personality profile.
    Similar to buddhist_radar but with 7 features instead of 6.
    
    Parameters:
    - w_vals: list of 7 scores (1-5 each) for Western personality features
    - w_type: personality type name (e.g., "Anxiety Prone")
    - w_conf: confidence percentage for the prediction
    
    Returns: Plotly Figure object with interactive radar chart
    """
    labels = [
        "Openness",      "Conscientiousness", "Extraversion",
        "Agreeableness", "Neuroticism",        "Stress Response",
        "Coping Skill",
    ]
    # Close the polygon
    r_user  = w_vals + [w_vals[0]]
    r_ideal = [5] * 7 + [5]
    theta   = labels + [labels[0]]

    fig = go.Figure()
    
    # Add ideal profile trace (green background)
    fig.add_trace(go.Scatterpolar(
        r=r_ideal, theta=theta, fill="toself",
        name="Ideal", line=dict(color=IDEAL_COLOR, width=1),
        fillcolor=IDEAL_FILL, opacity=0.6,
    ))
    
    # Add user's profile trace (teal)
    fig.add_trace(go.Scatterpolar(
        r=r_user, theta=theta, fill="toself",
        name=w_type,
        line=dict(color=RADAR_TEAL, width=2),
        fillcolor=RADAR_FILL2,
        marker=dict(size=7, color=RADAR_TEAL),
    ))
    
    # Configure chart layout
    fig.update_layout(
        title=dict(text=f"🧬 Western Profile<br><sup>{w_type} — {w_conf:.1f}% confidence</sup>",
                   font=dict(size=14), x=0.5),
        polar=dict(
            bgcolor="rgba(248,249,255,0.9)",
            radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=9),
                            gridcolor="#dde", linecolor="#dde"),
            angularaxis=dict(tickfont=dict(size=10), linecolor="#ccc"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=80, b=60, l=60, r=60),
        height=380,
    )
    return fig


def risk_gauge(risk_score: float, risk_lvl: str) -> go.Figure:
    """
    Create a gauge chart showing mental health risk level.
    Visually displays where the user sits on the 0-5 risk scale.
    
    Parameters:
    - risk_score: numeric score (0-5)
    - risk_lvl: text label ("Low Risk", "Moderate Risk", or "High Risk")
    
    Returns: Plotly Figure gauge chart
    
    Color zones:
    - Green (0-2.5): Low risk - maintain current practices
    - Yellow (2.5-3.5): Moderate risk - proactive intervention
    - Red (3.5-5.0): High risk - immediate professional help
    """
    # Select color based on risk level
    bar_color = {"Low Risk": "#27ae60", "Moderate Risk": "#f39c12", "High Risk": "#e74c3c"}
    color = bar_color.get(risk_lvl, "#7f8c8d")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(risk_score, 2),
        number=dict(font=dict(size=36, color=color), suffix=" / 5"),
        title=dict(
            text=f"⚕️ Mental Health Risk<br>"
                 f"<span style='font-size:1em;color:{color}'><b>{risk_lvl}</b></span>",
            font=dict(size=14),
        ),
        gauge=dict(
            axis=dict(range=[0, 5], tickwidth=1, tickcolor="#555",
                      tickvals=[0, 1, 2, 2.5, 3, 3.5, 4, 5],
                      ticktext=["0", "1", "2", "2.5", "3", "3.5", "4", "5"]),
            bar=dict(color=color, thickness=0.28),
            bgcolor="white",
            borderwidth=1,
            bordercolor="#ccc",
            # Define color zones
            steps=[
                dict(range=[0.0, 2.5], color="#d5f5e3"),   # Green zone (Low)
                dict(range=[2.5, 3.5], color="#fef9e7"),   # Yellow zone (Moderate)
                dict(range=[3.5, 5.0], color="#fdedec"),   # Red zone (High)
            ],
            threshold=dict(
                line=dict(color=color, width=4),
                thickness=0.82,
                value=risk_score,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=90, b=20, l=30, r=30),
        height=300,
    )
    return fig


def composite_bar(cs: float, mh: float, ws: float, cp: float) -> go.Figure:
    """
    Create horizontal bar chart showing composite domain scores.
    
    Parameters:
    - cs: Coping Skill score (1-5)
    - mh: Mental Health score (1-5)
    - ws: Work Performance score (1-5)
    - cp: Childhood score (1-5)
    
    Returns: Plotly Figure with horizontal bars
    
    Visual aids:
    - Each bar is a different color
    - Dotted line at 3.0 = neutral midpoint
    - Values above 3 = strength, below 3 = needs attention
    """
    categories = ["Coping Skill", "Work Performance", "Mental Health", "Childhood"]
    values     = [cp, ws, mh, cs]
    colors     = ["#c94b4b", "#f7971e", "#11998e", "#667eea"]

    fig = go.Figure(go.Bar(
        y=categories,
        x=values,
        orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=1)),
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(size=12, color="#333"),
        width=0.55,
    ))
    
    # Add reference line at 3.0 (neutral)
    fig.add_vline(x=3.0, line_dash="dot", line_color="#888",
                  annotation_text="Neutral (3.0)",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="#888"))
    
    fig.update_layout(
        title=dict(text="📊 Domain Score Summary", font=dict(size=14), x=0.5),
        xaxis=dict(range=[0, 5.8], title="Score (1–5)",
                   gridcolor="#eee", linecolor="#ddd"),
        yaxis=dict(tickfont=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,255,0.9)",
        margin=dict(t=50, b=40, l=130, r=60),
        height=300,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - Utility functions for calculations
# ═════════════════════════════════════════════════════════════════════════════

def reduce_num(n):
    """
    Reduce a number to single digit or master number (11, 22, 33).
    Used in numerology calculations.
    Example: 38 → 3+8=11 (master number, stop here)
    Example: 29 → 2+9=11 (master number, stop here)
    Example: 27 → 2+7=9 (single digit)
    """
    while n > 9 and n not in (11, 22, 33):
        n = sum(int(d) for d in str(n))
    return n


def life_path_number(dob: datetime.date) -> int:
    """
    Calculate Life Path Number from Date of Birth.
    Reduces day + month + year to single digit (or master number).
    Represents core personality traits and life themes in numerology.
    Example: Jan 15, 1990 → 1 + 1+5 + 1+9+9+0 = 26 → 2+6 = 8
    """
    day   = reduce_num(dob.day)  # Reduce day to single digit
    month = reduce_num(dob.month)  # Reduce month to single digit
    year  = reduce_num(sum(int(d) for d in str(dob.year)))  # Reduce year to single digit
    return reduce_num(day + month + year)  # Return final single digit or master number


def personal_number(dob: datetime.date) -> int:
    """
    Calculate Personal Number (Personal Day Number).
    Simply reduces the day of birth.
    Example: Born on 28th → 2+8 = 10 → 1+0 = 1
    """
    return reduce_num(dob.day)


def composite_scores(c_resp, m_resp, w_resp, cp_resp):
    """
    Calculate average scores for 4 survey domains.
    - Reverses negative items (those marked True) so higher is always better
    - Returns tuple: (childhood_score, mental_health_score, work_score, coping_score)
    Each score is on 1-5 scale where 5 = best and 1 = worst
    """
    def score(qs, resps):
        vals = []
        for (_, rev), r in zip(qs, resps):
            v = LIKERT_VAL[r]  # Convert response text to 1-5 number
            vals.append(6 - v if rev else v)  # Reverse if negative item
        return float(np.mean(vals))  # Return average
    
    return (score(CHILDHOOD_QS, c_resp), score(MENTAL_QS, m_resp),
            score(WORK_QS, w_resp),      score(COPING_QS, cp_resp))


def to_b_input(cs, mh, ws, cp):
    """
    Convert 4 domain scores to DataFrame for Buddhist model prediction.
    Maps the 4 general scores to 6 Buddhist-specific features.
    Example: childhood_support = raw childhood score
    """
    return pd.DataFrame([[
        int(np.clip(round(cs),          1, 5)),  # childhood_support
        int(np.clip(round(6 - mh),      1, 5)),  # stress_level (inverted mental health)
        int(np.clip(round(6 - ws),      1, 5)),  # work_pressure (inverted work score)
        int(np.clip(round((cs+mh)/2),   1, 5)),  # emotional_attachment (average)
        int(np.clip(round(mh),          1, 5)),  # mental_clarity
        int(np.clip(round(cp),          1, 5)),  # coping_skill
    ]], columns=["childhood_support","stress_level","work_pressure",
                 "emotional_attachment","mental_clarity","coping_skill"])


def to_w_input(cs, mh, ws, cp):
    """
    Convert 4 domain scores to DataFrame for Western model prediction.
    Maps the 4 general scores to 7 Western (Big Five) features.
    """
    return pd.DataFrame([[
        int(np.clip(round((cs+cp)/2),   1, 5)),  # openness
        int(np.clip(round(ws),          1, 5)),  # conscientiousness
        int(np.clip(round((mh+cs)/2),   1, 5)),  # extraversion
        int(np.clip(round(cp),          1, 5)),  # agreeableness
        int(np.clip(round(6 - mh),      1, 5)),  # neuroticism (inverted mental health)
        int(np.clip(round(6 - mh),      1, 5)),  # stress_level (inverted mental health)
        int(np.clip(round(cp),          1, 5)),  # coping_skill
    ]], columns=["openness","conscientiousness","extraversion","agreeableness",
                 "neuroticism","stress_level","coping_skill"])


def risk_level(cs, mh, ws, cp):
    """
    Calculate mental health risk level (1-5 scale).
    Uses weighted formula:
    - Mental health (40% weight): most important factor
    - Childhood (30% weight): formative experiences
    - Work (20% weight): occupational stress
    - Coping (10% weight): resilience ability
    Returns: (risk_level_text, numeric_score)
    """
    score = (6-mh)*0.4 + (6-cs)*0.3 + (6-ws)*0.2 + (6-cp)*0.1
    lvl   = "High Risk" if score >= 3.5 else "Moderate Risk" if score >= 2.5 else "Low Risk"
    return lvl, round(score, 2)


def build_schedule(b_type, w_type, risk_lvl):
    """
    Build a personalized 7-day recovery schedule.
    Combines:
    - Buddhist meditation practice (from b_type)
    - Western psychology activity (from w_type)
    - Risk-based interventions (from risk_lvl)
    Returns dictionary: {day_name: {time_slot: activity}}
    """
    mp  = BUDDHIST_INFO.get(b_type, {}).get("practice", "Mindfulness meditation (20 min)")  # Morning practice
    ev  = WESTERN_INFO.get(w_type,  {}).get("activity", "Reflective journalling")  # Evening activity
    ri  = RISK_INTERVENTIONS.get(risk_lvl, [])  # Risk-based interventions list
    get = lambda i: ri[i] if len(ri) > i else "Restorative leisure"  # Helper to get intervention by index

    # Create 7-day schedule with morning, afternoon, evening slots
    return {
        "Monday":    {"🌅 Morning": mp,                               "🕛 Afternoon": get(0),                            "🌙 Evening":  ev + " + Journalling"},
        "Tuesday":   {"🌅 Morning": "Light exercise 30 min + Gratitude writing", "🕛 Afternoon": get(1),            "🌙 Evening":  ev},
        "Wednesday": {"🌅 Morning": mp,                               "🕛 Afternoon": get(2),                            "🌙 Evening":  "Social connection (family / friend check-in)"},
        "Thursday":  {"🌅 Morning": "Body scan + Gratitude practice (15 min)", "🕛 Afternoon": "Weekly goals & progress review", "🌙 Evening": ev + " + Reading"},
        "Friday":    {"🌅 Morning": mp,                               "🕛 Afternoon": get(3),                            "🌙 Evening":  "Weekly review, planning & self-appreciation"},
        "Saturday":  {"🌅 Morning": "Nature walk or outdoor activity",         "🕛 Afternoon": get(4),                "🌙 Evening":  ev + " + Social activities"},
        "Sunday":    {"🌅 Morning": "Restorative rest + gentle yoga / stretching", "🕛 Afternoon": "Reflective journalling for the week", "🌙 Evening": "Prepare mentally & practically for the week ahead"},
    }


def render_card(content: str, css_class: str):
    """
    Render HTML card with custom styling (colored background, rounded corners, etc.)
    - content: HTML content to display inside the card
    - css_class: CSS class name from the style section (e.g., 'card-buddhist', 'card-risk-high')
    """
    st.markdown(f'<div class="result-card {css_class}">{content}</div>', unsafe_allow_html=True)


def prob_bars(classes, probas):
    """
    Display probability bars for all personality type predictions.
    Shows each type with percentage (e.g., "Type A: 45.2%")
    - classes: list of personality type names
    - probas: list of probability values (0.0 to 1.0)
    """
    for cls, p in sorted(zip(classes, probas), key=lambda x: -x[1]):  # Sort by highest probability first
        st.progress(float(p), text=f"{cls}: {p*100:.1f}%")  # Show progress bar with percentage


# ═════════════════════════════════════════════════════════════════════════════
# SURVEY TAB HELPER - Display survey questions with radio buttons
# ═════════════════════════════════════════════════════════════════════════════
def render_survey_tab(questions, key_prefix, section_name, caption_text):
    """
    Render a survey section with multiple questions.
    - questions: list of (question_text, is_reverse_scored) tuples
    - key_prefix: unique prefix for storing responses (e.g., "c", "m", "w", "cp")
    - section_name: heading text for the section
    - caption_text: small description text below heading
    Returns: list of user responses in order (e.g., ["Agree", "Strongly Agree", "Neutral"])
    """
    st.markdown(f"**{section_name}**")  # Display section heading
    st.caption(caption_text)  # Display description below heading
    
    responses = []  # Store user's answers
    for i, (q, rev) in enumerate(questions):  # Loop through each question
        # Add ↩️ prefix if question is reverse-scored (so higher number = different meaning)
        label = f"{'↩️ ' if rev else ''}Q{i+1}. {q}"
        # Create radio button with 5-point Likert scale (default = "Neutral" = index 2)
        r = st.radio(label, LIKERT_OPTS, index=2, key=f"{key_prefix}_{i}", horizontal=True)
        responses.append(r)  # Store the selected response
    
    return responses  # Return all responses in order


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP - Main function that orchestrates the entire app flow
# ═════════════════════════════════════════════════════════════════════════════
def main():
    """
    Main function - runs the entire Streamlit app.
    Flow:
    1. Load ML models (Buddhist classifier & Western classifier)
    2. Display header and welcome message
    3. Collect user personal information (name, age, gender, DOB)
    4. Display survey with 4 sections (24 questions total)
    5. When user clicks "Predict", calculate personality types & show results
    6. Display comprehensive profile analysis (charts, schedule, guidance)
    7. Allow user to download profile as CSV
    """
    
    # Load the pre-trained machine learning models
    clf_b, le_b, clf_w, le_w = load_models()

    # ── HEADER ────────────────────────────────────────────────────────────────
    # Display main title and description in custom HTML styling
    st.markdown("""
    <div class="app-header">
        <h1>🧠 Personality & Mental Health Profiler</h1>
        <p>Buddhist Psychology &nbsp;·&nbsp; Western Big Five &nbsp;·&nbsp;
           Numerology &nbsp;·&nbsp; 7-Day Wellness Schedule</p>
        <p style="font-size:0.85em; opacity:0.75; margin-top:6px;">
            MSc Research — The Influence of Childhood Experiences on Adult Mental Health,
            Personality Traits &amp; Occupational Performance
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── PERSONAL INFORMATION ──────────────────────────────────────────────────
    # Collect user's basic information needed for analysis
    st.markdown('<div class="section-label">👤 Personal Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)  # Create 3 columns for layout
    
    with c1:
        # Collect name and age in first column
        name       = st.text_input("Full Name", placeholder="Your name")
        age        = st.number_input("Age", min_value=15, max_value=100, value=28)
    
    with c2:
        # Collect gender and profession in second column
        gender     = st.selectbox("Gender", ["Male", "Female", "Prefer not to say"])
        profession = st.text_input("Profession / Field", placeholder="e.g. Software Engineer")
    
    with c3:
        # Collect date of birth in third column (needed for numerology calculation)
        dob = st.date_input(
            "Date of Birth",
            value=datetime.date(1995, 6, 15),
            min_value=datetime.date(1930, 1, 1),
            max_value=datetime.date.today(),
        )

    # ── SURVEY ────────────────────────────────────────────────────────────────
    # Display survey questionnaire with 4 sections
    st.markdown('<div class="section-label">📋 Survey Questionnaire</div>', unsafe_allow_html=True)
    st.markdown("*Rate each statement from **Strongly Disagree → Strongly Agree**. "
                "↩️ marks reverse-scored items.*")

    # Create 4 tabs for different survey sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Childhood Experiences",
        "🧠 Mental Health & Wellbeing",
        "💼 Work & Occupation",
        "🛡️ Coping & Resilience",
    ])
    
    # Display each survey section in its own tab
    with tab1:
        c_resp  = render_survey_tab(CHILDHOOD_QS, "c",  "Section A — Childhood Experiences",
                                    "Reflect on your upbringing and early-life environment.")
    with tab2:
        m_resp  = render_survey_tab(MENTAL_QS,    "m",  "Section B — Mental Health & Wellbeing",
                                    "Reflect on your current emotional and psychological state.")
    with tab3:
        w_resp  = render_survey_tab(WORK_QS,      "w",  "Section C — Work & Occupational Life",
                                    "Reflect on your professional environment and performance.")
    with tab4:
        cp_resp = render_survey_tab(COPING_QS,    "cp", "Section D — Coping & Resilience",
                                    "Reflect on how you handle stress and adversity.")

    # ── PREDICT BUTTON ────────────────────────────────────────────────────────
    # Add spacing and display large "Predict" button
    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])  # Center the button
    with mid:
        clicked = st.button("🔍 Generate My Personality Profile", use_container_width=True)

    # ── RESULTS ───────────────────────────────────────────────────────────────
    # Only process if user clicks the predict button
    if not clicked:
        return  # Exit function early if button not clicked

    # Calculate all predictions while showing loading spinner
    with st.spinner("Analysing your profile…"):
        # 1. Convert raw survey responses to 4 domain scores (1-5 scale)
        cs, mh, ws, cp = composite_scores(c_resp, m_resp, w_resp, cp_resp)

        # 2. BUDDHIST PREDICTION - Classify into 4 Buddhist personality types
        b_df   = to_b_input(cs, mh, ws, cp)  # Convert domain scores to Buddhist features
        b_enc  = clf_b.predict(b_df)[0]  # Get encoded prediction
        b_type = le_b.inverse_transform([b_enc])[0]  # Decode to personality type name
        b_prob = clf_b.predict_proba(b_df)[0]  # Get probability for each type
        b_conf = b_prob.max() * 100  # Get confidence % for top prediction

        # 3. WESTERN PREDICTION - Classify into 6 Western personality types
        w_df   = to_w_input(cs, mh, ws, cp)  # Convert domain scores to Western features
        w_enc  = clf_w.predict(w_df)[0]  # Get encoded prediction
        w_type = le_w.inverse_transform([w_enc])[0]  # Decode to personality type name
        w_prob = clf_w.predict_proba(w_df)[0]  # Get probability for each type
        w_conf = w_prob.max() * 100  # Get confidence % for top prediction

        # 4. MENTAL HEALTH RISK - Calculate risk level (Low/Moderate/High)
        risk_lvl, risk_score = risk_level(cs, mh, ws, cp)

        # 5. NUMEROLOGY - Calculate Life Path and Personal Numbers
        lpn      = life_path_number(dob)  # Life Path Number (derived from DOB)
        pn       = personal_number(dob)  # Personal Number (day of birth)
        num_info = NUMEROLOGY_DATA.get(lpn, NUMEROLOGY_DATA[9])  # Get numerology info

        # 6. CHILDHOOD ASSESSMENT - Label childhood environment quality
        c_env = "Supportive" if cs >= 4 else "Moderate" if cs >= 3 else "Adverse"

        # 7. BUILD PERSONALIZED 7-DAY SCHEDULE
        schedule = build_schedule(b_type, w_type, risk_lvl)

    # ── RESULTS DISPLAY ───────────────────────────────────────────────────────
    # Show confirmation message with user's name
    display_name = name if name else "You"
    st.success(f"✅ Profile generated for **{display_name}**")
    st.divider()

    # ── COMPOSITE SCORE SUMMARY ───────────────────────────────────────────────
    # Display 4 main domain scores in metric boxes
    st.markdown("## 📊 Your Profile at a Glance")
    m1, m2, m3, m4 = st.columns(4)  # Create 4 columns for metrics
    c_env_delta = {"Supportive": "↑ Positive", "Moderate": "→ Neutral", "Adverse": "↓ Needs attention"}
    
    # Display each score in a metric box with delta (comparison) indicator
    m1.metric("Childhood Environment", c_env,          c_env_delta[c_env])
    m2.metric("Mental Health Score",   f"{mh:.2f}/5",  "Higher = Better")
    m3.metric("Work Performance",      f"{ws:.2f}/5",  "Higher = Better")
    m4.metric("Coping Ability",        f"{cp:.2f}/5",  "Higher = Better")

    st.divider()
    st.markdown("## 🧩 Personality Predictions")

    # ── THREE PREDICTION COLUMNS ──────────────────────────────────────────────
    # Display Buddhist, Western, and Risk predictions side-by-side
    col_b, col_w, col_rn = st.columns(3)

    # BUDDHIST PERSONALITY COLUMN
    with col_b:
        bi = BUDDHIST_INFO.get(b_type, {})  # Get info for this Buddhist type
        # Render colored card with Buddhist prediction
        render_card(f"""
            <div style='font-size:2.4em'>{bi.get('emoji','🧘')}</div>
            <div style='font-size:0.8em; opacity:.8; margin:4px 0'>Buddhist Psychology</div>
            <div style='font-size:1.25em; font-weight:700'>{b_type}</div>
            <div style='font-size:0.88em; opacity:.9'>{b_conf:.1f}% confidence</div>
        """, "card-buddhist")
        
        # Display detailed information about Buddhist type
        st.markdown(f"**Core Trait:** {bi.get('core','')}")
        st.markdown(f"**Childhood Signal:** {bi.get('childhood','')}")
        st.markdown(f"**Mental Health Impact:** {bi.get('mental_impact','')}")
        st.markdown(f"**Work Behaviour:** {bi.get('work_behavior','')}")
        st.markdown(f"**Recommended Practice:** _{bi.get('practice','')}_")
        st.markdown("**Type Probabilities:**")  # Show confidence for all types
        prob_bars(le_b.classes_, b_prob)

    # WESTERN PERSONALITY COLUMN
    with col_w:
        wi = WESTERN_INFO.get(w_type, {})  # Get info for this Western type
        # Render colored card with Western prediction
        render_card(f"""
            <div style='font-size:2.4em'>{wi.get('emoji','🌍')}</div>
            <div style='font-size:0.8em; opacity:.8; margin:4px 0'>Western Psychology</div>
            <div style='font-size:1.25em; font-weight:700'>{w_type}</div>
            <div style='font-size:0.88em; opacity:.9'>{w_conf:.1f}% confidence</div>
        """, "card-western")
        
        # Display detailed information about Western type
        st.markdown(f"**Core Trait:** {wi.get('core','')}")
        st.markdown(f"**Strengths:** {wi.get('strengths','')}")
        st.markdown(f"**Risk Areas:** {wi.get('risks','')}")
        st.markdown(f"**CBT Intervention:** {wi.get('intervention','')}")
        st.markdown("**Type Probabilities:**")  # Show confidence for all types
        prob_bars(le_w.classes_, w_prob)

    # RISK & NUMEROLOGY COLUMN
    with col_rn:
        # Display Mental Health Risk card with color based on risk level
        risk_css = {"Low Risk": "card-risk-low", "Moderate Risk": "card-risk-mod",
                    "High Risk": "card-risk-high"}
        risk_emoji = {"Low Risk": "🟢", "Moderate Risk": "🟡", "High Risk": "🔴"}
        render_card(f"""
            <div style='font-size:2.4em'>{risk_emoji.get(risk_lvl,'⚠️')}</div>
            <div style='font-size:0.8em; opacity:.8; margin:4px 0'>Mental Health Risk</div>
            <div style='font-size:1.25em; font-weight:700'>{risk_lvl}</div>
            <div style='font-size:0.88em; opacity:.9'>Risk score: {risk_score}/5</div>
        """, risk_css.get(risk_lvl, "card-risk-mod"))

        # Display Numerology information (Life Path & Personal Number)
        render_card(f"""
            <div style='font-size:2.4em'>🔢</div>
            <div style='font-size:0.8em; opacity:.8; margin:4px 0'>Numerology</div>
            <div style='font-size:1.1em; font-weight:700'>{num_info['archetype']}</div>
            <div style='font-size:0.88em; opacity:.9'>Life Path: {lpn} &nbsp;|&nbsp; Personal No.: {pn}</div>
        """, "card-numerology")

        # Display numerology details and guidance
        st.markdown(f"**Life Theme:** {num_info['theme']}")
        st.markdown(f"**Traits:** {num_info['traits']}")
        st.markdown(f"**Life Challenge:** {num_info['challenge']}")
        st.markdown(f"**Affirmation:** _{num_info['affirmation']}_")
        st.markdown(f"**Guidance:** {num_info['guidance']}")

    st.divider()

    # ── VISUAL ANALYSIS ───────────────────────────────────────────────────────
    st.markdown("## 📈 Visual Analysis")

    v_tab1, v_tab2 = st.tabs(["🕸️ Trait Radar Charts", "📊 Score & Risk Summary"])

    with v_tab1:
        st.markdown(
            "Each axis shows a score from **1 (low) to 5 (high)**. "
            "The green background is the ideal profile. "
            "Your shape reveals where strengths and gaps lie."
        )
        rc1, rc2 = st.columns(2)

        # Build raw feature values lists for both models
        b_raw = b_df.values[0].tolist()   # [childhood_support, stress_level, ...]
        w_raw = w_df.values[0].tolist()   # [openness, conscientiousness, ...]

        with rc1:
            st.plotly_chart(
                buddhist_radar(b_raw, b_type, b_conf),
                use_container_width=True,
            )
            with st.expander("What do these axes mean?"):
                st.markdown("""
| Axis | Meaning |
|---|---|
| **Childhood Support** | How nurturing your early environment was |
| **Stress Resilience** | Inverted stress — higher = more resilient |
| **Work Capacity** | Inverted work pressure — higher = less burdened |
| **Emotional Attachment** | Tendency toward craving/attachment (Raga axis) |
| **Mental Clarity** | Lucidity and decision quality |
| **Coping Skill** | Ability to manage adversity |
                """)

        with rc2:
            st.plotly_chart(
                western_radar(w_raw, w_type, w_conf),
                use_container_width=True,
            )
            with st.expander("What do these axes mean?"):
                st.markdown("""
| Axis | Meaning |
|---|---|
| **Openness** | Curiosity, creativity, imagination |
| **Conscientiousness** | Self-discipline, organisation, reliability |
| **Extraversion** | Social energy, assertiveness, positivity |
| **Agreeableness** | Empathy, cooperation, trust |
| **Neuroticism** | Emotional instability / sensitivity |
| **Stress Response** | How stress manifests in behaviour |
| **Coping Skill** | Ability to manage adversity |
                """)

    with v_tab2:
        gc1, gc2 = st.columns(2)
        with gc1:
            st.plotly_chart(
                risk_gauge(risk_score, risk_lvl),
                use_container_width=True,
            )
            with st.expander("How is the risk score calculated?"):
                st.markdown("""
The risk score is a **weighted composite** of your four domain scores:

```
Risk = (6 - Mental Health) × 0.40
     + (6 - Childhood)     × 0.30
     + (6 - Work)          × 0.20
     + (6 - Coping)        × 0.10
```

| Range | Level |
|---|---|
| 0.0 – 2.49 | 🟢 Low Risk |
| 2.5 – 3.49 | 🟡 Moderate Risk |
| 3.5 – 5.0  | 🔴 High Risk |
                """)
        with gc2:
            st.plotly_chart(
                composite_bar(cs, mh, ws, cp),
                use_container_width=True,
            )
            with st.expander("What are the composite scores?"):
                st.markdown("""
Each composite score is the **average of 6 survey responses** in that domain,
with negative items reverse-scored so that higher always means better:

- **Childhood** — quality of early-life environment
- **Mental Health** — current emotional wellbeing
- **Work Performance** — effectiveness and satisfaction at work
- **Coping Skill** — resilience and healthy coping strategies

The dotted line at **3.0** is the neutral midpoint.
                """)

    st.divider()

    # ── 7-DAY RECOVERY SCHEDULE ───────────────────────────────────────────────
    # Display personalized weekly schedule with activities
    st.markdown("## 📅 Your Personalised 7-Day Recovery & Wellness Schedule")
    st.markdown(
        f"*Tailored for: **{b_type}** &nbsp;·&nbsp; **{w_type}** &nbsp;·&nbsp; **{risk_lvl}***"
    )

    # Color and emoji for each day of week (visual differentiation)
    day_borders = {
        "Monday":    "#667eea", "Tuesday": "#11998e", "Wednesday": "#f7971e",
        "Thursday":  "#c94b4b", "Friday":  "#0575E6", "Saturday":  "#764ba2",
        "Sunday":    "#27ae60",
    }
    day_emojis = {
        "Monday": "🌅", "Tuesday": "💪", "Wednesday": "🧘",
        "Thursday": "🙏", "Friday": "📊", "Saturday": "🌿", "Sunday": "☀️",
    }

    # Display schedule in 2-column layout
    cols = st.columns(2)
    for idx, (day, slots) in enumerate(schedule.items()):
        # Build HTML for time slots (Morning/Afternoon/Evening)
        slots_html = "".join(
            f'<div class="day-slot"><b>{slot}:</b> {activity}</div>'
            for slot, activity in slots.items()
        )
        # Create day card with color border
        html = f"""
        <div class="day-card" style="border-left-color:{day_borders.get(day,'#667eea')}">
            <strong>{day_emojis.get(day,'📅')} {day}</strong>
            {slots_html}
        </div>
        """
        # Alternate between left and right columns for layout
        with cols[idx % 2]:
            st.markdown(html, unsafe_allow_html=True)

    st.divider()

    # ── INTEGRATED PSYCHOLOGICAL GUIDANCE ────────────────────────────────────
    # Display 4 guidance boxes with recommendations from different frameworks
    st.markdown("## 🎯 Integrated Psychological Guidance")
    g1, g2 = st.columns(2)

    # LEFT COLUMN: BUDDHIST PRACTICE + NUMEROLOGY
    with g1:
        st.markdown('<div class="guidance-box">', unsafe_allow_html=True)
        st.markdown("### 🪷 Buddhist Practice")
        practice = BUDDHIST_INFO.get(b_type, {}).get("practice", "Daily mindfulness")
        st.info(
            f"**Recommended:** {practice}\n\n"
            f"This practice directly addresses the root mental formations associated "
            f"with the **{b_type}** tendency — helping cultivate clarity and equanimity."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="guidance-box">', unsafe_allow_html=True)
        st.markdown("### 🔢 Numerology Guidance")
        st.info(
            f"**Life Path {lpn} — {num_info['archetype']}**\n\n"
            f"{num_info['guidance']}\n\n"
            f"_\"{num_info['affirmation']}\"_"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT COLUMN: WESTERN PSYCHOLOGY + RISK INTERVENTIONS
    with g2:
        st.markdown('<div class="guidance-box">', unsafe_allow_html=True)
        st.markdown("### 🧬 Western Psychology Intervention")
        w_activity  = WESTERN_INFO.get(w_type, {}).get("activity",     "Regular self-care")
        w_interv    = WESTERN_INFO.get(w_type, {}).get("intervention",  "")
        st.info(
            f"**Evening Activity:** {w_activity}\n\n"
            f"**Clinical Focus:** {w_interv}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="guidance-box">', unsafe_allow_html=True)
        st.markdown("### ⚕️ Mental Health Risk Actions")
        actions = RISK_INTERVENTIONS.get(risk_lvl, [])  # Get actions for this risk level
        action_text = "\n\n".join(f"• {a}" for a in actions)  # Format as bullet list
        
        # Display with different styling based on risk level (error/warning/success)
        if risk_lvl == "High Risk":
            st.error(f"**{risk_lvl} — Immediate Action Recommended**\n\n{action_text}")
        elif risk_lvl == "Moderate Risk":
            st.warning(f"**{risk_lvl} — Proactive Steps Suggested**\n\n{action_text}")
        else:
            st.success(f"**{risk_lvl} — Maintain & Strengthen**\n\n{action_text}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── EXPORT ────────────────────────────────────────────────────────────────
    # Allow user to download profile results as CSV files
    st.markdown("## 📥 Export Your Profile")

    # Create dictionary with all profile information for CSV download
    profile_row = {
        "Name": display_name, "Age": age, "Gender": gender,
        "Date of Birth": dob.strftime("%Y-%m-%d"), "Profession": profession or "—",
        "Childhood Score": round(cs, 2), "Childhood Environment": c_env,
        "Mental Health Score": round(mh, 2), "Work Score": round(ws, 2),
        "Coping Score": round(cp, 2),
        "Buddhist Personality": b_type, "Buddhist Confidence (%)": round(b_conf, 1),
        "Western Personality":  w_type,  "Western Confidence (%)":  round(w_conf, 1),
        "Mental Health Risk": risk_lvl, "Risk Score": risk_score,
        "Life Path Number": lpn, "Personal Number": pn,
        "Numerology Archetype": num_info["archetype"],
    }

    # Create schedule rows for CSV download
    sched_rows = [
        {"Day": day, **{s.replace("🌅 ","").replace("🕛 ","").replace("🌙 ",""): act
                        for s, act in slots.items()}}
        for day, slots in schedule.items()
    ]

    # Create download buttons side-by-side
    dl1, dl2 = st.columns(2)
    with dl1:
        # Download button for profile summary
        st.download_button(
            "📊 Download Profile CSV",
            data=pd.DataFrame([profile_row]).to_csv(index=False),
            file_name=f"personality_profile_{display_name.replace(' ','_')}_{dob}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        # Download button for recovery schedule
        st.download_button(
            "📅 Download Weekly Schedule CSV",
            data=pd.DataFrame(sched_rows).to_csv(index=False),
            file_name=f"recovery_schedule_{display_name.replace(' ','_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── FOOTER ────────────────────────────────────────────────────────────────
    # Display disclaimer and copyright information
    st.markdown("""
    <div style="text-align:center; color:#aaa; font-size:0.8em; margin-top:30px; padding:20px 0">
        This tool is for research and educational purposes only.<br>
        It does not replace professional psychological or medical advice.<br>
        <em>MSc Research — Personality & Mental Health Profiler © 2025</em>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT - Script execution starts here
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # This block only runs if the script is executed directly (not imported as a module)
    # Calls main() to start the Streamlit app
    main()
