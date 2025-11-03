import streamlit as st
import joblib
import graphviz

# =============================
# Load trained model & data
# =============================
model = joblib.load("career_recommender_model.pkl")
job_roles = joblib.load("job_roles.pkl")
features = joblib.load("features.pkl")

# =============================
# App Header
# =============================
st.set_page_config(page_title="ALPHA - Career Path Finder", page_icon="⚡", layout="centered")
st.title("⚡ ALPHA")
st.markdown(
    "<h2 style='font-size:22px;'>Adaptive Learning Personalised Helper Algorithm</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='font-size:18px;'>Rate your interest from <b>0</b> (No Interest) to <b>5</b> (High Interest)</p>",
    unsafe_allow_html=True
)

# =============================
# Define questionnaire sections
# =============================
sections = {
    "🧠 Technical Skills": [
        "Programming",
        "Electronics",
        "Data Analysis",
        "Problem Solving",
        "Maths and Algorithms"
    ],
    "🎨 Creative Skills": [
        "Graphic Design",
        "Animation",
        "Storytelling",
        "Music / Audio",
        "UI/UX Design"
    ],
    "💬 Social & Communication": [
        "Teamwork",
        "Public Speaking",
        "Counselling",
        "Negotiation",
        "Teaching / Mentoring"
    ],
    "⚙️ Practical & Field Work": [
        "Mechanical Work",
        "Electrical Systems",
        "Robotics / IoT",
        "3D Modelling",
        "Research Experiments"
    ]
}

# =============================
# Collect user ratings
# =============================
user_ratings = {}

with st.form("career_form"):
    for section, items in sections.items():
        st.markdown(f"### {section}")
        cols = st.columns(2)
        for i, feature in enumerate(items):
            with cols[i % 2]:
                user_ratings[feature] = st.slider(feature, 0, 5, 3)
        st.markdown("---")
    submitted = st.form_submit_button("🔍 Find My Career Path")

# =============================
# Prediction + Visualization
# =============================
if submitted:
    predictions = []
    for job in job_roles:
        total = 0
        for feature, rating in user_ratings.items():
            try:
                total += model.predict(job, feature).est * rating
            except:
                continue
        predictions.append((job, total / len(user_ratings)))

    # Sort and select top results
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    best_match = sorted_preds[0][0]
    related_jobs = [j for j, _ in sorted_preds[1:4]]

    st.success(f"🎯 Your Recommended Career Path: **{best_match}**")

    # =============================
    # Visualize Career Flow
    # =============================
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', size='8,5')

    graph.node("You", "You", shape="circle", style="filled", color="#90CAF9")
    graph.node(best_match, best_match, shape="box", style="filled", color="#A5D6A7")
    graph.edge("You", best_match, label="Best Match", color="#1E88E5")

    for job in related_jobs:
        graph.node(job, job, shape="box", style="filled", color="#FFF59D")
        graph.edge(best_match, job, label="Related", color="#8E24AA")

    st.graphviz_chart(graph)

    # =============================
    # Expandable Career Paths
    # =============================
    from path import career_paths

    st.markdown("### 💼 Explore Career Paths")

    all_jobs = [best_match] + related_jobs

    for job in all_jobs:
        with st.expander(f"🚀 {job} — Click to view roadmap"):
            path_steps = career_paths.get(job, ["No roadmap available yet."])
            for i, step in enumerate(path_steps, 1):
                st.markdown(f"**{i}.** {step}")

else:
    st.info("⬆️ Adjust your interests above and click **Find My Career Path** to see recommendations.")
