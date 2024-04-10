import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image


# Application Backend

# To load medicine-dataframe from pickle in the form of dictionary
medicines_dict = pickle.load(open('medicine_dict.pkl', 'rb'))
medicines = pd.DataFrame(medicines_dict)
reasons_dict = pickle.load(open('uses.pkl', 'rb'))
reasons = pd.DataFrame(reasons_dict)

# To load similarity-vector-data from pickle in the form of dictionary
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Load descriptions of medicines
medicine_descriptions = pd.Series(medicines_dict['Description'], index=medicines_dict['Drug_Name'])

# Convert to lowercase only for string values
def custom_preprocessor(text):
    return text.lower() if isinstance(text, str) else str(text)

# Vectorize the medicine descriptions
vectorizer = TfidfVectorizer(stop_words='english', preprocessor=custom_preprocessor)
medicine_vectors = vectorizer.fit_transform(medicine_descriptions)

def recommend(medicine):
    medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_medicines = []
    for i in medicines_list:
        recommended_medicines.append(medicines.iloc[i[0]].Drug_Name)
    return recommended_medicines

def reason(medicine):
    medicine_index = reasons[reasons['Drug_Name'] == medicine]
    medicine_use = medicine_index['Description'].values[0]
    return medicine_use

def search_best_medicine(illness_description):
    # Vectorize the input illness description
    illness_vector = vectorizer.transform([illness_description])

    # Calculate cosine similarity between illness description and medicine descriptions
    similarity_scores = cosine_similarity(illness_vector, medicine_vectors).flatten()

    # Rank medicines based on similarity
    ranked_medicines = sorted(list(enumerate(similarity_scores)), reverse=True, key=lambda x: x[1])

    # Return the top recommended medicines
    recommended_medicines = []
    for i in ranked_medicines[:5]:
        recommended_medicines.append(medicines.iloc[i[0]].Drug_Name)

    return recommended_medicines

# Application Frontend

#logo = Image.open('logo.jpeg')
#st.image(logo)
# Set page configuration
st.set_page_config(
    page_title="KnowMeds - Your Medical Companion",
    page_icon=":pill:",
    layout="wide"
)

# Colored background with border for the title
st.markdown(
    """
    <style>
        .title-text {
            background-color: #3498db;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            border: 2px solid #2c3e50;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title-text'>KnowMeds</h1>", unsafe_allow_html=True)

# User instructions
st.info("Select a medicine and click 'Recommend Medicine' to get alternatives. Describe an illness and click 'Find Best Medicine' to get recommendations based on the illness.")

# Colored background with border for section headings
st.markdown(
    """
    <style>
        .section-heading {
            background-color: #2ecc71;
            color: white;
            padding: 0.5rem;
            border-radius: 0.3rem;
            border: 2px solid #27ae60;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Searchbox for selecting medicine
st.markdown("<h2 class='section-heading'>Select Medicine</h2>", unsafe_allow_html=True)
selected_medicine_name = st.selectbox(
    'Type your medicine name whose alternative is to be recommended',
    medicines['Drug_Name'].values
)

# Recommendation Program
if st.button('Recommend Medicine'):
    recommendations = recommend(selected_medicine_name)
    if recommendations:
        st.subheader('Top Recommended Medicines')
        for i, medicine in enumerate(recommendations, start=1):
            st.write(f"{i}. {medicine}")
            st.write(f"Purchase at [PharmEasy](https://pharmeasy.in/search/all?name={medicine})")
    else:
        st.warning("No alternative medicines found.")

# Uses of Selected Medicine
if st.button('Uses of Selected Medicine'):
    uses = reason(selected_medicine_name)
    uses = ' '.join(uses)
    st.subheader('Uses of Selected Medicine')
    st.write(uses)

# Divider
st.markdown("<hr>", unsafe_allow_html=True)

# Colored background with border for illness description
st.markdown(
    """
    <style>
        .illness-heading {
            background-color: #e74c3c;
            color: white;
            padding: 1rem;
            border-radius: 0.3rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<h2 class='illness-heading'>Describe Illness</h2>", unsafe_allow_html=True)

# Searchbox for entering illness description
illness_description = st.text_area("Describe the illness:", height=150)
illness_description = illness_description.lower()


# Recommendation Program for best medicine based on illness description
if st.button('Find Best Medicine'):
    recommendations = search_best_medicine(illness_description)
    if recommendations:
        st.subheader('Top Recommended Medicines and Their Uses')
        for i, medicine in enumerate(recommendations, start=1):
            # Create an expander for each medicine
            with st.expander(f"{i}. {medicine} - Click to view description and uses"):
                # Show the uses of the selected medicine
                uses = reason(medicine)
                uses = ' '.join(uses)
                st.write(uses)

# Image load
image = Image.open('logo.jpeg')
st.image(image, caption='Your Medical Companion', use_column_width=True)
