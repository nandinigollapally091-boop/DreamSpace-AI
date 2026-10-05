
import streamlit as st
import ollama
import chromadb

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from PIL import Image

import os
import uuid
from io import BytesIO
from urllib.parse import quote
from urllib.request import Request, urlopen


# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DreamSpace AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# TITLE
# =========================================================

st.title("🏠 DreamSpace AI")

st.subheader(
    "AI-Powered Interior & Space Design Assistant"
)

st.write(
    "Describe your dream space and DreamSpace AI "
    "will create an AI design concept and a realistic "
    "interior image."
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# =========================================================
# CHROMADB
# =========================================================

@st.cache_resource
def load_database():

    client = chromadb.PersistentClient(
        path="./dreamspace_db"
    )

    collection = client.get_or_create_collection(
        name="dreamspace_knowledge"
    )

    return collection


collection = load_database()


# =========================================================
# CREATE EMBEDDING
# =========================================================

def create_embedding(text):

    embedding = embedding_model.encode(text)

    return embedding.tolist()


# =========================================================
# ADD PDF CONTENT TO DATABASE
# =========================================================

def add_to_database(text, source):

    if not text.strip():
        return

    embedding = create_embedding(text)

    document_id = (
        source
        + "_"
        + uuid.uuid4().hex
    )

    collection.add(
        ids=[document_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[
            {
                "source": source
            }
        ]
    )


# =========================================================
# SEARCH KNOWLEDGE BASE
# =========================================================

def search_database(query):

    if collection.count() == 0:
        return ""

    query_embedding = create_embedding(query)

    number_of_results = min(
        3,
        collection.count()
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=number_of_results
    )

    documents = results.get(
        "documents",
        []
    )

    if not documents:
        return ""

    if not documents[0]:
        return ""

    return "\n\n".join(
        documents[0]
    )


# =========================================================
# READ PDF
# =========================================================

def read_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# GENERATE DESIGN CONCEPT USING OLLAMA
# =========================================================

def generate_design(
    space_type,
    design_style,
    mood,
    user_description,
    knowledge
):

    prompt = f"""
You are DreamSpace AI,
an expert interior designer.

Create a simple and practical interior design plan.

SPACE TYPE:
{space_type}

DESIGN STYLE:
{design_style}

MOOD:
{mood}

USER DESCRIPTION:
{user_description}

RELEVANT KNOWLEDGE:
{knowledge}

IMPORTANT OUTPUT RULES:

- Keep the explanation short and easy to understand.
- Do NOT write long paragraphs.
- Use bullet points for almost everything.
- Each bullet should be short and clear.
- Give practical suggestions.
- Avoid unnecessary explanations.
- Do not mention programming, APIs, AI models, or technical details.

Use exactly these sections:

## 🏡 Overall Design Concept

Write only ONE short and simple sentence
explaining the overall design idea.

## 🎨 Color Palette

- Main color:
- Secondary color:
- Accent color:

## 🛋️ Furniture

- Sofa:
- Coffee table:
- Storage:
- Other important furniture:

## 💡 Lighting

- Main lighting:
- Ambient lighting:
- Natural lighting:

## 🧱 Walls and Flooring

- Walls:
- Flooring:
- Feature wall:

## 🌿 Decorations

- Plants:
- Artwork:
- Rugs:
- Other decorations:

## 📐 Space Arrangement

- Furniture placement:
- Walking space:
- Main focal point:

## ✅ Practical Suggestions

- Suggestion 1:
- Suggestion 2:
- Suggestion 3:

Keep everything concise, realistic,
practical, and easy for a normal person to understand.
"""


    response = ollama.chat(
        model="llama3.2:latest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# =========================================================
# CREATE IMAGE PROMPT
# =========================================================

def create_image_prompt(
    space_type,
    design_style,
    mood,
    user_description,
    design
):

    prompt = f"""
Photorealistic professional interior
architecture photograph.

Create this exact space:

Space type:
{space_type}

Interior style:
{design_style}

Mood:
{mood}

User requirements:
{user_description}

Interior design concept:
{design}

VISUAL REQUIREMENTS:

- photorealistic interior
- professional architectural photography
- realistic furniture
- realistic materials
- realistic textures
- realistic proportions
- beautiful composition
- natural lighting
- detailed interior elements
- coherent color palette
- practical furniture arrangement
- spacious realistic interior
- premium interior design
- complete view of the requested space

DO NOT INCLUDE:

- people
- humans
- text
- letters
- logos
- watermarks
- fantasy elements
- unrealistic furniture
- distorted architecture
- extra rooms
- outdoor scenes

The final image should look like
a real professional interior-design photograph.
"""

    return prompt


# =========================================================
# GENERATE IMAGE
# =========================================================

def generate_image(image_prompt):

    encoded_prompt = quote(
        image_prompt
    )

    image_url = (
        "https://image.pollinations.ai/prompt/"
        + encoded_prompt
        + "?width=1024"
        + "&height=1024"
        + "&nologo=true"
    )

    request = Request(
        image_url,
        headers={
            "User-Agent": "DreamSpaceAI/1.0"
        }
    )

    with urlopen(
        request,
        timeout=180
    ) as response:

        image_bytes = response.read()

    image = Image.open(
        BytesIO(image_bytes)
    )

    return image


# =========================================================
# SIDEBAR - KNOWLEDGE BASE
# =========================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload a design PDF",
        type=["pdf"]
    )

    if uploaded_file:

        if st.button(
            "📥 Add PDF",
            use_container_width=True
        ):

            try:

                pdf_text = read_pdf(
                    uploaded_file
                )

                if not pdf_text.strip():

                    st.warning(
                        "No readable text was found in this PDF."
                    )

                else:

                    add_to_database(
                        pdf_text,
                        uploaded_file.name
                    )

                    st.success(
                        "PDF added successfully!"
                    )

            except Exception as error:

                st.error(
                    f"PDF error: {error}"
                )


# =========================================================
# DESIGN OPTIONS
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    space_type = st.selectbox(
        "🏠 Space Type",
        [
            "Bedroom",
            "Living Room",
            "Kitchen",
            "Dining Room",
            "Home Office",
            "Bathroom",
            "Studio Apartment",
            "Full House"
        ]
    )


with col2:

    design_style = st.selectbox(
        "🎨 Design Style",
        [
            "Modern",
            "Minimalist",
            "Luxury",
            "Scandinavian",
            "Industrial",
            "Bohemian",
            "Contemporary",
            "Traditional"
        ]
    )


with col3:

    mood = st.selectbox(
        "💡 Mood",
        [
            "Cozy",
            "Calm",
            "Elegant",
            "Warm",
            "Bright",
            "Luxury",
            "Relaxing",
            "Energetic"
        ]
    )


# =========================================================
# USER DESCRIPTION
# =========================================================

user_description = st.text_area(
    "✨ Describe your dream space",

    placeholder=(
        "Example: A cozy modern bedroom with "
        "warm lighting, wooden flooring, "
        "indoor plants, a large window and "
        "minimalist furniture."
    ),

    height=160
)


# =========================================================
# GENERATE DREAM DESIGN
# =========================================================

if st.button(
    "✨ Generate Dream Design",
    type="primary",
    use_container_width=True
):

    if not user_description.strip():

        st.warning(
            "Please describe your dream space first."
        )

        st.stop()


    try:

        # =================================================
        # STEP 1 - SEARCH KNOWLEDGE
        # =================================================

        with st.spinner(
            "📚 Searching design knowledge..."
        ):

            knowledge = search_database(
                user_description
            )


        # =================================================
        # STEP 2 - GENERATE DESIGN CONCEPT
        # =================================================

        with st.spinner(
            "🧠 Creating your design concept..."
        ):

            design = generate_design(
                space_type,
                design_style,
                mood,
                user_description,
                knowledge
            )


        # =================================================
        # STEP 3 - CREATE IMAGE PROMPT
        # =================================================

        image_prompt = create_image_prompt(
            space_type,
            design_style,
            mood,
            user_description,
            design
        )


        # =================================================
        # STEP 4 - GENERATE IMAGE
        # =================================================

        with st.spinner(
            "🎨 Generating your interior image..."
        ):

            generated_image = generate_image(
                image_prompt
            )


        # =================================================
        # STEP 5 - DISPLAY RESULT
        # =================================================

        st.success(
            "🎉 Your DreamSpace is ready!"
        )

        st.divider()


        image_column, design_column = st.columns(
            [1.1, 1]
        )


        # =================================================
        # GENERATED IMAGE
        # =================================================

        with image_column:

            st.subheader(
                "🖼️ Generated Interior"
            )

            st.image(
                generated_image,
                use_container_width=True
            )


        # =================================================
        # DESIGN CONCEPT
        # =================================================

        with design_column:

            st.subheader(
                "🏡 Design Concept"
            )

            st.markdown(
                design
            )


    except Exception as error:

        st.error(
            "Unable to generate your DreamSpace."
        )

        st.write(
            str(error)
        )

