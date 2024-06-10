import streamlit as st

# Import the Sentence Transformer library
from sentence_transformers import SentenceTransformer, util


MODEL_SENTENCE_TRANSFORMERS_ALL_MINI_LM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"


st.write('# TEXT SIMILARITY')

# There are several different Sentence Transformer models available on Hugging Face
model = SentenceTransformer(MODEL_SENTENCE_TRANSFORMERS_ALL_MINI_LM_L6_V2)

sentence_1 = st.text_area('Enter the first text: ')
sentence_1_embedding = model.encode(sentence_1, convert_to_tensor=True)

sentence_2 = st.text_area('Enter the second text: ')
sentence_2_embedding = model.encode(sentence_2, convert_to_tensor=True)

# Find the similarity between the two embeddings
cosine_similarity = util.pytorch_cos_sim(sentence_1_embedding, sentence_2_embedding)
print(" cosine_similarity: ", cosine_similarity)

st.write(cosine_similarity)
