# packages
import streamlit as st
import numpy as np
from copy import deepcopy

#from credentials import openai_api
import os

# New logo image URL
new_logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/Logo_univsemmelweis.svg/225px-Logo_univsemmelweis.svg.png"

# Display the new logo above the existing logo
st.sidebar.image(new_logo_url, use_column_width=True)

# Define the list of names
names = ["Name 1", "Name 2", "Name 3"]

# Define the list of names
names = ["Name 1", "Name 2", "Name 3"]

# Display a button to show the names and select one
if st.button("Choose a name"):
    selected_name = st.selectbox("Select a name:", names)
    st.write("Selected name:", selected_name)


# Define the previously executed prompt answer
predefined_prompt_answer = "This is the predefined prompt answer."

# Display the predefined prompt answer in a textbox
st.text(predefined_prompt_answer)
