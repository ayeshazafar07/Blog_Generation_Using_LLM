import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get response from llama model
def getLLamaResponse(input_text, no_words, blog_style):
    # LLama model calling using ctransformers
    llm = CTransformers(model='models\llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens':256,
                                'temperature':0.01})
    
    # Creating Prompt Template
    template="""
        Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words.
        """
    prompt = PromptTemplate(input_variables=["blog_style","input_text","no_words"],
                            template=template)
    
    # Generate the response from llama 2 model
    response = llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response


# Setting streamlit (can use flask too)
st.set_page_config(page_title="Generate Blogs",
                   page_icon="👽",
                   layout="centered",
                   initial_sidebar_state="collapsed"
                   )
st.header("Generate Blogs 👽")

# Creating input text field
input_text = st.text_input("Enter the Blog Topic")

# Creating input fields for no of words for blogs and style 
col1,col2 = st.columns([5,5]) # 5,5 is width for columns

with col1: 
    no_words = st.text_input("No of Words")
with col2:
    blog_style = st.selectbox("Writing the blog for", 
                              ("Researcher", "Data Scientist", "Student", "Other"), index=0) # Dropdown selection box

# Creating submit button
submit = st.button("Generate")

# Final response
if submit:
    st.write(getLLamaResponse(input_text, no_words, blog_style))