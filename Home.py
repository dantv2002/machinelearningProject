import streamlit as st

st.set_page_config(
    page_title="Trang chủ",
    page_icon="🚀"
)

st.write("# Welcome to website machine learning with Streamlit! 👋")

st.sidebar.success("Select a function above.")

st.image("https://github.com/HT-Tuan/MachineLearning/blob/main/images/streamlit_hero.jpg?raw=true", width=500)

st.markdown(
    """   
    Machine Learning and Data Science projects.
    
    👈 Select a function from the sidebar** to see some detection
    of what Streamlit can do!
    
    Own by: 
    1. Huynh Thanh Tuan - 20110120
    2. Tran Van Dan - 20110451
"""
)