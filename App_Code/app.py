# import the libaries 
import streamlit as st 

#------CODE-------

# Navigation
pages = {
    "Navigation": [
        st.Page("LinearRegression.py")
    ],
}
pg = st.navigation(pages)
pg.run()