from utils import *
import streamlit as st
# from streamlit_chat import message




def main():
    st.title("48 Laws of Power")

    # User input
    user_input = st.text_input("Ask your question:")
    
    if user_input:
        # Call your predict function
        response = predict(user_input)
        
        # Display the response
        st.text_area("Bot Response:", response, height=400)

if __name__ == "__main__":
    main()