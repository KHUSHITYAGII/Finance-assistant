import streamlit as st 
 
def main(): 
    st.title("Morning Market Brief Assistant") 
  
    if st.button("Get Market Brief"): 
        st.write("Market Brief will be generated here") 
 
if __name__ == "__main__": 
    main() 
