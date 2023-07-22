import streamlit as st

def option1_code():
    # Code for option 1
    st.write("Option 1 selected!")
    # Add your code here for option 1

def option2_code():
    # Code for option 2
    st.write("Option 2 selected!")
    # Add your code here for option 2

# Streamlit app code
def main():
    st.title("Select an Option")
    
    # Create a selectbox to choose between options
    selected_option = st.selectbox("Choose an option:", ("Option 1", "Option 2"))
    
    # Execute corresponding code based on the selected option
    if selected_option == "Option 1":
        option1_code()
    elif selected_option == "Option 2":
        option2_code()

if __name__ == "__main__":
    main()
