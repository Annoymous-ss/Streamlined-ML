import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, fetch_california_housing
from src.model1 import model, code # your custom code
import pandas as pd

#-----CODE-----

# Title
st.title("MultiLinear Regression")

# Brief
st.markdown(
    "This is the page of MultiLinear Regression. "
    "Here you can try out the model, evaluate its performance, "
    "copy the code, and see some additional info."
)

# Sections
sections = ["Try It Out", "Code", "Other Info"]
if "selected_section" not in st.session_state:
    st.session_state.selected_section = sections[0]

cols = st.columns(len(sections))
for i, sec in enumerate(sections):
    if cols[i].button(sec):
        st.session_state.selected_section = sec

# Prepare placeholders
df = None
X = None
y = None

if st.session_state.selected_section == "Try It Out":
    st.write("Here's a quick demo. Upload a dataset **OR** use a toy dataset.")

    # Let user pick a toy dataset or "None"
    toy_options = ["None", "Diabetes Data", "California Housing Data"]
    option = st.selectbox("Choose a Toy Dataset", toy_options, index=0)

    if option != "None":
        # Load the chosen toy dataset
        if option == "Diabetes Data":
            data = load_diabetes()
        else:  # "California Housing Data"
            data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="Target")
        df = pd.concat([X, y], axis=1)

    # Centered OR
    st.markdown(
        "<h4 style='text-align: center;'>OR</h4>",
        unsafe_allow_html=True
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file")
    if uploaded_file is not None:
        # If a CSV is uploaded, use that instead
        df = pd.read_csv(uploaded_file)
        # Handle missing values to avoid "Input contains NaN" error
        if df.isna().sum().sum() > 0:
            st.warning("Your CSV has missing values. Dropping rows with NaN...")
            df.dropna(inplace=True)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    # If we still have no data (user selected "None" and no file), do nothing
    if df is None:
        st.info("Please select a toy dataset or upload a CSV to proceed.")
    else:
        # Let user select hyperparameters
        col1, col2 = st.columns(2)
        with col1:
            # Dynamically adjust epoch slider based on dataset size
            if df.shape[0] < 1000:
                epoch_range = (100, 10000)
            elif df.shape[0] < 100000:
                epoch_range = (1000, 50000)
            else:
                epoch_range = (10, 1000)

            epoch = int(
                st.slider(
                    "Selected number of epochs", 
                    epoch_range[0], epoch_range[1], 
                    step=10, 
                    value=epoch_range[0]
                )
            )
            lr = st.slider("Choose the Learning Rate", 0.001, 0.1, step=0.001, value=0.001)

        with col2:
            split = float(st.slider("Select the Test Split %", 10, 90, value=50) / 100)
            seed = int(st.slider("Choose the seed value", 0, 100, value=42))

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split, random_state=seed
        )

        # Call your custom model function
        # (Make sure it does model.predict(...) internally 
        #  and plots Actual vs. Predicted with a scatter!)
        model(X_train, y_train, X_test, y_test, epoch, lr)

elif st.session_state.selected_section == "Code":
    # Show the user your code snippet (or entire file)
    st.markdown("**The code snippet goes here.**")
    st.code(code, language="python")

elif st.session_state.selected_section == "Other Info":
    st.write('''Mathematical Equations :
             \nThe equation of Linear Regression is y = m.x + b'''
    )
    st.write("To update the parameters m(slope) & b(intercept) we calculate the gradients of the MSE loss function :\n")
    st.latex(r"J(m, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2")
    st.write("The gradient for slope(m):")
    st.latex(r"\frac{\partial J}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - (mx_i + b))")
    st.write("The gradients for intercept(b):")
    st.latex(r"\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - (mx_i + b))")
    st.write("Gradient Descent Update Equations")
    st.latex(r"m = m - \alpha \cdot \frac{\partial J}{\partial m}")
    st.latex(r"b = b - \alpha \cdot \frac{\partial J}{\partial b}")
    st.write('''This approach is useful when,
             \n1. Large Dataset \n2. High Dimensional Data
             \n3. Online Streaming \n4. Memory Constraints 
    ''')

