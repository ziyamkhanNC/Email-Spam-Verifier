
import streamlit as st
import joblib

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.joblib"


st.set_page_config(
    page_title="Email Spam Verifier",
    page_icon="📧",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception:
    st.error("Could not load the trained model. Check the model file.")
    st.stop()


st.title("Email Spam Verifier")

st.write(
    "Check whether an email is spam or legitimate "
    "using a machine learning classifier."
)

st.divider()


with st.sidebar:
    st.header("Model Details")

    st.write("**Algorithm:** Multinomial Naive Bayes")
    st.write("**Vectorizer:** TF-IDF")
    st.write("**Test accuracy:** 99.00%")

    st.divider()

    st.caption("Email classification system")


st.subheader("Email Details")

with st.form("email_form"):

    subject = st.text_input(
        "Subject",
        placeholder="Enter email subject"
    )

    message = st.text_area(
        "Message",
        placeholder="Paste the email message here",
        height=220
    )

    submitted = st.form_submit_button(
        "Check Email",
        use_container_width=True
    )


if submitted:

    email_text = f"{subject} {message}".strip()

    if not email_text:
        st.warning("Please enter an email subject or message.")

    else:
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]

        class_probabilities = dict(
            zip(model.classes_, probabilities)
        )

        confidence = class_probabilities[prediction] * 100

        st.divider()
        st.subheader("Analysis Result")

        if prediction == "spam":
            st.error("Spam email detected.")

        else:
            st.success("This email appears to be legitimate.")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediction",
                prediction.upper()
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        st.subheader("Class Probabilities")

        for label in ["ham", "spam"]:
            probability = class_probabilities.get(label, 0) * 100

            st.write(f"{label.upper()}: {probability:.2f}%")
            st.progress(float(probability / 100))

        st.caption(
            "The confidence score represents the model's estimated "
            "probability and does not guarantee a correct prediction."
        )


st.divider()

st.caption("Email Spam Verifier | TF-IDF and Multinomial Naive Bayes")
