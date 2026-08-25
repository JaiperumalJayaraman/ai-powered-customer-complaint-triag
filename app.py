import os
import streamlit as st

from src.predict import predict

st.set_page_config(page_title="Complaint Triage AI", page_icon="📩", layout="centered")

st.title("📩 AI-Powered Customer Complaint Triage")
st.write("Enter a customer complaint to predict its category and priority.")

complaint = st.text_area(
    "Customer complaint",
    placeholder="Example: I was charged twice and need an urgent refund.",
    height=140,
)

if st.button("Predict", type="primary"):
    if not complaint.strip():
        st.warning("Please enter a complaint first.")
    else:
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        category_path = os.path.join(model_dir, "category_model.joblib")
        priority_path = os.path.join(model_dir, "priority_model.joblib")

        if not (os.path.exists(category_path) and os.path.exists(priority_path)):
            st.error("Models are not trained yet. Run: python src/train.py")
        else:
            result = predict(complaint)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Category", result["category"])
            with col2:
                st.metric("Predicted Priority", result["priority"])

            st.subheader("Model Confidence")
            st.write(f"Category: {result['category_confidence']:.1%}")
            st.progress(float(result["category_confidence"]))
            st.write(f"Priority: {result['priority_confidence']:.1%}")
            st.progress(float(result["priority_confidence"]))

            routing = {
                "Billing": "Billing & Finance Team",
                "Refund": "Refunds Team",
                "Returns": "Returns Team",
                "Payments": "Payments Team",
                "Fraud": "Fraud & Security Team",
                "Delivery": "Logistics / Delivery Team",
                "Account Access": "Customer Account Team",
                "Technical": "Technical Support Team",
            }
            st.info(f"Suggested routing team: **{routing.get(result['category'], 'Customer Support Team')}**")

st.caption("Portfolio demonstration only. Predictions should be reviewed by a human support agent.")
