import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from datetime import datetime

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)

# =============================
# CSS DESIGN
# =============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F5F9FF 0%, #FFFFFF 45%, #EEF6FF 100%);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E6EEF7;
}

.hero {
    background: linear-gradient(120deg, #0B3D91, #1E88E5);
    padding: 34px 38px;
    border-radius: 26px;
    color: white;
    box-shadow: 0 18px 40px rgba(11, 61, 145, 0.20);
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
    font-weight: 900;
}

.hero p {
    font-size: 17px;
    opacity: 0.95;
    max-width: 900px;
}

.card {
    background: rgba(255,255,255,0.95);
    border: 1px solid #E3EDF8;
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 10px 28px rgba(21, 71, 122, 0.08);
    margin-bottom: 18px;
}

.small-card {
    background: white;
    border: 1px solid #E6EEF7;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 22px rgba(21, 71, 122, 0.06);
}

.metric-label {
    color: #6A7D91;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
}

.metric-value {
    color: #0B2F5B;
    font-size: 28px;
    font-weight: 900;
    margin-top: 5px;
}

.result-title {
    color: #0B2F5B;
    font-size: 22px;
    font-weight: 900;
}

.result-chip {
    display: inline-block;
    background: #E8F5E9;
    color: #167A3B;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 800;
    margin-top: 8px;
}

.stage-chip {
    display: inline-block;
    background: #FFF3E0;
    color: #B05A00;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 800;
    margin-top: 8px;
}

.warning {
    background: #FFF8E1;
    border-left: 6px solid #F4B400;
    padding: 16px 18px;
    border-radius: 14px;
    color: #5E4B00;
    font-weight: 600;
}

.info-box {
    background: #EAF4FF;
    border-left: 6px solid #1E88E5;
    padding: 16px 18px;
    border-radius: 14px;
    color: #12355B;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: #7A8FA6;
    font-size: 13px;
    margin-top: 28px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# MODEL
# =============================
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model("tumor_model.h5", compile=False)

model = load_ai_model()
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

# =============================
# FUNCTIONS
# =============================
def estimate_stage(pred_class, confidence):
    if pred_class == "no_tumor":
        return "Aucun stade détecté"

    if confidence >= 0.95:
        return "Stade avancé estimé (III-IV)"
    elif confidence >= 0.80:
        return "Stade intermédiaire estimé (II)"
    return "Stade précoce estimé (I)"

def clinical_note(pred_class):
    notes = {
        "glioma": "Suspicion de tumeur gliale. Une analyse radiologique approfondie est recommandée.",
        "meningioma": "Suspicion de méningiome. La localisation et les contours doivent être vérifiés par un spécialiste.",
        "pituitary": "Suspicion de tumeur hypophysaire. Une interprétation endocrinologique et radiologique est recommandée.",
        "no_tumor": "Aucune tumeur détectée par le modèle sur cette image."
    }
    return notes.get(pred_class, "Interprétation clinique non disponible.")

def make_gradcam(original_image, img_array, pred_index):
    try:
        model_gc = tf.keras.models.clone_model(model)
        model_gc(tf.zeros((1, 128, 128, 3)))
        model_gc.set_weights(model.get_weights())

        last_conv_layer = model_gc.get_layer("conv2d_1")

        grad_model = tf.keras.models.Model(
            inputs=model_gc.inputs,
            outputs=[last_conv_layer.output, model_gc.outputs[0]]
        )

        img_tensor = tf.convert_to_tensor(img_array)

        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_tensor, training=False)
            loss = preds[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        img = original_image.resize((128, 128))
        img_cv = np.array(img)

        heatmap = cv2.resize(heatmap, (128, 128))
        heatmap = np.uint8(255 * heatmap)

        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        return cv2.addWeighted(img_cv, 0.58, heatmap_color, 0.42, 0)

    except Exception:
        return None

# =============================
# SIDEBAR
# =============================
st.sidebar.markdown("## 🧠 NeuroScan AI")
st.sidebar.caption("Clinical Deep Learning Assistant")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "MRI Analysis", "Clinical Report", "About"]
)

st.sidebar.markdown("---")
st.sidebar.success("Model loaded")
st.sidebar.info("CNN · Grad-CAM · FastAPI · Docker")

# =============================
# DASHBOARD
# =============================
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>NeuroScan AI</h1>
        <p>
        Plateforme intelligente d’aide au diagnostic pour l’analyse d’images IRM cérébrales.
        Le système combine classification par Deep Learning, estimation clinique et explicabilité par Grad-CAM.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="small-card"><div class="metric-label">Model</div><div class="metric-value">CNN</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="small-card"><div class="metric-label">Classes</div><div class="metric-value">4</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="small-card"><div class="metric-label">Explainability</div><div class="metric-value">Grad-CAM</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="small-card"><div class="metric-label">Status</div><div class="metric-value">Ready</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Classes supportées</h3>
        <p><b>Glioma</b> · <b>Meningioma</b> · <b>Pituitary</b> · <b>No Tumor</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning">
        ⚠️ NeuroScan AI est un outil académique d’aide à l’interprétation.
        Il ne remplace pas le diagnostic d’un médecin spécialiste.
    </div>
    """, unsafe_allow_html=True)

# =============================
# MRI ANALYSIS
# =============================
elif page == "MRI Analysis":
    st.markdown("""
    <div class="hero">
        <h1>Analyse IRM</h1>
        <p>Importer une image IRM afin d’obtenir une prédiction, un score de confiance, une estimation du stade et une carte Grad-CAM.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Importer une image IRM",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="info-box">
            📌 Importez une image IRM pour lancer l’analyse intelligente.
        </div>
        """, unsafe_allow_html=True)

    else:
        original_image = Image.open(uploaded_file).convert("RGB")

        img = original_image.resize((128, 128))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        pred_index = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][pred_index])
        pred_class = class_names[pred_index]

        stage = estimate_stage(pred_class, confidence)
        note = clinical_note(pred_class)

        left, right = st.columns([1.05, 1])

        with left:
            st.markdown('<div class="card"><h3>Image importée</h3>', unsafe_allow_html=True)
            st.image(original_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown(f"""
            <div class="card">
                <div class="result-title">Résultat de classification</div>
                <div class="result-chip">Type détecté : {pred_class}</div>
                <br><br>
                <div class="metric-label">Score de confiance</div>
                <div class="metric-value">{confidence:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(confidence, 1.0))

            st.markdown(f"""
            <div class="card">
                <div class="result-title">Interprétation clinique</div>
                <div class="stage-chip">{stage}</div>
                <p style="margin-top:14px;">{note}</p>
                <p><b>Date :</b> {datetime.now().strftime("%d/%m/%Y %H:%M")}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("## Explicabilité par Grad-CAM")

        gradcam_img = make_gradcam(original_image, img_array, pred_index)

        g1, g2 = st.columns(2)

        with g1:
            st.markdown('<div class="card"><h3>Image originale</h3>', unsafe_allow_html=True)
            st.image(original_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with g2:
            st.markdown('<div class="card"><h3>Carte Grad-CAM</h3>', unsafe_allow_html=True)
            if gradcam_img is not None:
                st.image(gradcam_img, use_container_width=True)
            else:
                st.warning("Grad-CAM non disponible pour cette image.")
            st.markdown('</div>', unsafe_allow_html=True)

# =============================
# CLINICAL REPORT
# =============================
elif page == "Clinical Report":
    st.markdown("""
    <div class="hero">
        <h1>Rapport clinique</h1>
        <p>Cette section présente la logique d’interprétation utilisée par NeuroScan AI.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Logique du système</h3>
        <p>
        Le modèle CNN réalise une classification de l’image IRM en quatre classes :
        glioma, meningioma, pituitary et no_tumor.
        </p>
        <p>
        Le stade tumoral est estimé à partir du score de confiance du modèle et de l’analyse visuelle
        fournie par la carte Grad-CAM.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning">
        ⚠️ Le stade affiché est une estimation académique. Un diagnostic médical réel nécessite
        l’expertise d’un radiologue, des examens complémentaires et des données cliniques.
    </div>
    """, unsafe_allow_html=True)

# =============================
# ABOUT
# =============================
elif page == "About":
    st.markdown("""
    <div class="hero">
        <h1>À propos</h1>
        <p>NeuroScan AI est une application développée pour un projet Deep Learning médical.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Technologies utilisées</h3>
        <ul>
            <li>TensorFlow / Keras</li>
            <li>CNN</li>
            <li>Grad-CAM</li>
            <li>FastAPI</li>
            <li>Streamlit</li>
            <li>Docker</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    NeuroScan AI · Medical Imaging · Deep Learning · Explainable AI
</div>
""", unsafe_allow_html=True)