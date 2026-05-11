import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "tumor_model.h5",
        compile=False
    )

model = load_model()

classes = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

# =========================================================
# FUNCTIONS
# =========================================================

def estimate_stage(pred_class, conf):

    if pred_class == "no_tumor":
        return "Aucun stade"

    if conf >= 0.95:
        return "Stade avancé (III-IV)"

    elif conf >= 0.80:
        return "Stade intermédiaire (II)"

    return "Stade précoce (I)"


def clinical_note(pred_class):

    notes = {

        "glioma":
        "Suspicion de tumeur gliale nécessitant une validation clinique.",

        "meningioma":
        "Suspicion de méningiome avec croissance localisée.",

        "pituitary":
        "Suspicion de tumeur hypophysaire.",

        "no_tumor":
        "Aucune tumeur détectée."
    }

    return notes.get(pred_class, "")


def generate_gradcam(original, img_array, pred_index):

    try:

        clone = tf.keras.models.clone_model(model)

        clone(tf.zeros((1,128,128,3)))

        clone.set_weights(model.get_weights())

        last_conv = clone.get_layer("conv2d_1")

        grad_model = tf.keras.models.Model(
            inputs=clone.inputs,
            outputs=[last_conv.output, clone.outputs[0]]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(
                tf.convert_to_tensor(img_array),
                training=False
            )

            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0,1,2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1
        )

        heatmap = tf.maximum(heatmap,0)

        heatmap /= tf.reduce_max(heatmap)

        heatmap = heatmap.numpy()

        img = original.resize((128,128))

        img_cv = np.array(img)

        heatmap = cv2.resize(heatmap,(128,128))

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        superimposed = cv2.addWeighted(
            img_cv,
            0.6,
            heatmap,
            0.4,
            0
        )

        return superimposed

    except:
        return None

# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

#MainMenu {
    visibility:hidden;
}

footer {
    visibility:hidden;
}

header {
    visibility:hidden;
}

.stApp{
    background:#edf2f9;
    color:#10213d;
}

/* MAIN */

.block-container{
    max-width:1350px;
    padding-top:1rem;
}

/* LOGO */

.logo-box{

    background:white;

    border-radius:24px;

    padding:24px 35px;

    box-shadow:
    0 10px 30px rgba(0,0,0,0.06);

    margin-bottom:18px;
}

.logo{

    font-size:42px;

    font-weight:1000;

    color:#1247b7;
}

/* NAVBAR */

div[data-baseweb="tab-list"] {

    background: #ffffff;

    padding: 14px;

    border-radius: 20px;

    box-shadow:
    0 12px 35px rgba(40,70,120,0.10);

    border: 1px solid #e6edf8;

    gap: 12px;

    margin-bottom: 20px;
}

button[data-baseweb="tab"] {

    background: #f4f7fc;

    border-radius: 16px;

    padding: 14px 24px;

    color: #32445e;

    font-weight: 900;

    border: 1px solid transparent;

    transition:0.3s;
}

button[data-baseweb="tab"]:hover {

    background: #ebf2ff;

    color: #0b4f9c;
}

button[data-baseweb="tab"][aria-selected="true"] {

    background:
    linear-gradient(
        135deg,
        #0b4f9c,
        #6c4cff
    );

    color: white;

    box-shadow:
    0 10px 25px rgba(92,76,255,0.25);
}

button[data-baseweb="tab"] p {

    font-size:16px;

    font-weight:900;
}

/* HERO */

.hero{

    background:
    linear-gradient(
        135deg,
        #0d3fa8,
        #6a4df5
    );

    border-radius:32px;

    padding:70px;

    color:white;

    position:relative;

    overflow:hidden;

    margin-bottom:30px;
}

.hero::after{

    content:"";

    position:absolute;

    right:40px;

    top:35px;

    width:340px;

    height:340px;

    background-image:url("https://images.unsplash.com/photo-1581595219315-a187dd40c322?q=80&w=800");

    background-size:cover;

    background-position:center;

    border-radius:30px;

    opacity:0.95;

    box-shadow:
    0 15px 40px rgba(0,0,0,0.25);
}
.badge{

    display:inline-block;

    padding:12px 22px;

    border-radius:999px;

    background:
    rgba(255,255,255,0.18);

    border:
    1px solid rgba(255,255,255,0.25);

    font-weight:800;

    margin-bottom:24px;
}

.hero h1{

    font-size:74px;

    font-weight:1000;

    margin-bottom:12px;
}

.hero h2{

    font-size:34px;

    font-weight:900;
}

.hero p{

    font-size:21px;

    line-height:1.9;

    max-width:760px;
}

/* CARDS */

.card{

    background:white;

    border-radius:24px;

    padding:28px;

    border:
    1px solid #e8eef8;

    box-shadow:
    0 12px 30px rgba(0,0,0,0.06);

    margin-bottom:24px;
}

.metric{

    font-size:42px;

    font-weight:1000;

    color:#123ea6;
}

.metric-label{

    font-size:13px;

    font-weight:900;

    color:#7b8798;
}

.section-title{

    font-size:28px;

    font-weight:1000;

    color:#12377e;

    margin-bottom:15px;
}

/* RESULTS */

.result-green{

    background:#eaf9ef;

    border-left:6px solid #20ad5d;

    padding:18px;

    border-radius:18px;

    color:#11763f;

    font-weight:900;

    font-size:22px;

    margin-bottom:15px;
}

.result-orange{

    background:#fff4e3;

    border-left:6px solid #f39c12;

    padding:18px;

    border-radius:18px;

    color:#9c5d00;

    font-weight:900;

    font-size:22px;

    margin-bottom:15px;
}

.note{

    background:#eef5ff;

    border-left:6px solid #2b7cff;

    padding:18px;

    border-radius:18px;

    color:#194b85;

    font-weight:700;
}

.warning{

    background:#fff7e5;

    border-left:6px solid #f0b400;

    padding:22px;

    border-radius:18px;

    font-weight:800;

    color:#765000;

    margin-top:30px;
}

.footer{

    text-align:center;

    margin-top:35px;

    color:#70819b;

    font-weight:700;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="logo-box">
<div class="logo">
🧠 NeuroScan AI
</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# NAVIGATION
# =========================================================

tabs = st.tabs([
    "🏠 Dashboard",
    "🔬 Analyse MRI",
    "🔥 Grad-CAM",
    "📄 Rapport Clinique",
    "ℹ️ À propos"
])

# =========================================================
# DASHBOARD
# =========================================================

with tabs[0]:

    st.markdown("""
    <div class="hero">

    <div class="badge">
    ⚕ Intelligence Artificielle Médicale
    </div>

    <h1>
    NeuroScan AI
    </h1>

    <h2>
    Système intelligent d’aide au diagnostic
    </h2>

    <p>
    Analyse avancée des images IRM cérébrales grâce au Deep Learning.
    Classification, estimation du stade et visualisation explicable avec Grad-CAM.
    </p>

    </div>
    """, unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)

    with m1:

        st.markdown("""
        <div class="card">
        <div class="metric-label">MODÈLE</div>
        <div class="metric">CNN</div>
        <p>Deep Learning</p>
        </div>
        """, unsafe_allow_html=True)

    with m2:

        st.markdown("""
        <div class="card">
        <div class="metric-label">CLASSES</div>
        <div class="metric">4</div>
        <p>Types de tumeurs</p>
        </div>
        """, unsafe_allow_html=True)

    with m3:

        st.markdown("""
        <div class="card">
        <div class="metric-label">EXPLICABILITÉ</div>
        <div class="metric">Grad-CAM</div>
        <p>Interprétation visuelle</p>
        </div>
        """, unsafe_allow_html=True)

    with m4:

        st.markdown("""
        <div class="card">
        <div class="metric-label">BACKEND</div>
        <div class="metric">FastAPI</div>
        <p>API RESTful</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# ANALYSE MRI
# =========================================================

with tabs[1]:

    left,right = st.columns([1,1])

    with left:

        st.markdown("""
        <div class="card">
        <div class="section-title">
        📤 Importer une image IRM
        </div>
        <p>Formats supportés : JPG, JPEG, PNG</p>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Choisir une image",
            type=["jpg","jpeg","png"]
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="card">
        <div class="section-title">
        📊 Résultats de l’analyse
        </div>
        """, unsafe_allow_html=True)

        if uploaded is None:

            st.info(
                "Veuillez importer une image IRM."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:

        image = Image.open(uploaded).convert("RGB")

        resized = image.resize((128,128))

        arr = np.array(resized).astype("float32") / 255.0

        arr = np.expand_dims(arr, axis=0)

        prediction = model.predict(arr, verbose=0)

        pred_index = int(np.argmax(prediction[0]))

        conf = float(prediction[0][pred_index])

        pred_class = classes[pred_index]

        stage = estimate_stage(pred_class, conf)

        note = clinical_note(pred_class)

        cam = generate_gradcam(
            image,
            arr,
            pred_index
        )

        st.session_state["image"] = image
        st.session_state["cam"] = cam
        st.session_state["pred"] = pred_class
        st.session_state["conf"] = conf
        st.session_state["stage"] = stage
        st.session_state["note"] = note

        c1,c2 = st.columns([1,1])

        with c1:

            st.markdown("""
            <div class="card">
            <div class="section-title">
            Image originale
            </div>
            """, unsafe_allow_html=True)

            st.image(
                image,
                use_container_width=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

        with c2:

            st.markdown("""
            <div class="card">
            <div class="section-title">
            Diagnostic assisté
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-green">
            Type détecté : {pred_class}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-orange">
            Stade estimé : {stage}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <h2 style="color:#10377c;">
            Confiance : {conf:.4f}
            </h2>
            """, unsafe_allow_html=True)

            st.progress(conf)

            st.markdown(f"""
            <div class="note">
            {note}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# GRAD CAM
# =========================================================

with tabs[2]:

    st.markdown("""
    <div class="card">
    <div class="section-title">
    🔥 Visualisation Grad-CAM
    </div>
    </div>
    """, unsafe_allow_html=True)

    if "cam" not in st.session_state:

        st.info(
            "Veuillez analyser une image d’abord."
        )

    else:

        g1,g2 = st.columns(2)

        with g1:

            st.markdown("""
            <div class="card">
            <div class="section-title">
            Image originale
            </div>
            """, unsafe_allow_html=True)

            st.image(
                st.session_state["image"],
                use_container_width=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

        with g2:

            st.markdown("""
            <div class="card">
            <div class="section-title">
            Carte Grad-CAM
            </div>
            """, unsafe_allow_html=True)

            st.image(
                st.session_state["cam"],
                use_container_width=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# RAPPORT
# =========================================================

with tabs[3]:

    st.markdown("""
    <div class="card">
    <div class="section-title">
    📄 Rapport Clinique
    </div>
    </div>
    """, unsafe_allow_html=True)

    if "pred" not in st.session_state:

        st.info(
            "Aucune analyse disponible."
        )

    else:

        st.markdown(f"""
        <div class="card">

        <p><b>Type détecté :</b> {st.session_state["pred"]}</p>

        <p><b>Confiance :</b> {st.session_state["conf"]:.4f}</p>

        <p><b>Stade estimé :</b> {st.session_state["stage"]}</p>

        <p><b>Note clinique :</b> {st.session_state["note"]}</p>

        <p><b>Date :</b> {datetime.now().strftime("%d/%m/%Y %H:%M")}</p>

        </div>
        """, unsafe_allow_html=True)

# =========================================================
# ABOUT
# =========================================================

with tabs[4]:

    st.markdown("""
    <div class="card">

    <div class="section-title">
    ℹ️ À propos de NeuroScan AI
    </div>

    <p>
    NeuroScan AI est une plateforme médicale basée sur le Deep Learning
    pour la classification des tumeurs cérébrales à partir d’IRM.
    </p>

    <ul>
    <li>✔ CNN pour la classification</li>
    <li>✔ Grad-CAM pour l’explicabilité</li>
    <li>✔ FastAPI + Docker</li>
    <li>✔ Streamlit Interface</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# WARNING
# =========================================================

st.markdown("""
<div class="warning">
⚠️ Cette application est un outil académique d’aide à la décision.
Elle ne remplace pas l’avis d’un médecin spécialiste.
</div>
""", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

NeuroScan AI • Deep Learning • TensorFlow • FastAPI • Docker • Streamlit

</div>
""", unsafe_allow_html=True)