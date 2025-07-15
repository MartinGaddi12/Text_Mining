# ─────────────────────────────────────────────────────────────────────────────
#  asistente_farmacologico.py
# ─────────────────────────────────────────────────────────────────────────────
import os, json, pickle, re, textwrap
from datetime import datetime

import faiss, numpy as np, streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
import functools
import time

# ─────────────────────────  CONFIGURACIÓN GENERAL  ──────────────────────────
st.set_page_config(
    page_title="💊🧠 Asistente Farmacológico",
    layout="wide",
    initial_sidebar_state="collapsed",
)

INDEX_DIR        = "./faiss_index_recursive"
JSON_DRUGS_PATH  = "./Medicamentos.json"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
TOP_K            = 4  # Reducido para respuestas más rápidas

BASE_TOKENS      = 150          # tokens "fijos" por mensaje (reducido)
TOKENS_PER_WORD  = 1.0          # tokens extra por palabra en la pregunta (reducido)
MAX_TOKENS       = 600          # límite duro de salida (reducido para respuestas más rápidas)

AUTORES = (
    "**Realizado por: Lucas Brusasca • Pedro Durán • Martin Gaddi • "
    " Paul Lijtmaer • Nicolás Palavecino **"
)

# ─────────────────────────────  OPTIMIZACIONES  ─────────────────────────────
@st.cache_data(ttl=3600)  # Cache por 1 hora
def cached_translate(text: str, source: str, target: str) -> str:
    """Traducción con caché para mejorar rendimiento."""
    trans_start = time.time()
    try:
        result = GoogleTranslator(source=source, target=target).translate(text)
        trans_time = time.time() - trans_start
        print(f"        🌐 GoogleTranslator ({source}→{target}): {len(text)} chars ({trans_time:.3f}s)")
        return result
    except Exception as e:
        trans_time = time.time() - trans_start
        print(f"        ❌ Error traducción ({trans_time:.3f}s): {e}")
        return text  # Retorna el texto original si falla

@st.cache_data(ttl=3600)  # Cache por 1 hora
def detect_language(text: str) -> str:
    """Detecta el idioma del texto de forma eficiente."""
    # Detección simple basada en patrones
    spanish_indicators = ['qué', 'cómo', 'cuál', 'dónde', 'cuándo', 'por qué', 'para qué', 'efectos', 'medicamento', 'fármaco']
    text_lower = text.lower()
    
    for indicator in spanish_indicators:
        if indicator in text_lower:
            return 'es'
    
    # Si no detecta español, asume inglés
    return 'en'

@st.cache_data(ttl=3600)  # Cache por 1 hora
def optimized_find_drugs(text: str) -> list:
    """Versión optimizada de búsqueda de medicamentos."""
    # Buscar directamente en el texto sin traducir cada palabra
    results, seen_urls = [], set()
    
    # Buscar medicamentos conocidos directamente
    for drug_name, url in DRUGS.items():
        if drug_name in text.lower() and url not in seen_urls:
            seen_urls.add(url)
            results.append((drug_name.title(), url))
    
    return results[:5]  # Limitar a 5 resultados para mejorar rendimiento

# ───────────────────────────────  ESTILOS  ──────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { 
        background: #343541; 
        color: #ffffff; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .chat-container {
        background: #444654;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: #343541;
        color: #ffffff;
        padding: 15px 20px;
        border-radius: 18px;
        margin: 10px 0;
        margin-left: 20%;
        border: 1px solid #565869;
    }
    
    .assistant-message {
        background: #444654;
        color: #ffffff;
        padding: 15px 20px;
        border-radius: 18px;
        margin: 10px 0;
        margin-right: 20%;
        border: 1px solid #565869;
    }
    
    .message-header {
        font-weight: bold;
        font-size: 0.9rem;
        margin-bottom: 8px;
        opacity: 0.8;
    }
    
    .stTextInput > div > div > input {
        background: #40414f;
        color: white;
        border: 1px solid #565869;
        border-radius: 25px;
        padding: 15px 20px;
        font-size: 16px;
    }
    
    .stButton > button {
        background: #10a37f;
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: #0d8f6f;
        transform: translateY(-1px);
    }
    
    .settings-panel {
        background: #40414f;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #565869;
    }
    
    .stSelectbox > div > div {
        background: #40414f;
        color: white;
        border: 1px solid #565869;
        border-radius: 8px;
    }
    
    .header-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 30px;
        color: #10a37f;
    }
    
    .chat-history {
        max-height: 600px;
        overflow-y: auto;
        margin-bottom: 20px;
    }
    
    .input-container {
        position: sticky;
        bottom: 0;
        background: #343541;
        padding: 20px 0;
        border-top: 1px solid #565869;
    }
    
    .stMarkdown a { color: #10a37f; }
    
    /* Ocultar elementos innecesarios */
    .stDeployButton { display: none; }
    footer { display: none; }
    header { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────  FAISS  +  CORPUS  ─────────────────────────────
@st.cache_resource(show_spinner=False)
def load_faiss():
    idx  = faiss.read_index(os.path.join(INDEX_DIR, "index_recursive.faiss"))
    with open(os.path.join(INDEX_DIR, "texts_recursive.pkl"), "rb") as f:
        texts = pickle.load(f)
    return idx, texts

print("🔄 Cargando recursos...")
startup_time = time.time()
faiss_index, corpus = load_faiss()
startup_elapsed = time.time() - startup_time
print(f"✅ FAISS cargado: {len(corpus)} docs ({startup_elapsed:.3f}s)")

# ───────────────────────  EMBEDDINGS + ​​RETRIEVAL  ─────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

embed_startup = time.time()
embedder = load_embedder()
embed_elapsed = time.time() - embed_startup
print(f"Embedder cargado: {EMBEDDING_MODEL} ({embed_elapsed:.3f}s)")

def embed(q: str) -> np.ndarray:
    return np.asarray(embedder.encode(q)).astype("float32")

@st.cache_data(ttl=300)  # Cache por 5 minutos
def retrieve(query: str, k: int = TOP_K):
    # Timer para embedding
    embed_start = time.time()
    qvec = embed(query).reshape(1, -1)
    embed_time = time.time() - embed_start
    print(f"        Embedding: ({embed_time:.3f}s)")
    
    # Timer para búsqueda FAISS
    search_start = time.time()
    _, idx = faiss_index.search(qvec, k)
    search_time = time.time() - search_start
    print(f"        FAISS search: ({search_time:.3f}s)")
    
    # Timer para procesar resultados
    process_start = time.time()
    results = []
    for i in idx[0]:
        if i < len(corpus):
            # Verificar si corpus[i] es una cadena o una tupla/lista
            text = corpus[i]
            if isinstance(text, (list, tuple)):
                # Si es una tupla/lista, tomar el primer elemento o concatenar
                text = text[0] if len(text) > 0 else ""
            
            # Verificar que sea una cadena y tenga suficiente longitud
            if isinstance(text, str) and len(text) > 150:  # Reducido el mínimo
                results.append(text[:1000])  # Limitar longitud de fragmentos
    
    process_time = time.time() - process_start
    print(f"        Procesamiento: {len(results)} docs ({process_time:.3f}s)")
    
    return results

# ─────────────────────────────  LLM + PROMPT  ───────────────────────────────
@st.cache_resource
def get_llm(model_name: str, max_tokens: int):
    return OllamaLLM(
        model=model_name, 
        temperature=0.5,  # Reducido para respuestas más consistentes
        num_predict=max_tokens,
        stop=["RESPUESTA:", "PREGUNTA:", "CONTEXTO:"]  # Parar en estos tokens
    )

PROMPT = PromptTemplate.from_template(textwrap.dedent("""
    Responde basándote únicamente en el siguiente contexto. 
    Si no hay información suficiente, dilo brevemente.

    CONTEXTO:
    {context}

    PREGUNTA:
    {question}

    RESPUESTA:
"""))

def rag_answer(question_en: str, model: str, max_tokens: int):
    # Timer para retrieve
    retrieve_start = time.time()
    retrieved_docs = retrieve(question_en)
    retrieve_time = time.time() - retrieve_start
    print(f"    🔍 Retrieve: {len(retrieved_docs)} docs ({retrieve_time:.3f}s)")
    
    # Timer para preparar contexto
    context_start = time.time()
    ctx = "\n\n".join(retrieved_docs)
    context_time = time.time() - context_start
    print(f"    📝 Contexto: {len(ctx)} chars ({context_time:.3f}s)")
    
    # Timer para LLM
    llm_start = time.time()
    prompt = PROMPT.format(context=ctx, question=question_en)
    llm = get_llm(model, max_tokens)
    response = llm.invoke(prompt).strip()
    llm_time = time.time() - llm_start
    print(f"    🤖 LLM: {len(response)} chars ({llm_time:.3f}s)")
    
    return response, ctx

# ────────────────────────  JSON → diccionario fármacos  ─────────────────────
@st.cache_data(show_spinner=False)
def load_drug_dict():
    if not os.path.isfile(JSON_DRUGS_PATH):
        st.error("No se encontró Medicamentos.json.")
        return {}
    with open(JSON_DRUGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {d["name"].lower(): d["url"] for d in data}

drugs_startup = time.time()
DRUGS = load_drug_dict()
drugs_elapsed = time.time() - drugs_startup
print(f"Diccionario medicamentos: {len(DRUGS)} drugs ({drugs_elapsed:.3f}s)\n")

def find_drugs(text: str):
    """Función legacy - usar optimized_find_drugs en su lugar."""
    return optimized_find_drugs(text)

# ───────────────────────────  INTERFAZ  ─────────────────────────────────────

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Inicializar configuración
if "model" not in st.session_state:
    st.session_state.model = "llama3.1"
if "lang_out" not in st.session_state:
    st.session_state.lang_out = "es"

# Título principal
st.markdown('<div class="header-title">💊🧠 Asistente Farmacológico</div>', unsafe_allow_html=True)

# Panel de configuración (colapsable)
with st.expander("⚙️ Configuración", expanded=False):
    st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.model = st.selectbox("🤖 Modelo:", ["llama3.1", "llama3", "mistral"], index=0)
    with col2:
        st.session_state.lang_out = st.selectbox("🌍 Idioma de salida:", ["es", "en"], index=0)
    st.markdown('</div>', unsafe_allow_html=True)

# Función para renderizar mensajes
def render_message(role, content, timestamp=None):
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <div class="message-header">Tú</div>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <div class="message-header">Asistente Farmacológico</div>
            {content}
        </div>
        """, unsafe_allow_html=True)

# Contenedor del historial de chat
st.markdown('<div class="chat-history">', unsafe_allow_html=True)

# Mostrar mensajes del historial
for message in st.session_state.messages:
    render_message(message["role"], message["content"], message.get("timestamp"))

st.markdown('</div>', unsafe_allow_html=True)

# Contenedor de entrada
st.markdown('<div class="input-container">', unsafe_allow_html=True)

# Input para nueva pregunta
user_input = st.text_input(
    "Escribe tu pregunta sobre medicamentos...",
    key="user_input",
    placeholder="Ejemplo: ¿Qué efectos secundarios tiene el ibuprofeno?",
    label_visibility="collapsed"
)

# Columnas para botón y funciones
col1, col2, col3 = st.columns([4, 1, 1])
with col2:
    send_button = st.button("Enviar", key="send_btn")
with col3:
    clear_button = st.button("Limpiar", key="clear_btn")

st.markdown('</div>', unsafe_allow_html=True)

# Lógica de procesamiento optimizada
if send_button and user_input.strip():
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    with st.spinner("🧠 Generando respuesta…"):
        try:
            # ⏱️ TIMER INICIO
            start_time = time.time()
            print(f"\n🚀 === INICIO DEL PROCESAMIENTO ===")
            print(f"📝 Pregunta: {user_input}")
            
            # 1) Detectar idioma de entrada
            step_start = time.time()
            input_lang = detect_language(user_input)
            step_time = time.time() - step_start
            print(f"🔤 Detección de idioma: {input_lang} ({step_time:.3f}s)")
            
            # 2) Traducir solo si es necesario (para embeddings)
            step_start = time.time()
            if input_lang == 'es':
                q_en = cached_translate(user_input, "es", "en")
                step_time = time.time() - step_start
                print(f"🌐 Traducción ES→EN: {step_time:.3f}s")
            else:
                q_en = user_input
                step_time = time.time() - step_start
                print(f"🌐 Sin traducción necesaria: {step_time:.3f}s")

            # 3) Tokens dinámicos optimizados
            step_start = time.time()
            max_toks = min(MAX_TOKENS, BASE_TOKENS + len(q_en.split()) * 2)
            step_time = time.time() - step_start
            print(f"🎯 Cálculo de tokens: {max_toks} tokens ({step_time:.3f}s)")

            # 4) RAG + generación
            step_start = time.time()
            ans_en, _ = rag_answer(q_en, st.session_state.model, max_toks)
            step_time = time.time() - step_start
            print(f"🧠 RAG + LLM: {step_time:.3f}s")

            # 5) Traducir respuesta solo si es necesario
            step_start = time.time()
            if st.session_state.lang_out == 'es' and input_lang == 'en':
                answer = cached_translate(ans_en, "en", "es")
                step_time = time.time() - step_start
                print(f"🌍 Traducción respuesta EN→ES: {step_time:.3f}s")
            elif st.session_state.lang_out == 'en' and input_lang == 'es':
                answer = cached_translate(ans_en, "en", "en")  # Ya está en inglés
                step_time = time.time() - step_start
                print(f"🌍 Traducción respuesta ES→EN: {step_time:.3f}s")
            else:
                answer = ans_en
                step_time = time.time() - step_start
                print(f"🌍 Sin traducción de respuesta: {step_time:.3f}s")

            # 6) Enlaces útiles optimizados
            step_start = time.time()
            links = optimized_find_drugs(user_input + " " + answer)
            if links:
                answer += (
                    "\n\n🔗 **Enlaces útiles:**\n"
                    + "\n".join(f"- [{name}]({url})" for name, url in links)
                )
            step_time = time.time() - step_start
            print(f"🔗 Búsqueda de enlaces: {len(links)} encontrados ({step_time:.3f}s)")

            # ⏱️ TIMER TOTAL
            total_time = time.time() - start_time
            print(f"TIEMPO TOTAL: {total_time:.3f}s")
            
            # 📊 RESUMEN DE RENDIMIENTO
            print(f"   RESUMEN:")
            print(f"   • Modelo: {st.session_state.model}")
            print(f"   • Tokens máximos: {max_toks}")
            print(f"   • Idioma entrada: {input_lang}")
            print(f"   • Idioma salida: {st.session_state.lang_out}")
            print(f"   • Longitud pregunta: {len(user_input)} chars")
            print(f"   • Longitud respuesta: {len(answer)} chars")
            print(f"   • Enlaces encontrados: {len(links)}")
            print(f"   • Velocidad: {len(answer)/total_time:.1f} chars/s")
            print(f"Listo\n")

            # Agregar respuesta al historial
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"ERROR después de {error_time:.3f}s: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error al procesar la consulta: {str(e)}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Limpiar input y rerun
    st.rerun()

# Limpiar historial
if clear_button:
    st.session_state.messages = []
    st.rerun()

# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(f'<div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">{AUTORES}</div>', unsafe_allow_html=True)
