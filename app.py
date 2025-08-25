import streamlit as st
from langchain_groq import ChatGroq # Changed import
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configuración de la página
st.set_page_config(page_title="Agente de Economia", page_icon="🏦")
st.title("🏦 Agente de Economia con LangChain y Groq") # Updated title
st.markdown("Pregunta sobre cultivos, fertilización, enfermedades de plantas y buenas prácticas agrícolas.")

# Token de Groq (configurado en Streamlit Secrets)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] # Changed secret key name

# Inicializar el modelo usando ChatGroq
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192",  # Un modelo rápido y potente disponible en Groq
    temperature=0.5
)

# Plantilla de prompt
template = """
Eres un Agente de Economía y Finanzas cuya misión es analizar tendencias económicas, explicar conceptos financieros de forma clara con ejemplos prácticos y simular escenarios de mercado considerando variables como tasas de interés, políticas fiscales, precios de materias primas o choques externos. Debes responder con un lenguaje profesional y accesible, adaptando la profundidad al nivel del usuario, organizando la información en secciones cuando sea útil (contexto, variables clave, escenarios, implicaciones) y fundamentando siempre tus explicaciones en principios económicos sólidos. Responde la siguiente pregunta de forma clara y breve.
Pregunta: {pregunta}
Respuesta:
"""
prompt = PromptTemplate(input_variables=["pregunta"], template=template)
chain = LLMChain(llm=llm, prompt=prompt)

# Entrada del usuario
query = st.text_input("Escribe tu pregunta:")

# Botón para generar respuesta
if st.button("Obtener respuesta"):
    if query.strip() == "":
        st.warning("Por favor escribe una pregunta.")
    else:
        try:
            respuesta = chain.run(pregunta=query)
            st.success(respuesta)
        except Exception as e:
            st.error(f"Error al generar la respuesta: {e}")

# Ejemplos de preguntas para probar
st.markdown("**Ejemplos de preguntas:**")
st.markdown("""
- ¿Cuáles son las mejores prácticas para fertilizar maíz?
- ¿Cómo detectar y controlar la roya en el café?
- ¿Qué cultivos son recomendables en suelos ácidos?
- ¿Cuándo es el mejor momento para sembrar trigo en clima templado?
- ¿Qué enfermedades afectan al tomate y cómo prevenirlas?
""")
