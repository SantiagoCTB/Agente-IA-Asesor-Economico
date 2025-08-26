# app.py
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# -----------------------------
# Configuración general
# -----------------------------
st.set_page_config(page_title="Agente de Economía + EDA", page_icon="🏦", layout="wide")

# =========================================================
# PARTE 1 — Tu agente LLM (tal cual lo tenías)
# =========================================================
st.title("🏦 Agente de Economía con LangChain y Groq")
st.markdown("Pregunta tendencias económicas, conceptos financieros o escenarios de mercado.")

# Token de Groq (Streamlit secrets)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Inicializar modelo para Parte 1
llm_parte1 = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192",
    temperature=0.5
)

# Prompt ORIGINAL (tal cual)
template_parte1 = """
Eres un Agente de Economía y Finanzas cuya misión es analizar tendencias económicas, explicar conceptos financieros de forma clara con ejemplos prácticos y simular escenarios de mercado considerando variables como tasas de interés, políticas fiscales, precios de materias primas o choques externos. Debes responder con un lenguaje profesional y accesible, adaptando la profundidad al nivel del usuario, organizando la información en secciones cuando sea útil (contexto, variables clave, escenarios, implicaciones) y fundamentando siempre tus explicaciones en principios económicos sólidos. Responde la siguiente pregunta de forma clara y breve.
Pregunta: {pregunta}
Respuesta:
"""
prompt_parte1 = PromptTemplate(input_variables=["pregunta"], template=template_parte1)
chain_parte1 = LLMChain(llm=llm_parte1, prompt=prompt_parte1)

# UI Parte 1
query = st.text_input("Escribe tu pregunta:")
if st.button("Obtener respuesta", key="btn_p1"):
    if query.strip() == "":
        st.warning("Por favor escribe una pregunta.")
    else:
        try:
            respuesta = chain_parte1.run(pregunta=query)
            st.success(respuesta)
        except Exception as e:
            st.error(f"Error al generar la respuesta: {e}")

st.markdown("**Ejemplos de preguntas:**")
st.markdown("""
- ¿Qué impacto tendría un aumento de la tasa de interés en el consumo y la inversión?
- ¿Cómo afecta la inflación al poder adquisitivo de los hogares?
- ¿Cuáles son las diferencias entre política fiscal y política monetaria?
- ¿Qué escenarios podrían darse si el precio del petróleo cae un 20%?
- ¿Cómo funciona el PIB y por qué es un indicador clave en la economía?
""")

st.markdown("---")

# =========================================================
# PARTE 2 — EDA de CSV + Experto LLM contextualizado
# =========================================================
st.header("📈 EDA de un CSV y asesor LLM contextualizado")

st.markdown("Sube un **CSV** (p. ej. AAPL) para hacer EDA. Después podrás **hablar con un experto** que usará como contexto el **resumen del EDA**.")

# ---------- utilidades ----------
def parse_dates_auto(df: pd.DataFrame):
    candidates = ["Date","date","Fecha","fecha","Datetime","datetime","Time","time"]
    for c in candidates:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="raise")
                return df, c
            except Exception:
                pass
    return df, None

def detect_numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def compute_basic_metrics(df: pd.DataFrame, price_col="Adj Close"):
    out = {}
    if price_col not in df.columns:
        if "Close" in df.columns:
            price_col = "Close"
        else:
            return {"note": "No se encontró columna de precio ('Adj Close' o 'Close')."}
    series = df[price_col].dropna()
    if series.empty:
        return {"note": f"Columna {price_col} vacía tras dropna()."}

    out["first_price"] = float(series.iloc[0])
    out["last_price"]  = float(series.iloc[-1])
    out["change_pct"]  = (out["last_price"]/out["first_price"] - 1.0) * 100.0

    returns = series.pct_change().dropna()
    if not returns.empty:
        out["mean_daily_return_pct"] = float(returns.mean()*100)
        out["vol_annual_pct"] = float(returns.std()*np.sqrt(252)*100)
        cummax = series.cummax()
        drawdown = (series/cummax) - 1.0
        out["max_drawdown_pct"] = float(drawdown.min()*100)
    return out

def make_eda_summary_text(df: pd.DataFrame, date_col: str | None):
    cols = list(df.columns)
    n_rows, n_cols = df.shape
    nulls = df.isna().sum().sort_values(ascending=False)
    top_nulls = nulls[nulls > 0].head(5).to_dict()

    price_col = "Adj Close" if "Adj Close" in cols else ("Close" if "Close" in cols else None)
    metrics = compute_basic_metrics(df, price_col=price_col if price_col else "Adj Close")

    # rango temporal
    time_range = "No detectado"
    if date_col:
        dmin = pd.to_datetime(df[date_col], errors="coerce").min()
        dmax = pd.to_datetime(df[date_col], errors="coerce").max()
        if pd.notna(dmin) and pd.notna(dmax):
            time_range = f"{dmin.date()} a {dmax.date()}"

    lines = [
        f"Filas/Columnas: {n_rows}/{n_cols}",
        f"Columnas: {', '.join(cols)}",
        f"Rango temporal: {time_range}",
    ]
    if top_nulls:
        lines.append(f"Nulos (top5): {json.dumps(top_nulls, ensure_ascii=False)}")
    if price_col and metrics:
        if "first_price" in metrics and "last_price" in metrics:
            lines.append(
                f"Precio {price_col}: inicial={metrics['first_price']:.2f}, "
                f"final={metrics['last_price']:.2f}, variación={metrics.get('change_pct', float('nan')):.2f}%"
            )
        if "mean_daily_return_pct" in metrics:
            lines.append(f"Retorno diario medio≈{metrics['mean_daily_return_pct']:.3f}%")
        if "vol_annual_pct" in metrics:
            lines.append(f"Volatilidad anual≈{metrics['vol_annual_pct']:.2f}%")
        if "max_drawdown_pct" in metrics:
            lines.append(f"Máximo drawdown≈{metrics['max_drawdown_pct']:.2f}%")
    return "\n".join(lines)

# ---------- carga de CSV ----------
uploaded = st.file_uploader("📤 Sube tu archivo CSV", type=["csv"], key="csv_p2")
if uploaded:
    try:
        raw = uploaded.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        st.error(f"No se pudo leer el CSV: {e}")
        st.stop()

    df, date_col = parse_dates_auto(df)
    if date_col:
        df = df.sort_values(by=date_col).reset_index(drop=True)

    st.success("CSV cargado correctamente.")
    st.write("Vista previa:")
    st.dataframe(df.head(15), use_container_width=True)

    # ---------- EDA ----------
    st.subheader("🔎 EDA")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Filas", f"{df.shape[0]:,}")
    with c2:
        st.metric("Columnas", f"{df.shape[1]:,}")
    with c3:
        st.metric("Celdas nulas", f"{int(df.isna().sum().sum()):,}")

    with st.expander("📋 Estadísticas descriptivas"):
        # Seleccionamos solo columnas numéricas
        num_df = df.select_dtypes(include=[np.number])
        st.dataframe(num_df.describe(), use_container_width=True)  # Solo numéricas

    with st.expander("🧹 Nulos por columna"):
        nulls = df.isna().sum().sort_values(ascending=False)
        st.dataframe(nulls.to_frame("nulos"), use_container_width=True)
        st.plotly_chart(px.bar(nulls[nulls > 0], title="Nulos por columna"),
                        use_container_width=True)

    num_cols = detect_numeric_columns(df)
    if len(num_cols) >= 2:
        with st.expander("📈 Matriz de correlación"):
            corr = df[num_cols].corr(numeric_only=True)
            st.plotly_chart(px.imshow(corr, aspect="auto", title="Correlación"),
                            use_container_width=True)

    # Si es dataset financiero típico
    if date_col and ("Adj Close" in df.columns or "Close" in df.columns):
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        st.subheader("📉 Precio vs tiempo")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df[date_col], y=df[price_col], mode="lines", name=price_col))
        fig_price.update_layout(title=f"{price_col} vs tiempo", xaxis_title="Fecha", yaxis_title="Precio")
        st.plotly_chart(fig_price, use_container_width=True)

        # Retornos
        if "Return" in df.columns:
            returns = df["Return"]
        else:
            returns = df[price_col].pct_change()

        with st.expander("📊 Histograma de retornos diarios"):
            st.plotly_chart(px.histogram(returns.dropna(), nbins=50, title="Retornos diarios"),
                            use_container_width=True)

        with st.expander("📦 Boxplots (precio/volumen/retorno)"):
            cols_for_box = [c for c in [price_col, "Volume", "Return"] if c in df.columns]
            if cols_for_box:
                # Crear tantas columnas como variables a graficar
                col_objs = st.columns(len(cols_for_box))
                for i, col in enumerate(cols_for_box):
                    with col_objs[i]:
                        st.plotly_chart(
                            px.box(df, y=col, title=f"Boxplot de {col}"),
                            use_container_width=True
                        )
            else:
                st.info("No hay columnas típicas para boxplot.")

        roll_cols = [c for c in ["RollingMean20", "RollingVol20"] if c in df.columns]
        if roll_cols:
            st.subheader("📐 Indicadores móviles (del CSV)")
            for c in roll_cols:
                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(x=df[date_col], y=df[c], mode="lines", name=c))
                fig_roll.update_layout(title=c, xaxis_title="Fecha", yaxis_title=c)
                st.plotly_chart(fig_roll, use_container_width=True)
    else:
        # Caso general
        with st.expander("📈 Línea (elige X e Y)"):
            x_col = st.selectbox("Eje X", options=df.columns, index=0, key="x_line")
            num_cols2 = detect_numeric_columns(df)
            y_col = st.selectbox("Eje Y (numérica)", options=num_cols2, index=0 if num_cols2 else None, key="y_line")
            if x_col and y_col:
                st.plotly_chart(px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}"),
                                use_container_width=True)

        with st.expander("📊 Histogramas rápidos"):
            num_to_plot = st.multiselect("Columnas numéricas", options=num_cols, default=num_cols[:3])
            for col in num_to_plot:
                st.plotly_chart(px.histogram(df[col].dropna(), nbins=40, title=f"Histograma - {col}"),
                                use_container_width=True)

    # Resumen EDA -> contexto del LLM
    eda_summary = make_eda_summary_text(df, date_col=date_col)
    with st.expander("🧾 Resumen EDA que se usará como contexto"):
        st.code(eda_summary, language="markdown")

    # -----------------------------
    # 3) Agente LLM (experto contextualizado)
    # -----------------------------
    st.subheader("💬 ¿Quieres hablar con un experto sobre estos datos?")

    # Inicializar modelo para Parte 2
    llm_parte2 = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0.3
    )

    # Prompt contextualizado
    system_template = """Eres un asesor financiero que responde de forma clara, concisa y basada en datos.
    Usa el siguiente contexto (resumen del EDA) para personalizar tus respuestas.
    Si la pregunta excede el contexto, apóyate en principios financieros generales, indicándolo explícitamente.

    === CONTEXTO EDA ===
    {eda_summary}
    ====================
    Responde en español y, cuando sea útil, organiza en: contexto, análisis, implicaciones, riesgos y escenarios.
    """
    user_template = "Pregunta: {pregunta}"

    prompt_parte2 = PromptTemplate(
        input_variables=["eda_summary", "pregunta"],
        template=system_template + "\n" + user_template
    )
    chain_parte2 = LLMChain(llm=llm_parte2, prompt=prompt_parte2)

    # Entrada directa del usuario (igual a tu Parte 1)
    query2 = st.text_input(
        "Escribe tu pregunta al experto:",
        value="¿Cómo interpretarías la tendencia y la volatilidad en este período?"
    )
    if st.button("Obtener respuesta del experto"):
        if query2.strip() == "":
            st.warning("Por favor escribe una pregunta.")
        else:
            try:
                respuesta2 = chain_parte2.run(eda_summary=eda_summary, pregunta=query2)
                st.success(respuesta2)
            except Exception as e:
                st.error(f"Error al generar la respuesta: {e}")

else:
    st.info("Sube un CSV para ejecutar el análisis.")
