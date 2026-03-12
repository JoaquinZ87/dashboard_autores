#!/usr/bin/env python3
"""
Dashboard Bibliográfico Interactivo — Analizador de Bibliografía UNSAM
Visualización de la composición bibliográfica de Sociología y Antropología.

Uso: python dashboard_bibliografia.py
Abre en localhost:7861
"""

import os
import warnings

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(BASE_DIR, "data", "0510_MatrizProgramas_TABLA_MAESTRA_BIBLIO.csv")
METADATOS_CSV = os.path.join(BASE_DIR, "Autores_Metadatos.csv")
OBRAS_CSV = os.path.join(BASE_DIR, "obras_año_original.csv")

# ─── Nombres largos de columnas del master ────────────────────────────────────
COL_CARRERA = "Carrera en la que se oferta la materia: Sociología (S), Antropología (A), Sociología y Antropología (SA)"
COL_ANIO_SOC = "Año en el plan de estudios de sociología (1, 2, 3, 4, 5)"
COL_ANIO_ANTRO = "Año en el plan de estudios de antropología (1, 2, 3, 4, 5)"
COL_MATERIA = "Nombre de la materia en el plan de estudios"

# ─── Épocas ───────────────────────────────────────────────────────────────────
EPOCH_BINS = [-9999, 1880, 1920, 1960, 1980, 2000, 9999]
EPOCH_LABELS = ["< 1880", "1880–1920", "1920–1960", "1960–1980", "1980–2000", "2000+"]
EPOCH_NOMBRES = [
    "Precursores", "Clásicos", "Institucionalización",
    "Crisis y renovación", "Giro cultural-reflexivo", "Contemporáneo",
]

# ─── Agrupamiento de disciplinas ──────────────────────────────────────────────
GRUPOS_DISCIPLINA = [
    ("Sociología", ["sociolog"]),
    ("Antropología", ["antropolog", "etnolog", "etnograf", "etnolin", "etnomusicolog", "folklore"]),
    ("Ciencia Política", ["ciencia polít", "ciencias polít", "relaciones internacionales"]),
    ("Historia", ["histori"]),
    ("Filosofía", ["filosof", "teolog"]),
    ("Economía", ["econom"]),
    ("Literatura", ["literatur", "crítica literar", "teoría literar", "drama"]),
    ("Lingüística y Semiótica", ["lingüíst", "semiótic", "semiolog", "análisis del discurso",
                                  "sociolingüíst", "glotopolít"]),
    ("Comunicación y Periodismo", ["periodism", "comunicaci", "medios"]),
    ("Educación", ["educaci", "pedagog", "didáctic"]),
    ("Demografía", ["demograf", "estadístic", "estudios de población"]),
    ("Psicología", ["psicolog", "psicoanálisis", "psiquiatr"]),
    ("Derecho", ["derech", "criminolog"]),
    ("Estudios de Género", ["géner", "feminism", "queer", "estudios de la mujer"]),
    ("Geografía y Urbanismo", ["geograf", "urban", "planificación"]),
]


def _mapear_grupo(texto):
    if not texto or texto == "Sin datos":
        return "Sin datos"
    t = texto.strip().lower()
    for grupo, patrones in GRUPOS_DISCIPLINA:
        if any(p in t for p in patrones):
            return grupo
    return "Otras"


# ─── Paleta UNSAM ─────────────────────────────────────────────────────────────
COLORES = {
    "azul_oscuro": "#003D7A",
    "azul_material": "#2196F3",
    "celeste": "#2DD7FF",
    "naranja": "#F89B34",
    "gris": "#F5F5F5",
    "azul_profundo": "#1A237E",
}

PALETA_EPOCAS = ["#1A237E", "#003D7A", "#1976D2", "#2196F3", "#2DD7FF", "#F89B34"]

PALETA_GENERO = {
    "masculino": "#1976D2",
    "femenino": "#F89B34",
    "desconocido": "#B0BEC5",
    "Sin datos": "#E0E0E0",
}

# ─── Layout base Plotly ──────────────────────────────────────────────────────
LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="Calibri, sans-serif", size=13, color="#333"),
    title_font=dict(size=17, color=COLORES["azul_profundo"]),
    title_x=0.5,
    margin=dict(l=20, r=20, t=55, b=20),
    hoverlabel=dict(
        bgcolor="#1A237E", font_size=13, font_color="white",
        font_family="Calibri, sans-serif", bordercolor="#2DD7FF",
    ),
    legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
    plot_bgcolor="white",
    paper_bgcolor="white",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

_cache = {}


def cargar_datos():
    if "master" in _cache:
        return _cache["master"], _cache["metadatos"], _cache["obras"]

    master = pd.read_csv(MASTER_CSV, encoding="utf-8-sig")
    metadatos = pd.read_csv(METADATOS_CSV, encoding="utf-8")
    obras = pd.read_csv(OBRAS_CSV, encoding="utf-8")

    master[COL_MATERIA] = master[COL_MATERIA].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    master = master[master["Autores"].notna() & (master["Autores"].str.strip().str.lower() != "nc")]

    _cache["master"] = master
    _cache["metadatos"] = metadatos
    _cache["obras"] = obras
    return master, metadatos, obras


def construir_tabla_expandida():
    if "expandida" in _cache:
        return _cache["expandida"]

    master, metadatos, obras = cargar_datos()

    cols_master = [
        COL_CARRERA, COL_ANIO_SOC, COL_ANIO_ANTRO, COL_MATERIA,
        "Autores", "Nombre_Publicacion", "Año_pub", "Titulo",
    ]
    df = master[cols_master].copy()

    # Join a obras_año_original ANTES de explotar autores
    df["_autores_lc"] = df["Autores"].str.strip().str.lower()
    df["_pub_lc"] = df["Nombre_Publicacion"].fillna("").str.strip().str.lower()
    obras_j = obras[["Autores", "Nombre_Publicacion", "Año_original"]].copy()
    obras_j["_autores_lc"] = obras_j["Autores"].str.strip().str.lower()
    obras_j["_pub_lc"] = obras_j["Nombre_Publicacion"].fillna("").str.strip().str.lower()
    obras_j = obras_j.drop_duplicates(subset=["_autores_lc", "_pub_lc"])

    df = df.merge(obras_j[["_autores_lc", "_pub_lc", "Año_original"]],
                  on=["_autores_lc", "_pub_lc"], how="left")
    df.drop(columns=["_autores_lc", "_pub_lc"], inplace=True)

    df["año_grafico"] = df["Año_original"].fillna(df["Año_pub"])
    df["año_grafico"] = pd.to_numeric(df["año_grafico"], errors="coerce")

    df["autor_lista"] = df["Autores"].str.split(", ")
    df = df.explode("autor_lista")
    df["autor_lc"] = df["autor_lista"].str.strip().str.lower()

    meta_j = metadatos.copy()
    meta_j["autor_lc"] = meta_j["nombre_autor"].str.strip().str.lower()
    meta_j = meta_j.drop_duplicates(subset=["autor_lc"])

    meta_cols = ["autor_lc", "genero", "region", "disciplina", "institucion"]
    if "pais_institucion" in meta_j.columns:
        meta_cols.append("pais_institucion")

    df = df.merge(meta_j[meta_cols], on="autor_lc", how="left")

    for col in ["genero", "region", "disciplina", "institucion"]:
        df[col] = df[col].fillna("Sin datos")
        df[col] = df[col].replace({"desconocido": "Sin datos", "": "Sin datos"})

    if "pais_institucion" in df.columns:
        df["pais_institucion"] = df["pais_institucion"].fillna("Sin datos")
        df["pais_institucion"] = df["pais_institucion"].replace({"desconocido": "Sin datos", "": "Sin datos"})

    df["epoca"] = pd.cut(df["año_grafico"], bins=EPOCH_BINS, labels=EPOCH_LABELS, right=False)
    df[COL_ANIO_SOC] = df[COL_ANIO_SOC].astype(str).str.strip()
    df[COL_ANIO_ANTRO] = df[COL_ANIO_ANTRO].astype(str).str.strip()

    _cache["expandida"] = df
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  FILTROS
# ═══════════════════════════════════════════════════════════════════════════════

def aplicar_filtros(df, carrera, anio_plan, materias):
    filtrado = df
    if carrera == "Sociología":
        filtrado = filtrado[filtrado[COL_CARRERA].isin(["S", "SA"])]
    elif carrera == "Antropología":
        filtrado = filtrado[filtrado[COL_CARRERA].isin(["A", "SA"])]

    if anio_plan and anio_plan != "Todos":
        col_anio = COL_ANIO_SOC if carrera == "Sociología" else COL_ANIO_ANTRO
        filtrado = filtrado[filtrado[col_anio] == anio_plan]

    if materias:
        filtrado = filtrado[filtrado[COL_MATERIA].isin(materias)]
    return filtrado


def obtener_materias(df, carrera, anio_plan):
    filtrado = df
    if carrera == "Sociología":
        filtrado = filtrado[filtrado[COL_CARRERA].isin(["S", "SA"])]
    elif carrera == "Antropología":
        filtrado = filtrado[filtrado[COL_CARRERA].isin(["A", "SA"])]

    if anio_plan and anio_plan != "Todos":
        col_anio = COL_ANIO_SOC if carrera == "Sociología" else COL_ANIO_ANTRO
        filtrado = filtrado[filtrado[col_anio] == anio_plan]

    return sorted(filtrado[COL_MATERIA].dropna().unique().tolist())


# ═══════════════════════════════════════════════════════════════════════════════
#  GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════

def _layout_compacto(layout):
    layout["height"] = layout.get("height", 300)
    layout["font"] = dict(family="Calibri, sans-serif", size=11, color="#333")
    layout["title_font"] = dict(size=14, color=COLORES["azul_profundo"])
    layout["margin"] = dict(l=15, r=15, t=45, b=15)
    return layout


def _fig_vacia(titulo="Sin datos", compacto=False):
    fig = go.Figure()
    layout = {**LAYOUT_BASE, "title": titulo}
    if compacto:
        _layout_compacto(layout)
    fig.update_layout(**layout)
    fig.add_annotation(text="Sin datos", xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=14, color="#999"))
    return fig


def _gradiente(n, r1, g1, b1, r2, g2, b2):
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        colors.append(f"rgb({int(r1+t*(r2-r1))},{int(g1+t*(g2-g1))},{int(b1+t*(b2-b1))})")
    return colors


def chart_epocas(df, compacto=False):
    if df.empty or df["epoca"].isna().all():
        return _fig_vacia("Textos por época", compacto)

    textos = df.drop_duplicates(subset=["Autores", "Nombre_Publicacion", "Titulo"])
    conteo = textos["epoca"].value_counts().reindex(EPOCH_LABELS).fillna(0).astype(int)
    total = conteo.sum()

    hover_texts = [
        f"<b>{nombre}</b> ({label})<br>Textos: {v}<br>{v/total*100:.1f}%"
        if total > 0 else f"<b>{nombre}</b> ({label})<br>Textos: 0"
        for label, nombre, v in zip(EPOCH_LABELS, EPOCH_NOMBRES, conteo.values)
    ]

    pct_texts = [f"{v}  ({v/total*100:.1f}%)" if total > 0 else f"{v}"
                  for v in conteo.values]
    fig = go.Figure(go.Bar(
        x=conteo.index, y=conteo.values,
        marker=dict(color=PALETA_EPOCAS, line=dict(width=0), cornerradius=6),
        text=pct_texts,
        textposition="outside",
        textfont=dict(size=11 if compacto else 12, color="#333"),
        hovertext=hover_texts, hoverinfo="text",
    ))

    layout = {**LAYOUT_BASE, "title": "Textos por época", "bargap": 0.25,
              "xaxis": dict(title="", tickangle=0),
              "yaxis": dict(title="Cantidad", gridcolor="#E8EAF6", gridwidth=1)}
    if compacto:
        layout["height"] = 300
        _layout_compacto(layout)
    fig.update_layout(**layout)
    return fig


def chart_genero(df, compacto=False):
    if df.empty:
        return _fig_vacia("Género del autor/a", compacto)

    textos = df.drop_duplicates(subset=["Autores", "Nombre_Publicacion", "Titulo", "autor_lc"])
    conteo = textos["genero"].value_counts()
    total = conteo.sum()

    orden = ["masculino", "femenino", "Sin datos"]
    labels = [g for g in orden if g in conteo.index]
    labels += [g for g in conteo.index if g not in labels]
    values = [conteo[g] for g in labels]
    colors = [PALETA_GENERO.get(g, "#B0BEC5") for g in labels]

    fig = go.Figure(go.Pie(
        labels=[l.capitalize() for l in labels], values=values, hole=0.45,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+percent", textfont=dict(size=12 if not compacto else 10),
        hovertemplate="<b>%{label}</b><br>Textos: %{value}<br>%{percent}<extra></extra>",
        rotation=90,
    ))

    layout = {**LAYOUT_BASE, "title": "Género del autor/a",
              "annotations": [dict(text=f"<b>{total}</b><br>textos", x=0.5, y=0.5,
                                   font_size=14 if not compacto else 12,
                                   font_color=COLORES["azul_oscuro"], showarrow=False)]}
    if compacto:
        layout["height"] = 300
        _layout_compacto(layout)
    fig.update_layout(**layout)
    return fig


def _chart_barras_h(series, titulo, compacto=False, max_items=15,
                    color_from=(45, 215, 255), color_to=(0, 61, 122)):
    conteo = series.value_counts().head(max_items)
    if conteo.empty:
        return _fig_vacia(titulo, compacto)

    total = conteo.sum()
    conteo = conteo.iloc[::-1]
    n = len(conteo)
    colors = _gradiente(n, *color_from, *color_to)

    fig = go.Figure(go.Bar(
        x=conteo.values, y=conteo.index, orientation="h",
        marker=dict(color=colors, line=dict(width=0), cornerradius=4),
        text=[f"  {v}  ({v/total*100:.1f}%)" for v in conteo.values],
        textposition="outside",
        textfont=dict(size=10 if compacto else 11, color="#555"),
        hovertemplate="<b>%{y}</b><br>Textos: %{x}<br>%{customdata:.1f}%<extra></extra>",
        customdata=[v / total * 100 if total > 0 else 0 for v in conteo.values],
    ))

    h = max(300 if compacto else 350, n * (24 if compacto else 32) + 100)
    layout = {**LAYOUT_BASE, "title": titulo, "height": h,
              "xaxis": dict(title="", showgrid=False, showticklabels=False),
              "yaxis": dict(title="")}
    if compacto:
        _layout_compacto(layout)
        layout["height"] = h
    fig.update_layout(**layout)
    return fig


def chart_region(df, compacto=False):
    if df.empty:
        return _fig_vacia("Región del autor/a", compacto)
    textos = df.drop_duplicates(subset=["Autores", "Nombre_Publicacion", "Titulo", "autor_lc"])
    return _chart_barras_h(textos["region"], "Región del autor/a", compacto)


def chart_region_instituciones(df, compacto=False):
    if df.empty or "pais_institucion" not in df.columns:
        return _fig_vacia("País de las instituciones", compacto)
    textos = df.drop_duplicates(subset=["Autores", "Nombre_Publicacion", "Titulo", "autor_lc"])
    paises = textos["pais_institucion"].str.split(",").explode().str.strip()
    paises = paises[paises != "Sin datos"]
    if paises.empty:
        return _fig_vacia("País de las instituciones", compacto)
    return _chart_barras_h(paises, "País de las instituciones", compacto,
                           color_from=(248, 155, 52), color_to=(0, 61, 122))


def chart_institucion(df, compacto=False):
    if df.empty:
        return _fig_vacia("Top 15 instituciones", compacto)
    textos = df.drop_duplicates(subset=["Autores", "Nombre_Publicacion", "Titulo", "autor_lc"])
    inst = textos["institucion"].str.split(",").explode().str.strip()
    inst = inst[inst != "Sin datos"]
    if inst.empty:
        return _fig_vacia("Top 15 instituciones", compacto)
    return _chart_barras_h(inst, "Top 15 instituciones", compacto)


def chart_disciplina(df, compacto=False):
    if df.empty:
        return _fig_vacia("Disciplina del autor/a", compacto)

    textos = df.drop_duplicates(subset=["Autores", "Nombre_Publicacion", "Titulo", "autor_lc"])
    disc = textos["disciplina"].str.split(",").explode().str.strip()
    serie = disc.apply(_mapear_grupo)

    conteo = serie.value_counts()
    sin_datos_n = conteo.get("Sin datos", 0)
    if len(conteo) > 1 and "Sin datos" in conteo.index:
        conteo = conteo.drop("Sin datos")
    if conteo.empty:
        return _fig_vacia("Disciplina del autor/a", compacto)

    total = conteo.sum() + sin_datos_n
    conteo = conteo.iloc[::-1]
    n = len(conteo)
    colors = _gradiente(n, 248, 155, 52, 0, 61, 122)

    fig = go.Figure(go.Bar(
        x=conteo.values, y=conteo.index, orientation="h",
        marker=dict(color=colors, line=dict(width=0), cornerradius=4),
        text=[f"  {v}  ({v/total*100:.1f}%)" for v in conteo.values],
        textposition="outside",
        textfont=dict(size=10 if compacto else 11, color="#555"),
        hovertemplate="<b>%{y}</b><br>Textos: %{x}<br>%{customdata:.1f}%<extra></extra>",
        customdata=[v / total * 100 if total > 0 else 0 for v in conteo.values],
    ))

    nota = f"  (+ {sin_datos_n} sin datos)" if sin_datos_n > 0 else ""
    h = max(300 if compacto else 350, n * (24 if compacto else 32) + 100)
    layout = {**LAYOUT_BASE, "title": f"Disciplina del autor/a{nota}", "height": h,
              "xaxis": dict(title="", showgrid=False, showticklabels=False),
              "yaxis": dict(title="")}
    if compacto:
        _layout_compacto(layout)
        layout["height"] = h
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _generar_resumen(df, label):
    n_textos = df.drop_duplicates(subset=["Autores", "Nombre_Publicacion", "Titulo"]).shape[0]
    n_autores = df["autor_lc"].nunique()
    n_materias = df[COL_MATERIA].nunique()
    return f"### {label}\n**{n_textos}** textos · **{n_autores}** autores · **{n_materias}** materias"


def _generar_graficos(df, compacto=False):
    return (
        chart_epocas(df, compacto),
        chart_genero(df, compacto),
        chart_region(df, compacto),
        chart_region_instituciones(df, compacto),
        chart_institucion(df, compacto),
        chart_disciplina(df, compacto),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

def callback_principal(carrera, anio_plan, materias):
    df_full = construir_tabla_expandida()
    materias_disp = obtener_materias(df_full, carrera, anio_plan)
    if materias:
        materias = [m for m in materias if m in materias_disp]

    df = aplicar_filtros(df_full, carrera, anio_plan, materias)
    label_anio = f" · Año {anio_plan}" if anio_plan and anio_plan != "Todos" else ""
    resumen = _generar_resumen(df, f"{carrera}{label_anio}")
    figs = _generar_graficos(df)
    # Gradio 5: devolver gr.Dropdown para actualizar choices
    return (*figs, resumen, gr.Dropdown(choices=materias_disp, value=materias if materias else None))


def callback_comparar(carrera_a, anio_a, materias_a, carrera_b, anio_b, materias_b):
    """Tab Comparar: genera 6 gráficos por lado + resúmenes + dropdowns."""
    df_full = construir_tabla_expandida()

    mat_disp_a = obtener_materias(df_full, carrera_a, anio_a)
    mat_disp_b = obtener_materias(df_full, carrera_b, anio_b)
    if materias_a:
        materias_a = [m for m in materias_a if m in mat_disp_a]
    if materias_b:
        materias_b = [m for m in materias_b if m in mat_disp_b]

    df_a = aplicar_filtros(df_full, carrera_a, anio_a, materias_a)
    df_b = aplicar_filtros(df_full, carrera_b, anio_b, materias_b)

    anio_suf_a = f" · Año {anio_a}" if anio_a and anio_a != "Todos" else ""
    anio_suf_b = f" · Año {anio_b}" if anio_b and anio_b != "Todos" else ""
    label_a = f"{carrera_a}{anio_suf_a}"
    label_b = f"{carrera_b}{anio_suf_b}"
    if materias_a:
        label_a = ", ".join(materias_a[:2]) + ("..." if len(materias_a) > 2 else "")
    if materias_b:
        label_b = ", ".join(materias_b[:2]) + ("..." if len(materias_b) > 2 else "")

    figs_a = _generar_graficos(df_a, compacto=True)
    figs_b = _generar_graficos(df_b, compacto=True)

    return (
        *figs_a, _generar_resumen(df_a, label_a),
        gr.Dropdown(choices=mat_disp_a, value=materias_a if materias_a else None),
        *figs_b, _generar_resumen(df_b, label_b),
        gr.Dropdown(choices=mat_disp_b, value=materias_b if materias_b else None),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERFAZ GRADIO
# ═══════════════════════════════════════════════════════════════════════════════

# Forzar light mode (quita clase 'dark' que Gradio agrega según OS preference)
JS_FORCE_LIGHT = """
() => {
    document.body.classList.remove('dark');
    document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
    const observer = new MutationObserver(() => {
        document.body.classList.remove('dark');
        document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
    });
    observer.observe(document.body, {attributes: true, attributeFilter: ['class']});
}
"""

CSS = """
/* Forzar fondo claro */
body, .gradio-container, .main, .wrap, .contain {
    background-color: #FAFBFF !important;
    color: #333 !important;
}
.dashboard-header {
    background: linear-gradient(135deg, #003D7A 0%, #1A237E 100%);
    color: white; padding: 20px 28px; border-radius: 10px; margin-bottom: 12px;
}
.dashboard-header h1 {
    margin: 0 0 4px 0; font-size: 1.5em; color: white !important;
}
.dashboard-header p {
    margin: 0; opacity: 0.85; font-size: 0.9em; color: white !important;
}
.resumen-box {
    background: #E8EAF6 !important; border-left: 4px solid #003D7A;
    padding: 10px 16px; border-radius: 8px; color: #333 !important;
}
.lado-a {
    background: #E3F2FD !important; padding: 8px 14px; border-radius: 8px;
    border-left: 3px solid #1976D2; margin-bottom: 6px; color: #333 !important;
}
.lado-b {
    background: #FFF3E0 !important; padding: 8px 14px; border-radius: 8px;
    border-left: 3px solid #F89B34; margin-bottom: 6px; color: #333 !important;
}
/* Labels y textos legibles */
label, .label-wrap, .label-text, span, p {
    color: #333 !important;
}
/* Inputs con fondo claro */
input, select, textarea, .wrap, .secondary-wrap {
    background-color: white !important;
    color: #333 !important;
    border-color: #ddd !important;
}
/* Tabs */
.tab-nav button {
    color: #333 !important;
}
.tab-nav button.selected {
    color: #003D7A !important;
    border-color: #003D7A !important;
}
"""

ANIOS_CHOICES = ["Todos", "1", "2", "3", "4", "5"]


def crear_dashboard():
    df_full = construir_tabla_expandida()
    materias_soc = obtener_materias(df_full, "Sociología", "Todos")
    materias_antro = obtener_materias(df_full, "Antropología", "Todos")

    theme = gr.themes.Soft(
        primary_hue="blue", secondary_hue="orange",
        font=["Calibri", "sans-serif"],
    )

    with gr.Blocks(theme=theme, css=CSS, js=JS_FORCE_LIGHT,
                    title="Dashboard Bibliográfico") as demo:

        gr.HTML("""
        <div class="dashboard-header">
            <h1>Dashboard Bibliográfico</h1>
            <p>Exploración interactiva de la bibliografía de Sociología y Antropología — UNSAM</p>
        </div>
        """)

        with gr.Tabs():

            # ══════════════════════ TAB DASHBOARD ══════════════════════
            with gr.Tab("Dashboard"):
                with gr.Row():
                    with gr.Column(scale=1):
                        filtro_carrera = gr.Radio(
                            choices=["Sociología", "Antropología"],
                            value="Sociología", label="Carrera",
                        )
                    with gr.Column(scale=1):
                        filtro_anio = gr.Dropdown(
                            choices=ANIOS_CHOICES, value="Todos",
                            label="Año del plan de estudios",
                        )
                    with gr.Column(scale=2):
                        filtro_materias = gr.Dropdown(
                            choices=materias_soc, value=[],
                            multiselect=True, label="Materias (vacío = todas)",
                        )

                resumen_md = gr.Markdown(elem_classes=["resumen-box"])

                with gr.Row():
                    with gr.Column(scale=3):
                        plot_epocas = gr.Plot(label="Épocas")
                    with gr.Column(scale=2):
                        plot_genero = gr.Plot(label="Género")

                with gr.Row():
                    with gr.Column():
                        plot_region = gr.Plot(label="Región autores")
                    with gr.Column():
                        plot_region_inst = gr.Plot(label="País instituciones")

                with gr.Row():
                    with gr.Column():
                        plot_institucion = gr.Plot(label="Instituciones")
                    with gr.Column():
                        plot_disciplina = gr.Plot(label="Disciplina")

                dash_outputs = [
                    plot_epocas, plot_genero, plot_region, plot_region_inst,
                    plot_institucion, plot_disciplina,
                    resumen_md, filtro_materias,
                ]
                dash_inputs = [filtro_carrera, filtro_anio, filtro_materias]

                filtro_carrera.change(callback_principal, dash_inputs, dash_outputs)
                filtro_anio.change(callback_principal, dash_inputs, dash_outputs)
                filtro_materias.change(callback_principal, dash_inputs, dash_outputs)
                demo.load(callback_principal, dash_inputs, dash_outputs)

            # ══════════════════════ TAB COMPARAR ══════════════════════
            with gr.Tab("Comparar"):
                with gr.Row():
                    # ─── LADO A ───
                    with gr.Column():
                        gr.HTML('<div class="lado-a"><b>Lado A</b></div>')
                        comp_carrera_a = gr.Radio(
                            choices=["Sociología", "Antropología"],
                            value="Sociología", label="Carrera",
                        )
                        comp_anio_a = gr.Dropdown(
                            choices=ANIOS_CHOICES, value="Todos",
                            label="Año del plan",
                        )
                        comp_materias_a = gr.Dropdown(
                            choices=materias_soc, value=[],
                            multiselect=True, label="Materias (vacío = todas)",
                        )
                        comp_resumen_a = gr.Markdown(elem_classes=["resumen-box"])
                        comp_epocas_a = gr.Plot()
                        comp_genero_a = gr.Plot()
                        comp_region_a = gr.Plot()
                        comp_region_inst_a = gr.Plot()
                        comp_inst_a = gr.Plot()
                        comp_disc_a = gr.Plot()

                    # ─── LADO B ───
                    with gr.Column():
                        gr.HTML('<div class="lado-b"><b>Lado B</b></div>')
                        comp_carrera_b = gr.Radio(
                            choices=["Sociología", "Antropología"],
                            value="Antropología", label="Carrera",
                        )
                        comp_anio_b = gr.Dropdown(
                            choices=ANIOS_CHOICES, value="Todos",
                            label="Año del plan",
                        )
                        comp_materias_b = gr.Dropdown(
                            choices=materias_antro, value=[],
                            multiselect=True, label="Materias (vacío = todas)",
                        )
                        comp_resumen_b = gr.Markdown(elem_classes=["resumen-box"])
                        comp_epocas_b = gr.Plot()
                        comp_genero_b = gr.Plot()
                        comp_region_b = gr.Plot()
                        comp_region_inst_b = gr.Plot()
                        comp_inst_b = gr.Plot()
                        comp_disc_b = gr.Plot()

                comp_inputs = [
                    comp_carrera_a, comp_anio_a, comp_materias_a,
                    comp_carrera_b, comp_anio_b, comp_materias_b,
                ]
                comp_outputs = [
                    comp_epocas_a, comp_genero_a, comp_region_a,
                    comp_region_inst_a, comp_inst_a, comp_disc_a,
                    comp_resumen_a, comp_materias_a,
                    comp_epocas_b, comp_genero_b, comp_region_b,
                    comp_region_inst_b, comp_inst_b, comp_disc_b,
                    comp_resumen_b, comp_materias_b,
                ]

                for inp in comp_inputs:
                    inp.change(callback_comparar, comp_inputs, comp_outputs)
                # Cargar gráficos iniciales al abrir
                demo.load(callback_comparar, comp_inputs, comp_outputs)

    return demo


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Cargando datos...")
    cargar_datos()
    print("Construyendo tabla expandida...")
    construir_tabla_expandida()
    print("Lanzando dashboard en http://localhost:7861")
    crear_dashboard().launch(server_port=7861, inbrowser=True)
