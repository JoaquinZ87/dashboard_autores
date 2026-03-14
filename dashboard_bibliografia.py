#!/usr/bin/env python3
"""
Dashboard Bibliográfico Interactivo — Analizador de Bibliografía UNSAM
Visualización de la composición bibliográfica de Sociología y Antropología.

Uso: python dashboard_bibliografia.py
Abre en localhost:7861
"""

import os
import re
import warnings

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import Counter

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
COL_FUND = "Fundamentación (se copia tal cómo está en el programa)"
COL_OBJ = "Objetivos (se copia tal cómo está en el programa)"
COL_EVAL = "Forma de evaluación (Parcial escrito presencial individual, Parcial escrito domiciliario individual, TP presencial individual, TP domiciliario individual, TP presencial colectivo, TP domiciliario colectivo, Final oral, Final escrito, Control continuo)"
COL_MODALIDAD = "Modalidad de aprobación"
COL_PAGINAS = "Cantidad_Paginas_Lectura"
COL_SELECCION = "Es_Seleccion"

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

PAIS_REGION = {
    'Argentina': 'Argentina',
    'Brasil': 'Latinoamérica', 'México': 'Latinoamérica', 'Chile': 'Latinoamérica',
    'Colombia': 'Latinoamérica', 'Perú': 'Latinoamérica', 'Uruguay': 'Latinoamérica',
    'Venezuela': 'Latinoamérica', 'Bolivia': 'Latinoamérica', 'Ecuador': 'Latinoamérica',
    'Costa Rica': 'Latinoamérica', 'Panamá': 'Latinoamérica', 'Paraguay': 'Latinoamérica',
    'Cuba': 'Caribe', 'Haití': 'Caribe', 'Jamaica': 'Caribe',
    'Martinica': 'Caribe', 'Puerto Rico': 'Caribe',
    'Estados Unidos': 'Anglosajón', 'Reino Unido': 'Anglosajón', 'Canadá': 'Anglosajón',
    'Australia': 'Anglosajón', 'Nueva Zelanda': 'Anglosajón', 'Inglaterra': 'Anglosajón',
    'Escocia': 'Anglosajón', 'Irlanda': 'Anglosajón',
    'Francia': 'Europa', 'Alemania': 'Europa', 'Italia': 'Europa', 'España': 'Europa',
    'Países Bajos': 'Europa', 'Bélgica': 'Europa', 'Suiza': 'Europa', 'Austria': 'Europa',
    'Portugal': 'Europa', 'Grecia': 'Europa', 'Polonia': 'Europa', 'Hungría': 'Europa',
    'Rumania': 'Europa', 'Rumanía': 'Europa', 'Bulgaria': 'Europa', 'Suecia': 'Europa',
    'Dinamarca': 'Europa', 'Noruega': 'Europa', 'Islandia': 'Europa', 'Rusia': 'Europa',
    'India': 'Asia', 'China': 'Asia', 'Japón': 'Asia', 'Israel': 'Asia',
    'Indonesia': 'Asia', 'Singapur': 'Asia',
    'Argelia': 'África', 'Sudáfrica': 'África', 'Túnez': 'África',
    'Kenia': 'África', 'Tanzania': 'África', 'Uganda': 'África',
}

# ─── Layout base Plotly ──────────────────────────────────────────────────────
LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="Calibri, sans-serif", size=13, color="#333"),
    title_font=dict(size=17, color=COLORES["azul_profundo"]),
    title_x=0.5,
    margin=dict(l=20, r=80, t=55, b=20),
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

    meta_cols = ["autor_lc", "genero", "region", "pais_nacimiento", "disciplina", "institucion"]
    if "pais_institucion" in meta_j.columns:
        meta_cols.append("pais_institucion")
    meta_cols = [c for c in meta_cols if c in meta_j.columns]

    df = df.merge(meta_j[meta_cols], on="autor_lc", how="left")

    for col in ["genero", "region", "pais_nacimiento", "disciplina", "institucion"]:
        if col in df.columns:
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
    layout["margin"] = dict(l=15, r=80, t=45, b=15)
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
        cliponaxis=False,
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
        cliponaxis=False,
        textfont=dict(size=10 if compacto else 11, color="#555"),
        hovertemplate="<b>%{y}</b><br>Textos: %{x}<br>%{customdata:.1f}%<extra></extra>",
        customdata=[v / total * 100 if total > 0 else 0 for v in conteo.values],
    ))

    max_val = conteo.values.max()
    h = max(300 if compacto else 350, n * (24 if compacto else 32) + 100)
    layout = {**LAYOUT_BASE, "title": titulo, "height": h,
              "xaxis": dict(title="", showgrid=False, showticklabels=False,
                            range=[0, max_val * 1.35]),
              "yaxis": dict(title="")}
    if compacto:
        _layout_compacto(layout)
        layout["height"] = h
    fig.update_layout(**layout)
    return fig


def _chart_barras_h_dual(series_a, series_b, titulo, label_a, label_b,
                         compacto=False, max_items=15,
                         color_from_a=(45, 215, 255), color_to_a=(0, 61, 122),
                         color_from_b=(45, 215, 255), color_to_b=(0, 61, 122)):
    """Gráfico de barras H con dos vistas alternables por botones Plotly."""
    conteo_a = series_a.value_counts().head(max_items)
    conteo_b = series_b.value_counts().head(max_items)
    if conteo_a.empty and conteo_b.empty:
        return _fig_vacia(titulo, compacto)

    fig = go.Figure()

    # ─── Vista A (visible por defecto) ───
    if not conteo_a.empty:
        total_a = conteo_a.sum()
        ca = conteo_a.iloc[::-1]
        na = len(ca)
        colors_a = _gradiente(na, *color_from_a, *color_to_a)
        fig.add_trace(go.Bar(
            x=ca.values, y=ca.index, orientation="h",
            marker=dict(color=colors_a, line=dict(width=0), cornerradius=4),
            text=[f"  {v}  ({v/total_a*100:.1f}%)" for v in ca.values],
            textposition="outside", cliponaxis=False,
            textfont=dict(size=10 if compacto else 11, color="#555"),
            hovertemplate="<b>%{y}</b><br>Textos: %{x}<br>%{customdata:.1f}%<extra></extra>",
            customdata=[v / total_a * 100 for v in ca.values],
            visible=True,
        ))
        max_a = ca.values.max()
    else:
        na = 0
        max_a = 1

    # ─── Vista B (oculta) ───
    if not conteo_b.empty:
        total_b = conteo_b.sum()
        cb = conteo_b.iloc[::-1]
        nb = len(cb)
        colors_b = _gradiente(nb, *color_from_b, *color_to_b)
        fig.add_trace(go.Bar(
            x=cb.values, y=cb.index, orientation="h",
            marker=dict(color=colors_b, line=dict(width=0), cornerradius=4),
            text=[f"  {v}  ({v/total_b*100:.1f}%)" for v in cb.values],
            textposition="outside", cliponaxis=False,
            textfont=dict(size=10 if compacto else 11, color="#555"),
            hovertemplate="<b>%{y}</b><br>Textos: %{x}<br>%{customdata:.1f}%<extra></extra>",
            customdata=[v / total_b * 100 for v in cb.values],
            visible=False,
        ))
        max_b = cb.values.max()
    else:
        nb = 0
        max_b = 1

    n_max = max(na, nb, 1)
    h = max(300 if compacto else 350, n_max * (24 if compacto else 32) + 100)

    layout = {**LAYOUT_BASE, "title": titulo, "height": h,
              "xaxis": dict(title="", showgrid=False, showticklabels=False,
                            range=[0, max_a * 1.35]),
              "yaxis": dict(title=""),
              "updatemenus": [dict(
                  type="buttons",
                  direction="right",
                  x=1.0, xanchor="right",
                  y=1.12, yanchor="top",
                  buttons=[
                      dict(label=f"  {label_a}  ",
                           method="update",
                           args=[{"visible": [True, False]},
                                 {"xaxis.range": [0, max_a * 1.35]}]),
                      dict(label=f"  {label_b}  ",
                           method="update",
                           args=[{"visible": [False, True]},
                                 {"xaxis.range": [0, max_b * 1.35]}]),
                  ],
                  bgcolor="#E8EAF6",
                  activecolor="#2196F3",
                  font=dict(size=11, color="#333"),
              )]}
    if compacto:
        _layout_compacto(layout)
        layout["height"] = h
        layout["updatemenus"] = layout.get("updatemenus")
    fig.update_layout(**layout)
    return fig


def chart_origen_autor(df, compacto=False):
    """Origen del autor: toggle Regiones / Países."""
    if df.empty:
        return _fig_vacia("Origen del autor/a", compacto)
    textos = df.drop_duplicates(subset=["Autores", "Nombre_Publicacion", "Titulo", "autor_lc"])
    regiones = textos["region"]
    paises = textos["pais_nacimiento"] if "pais_nacimiento" in textos.columns else textos["region"]
    return _chart_barras_h_dual(
        regiones, paises, "Origen del autor/a",
        "Regiones", "Países", compacto,
    )


def chart_origen_inst(df, compacto=False):
    """Origen de las instituciones: toggle Regiones / Países."""
    if df.empty or "pais_institucion" not in df.columns:
        return _fig_vacia("Origen de las instituciones", compacto)
    textos = df.drop_duplicates(subset=["Autores", "Nombre_Publicacion", "Titulo", "autor_lc"])
    paises = textos["pais_institucion"].str.split(",").explode().str.strip()
    paises = paises[paises != "Sin datos"]
    if paises.empty:
        return _fig_vacia("Origen de las instituciones", compacto)
    regiones = paises.map(PAIS_REGION).fillna("Otro")
    return _chart_barras_h_dual(
        regiones, paises, "Origen de las instituciones",
        "Regiones", "Países", compacto,
        color_from_a=(248, 155, 52), color_to_a=(0, 61, 122),
        color_from_b=(248, 155, 52), color_to_b=(0, 61, 122),
    )


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
        cliponaxis=False,
        textfont=dict(size=10 if compacto else 11, color="#555"),
        hovertemplate="<b>%{y}</b><br>Textos: %{x}<br>%{customdata:.1f}%<extra></extra>",
        customdata=[v / total * 100 if total > 0 else 0 for v in conteo.values],
    ))

    max_val = conteo.values.max()
    nota = f"  (+ {sin_datos_n} sin datos)" if sin_datos_n > 0 else ""
    h = max(300 if compacto else 350, n * (24 if compacto else 32) + 100)
    layout = {**LAYOUT_BASE, "title": f"Disciplina del autor/a{nota}", "height": h,
              "xaxis": dict(title="", showgrid=False, showticklabels=False,
                            range=[0, max_val * 1.35]),
              "yaxis": dict(title="")}
    if compacto:
        _layout_compacto(layout)
        layout["height"] = h
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DE TEXTO (Fundamentación / Objetivos)
# ═══════════════════════════════════════════════════════════════════════════════

# Stopwords español extendidas con términos académicos genéricos
_STOPWORDS_EXTRA = {
    "así", "además", "cada", "cual", "sino", "partir", "través", "dentro",
    "entre", "sobre", "desde", "hacia", "hasta", "según", "durante", "ante",
    "tras", "mediante", "respecto", "acerca", "forma", "modo", "manera",
    "puede", "pueden", "poder", "hacer", "hace", "ser", "sido", "siendo",
    "tiene", "tienen", "tener", "haber", "hay", "está", "están", "estar",
    "también", "solo", "sólo", "bien", "dos", "tres", "gran", "mayor",
    "menor", "parte", "vez", "veces", "año", "años", "cuenta",
    # Académicos genéricos
    "materia", "programa", "curso", "clase", "clases", "estudiantes",
    "alumno", "alumnos", "alumna", "alumnas", "docente", "cátedra",
    "bibliografía", "texto", "textos", "lectura", "lecturas", "unidad",
    "trabajo", "trabajos", "tema", "temas", "contenido", "contenidos",
    "propone", "propuesta", "perspectiva", "perspectivas", "enfoque",
    "abordaje", "permite", "permiten", "busca", "objetivo", "objetivos",
    "general", "generales", "específico", "específicos", "particular",
    "diferentes", "diferente", "distintos", "distintas", "diversos",
    "diversas", "principales", "principal", "nuevo", "nueva", "nuevos",
    "nuevas", "propio", "propia", "propios", "mismo", "misma",
}


def _get_stopwords():
    """Carga stopwords de NLTK + extras."""
    try:
        from nltk.corpus import stopwords
        sw = set(stopwords.words("spanish"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        sw = set(stopwords.words("spanish"))
    return sw | _STOPWORDS_EXTRA


def _get_stemmer():
    from nltk.stem import SnowballStemmer
    return SnowballStemmer("spanish")


def _tokenizar(texto, stopwords_set, stemmer):
    """Tokeniza, limpia, quita stopwords y aplica stemming."""
    texto = re.sub(r"[^\w\sáéíóúüñ]", " ", texto.lower())
    texto = re.sub(r"\d+", " ", texto)
    tokens = texto.split()
    tokens = [t for t in tokens if len(t) > 2 and t not in stopwords_set]
    return [stemmer.stem(t) for t in tokens]


def _tokens_a_palabras(textos_originales, stopwords_set, stemmer):
    """Devuelve Counter de stems y un mapeo stem→palabra más frecuente."""
    stem_counter = Counter()
    stem_to_word = {}  # stem → {palabra: count}
    for texto in textos_originales:
        if not isinstance(texto, str):
            continue
        texto_limpio = re.sub(r"[^\w\sáéíóúüñ]", " ", texto.lower())
        texto_limpio = re.sub(r"\d+", " ", texto_limpio)
        tokens = texto_limpio.split()
        tokens = [t for t in tokens if len(t) > 2 and t not in stopwords_set]
        for t in tokens:
            stem = stemmer.stem(t)
            stem_counter[stem] += 1
            if stem not in stem_to_word:
                stem_to_word[stem] = Counter()
            stem_to_word[stem][t] += 1
    # Mapear cada stem a la palabra original más común
    best_word = {stem: wc.most_common(1)[0][0] for stem, wc in stem_to_word.items()}
    return stem_counter, best_word


def _obtener_textos_campo(df_master, carrera, anio_plan, campo_col, materias_sel=None):
    """Filtra materias únicas y devuelve los textos del campo indicado."""
    filtrado = df_master.copy()
    if carrera == "Sociología":
        filtrado = filtrado[filtrado[COL_CARRERA].isin(["S", "SA"])]
    elif carrera == "Antropología":
        filtrado = filtrado[filtrado[COL_CARRERA].isin(["A", "SA"])]
    if anio_plan and anio_plan != "Todos":
        col_anio = COL_ANIO_SOC if carrera == "Sociología" else COL_ANIO_ANTRO
        filtrado = filtrado[filtrado[col_anio].astype(str).str.strip() == anio_plan]
    if materias_sel:
        filtrado = filtrado[filtrado[COL_MATERIA].isin(materias_sel)]
    # Una sola entrada por materia (los campos son por programa, no por texto)
    materias = filtrado.drop_duplicates(subset=[COL_MATERIA])
    textos = materias[campo_col].dropna().tolist()
    return textos


def _obtener_materias_master(df_master, carrera, anio_plan):
    """Obtiene lista de materias del master (para filtrar Fund/Obj)."""
    filtrado = df_master.copy()
    if carrera == "Sociología":
        filtrado = filtrado[filtrado[COL_CARRERA].isin(["S", "SA"])]
    elif carrera == "Antropología":
        filtrado = filtrado[filtrado[COL_CARRERA].isin(["A", "SA"])]
    if anio_plan and anio_plan != "Todos":
        col_anio = COL_ANIO_SOC if carrera == "Sociología" else COL_ANIO_ANTRO
        filtrado = filtrado[filtrado[col_anio].astype(str).str.strip() == anio_plan]
    return sorted(filtrado[COL_MATERIA].dropna().unique().tolist())


def chart_palabras_frecuentes(textos, titulo, top_n=25):
    """Gráfico de barras horizontales con las palabras más frecuentes."""
    if not textos:
        return _fig_vacia(titulo)
    sw = _get_stopwords()
    stemmer = _get_stemmer()
    stem_counter, best_word = _tokens_a_palabras(textos, sw, stemmer)
    if not stem_counter:
        return _fig_vacia(titulo)

    top = stem_counter.most_common(top_n)
    palabras = [best_word[stem] for stem, _ in top]
    counts = [c for _, c in top]
    total = sum(counts)

    # Invertir para que la más frecuente quede arriba
    palabras = palabras[::-1]
    counts = counts[::-1]
    n = len(palabras)
    colors = _gradiente(n, 45, 215, 255, 0, 61, 122)

    fig = go.Figure(go.Bar(
        x=counts, y=palabras, orientation="h",
        marker=dict(color=colors, line=dict(width=0), cornerradius=4),
        text=[f"  {v}  ({v/total*100:.1f}%)" for v in counts],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=11, color="#555"),
        hovertemplate="<b>%{y}</b><br>Frecuencia: %{x}<br>%{customdata:.1f}%<extra></extra>",
        customdata=[v / total * 100 for v in counts],
    ))
    max_val = max(counts)
    h = max(400, n * 28 + 100)
    layout = {**LAYOUT_BASE, "title": titulo, "height": h,
              "xaxis": dict(title="", showgrid=False, showticklabels=False,
                            range=[0, max_val * 1.35]),
              "yaxis": dict(title="")}
    fig.update_layout(**layout)
    return fig


def _extraer_topics(textos_fund, textos_obj, n_topics=5, n_words=7):
    """Extrae temas con NMF + TF-IDF. Devuelve lista de dicts con palabras y pesos, o None."""
    textos = []
    for t in (textos_fund or []) + (textos_obj or []):
        if isinstance(t, str) and len(t.strip()) > 20:
            textos.append(t)
    if len(textos) < 3:
        return None

    sw = _get_stopwords()
    stemmer = _get_stemmer()
    docs_tokens = [_tokenizar(t, sw, stemmer) for t in textos]
    docs_tokens = [d for d in docs_tokens if len(d) > 3]
    if len(docs_tokens) < 3:
        return None

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    docs_str = [" ".join(d) for d in docs_tokens]
    vectorizer = TfidfVectorizer(max_features=500, max_df=0.85, min_df=2)
    try:
        tfidf = vectorizer.fit_transform(docs_str)
    except ValueError:
        return None

    feature_names = vectorizer.get_feature_names_out()
    _, best_word = _tokens_a_palabras(textos, sw, stemmer)

    n_topics = min(n_topics, max(2, len(docs_tokens) // 3))
    model = NMF(n_components=n_topics, random_state=42, max_iter=300)
    model.fit(tfidf)

    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_words:]
        words = [best_word.get(feature_names[i], feature_names[i]) for i in top_indices]
        weights = [topic[i] for i in top_indices]
        # Auto-label: top 3 palabras más relevantes
        top3 = [best_word.get(feature_names[j], feature_names[j])
                for j in topic.argsort()[-3:][::-1]]
        auto_label = ", ".join(top3)
        topics.append({"words": words, "weights": weights, "auto_label": auto_label})
    return topics


def chart_topics(topics_data, labels=None):
    """Genera el gráfico de topics con etiquetas opcionales."""
    if not topics_data:
        return _fig_vacia("Topic Modeling")

    n_topics = len(topics_data)
    topic_colors = ["#003D7A", "#1976D2", "#2DD7FF", "#F89B34", "#1A237E",
                    "#4CAF50", "#E91E63", "#9C27B0"]

    # Determinar título de cada subplot
    subtitles = []
    for i, td in enumerate(topics_data):
        if labels and i < len(labels) and labels[i].strip():
            subtitles.append(f"Tema {i+1}: {labels[i].strip()}")
        else:
            subtitles.append(f"Tema {i+1}: {td['auto_label']}")

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=n_topics, cols=1,
        subplot_titles=subtitles,
        vertical_spacing=0.08,
    )

    for topic_idx, td in enumerate(topics_data):
        color = topic_colors[topic_idx % len(topic_colors)]
        fig.add_trace(go.Bar(
            y=td["words"], x=td["weights"], orientation="h",
            marker=dict(color=color, line=dict(width=0), cornerradius=3),
            text=[f"  {w:.2f}" for w in td["weights"]],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=10, color="#555"),
            hovertemplate="<b>%{y}</b><br>Peso: %{x:.3f}<extra></extra>",
            showlegend=False,
        ), row=topic_idx + 1, col=1)
        max_w = max(td["weights"]) if td["weights"] else 1
        fig.update_xaxes(showticklabels=False, showgrid=False,
                         range=[0, max_w * 1.35], row=topic_idx + 1, col=1)
        fig.update_yaxes(tickfont=dict(size=11), row=topic_idx + 1, col=1)

    h = n_topics * 180 + 80
    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("margin",)},
        title=f"Topic Modeling — {n_topics} temas (Fundamentación + Objetivos)",
        height=h,
        margin=dict(l=20, r=80, t=60, b=20),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color=COLORES["azul_oscuro"])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  GRÁFICOS — PROGRAMAS (Evaluación, Páginas, Selección)
# ═══════════════════════════════════════════════════════════════════════════════

def _filtrar_master(df_master, carrera, anio_plan, materias=None):
    """Filtra el master por carrera, año y materias."""
    f = df_master.copy()
    if carrera == "Sociología":
        f = f[f[COL_CARRERA].isin(["S", "SA"])]
    elif carrera == "Antropología":
        f = f[f[COL_CARRERA].isin(["A", "SA"])]
    if anio_plan and anio_plan != "Todos":
        col_anio = COL_ANIO_SOC if carrera == "Sociología" else COL_ANIO_ANTRO
        f = f[f[col_anio].astype(str).str.strip() == anio_plan]
    if materias:
        f = f[f[COL_MATERIA].isin(materias)]
    return f


def chart_modalidad(df_master, carrera, anio_plan, materias=None):
    """Donut de modalidades de aprobación."""
    f = _filtrar_master(df_master, carrera, anio_plan, materias)
    # Una vez por materia
    materias_unicas = f.drop_duplicates(subset=[COL_MATERIA])
    conteo = materias_unicas[COL_MODALIDAD].value_counts()
    if conteo.empty:
        return _fig_vacia("Modalidad de aprobación")

    total = conteo.sum()
    colors_mod = {"Promocionable": "#2196F3", "Final obligatorio": "#F89B34",
                  "No se informa": "#B0BEC5"}
    labels = conteo.index.tolist()
    values = conteo.values.tolist()
    colors = [colors_mod.get(l, "#E0E0E0") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.45,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+percent", textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>Materias: %{value}<br>%{percent}<extra></extra>",
        rotation=90,
    ))
    layout = {**LAYOUT_BASE, "title": "Modalidad de aprobación",
              "annotations": [dict(text=f"<b>{total}</b><br>materias", x=0.5, y=0.5,
                                   font_size=14, font_color=COLORES["azul_oscuro"],
                                   showarrow=False)]}
    fig.update_layout(**layout)
    return fig


def _clasificar_eval_eje(forma, eje):
    """Clasifica una forma de evaluación en un eje: 'formato', 'lugar', 'modalidad'."""
    f = forma.lower().strip()
    if f in ("nc", ""):
        return None
    if eje == "formato":  # Oral vs Escrito
        if any(k in f for k in ("oral", "exposición", "exposicion",
                                 "coloquio", "presentaciones orales")):
            return "Oral"
        if "escrito" in f:
            return "Escrito"
        return "S/D"
    elif eje == "lugar":  # Presencial vs Domiciliario
        if "presencial" in f:
            return "Presencial"
        if "domiciliario" in f:
            return "Domiciliario"
        return "S/D"
    elif eje == "modalidad":  # Individual vs Grupal
        if "individual" in f or "indivual" in f:  # incluye typo
            return "Individual"
        if any(k in f for k in ("grupal", "colectivo", "grupo")):
            return "Grupal"
        return "S/D"
    return "S/D"


def _chart_eval_eje(df_master, carrera, anio_plan, eje, titulo, materias=None):
    """Donut de formas de evaluación clasificadas por un eje."""
    f = _filtrar_master(df_master, carrera, anio_plan, materias)
    materias_unicas = f.drop_duplicates(subset=[COL_MATERIA])
    formas = materias_unicas[COL_EVAL].dropna()
    if formas.empty:
        return _fig_vacia(titulo)

    exploded = formas.str.split(",").explode().str.strip()
    clasificado = exploded.map(lambda x: _clasificar_eval_eje(x, eje)).dropna()
    if clasificado.empty:
        return _fig_vacia(titulo)

    conteo = clasificado.value_counts()
    total = conteo.sum()

    # Colores fijos por categoría
    colores_map = {
        "Oral": "#F89B34", "Escrito": "#1976D2",
        "Presencial": "#2196F3", "Domiciliario": "#F89B34",
        "Individual": "#003D7A", "Grupal": "#2DD7FF",
        "S/D": "#B0BEC5",
    }
    cols = [colores_map.get(c, "#B0BEC5") for c in conteo.index]

    fig = go.Figure(go.Pie(
        labels=conteo.index, values=conteo.values,
        hole=0.5,
        marker=dict(colors=cols),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>%{value} menciones (%{percent})<extra></extra>",
    ))
    layout = {**LAYOUT_BASE, "title": f"{titulo} ({total} menciones)", "height": 380,
              "showlegend": True,
              "legend": dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5)}
    fig.update_layout(**layout)
    return fig


def chart_paginas(df_master, carrera, anio_plan, materias=None):
    """Indicadores agregados de páginas de lectura."""
    f = _filtrar_master(df_master, carrera, anio_plan, materias)
    pags = f[[COL_MATERIA, COL_PAGINAS]].copy()
    pags[COL_PAGINAS] = pags[COL_PAGINAS].astype(str).str.strip().str.lower()
    pags = pags[~pags[COL_PAGINAS].isin(["nc", "nan", ""])]
    pags[COL_PAGINAS] = pd.to_numeric(pags[COL_PAGINAS], errors="coerce")
    pags = pags.dropna(subset=[COL_PAGINAS])
    if pags.empty:
        return _fig_vacia("Páginas de lectura")

    total = int(pags[COL_PAGINAS].sum())
    n_textos = len(pags)
    promedio_texto = total / n_textos if n_textos else 0
    por_materia = pags.groupby(COL_MATERIA)[COL_PAGINAS].sum()
    n_mat = len(por_materia)
    promedio_materia = total / n_mat if n_mat else 0

    fig = go.Figure()
    labels = ["Total páginas", "Promedio\npor materia", "Promedio\npor texto"]
    values = [total, promedio_materia, promedio_texto]
    colors = ["#003D7A", "#2196F3", "#2DD7FF"]

    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(width=0), cornerradius=6),
        text=[f"{int(v):,}" if v >= 10 else f"{v:.1f}" for v in values],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=14, color="#333"),
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>",
    ))
    layout = {**LAYOUT_BASE, "title": f"Páginas de lectura ({n_textos} textos, {n_mat} materias)",
              "height": 380,
              "yaxis": dict(title="", showgrid=True, gridcolor="#E8EAF6"),
              "xaxis": dict(title="")}
    fig.update_layout(**layout)
    return fig


def chart_seleccion(df_master, carrera, anio_plan, materias=None):
    """Donut: proporción selecciones vs textos completos (agregado)."""
    f = _filtrar_master(df_master, carrera, anio_plan, materias)
    if f.empty or COL_SELECCION not in f.columns:
        return _fig_vacia("Selecciones vs. textos completos")

    sel = f[COL_SELECCION].astype(str).str.strip().str.lower()
    n_sel = (sel == "sí").sum()
    n_comp = (sel == "no").sum()
    n_nc = len(sel) - n_sel - n_comp

    labels_vals = []
    if n_comp:
        labels_vals.append(("Texto completo", n_comp, "#1976D2"))
    if n_sel:
        labels_vals.append(("Selección", n_sel, "#F89B34"))
    if n_nc:
        labels_vals.append(("Sin dato", n_nc, "#B0BEC5"))

    if not labels_vals:
        return _fig_vacia("Selecciones vs. textos completos")

    labs, vals, cols = zip(*labels_vals)
    total = sum(vals)

    fig = go.Figure(go.Pie(
        labels=labs, values=vals,
        hole=0.5,
        marker=dict(colors=cols),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>%{value} textos (%{percent})<extra></extra>",
    ))
    layout = {**LAYOUT_BASE,
              "title": f"Selecciones vs. textos completos ({total} textos)",
              "height": 400,
              "showlegend": True,
              "legend": dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5)}
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
        chart_origen_autor(df, compacto),
        chart_origen_inst(df, compacto),
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


def callback_texto(carrera, anio_plan, materias):
    """Tab Análisis de Texto: palabras frecuentes + topic modeling."""
    master, _, _ = cargar_datos()
    materias_disp = _obtener_materias_master(master, carrera, anio_plan)
    if materias:
        materias = [m for m in materias if m in materias_disp]
    textos_fund = _obtener_textos_campo(master, carrera, anio_plan, COL_FUND, materias)
    textos_obj = _obtener_textos_campo(master, carrera, anio_plan, COL_OBJ, materias)
    label = carrera
    if anio_plan and anio_plan != "Todos":
        label += f" · Año {anio_plan}"
    if materias:
        label = ", ".join(materias[:3]) + ("..." if len(materias) > 3 else "")
    n_materias = len(textos_fund) or len(textos_obj)
    resumen = f"### {label}\nAnalizando textos de **{n_materias}** materias"
    fig_fund = chart_palabras_frecuentes(textos_fund, "Palabras más frecuentes — Fundamentación")
    fig_obj = chart_palabras_frecuentes(textos_obj, "Palabras más frecuentes — Objetivos")

    topics_data = _extraer_topics(textos_fund, textos_obj)
    fig_topics = chart_topics(topics_data)

    # Tabla de etiquetas editable
    if topics_data:
        labels_df = pd.DataFrame({
            "Tema": [f"Tema {i+1}" for i in range(len(topics_data))],
            "Etiqueta": [td["auto_label"] for td in topics_data],
        })
    else:
        labels_df = pd.DataFrame({"Tema": [], "Etiqueta": []})

    # Guardar topics_data en cache para re-render al editar etiquetas
    _cache["last_topics"] = topics_data

    return (resumen, fig_fund, fig_obj, fig_topics, labels_df,
            gr.Dropdown(choices=materias_disp, value=materias if materias else None))


def callback_editar_labels(labels_df):
    """Re-renderiza el chart de topics con las etiquetas editadas."""
    topics_data = _cache.get("last_topics")
    if topics_data is None or labels_df is None or labels_df.empty:
        return chart_topics(topics_data)
    labels = labels_df["Etiqueta"].tolist() if "Etiqueta" in labels_df.columns else []
    return chart_topics(topics_data, labels=labels)


def callback_programas(carrera, anio_plan, materias):
    """Tab Programas: modalidad, 3 ejes evaluación, páginas, selecciones."""
    master, _, _ = cargar_datos()
    materias_disp = _obtener_materias_master(master, carrera, anio_plan)
    if materias:
        materias = [m for m in materias if m in materias_disp]

    fig_mod = chart_modalidad(master, carrera, anio_plan, materias)
    fig_formato = _chart_eval_eje(master, carrera, anio_plan, "formato", "Oral vs. Escrito", materias)
    fig_lugar = _chart_eval_eje(master, carrera, anio_plan, "lugar", "Presencial vs. Domiciliario", materias)
    fig_modalidad = _chart_eval_eje(master, carrera, anio_plan, "modalidad", "Individual vs. Grupal", materias)
    fig_pag = chart_paginas(master, carrera, anio_plan, materias)
    fig_sel = chart_seleccion(master, carrera, anio_plan, materias)

    label = carrera
    if anio_plan and anio_plan != "Todos":
        label += f" · Año {anio_plan}"
    if materias:
        label = ", ".join(materias[:3]) + ("..." if len(materias) > 3 else "")
    df_f = _filtrar_master(master, carrera, anio_plan, materias)
    n_mat = df_f[COL_MATERIA].nunique() if COL_MATERIA in df_f.columns else 0
    resumen = f"### {label}\n**{n_mat}** materias"

    return (fig_mod, fig_formato, fig_lugar, fig_modalidad, fig_pag, fig_sel, resumen,
            gr.Dropdown(choices=materias_disp, value=materias if materias else None))


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
            with gr.Tab("Información bibliográfica"):
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
                        plot_origen_autor = gr.Plot(label="Origen autores")
                    with gr.Column():
                        plot_origen_inst = gr.Plot(label="Origen instituciones")

                with gr.Row():
                    with gr.Column():
                        plot_institucion = gr.Plot(label="Instituciones")
                    with gr.Column():
                        plot_disciplina = gr.Plot(label="Disciplina")

                dash_outputs = [
                    plot_epocas, plot_genero, plot_origen_autor, plot_origen_inst,
                    plot_institucion, plot_disciplina,
                    resumen_md, filtro_materias,
                ]
                dash_inputs = [filtro_carrera, filtro_anio, filtro_materias]

                filtro_carrera.change(callback_principal, dash_inputs, dash_outputs)
                filtro_anio.change(callback_principal, dash_inputs, dash_outputs)
                filtro_materias.change(callback_principal, dash_inputs, dash_outputs)
                demo.load(callback_principal, dash_inputs, dash_outputs)

            # ══════════════════════ TAB COMPARAR ══════════════════════
            with gr.Tab("Comparador"):
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
                        comp_origen_autor_a = gr.Plot()
                        comp_origen_inst_a = gr.Plot()
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
                        comp_origen_autor_b = gr.Plot()
                        comp_origen_inst_b = gr.Plot()
                        comp_inst_b = gr.Plot()
                        comp_disc_b = gr.Plot()

                comp_inputs = [
                    comp_carrera_a, comp_anio_a, comp_materias_a,
                    comp_carrera_b, comp_anio_b, comp_materias_b,
                ]
                comp_outputs = [
                    comp_epocas_a, comp_genero_a, comp_origen_autor_a,
                    comp_origen_inst_a, comp_inst_a, comp_disc_a,
                    comp_resumen_a, comp_materias_a,
                    comp_epocas_b, comp_genero_b, comp_origen_autor_b,
                    comp_origen_inst_b, comp_inst_b, comp_disc_b,
                    comp_resumen_b, comp_materias_b,
                ]

                for inp in comp_inputs:
                    inp.change(callback_comparar, comp_inputs, comp_outputs)
                # Cargar gráficos iniciales al abrir
                demo.load(callback_comparar, comp_inputs, comp_outputs)

            # ══════════════════════ TAB ANÁLISIS DE TEXTO ══════════════════════
            with gr.Tab("Análisis textual de programas"):
                gr.Markdown("Análisis de las secciones **Fundamentación** y **Objetivos** "
                            "de los programas de las materias.")
                with gr.Row():
                    with gr.Column(scale=1):
                        txt_carrera = gr.Radio(
                            choices=["Sociología", "Antropología"],
                            value="Sociología", label="Carrera",
                        )
                    with gr.Column(scale=1):
                        txt_anio = gr.Dropdown(
                            choices=ANIOS_CHOICES, value="Todos",
                            label="Año del plan",
                        )
                    with gr.Column(scale=2):
                        txt_materias = gr.Dropdown(
                            choices=materias_soc, value=[],
                            multiselect=True, label="Materias (vacío = todas)",
                        )

                txt_resumen = gr.Markdown(elem_classes=["resumen-box"])

                with gr.Row():
                    with gr.Column():
                        plot_palabras_fund = gr.Plot(label="Fundamentación")
                    with gr.Column():
                        plot_palabras_obj = gr.Plot(label="Objetivos")

                with gr.Row():
                    with gr.Column(scale=3):
                        plot_topics = gr.Plot(label="Topic Modeling")
                    with gr.Column(scale=1):
                        gr.Markdown("**Etiquetas de temas** *(editá y presioná Enter)*")
                        topic_labels = gr.Dataframe(
                            headers=["Tema", "Etiqueta"],
                            col_count=(2, "fixed"),
                            interactive=True,
                            wrap=True,
                        )

                txt_inputs = [txt_carrera, txt_anio, txt_materias]
                txt_outputs = [txt_resumen, plot_palabras_fund, plot_palabras_obj,
                               plot_topics, topic_labels, txt_materias]
                txt_carrera.change(callback_texto, txt_inputs, txt_outputs)
                txt_anio.change(callback_texto, txt_inputs, txt_outputs)
                txt_materias.change(callback_texto, txt_inputs, txt_outputs)
                demo.load(callback_texto, txt_inputs, txt_outputs)
                # Re-render chart when labels are edited
                topic_labels.change(callback_editar_labels, [topic_labels], [plot_topics])

            # ══════════════════════ TAB PROGRAMAS ══════════════════════
            with gr.Tab("Análisis estructural de programas"):
                gr.Markdown("Análisis de **modalidades de aprobación**, **formas de evaluación**, "
                            "**páginas de lectura** y **selecciones vs. textos completos**.")
                with gr.Row():
                    with gr.Column(scale=1):
                        prog_carrera = gr.Radio(
                            choices=["Sociología", "Antropología"],
                            value="Sociología", label="Carrera",
                        )
                    with gr.Column(scale=1):
                        prog_anio = gr.Dropdown(
                            choices=ANIOS_CHOICES, value="Todos",
                            label="Año del plan",
                        )
                    with gr.Column(scale=2):
                        prog_materias = gr.Dropdown(
                            choices=materias_soc, value=[],
                            multiselect=True, label="Materias (vacío = todas)",
                        )

                prog_resumen = gr.Markdown(elem_classes=["resumen-box"])

                with gr.Row():
                    with gr.Column():
                        plot_modalidad = gr.Plot(label="Modalidad de aprobación")

                gr.Markdown("#### Formas de evaluación")
                with gr.Row():
                    with gr.Column():
                        plot_eval_formato = gr.Plot(label="Oral vs. Escrito")
                    with gr.Column():
                        plot_eval_lugar = gr.Plot(label="Presencial vs. Domiciliario")
                    with gr.Column():
                        plot_eval_modalidad = gr.Plot(label="Individual vs. Grupal")

                with gr.Row():
                    with gr.Column():
                        plot_paginas = gr.Plot(label="Páginas de lectura")
                    with gr.Column():
                        plot_seleccion = gr.Plot(label="Selecciones vs. textos completos")

                prog_inputs = [prog_carrera, prog_anio, prog_materias]
                prog_outputs = [plot_modalidad, plot_eval_formato, plot_eval_lugar,
                                plot_eval_modalidad, plot_paginas, plot_seleccion,
                                prog_resumen, prog_materias]
                prog_carrera.change(callback_programas, prog_inputs, prog_outputs)
                prog_anio.change(callback_programas, prog_inputs, prog_outputs)
                prog_materias.change(callback_programas, prog_inputs, prog_outputs)
                demo.load(callback_programas, prog_inputs, prog_outputs)

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
