#!/usr/bin/env python3
"""Entry point para HuggingFace Spaces."""

from dashboard_bibliografia import cargar_datos, construir_tabla_expandida, crear_dashboard

print("Cargando datos...")
cargar_datos()
print("Construyendo tabla expandida...")
construir_tabla_expandida()
print("Lanzando dashboard...")
demo = crear_dashboard()
demo.launch()
