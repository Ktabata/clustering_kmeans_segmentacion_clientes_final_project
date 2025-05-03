import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
#import altair as alt
#import plotly.express as px


with open('clustering_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Segmentación')

# 2. Definir la función de predicción
def predict_customer_cluster(data):
    """
    Realiza la predicción del cluster de un cliente a partir de los datos ingresados.

    Args:
        data (dict): Un diccionario que contiene los datos del cliente, incluyendo
                      'Recency', 'Frequency', 'Monetary', 'CLTV' y el gasto en
                      diferentes categorías de productos.

    Returns:
        int: El cluster predicho para el cliente.
    """
    # 3. Convertir el diccionario a DataFrame
    df = pd.DataFrame([data])

    # 4. Preprocesar los datos
    df_processed = preprocess_data(df)  # Usa tu función de preprocesamiento

    # 5. Realizar la predicción
    cluster = predict_cluster(model, df_processed)  # Usa tu función de predicción

    return cluster

def main():
    st.title("Predicción de Cluster de Clientes")

    # Entradas del usuario
    st.sidebar.header("Ingrese los datos del cliente:")
    recency = st.sidebar.number_input("Recency (días desde la última compra):", min_value=0, value=30)
    frequency = st.sidebar.number_input("Frequency (número de compras):", min_value=0, value=10)
    monetary = st.sidebar.number_input("Monetary (gasto total):", min_value=0.0, value=100.0)
    cltv = st.sidebar.number_input("CLTV (Customer Lifetime Value):", min_value=0.0, value=500.0)
    # Gasto por categoría de producto (puedes ajustar esto según tus categorías)
    spending_other = st.sidebar.number_input("Gasto en Otros:", min_value=0.0, value=0.0)
    spending_christmas_seasonal_decor = st.sidebar.number_input("Gasto en Decoración Navideña:", min_value=0.0, value=0.0)
    spending_decorations_and_ornaments = st.sidebar.number_input("Gasto en Decoraciones y Ornamentos:", min_value=0.0, value=0.0)
    spending_decorative_collectibles = st.sidebar.number_input("Gasto en Coleccionables Decorativos:", min_value=0.0, value=0.0)
    spending_discount_and_fees = st.sidebar.number_input("Gasto en Descuentos y Tarifas:", min_value=0.0, value=0.0)
    spending_fashion_accessories = st.sidebar.number_input("Gasto en Accesorios de Moda:", min_value=0.0, value=0.0)
    spending_games_and_amusements = st.sidebar.number_input("Gasto en Juegos y Entretenimiento:", min_value=0.0, value=0.0)
    spending_garden_accessories = st.sidebar.number_input("Gasto en Accesorios de Jardín:", min_value=0.0, value=0.0)
    spending_gifts_and_novelties = st.sidebar.number_input("Gasto en Regalos y Novedades:", min_value=0.0, value=0.0)
    spending_home_decor = st.sidebar.number_input("Gasto en Decoración del Hogar:", min_value=0.0, value=0.0)
    spending_kitchen_and_dining = st.sidebar.number_input("Gasto en Cocina y Comedor:", min_value=0.0, value=0.0)
    spending_money_banks = st.sidebar.number_input("Gasto en Huchas:", min_value=0.0, value=0.0)
    spending_office_and_stationery = st.sidebar.number_input("Gasto en Oficina y Papelería:", min_value=0.0, value=0.0)
    spending_party_and_celebration = st.sidebar.number_input("Gasto en Artículos de Fiesta:", min_value=0.0, value=0.0)
    spending_personal_care = st.sidebar.number_input("Gasto en Cuidado Personal:", min_value=0.0, value=0.0)
    spending_storage_and_organization = st.sidebar.number_input("Gasto en Almacenamiento y Organización:", min_value=0.0, value=0.0)
    spending_textiles_and_warmth = st.sidebar.number_input("Gasto en Textiles:", min_value=0.0, value=0.0)
    spending_tools_and_hardware = st.sidebar.number_input("Gasto en Herramientas:", min_value=0.0, value=0.0)
    spending_toys_and_childrens_items = st.sidebar.number_input("Gasto en Juguetes:", min_value=0.0, value=0.0)

       # Botón de predicción
    if st.button("Predecir Cluster"):
        # 7. Preparar los datos para la predicción
        customer_data = {
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary,
            'CLTV': cltv,
            'Spending_Other': spending_other,
            'Spending_christmas_seasonal_decor': spending_christmas_seasonal_decor,
            'Spending_decorations_and_ornaments': spending_decorations_and_ornaments,
            'Spending_decorative_collectibles': spending_decorative_collectibles,
            'Spending_discount_and_fees': spending_discount_and_fees,
            'Spending_fashion_accessories': spending_fashion_accessories,
            'Spending_games_and_amusements': spending_games_and_amusements,
            'Spending_garden_accessories': spending_garden_accessories,
            'Spending_gifts_and_novelties': spending_gifts_and_novelties,
            'Spending_home_decor': spending_home_decor,
            'Spending_kitchen_and_dining': spending_kitchen_and_dining,
            'Spending_money_banks': spending_money_banks,
            'Spending_office_and_stationery': spending_office_and_stationery,
            'Spending_party_and_celebration': spending_party_and_celebration,
            'Spending_personal_care': spending_personal_care,
            'Spending_storage_and_organization': spending_storage_and_organization,
            'Spending_textiles_and_warmth': spending_textiles_and_warmth,
            'Spending_tools_and_hardware': spending_tools_and_hardware,
            'Spending_toys_and_childrens_items': spending_toys_and_childrens_items,
        }

        # 8. Obtener la predicción
        cluster = predict_customer_cluster(customer_data)

        # 9. Mostrar el resultado
        st.subheader("Cluster Predicho:")
        if cluster != -1:
            st.write(f"El cliente pertenece al Cluster {cluster}")
            st.write("Los clusters van del 0 al 4, donde 0 es el cluster de clientes con mayor valor y 4 el de menor valor")
        else:
            st.write("Error al predecir el cluster. Por favor, revise los datos ingresados e intente nuevamente.")


if __name__ == '__main__':
    main()