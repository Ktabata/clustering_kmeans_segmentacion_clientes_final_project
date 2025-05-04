
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---

# Define la ruta al archivo del modelo
MODEL_FILE_PATH = 'src/clustering_model.pkl'

def load_model_and_scaler(file_path):
    """Carga el modelo de clustering, el scaler y los predictores desde un archivo pickle."""
    st.write(f"DEBUG: Intentando cargar archivo: {file_path}")
    st.write(f"DEBUG: Directorio de trabajo actual: {os.getcwd()}")
    if not os.path.exists(file_path):
        st.error(f"Error: No se encontró el archivo del modelo en la ruta: '{file_path}'. Asegúrate de que la ruta sea correcta.")
        return None, None, None
    try:
        with open(file_path, 'rb') as file:
            model_info = pickle.load(file)
        model = model_info.get('model')
        scaler = model_info.get('scaler')
        #scaled_features_train = model_info.get('scaled_features_train')
        #cluster_labels_train = model_info.get('cluster_labels_train')
        predictors = model_info.get('predictors')
        if model is None:
            st.error("Error: El archivo del modelo no contiene el modelo entrenado ('model').")
            return None, None, None
        if scaler is None:
            st.warning("Advertencia: El archivo del modelo no contiene el scaler ('scaler'). Las predicciones podrían ser menos precisas si los datos requieren escalado.")
            # Considera cargar un scaler por defecto o informar al usuario.
        if predictors is None:
            st.warning("Advertencia: El archivo del modelo no contiene la lista de predictores ('predictors'). Asegúrate de que el orden de las entradas del usuario coincida con el orden utilizado durante el entrenamiento.")
            # Considera definir un orden de predictores por defecto si es seguro hacerlo.
        st.success("Modelo cargado exitosamente.")
        return model, scaler, predictors
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None, None
    
def predict_cluster(data, model, scaler, predictors):
    """Preprocesa los datos y predice el clúster."""
    new_customer_data = pd.DataFrame([data], columns=predictors)  # Crear DataFrame con los datos del usuario
    scaled_new_customer_data = scaler.transform(new_customer_data)  # Escalar los datos
    cluster = model.predict(scaled_new_customer_data)[0]
    return model, scaler, predictors, #scaled_features_train, cluster_labels_train


def plot_pair_plots_with_new_customer(
    data_train_scaled, cluster_labels_train, new_customer_scaled, predictors, predicted_cluster
):
    """
    Crea una matriz de scatter plots para visualizar los clusters y el nuevo cliente.

    Args:
        data_train_scaled (np.ndarray): Datos de entrenamiento escalados.
        cluster_labels_train (np.ndarray): Etiquetas de los clusters para los datos de entrenamiento.
        new_customer_scaled (np.ndarray): Datos del nuevo cliente escalados.
        predictors (list): Lista de nombres de las características.
        predicted_cluster (int): El cluster predicho para el nuevo cliente.
    """

    df_train_plot = pd.DataFrame(data_train_scaled, columns=predictors)
    df_train_plot['cluster'] = cluster_labels_train
    new_customer_df = pd.DataFrame(new_customer_scaled, columns=predictors)
    new_customer_df['cluster'] = f'New Customer (Cluster {predicted_cluster})'

    df_plot = pd.concat([df_train_plot, new_customer_df], ignore_index=True)

    sns.pairplot(df_plot, hue='cluster', palette='viridis', diag_kind='kde')  # Puedes ajustar la paleta
    plt.suptitle('Visualización de Clusters y Nuevo Cliente (Pair Plots)', y=1.02)
    st.pyplot(plt)

# Carga el modelo, el scaler y los predictores al inicio del script
#model, scaler, predictors, scaled_features_train, cluster_labels_train = load_model_and_scaler(MODEL_FILE_PATH)
model, scaler, predictors = load_model_and_scaler(MODEL_FILE_PATH)

# Ahora puedes usar las variables 'model', 'scaler' y 'predictors' en el resto de tu script.

# Ejemplo de cómo podrías verificar si el modelo se cargó correctamente:
if model is None:
    st.stop() # Detiene la ejecución si el modelo no se cargó


# --- Streamlit App Interface ---
st.title("Customer Cluster Prediction")
st.write("Enter the customer details below to predict their cluster.")
st.write(f"*Based on the model trained in `segmentación_clientes_clustering_kmeans.ipynb`*")  # Cite the source notebook

if model and predictors:  # Only proceed if model and predictors are loaded
    st.sidebar.header("Input Customer Features")
    input_data = {}

    # Create input fields dynamically based on predictors list
    for feature in predictors:
        # Use number_input for numerical features
        input_data[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0, format="%.2f")

    # --- Prediction ---
    if st.sidebar.button("Predict Cluster"):
        try:
            # Create a DataFrame from the input data
            # The order of columns must match the order used during training/scaling
            input_df = pd.DataFrame([input_data], columns=predictors)

            # Scale and Predict
            predicted_cluster, scaled_input_data = predict_cluster(input_data, model, scaler, predictors)

            st.subheader("Input Data:")
            st.dataframe(input_df)

            st.subheader("Prediction Result:")
            st.success(f"The predicted cluster for this customer is: **Cluster {predicted_cluster}**")

            # Optional: Add interpretation of the cluster based on notebook analysis
            st.markdown("---")
            st.write("**Cluster Interpretation (Example based on typical RFM analysis):**")
            if predicted_cluster == 0:  # Adjust based on actual cluster profiles from the notebook
                st.write(
                    "Cluster 0 represent customers who are less engaged with the business. " \
                    "They have purchased recently, purchase infrequently, spend less money, "
                    "and have a lower lifetime value. " \
                    "They could be casual customers or customers at risk of churn."
                )
            elif predicted_cluster == 1:  # Adjust based on actual cluster profiles from the notebook
                st.write("Cluster 1 represents the most valuable and engaged customers. " \
                "They have purchased recently, purchase frequently, spend more money, and have a " \
                "higher lifetime value. They are the most important customers for the business.")

            # Add more elif conditions for other clusters found in the notebook analysis
            else:
                st.write(
                    f"Interpretation for Cluster {predicted_cluster} needs to be defined based on notebook analysis."
                )

            # Visualizar (If you have the training data available)
            # Assuming you have 'scaled_features_train' and 'cluster_labels_train' available
            # You'll need to modify load_model_and_scaler to load these from your pickle file
            #   or calculate them here if possible.
            st.markdown("---")
            st.title('Predicción de Clúster de Cliente')

            if scaled_features_train is not None and cluster_labels_train is not None:
                 plot_pair_plots_with_new_customer(
                     scaled_features_train, cluster_labels_train, scaled_input_data, predictors, predicted_cluster
                 )
            else:
                 st.warning("Training data not available for visualization.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure the input values are valid numbers and the model file is correct.")

elif not model:
    st.warning("Model could not be loaded. Please check the model file and path.")

st.markdown("---")
#st.write("Note: This application requires the `clustering_model.pkl` file generated by the `codigo_limpio.ipynb` notebook. Ensure this file is in the same directory as the Streamlit script or provide the correct path.")