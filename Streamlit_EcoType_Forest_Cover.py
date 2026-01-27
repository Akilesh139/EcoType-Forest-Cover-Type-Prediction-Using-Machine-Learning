import streamlit as st
import pickle
import pandas as pd


#Importing pickle files
with open('target_encoder.pkl', 'rb') as f:
    target_encoder=pickle.load(f)
with open('wilderness_encoder.pkl', 'rb') as f:
    wilderness_encoder=pickle.load(f)
with open('soil_encoder.pkl', 'rb') as f:
    soil_encoder=pickle.load(f)
with open("best_model.pkl", "rb") as f:
    model=pickle.load(f)

st.title("🌲FOREST COVER PREDICTION🌲")
st.header("📝Instructions")
st.markdown("#### 🗄️Enter the values below for the Prediction")

# Values Assignment
Elevation=st.slider("Elevation",2000,3500)
Water_Elevation=st.slider("Water_Elevation",2000,3500) 
Aspect=st.number_input("Aspect")
Slope=st.number_input("Slope")
HD_To_Hydro=st.number_input("Horizontal_Distance_To_Hydrology")
VD_To_Hydro=st.number_input("Vertical_Distance_To_Hydrology")
HD_To_Roadways=st.number_input("Horizontal_Distance_To_Roadways")
HD_To_FirePoint=st.number_input("Horizontal_Distance_To_Fire_Points")
D_To_Hydrology=st.number_input("Distance_To_Hydrology")
Area_Steepness=st.number_input("Area_Steepness")
Hillshade_Avg=st.number_input("Hillshade_Avg")
Wilderness=st.selectbox("Wilderness",
                        ['Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4'])
Soil_Type=st.selectbox("Soil_Type",
                       ['Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39', 'Soil_Type_40'])

# Prediction
Submit=st.button("Submit")
if Submit:
    input_df=pd.DataFrame({ "Elevation":[Elevation],
                            "Aspect":[Aspect],
                            "Slope":[Slope],
                            "Horizontal_Distance_To_Hydrology":[HD_To_Hydro],
                            "Vertical_Distance_To_Hydrology":[VD_To_Hydro],
                            "Horizontal_Distance_To_Roadways":[HD_To_Roadways],
                            "Horizontal_Distance_To_Fire_Points":[HD_To_FirePoint],
                            "Distance_To_Hydrology":[D_To_Hydrology],
                            "Water_Elevation":[Water_Elevation],
                            "Area_Steepness":[Area_Steepness],
                            "Hillshade_Avg":[Hillshade_Avg],
                            "Wilderness": [Wilderness],
                            "Soil_Type":[Soil_Type]})

    input_df['Wilderness']=wilderness_encoder.transform(input_df['Wilderness'])
    input_df['Soil_Type']=soil_encoder.transform(input_df['Soil_Type'])

    Forest_type=model.predict(input_df)
    Org_Forest_type=target_encoder.inverse_transform(Forest_type)
    st.write("Predicted Forest Cover Type is:",Org_Forest_type)
# Feedback
st.markdown("#### Rating:")
feedback=st.feedback("stars")
if feedback is not None:
    st.markdown("##### 💐Thank You for Rating✨")