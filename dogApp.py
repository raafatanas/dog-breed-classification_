import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications import NASNetLarge

# List of dog breeds
dog_breeds = [
    "affenpinscher", "afghan_hound", "african_hunting_dog", "airedale", "american_staffordshire_terrier", 
    "appenzeller", "australian_terrier", "basenji", "basset", "beagle", "bedlington_terrier", 
    "bernese_mountain_dog", "black-and-tan_coonhound", "blenheim_spaniel", "bloodhound", "bluetick", 
    "border_collie", "border_terrier", "borzoi", "boston_bull", "bouvier_des_flandres", "boxer", 
    "brabancon_griffon", "briard", "brittany_spaniel", "bull_mastiff", "cairn", "cardigan", 
    "chesapeake_bay_retriever", "chihuahua", "chow", "clumber", "cocker_spaniel", "collie", 
    "curly-coated_retriever", "dandie_dinmont", "dhole", "dingo", "doberman", "english_foxhound", 
    "english_setter", "english_springer", "entlebucher", "eskimo_dog", "flat-coated_retriever", 
    "french_bulldog", "german_shepherd", "german_short-haired_pointer", "giant_schnauzer", 
    "golden_retriever", "gordon_setter", "great_dane", "great_pyrenees", "greater_swiss_mountain_dog", 
    "groenendael", "ibizan_hound", "irish_setter", "irish_terrier", "irish_water_spaniel", 
    "irish_wolfhound", "italian_greyhound", "japanese_spaniel", "keeshond", "kelpie", 
    "kerry_blue_terrier", "komondor", "kuvasz", "labrador_retriever", "lakeland_terrier", 
    "leonberg", "lhasa", "malamute", "malinois", "maltese_dog", "mexican_hairless", 
    "miniature_pinscher", "miniature_poodle", "miniature_schnauzer", "newfoundland", "norfolk_terrier", 
    "norwegian_elkhound", "norwich_terrier", "old_english_sheepdog", "otterhound", "papillon", 
    "pekinese", "pembroke", "pomeranian", "pug", "redbone", "rhodesian_ridgeback", "rottweiler", 
    "saint_bernard", "saluki", "samoyed", "schipperke", "scotch_terrier", "scottish_deerhound", 
    "sealyham_terrier", "shetland_sheepdog", "shih-tzu", "siberian_husky", "silky_terrier", 
    "soft-coated_wheaten_terrier", "staffordshire_bullterrier", "standard_poodle", "standard_schnauzer", 
    "sussex_spaniel", "tibetan_mastiff", "tibetan_terrier", "toy_poodle", "toy_terrier", "vizsla", 
    "walker_hound", "weimaraner", "welsh_springer_spaniel", "west_highland_white_terrier", "whippet", 
    "wire-haired_fox_terrier", "yorkshire_terrier"
]

# Organize dog breeds into a DataFrame starting from index 1
dog_breeds_df = pd.DataFrame(dog_breeds, columns=["Dog Breed"]).reset_index(drop=True)
dog_breeds_df.index += 1

def disable(b):
    st.session_state["disabled"] = b

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_2.keras")
    return model

def read_image(file, size):
    image = np.array(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

def main():
    st.title("Dog Breed Classification")
    st.write("Upload an image of a dog, and the model will predict its breed.")
    
    st.sidebar.header('App Details')
    st.sidebar.write("Please upload a photo of one of the following 120 dog breeds.")
    st.sidebar.table(dog_breeds_df)

    model = load_model()
    
    labels_path = "labels.csv"
    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    id2breed = {i: name for i, name in enumerate(breed)}

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        size = 331
        image = read_image(uploaded_file, size)
        image_expanded = np.expand_dims(image, axis=0)
        
        pred = model.predict(image_expanded)[0]
        label_idx = np.argmax(pred)
        breed_name = id2breed[label_idx]

        image = (image * 255).astype(np.uint8)
        image = cv2.putText(image, breed_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Center the image on the page
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            st.write("")
        with col2:
            st.image(image, channels="BGR", caption=f"Predicted Breed: {breed_name}", width=400)
        with col3:
            st.write("")

if __name__ == "__main__":
    main()
