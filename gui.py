import argparse
from pathlib import Path

import streamlit as st

from app.utils.parse import yaml_load
from menu import menu


parser = argparse.ArgumentParser(description="Visualize the analysis")
parser.add_argument("--folder", help="path of the analyzed outputs", type=str)
parser.add_argument("--subset", help="partition of the dataset: train, val or test", type=str)
parser.add_argument("--yaml_path", help="the parameters of the dataset", type=str)
args = parser.parse_args()

st.set_page_config(
    page_title="Identify Box Error Types App",
    page_icon="🧐",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("<h1 style='text-align: center;'>Visualize the analysis</h1>", unsafe_allow_html=True)
description = 'Please select the kind of analysis you would like to see on the sidebar.'
st.markdown(f"<h4 style='text-align: center'>{description}</h4>", unsafe_allow_html=True)

# Initialize st.session_state variables
if "analysis_folder_path" not in st.session_state:
    st.session_state['analysis_folder_path'] = Path(args.folder)
if "data_subset" not in st.session_state:
    st.session_state['data_subset'] = args.subset
if "class_names" not in st.session_state:
    st.session_state['class_names'] = yaml_load(args.yaml_path).get('names')
if "target_classes" not in st.session_state:
    st.session_state['target_classes'] = []


menu()
