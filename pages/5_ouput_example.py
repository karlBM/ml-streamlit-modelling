# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from urllib.error import URLError

import pandas as pd
import pydeck as pdk

import streamlit as st
from streamlit.hello.utils import show_code


def mapping_rules():
    @st.cache_data
    def get_dataset_data():
        # load dataset
        df = pd.read_csv("/workspaces/ml-streamlit-modelling/datasets/karl_mccabe_grey_matter_31032023.csv", delimiter=',' )

        return df

    try:
        ALL_LAYERS = {
            "studnets": pdk.Layer(
                "HexagonLayer",
                data=get_dataset_data(),
                get_position=["longitude", "latitude"],
                radius=2000,
                elevation_scale=10,
                elevation_range=[0, 1000],
                extruded=True,
            ),
        }
        st.sidebar.markdown("### Map Layers")
        selected_layers = [
            layer
            for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)
        ]
        if selected_layers:
            st.pydeck_chart(
                pdk.Deck(
                    map_style=None,
                    initial_view_state={
                        "latitude": 51.5,
                        "longitude": 0.1276,
                        "zoom": 5,
                        "pitch": 50,
                    },
                    layers=selected_layers,
                )
            )
        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error(
            """
            **This example requires internet access.**
            Connection error: %s
        """
            % e.reason
        )


st.set_page_config(page_title="Mapping Rules Example", page_icon="🌍")
st.markdown("# Output Example")
st.sidebar.header("Output Example")
st.write(
    """This example is to showcase what you have 
    now created as you have introduced some 
    aspects of: GEO, TIME and FEATURE into an ML model."""
)

mapping_rules()

show_code(mapping_rules)
