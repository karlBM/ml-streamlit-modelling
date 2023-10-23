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

import altair as alt
import pandas as pd

import streamlit as st
from streamlit.hello.utils import show_code


def pre_proc_example():

    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    # load dataset
    pre_proc_dataset = pd.read_csv("datasets/diabetes.csv", header=1, names=col_names)
    pre_proc_dataset
    #split dataset in features and target variable
    feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']

    @st.cache_data
    def get_dataset_data():
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        # load dataset
        pre_proc_dataset = pd.read_csv("datasets/diabetes.csv", header=1, names=col_names)
        return pre_proc_dataset.reset_index()

    try:
        df = get_dataset_data()
        features = st.multiselect(
            "What are your favorite colors",
        options=['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'], # convert to list

        )
        if not features:
            st.error("Please select at least one feature.")
        else:
            data = df.loc[features]
            st.write("### Diabetes Dataset", data.sort_index())

            data = data.T.reset_index()
            data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
            )
            chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x="year:T",
                    y=alt.Y("Diabetes age vs. bp ($B):Q", stack=None),
                    color="Region:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error(
            """
            **This example requires internet access.**
            Connection error: %s
        """
            % e.reason
        )


st.set_page_config(page_title="Pre-Processing Example", page_icon="📊")
st.markdown("# Pre-Processing Example")
st.sidebar.header("Pre-Processing Example")
st.write(
    """This example shows you how some type of  ml pre_processing works. So if a DS nerd says 
    "this is a pre-processed dataset" now you know what we are talking about !"""
)

pre_proc_example()

show_code(pre_proc_example)
