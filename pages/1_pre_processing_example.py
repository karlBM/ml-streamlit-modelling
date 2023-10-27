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
import seaborn as sns
import matplotlib.pyplot as plt


def eda():
    def get_data():
        data = {"Publishers":["AngryBirds", "Tumblr", "Guardian", "EdinburghLive", "LondonBus"], "Monday":[19284, 124654 ,2362356, 134643, 13461], 
        "Tuesday":[13464, 845686, 458, 4583, 25626], "Wednesday":[136336, 3468 ,659794, 25472, 1157], 
        "Thursday":[4574, 3568, 356869, 2562, 13436], "Friday":[13646, 574747 ,45769, 679479, 4797947], 
        "Saturday":[245722, 5724754, 2577865, 68372, 57], "Sunday":[2447257, 24576 ,769797, 6738, 794697]}
        df = pd.DataFrame(data)
        return df.set_index("Publishers")
    
    try:
        df = get_data()

        publishers = st.multiselect(
            "Choose publishers", list(df.index), ["Guardian", "EdinburghLive"]
        )
        if not publishers:
            st.error("Please select at least one country.")
        else:
            data = df.loc[publishers]
            data /= 100.0
            st.write("### Publishers Frequency", data.sort_index())

        data = data.T.reset_index()
        data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "Day", "value": "Publishers Frequency"}
            )
        chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x=alt.X("Day:N", sort=None),
                    y=alt.Y("Publishers Frequency", stack=None),
                    color="Publishers:N",
                )
            )
        st.altair_chart(chart, use_container_width=True)


        st.write("###### So now we have some understanding of what the data looks like, let's continue to get a bit smarter with how \
                 we can represent our data for modelling")

    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**
            Connection error: %s
        """
            % e.reason
        )


st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📊")
st.markdown("# Exploratory Data Analysis")
st.sidebar.header("Exploratory Data Analysis")
st.write(
    """This example is how you can look at different aspects of some data. This is some fake data generated to 
    mimic the frequency of some publishers that we can observe in our landscape over one week"""
)

eda()

show_code(eda)