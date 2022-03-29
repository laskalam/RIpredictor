import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import streamlit as st
from base64 import b64encode
from datetime import datetime, date, time
import io
from functions import xception, vgg, image_prediction_2
from footer import footer
from netCDF4 import Dataset
from combine_images import convert_image

DEFAULT_IMAGE_NAME = 'image_file.nc'
st.set_page_config(layout='wide')

def save_image(x, file_name) -> None:

    """
    Save image under file_name. The image can be displayed subsequently.
    """
    sizes = np.shape(x)
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(x, vmin=-1, vmax=1, cmap='binary')
    plt.savefig(file_name, dpi=400)

    return

def image_with_colorbar(x, file_name, ax=None,
                        min_x=175, max_x=325, 
                        name='Temperature [K]',
                        cmap='viridis', norm=None):
    """given a MxN array, plot a 2d map with a colorbar"""
    if ax==None:
        fig, ax = plt.subplots(1,1, figsize=(5,5))
    if norm==None:
        vmin, vmax = min_x, max_x
    else:
        vmin, vmax = None, None
    im = ax.imshow(x,
                   norm=norm,
                   vmin=vmin,
                   vmax=vmax,
                   origin='lower',
                   cmap=cmap)
    #Colorbar
    div = make_axes_locatable(ax)
    cax = div.append_axes("bottom", size="5%", pad=0.05)
    cb  = plt.colorbar(im,cax=cax, ticks=[], orientation='horizontal')
    cb.ax.text(0, -0.15, f'{min_x:.0f}',
               transform=cb.ax.transAxes,
               rotation=0,
               va='top', ha='left')
    cb.ax.text(1, -0.15, f'{max_x:.0f}',
               transform=cb.ax.transAxes,
               rotation=0,
               va='top', ha='right')
    cb.ax.text(0.5, -0.15, f'{name}',
               transform=cb.ax.transAxes,
               rotation=0,
               va='top', ha='center')
    ax.set_axis_off()
    plt.savefig(file_name, dpi=400, bbox_inches='tight')


@st.cache(ttl=86400)
def getMap(*args, **argvs):
    return np.full((args[0], args[0]), 250)


@st.cache(ttl=86400)
def getXception(*args, **argvs):
    return xception()


@st.cache(ttl=86400)
def getVgg():
    return vgg()


def vgg_predict(image):
    net = getVgg()
    image = image[np.newaxis, :, np.newaxis]
    output = net.predict(image)
    return output[0]


#@st.cache(ttl=86400)
def getImage(path, variable_name='IRWIN'):
    if path==None:
        return  getMap(100)
    try:
        with Dataset(path, 'r') as d:
            image = np.squeeze(d[variable_name][:])
    except:
        error = """<p style='font-family:sans-serif; color:Red; font-size: 24px;'>
        Please make sure your chosen file has a netCDF format with the appropriate variable name. Variable name is IRWIN by default but please change according to the variable name of the IR temperature in your file.
        </p>"""
        st.markdown(error, unsafe_allow_html=True)
        st.stop()

    return image


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Add title
st.markdown("<h1 style='text-align: center; color:grey;'>Tropical cyclone rapid intensification (RI) predictor</h1>", unsafe_allow_html=True)

footer()

with st.form(key='form0'):



    st.markdown("This tool can be used to predict if there will be a rapid intensification of the tropical cyclone or not in the next 24h. \n"
                "The input required for the prediction is an IR temperature 2D map. The input file should be in a netcdf format with \n"
                "the variable name **IRWIN**. The input shape can be any of the following formats: MxN, 1xMxN or MxNx1 where M and N are \n"
                "preferably close. If M and N are very different, the processed image will be distorted. The map should include all of \n"
                r"the tropical cyclone IR temperature feature, about $20\times20~\mathrm{deg}^2$ around the eye was used during the training of the models."
                "  \n We present two predictions from two different DNN models."
                )
    st.write("You can download a data sample [here](https://www.dropbox.com/sh/h33g391kk8xalbd/AABefWFa13mfvHuMkDekYqbHa?dl=1). **unzip** to extract the data.")
    a0, a1 = st.columns((8, 2))
    uploaded_file = a0.file_uploader("Choose a file")
    variable_name = a1.text_input("Variable Name (case sensitive)", value="IRWIN")
    if variable_name.strip()=='' or variable_name==None:
        variable_name='IRWIN'
    _ , t, _ = st.columns((4,1,4))
    submit_button = t.form_submit_button(label='Predict')
    
    #make a nice progress bar
    long_task_progress = st.empty()

    c0, c1, c2 = st.columns((2, 2, 2))

    if submit_button:
        try:
            os.remove('x.png')
        except:
            pass
        if uploaded_file is not None:
            with open(DEFAULT_IMAGE_NAME,"wb") as f:
                f.write(uploaded_file.getbuffer())
            image = getImage(DEFAULT_IMAGE_NAME, variable_name=variable_name)
        else:
            image = getImage(None)

        c0.header("Original data")
        image_with_colorbar(image, 'orig.png')
        c0.image('orig.png', use_column_width=True)

        with c1:
            _ = long_task_progress.progress(0)
            # get the model and preprocess the input
            _xception, preprocess_input, input_size, _ = getXception()
            _vgg = getVgg()
            shaped_image = convert_image(image)
            X = preprocess_input(shaped_image)
            _ = long_task_progress.progress(2)
            X = np.expand_dims(np.average(X, axis=-1), axis=-1)
            X = np.expand_dims(X, axis=0)
            _ = long_task_progress.progress(30)
            # get RI probabilities
            xception_prediction = _xception.predict(X)[0][0]
            vgg_prediction = _vgg.predict(shaped_image[np.newaxis, :, :, :1])[0][1]
            _ = long_task_progress.progress(90)
            
            #plot the processed image
            _map = np.squeeze(X)
            image_with_colorbar(_map, 'x.png', min_x=-1, max_x=1, name='Scaled', cmap='binary')

            _ = long_task_progress.progress(100)
            c1.header("Processed map")
            c1.image('x.png', use_column_width=True)
            os.remove('x.png')
            c2.header("24h-prediction")
            image_prediction_2([xception_prediction, vgg_prediction], 'x.png')
            #image_prediction(probability, 'x.png')
            c2.image('x.png', use_column_width=True)
            os.remove('x.png')
st.write("If our work inspires you to advance in your research, please cite our preprint paper [here](https://eartharxiv.org/repository/view/2195/).")
st.write("If you are interested in our source code, please find them [here](https://github.com/laskalam/RIpredictor).")
