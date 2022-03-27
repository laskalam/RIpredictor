import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import streamlit as st
from base64 import b64encode
from datetime import datetime, date, time
import io
from functions import (xception, vgg, tensor, float32,
                       F, image_prediction, image_prediction_2)
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
    image = np.repeat(image[np.newaxis,np.newaxis,:], 2, axis=0)
    image = tensor(image, dtype=float32)
    output = F.softmax(net(image), dim=-1)
    return float(output[0][1])


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



    st.markdown("This tool can be used to predict if there will be a rapid intensification of the tropical cyclone in the next 24h or not. \n"
                "The input required for the prediction is an IR temperature 2D map. The input file should be in a netcdf format with \n"
                "the variable name **IRWIN**. The input shape can be of the following format: MxN, 1xMxN or MxNx1 where M and N are \n"
                "preferably close. If M and N are very different, the processed image will be distorted. The map should include all of \n"
                "the tropical cyclone IR temperature feature, about 20x20deg^2 was used during the training of the models. \n"
                "We present 2 predictions from 2 DNN different models."
                )
    a0, a1 = st.columns((8, 2))
    uploaded_file = a0.file_uploader("Choose a file")
    variable_name = a1.text_input("Variable Name (case sensitive)", value="IRWIN")
    if variable_name.strip()=='' or variable_name==None:
        variable_name='IRWIN'
    _ , t, _ = st.columns((4,1,4))
    submit_button = t.form_submit_button(label='submit')
    
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
            model, preprocess_input, input_size, _ = getXception()
            X = convert_image(image)
            X = preprocess_input(X)
            _ = long_task_progress.progress(2)
            X = np.expand_dims(np.average(X, axis=-1), axis=-1)
            X = np.expand_dims(X, axis=0)
            _ = long_task_progress.progress(30)
            prediction = model.predict(X)
            image_ = convert_image(image)[:,:,0]
            vgg_prediction = vgg_predict(image_)
            _ = long_task_progress.progress(90)
            
            #plot the processed image
            _map = np.squeeze(X)
            image_with_colorbar(_map, 'x.png', min_x=-1, max_x=1, name='Scaled', cmap='binary')

            _ = long_task_progress.progress(100)
            c1.header("Processed map")
            c1.image('x.png', use_column_width=True)
            os.remove('x.png')
            c2.header("24h-prediction")
            probability = prediction[0][0]
            image_prediction_2([probability, vgg_prediction], 'x.png')
            c2.image('x.png', use_column_width=True)
            os.remove('x.png')
