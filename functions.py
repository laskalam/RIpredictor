import os
import numpy as np
from download import download_requirements
download_requirements()
from xception_model import ModelFactory
from vgg_model import ModelVGG as model
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"]="0"
nd = 1
num_classes = 2

def image_prediction_2(prob, file_name):
    """Create the image showing the RI probability from the 2 models
    """
    font = {'family' : 'DejaVu Sans',
            'weight' : 'bold',
            'size'   : 22}
    font1 = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 15}
    size = 0.2
    outersize = 0.8
    classes = ['RI', 'non-RI']
    mycolors = ['red', 'green']
    fig, axes = plt.subplots(2,1,figsize=(10,10))

    #----
    y = [prob[0], 1-prob[0]]
    argmax = np.argmax(y)
    mylabels = [f'{100*_y:1.1f}% {c}' for _y, c in zip(y, classes)]
    wedges, texts = axes[0].pie(y, radius=outersize, labels = mylabels,
                                wedgeprops=dict(width=size, edgecolor='w'),
                                colors=mycolors
                                             )
    plt.setp(texts, size=24, fontweight=600)
    for i, patch in enumerate(wedges):
        texts[i].set_color(patch.get_facecolor())
    axes[0].add_patch(plt.Circle((0, 0), outersize-1.1*size, color=mycolors[argmax], alpha=0.7))
    axes[0].text(0.5, 0.5, classes[np.argmax(y)],
                 color='white', weight ="bold", size=32,
                 va='center', ha='center', transform=axes[0].transAxes)
    #----
    y = [prob[1], 1-prob[1]]
    argmax = np.argmax(y)
    mylabels = [f'{100*_y:1.1f}% {c}' for _y, c in zip(y, classes)]
    wedges, texts = axes[1].pie(y, radius=outersize, labels = mylabels,
                                              wedgeprops=dict(width=size, edgecolor='w'),
                                              colors=mycolors

                                             )
    plt.setp(texts, size=24, fontweight=600)
    for i, patch in enumerate(wedges):
        texts[i].set_color(patch.get_facecolor())
    axes[1].add_patch(plt.Circle((0, 0), outersize-1.1*size, color=mycolors[argmax], alpha=0.7))
    axes[1].text(0.5, 0.5, classes[np.argmax(y)],
                   color='white', weight ="bold", size=32,
                   va='center', ha='center', transform=axes[1].transAxes)
    #----


    axes[0].set_title('Model 1 prediction', fontdict=font, horizontalalignment='center',
                verticalalignment='center', transform=axes[0].transAxes)
    axes[1].set_title('Model 2 prediction', fontdict=font, horizontalalignment='center',
                verticalalignment='center', transform=axes[1].transAxes)

    axes[0].axis('off')
    axes[1].axis('off')

    plt.savefig(file_name, dpi=360)#bbox_inches='tight', dpi=500)

    return 


def image_prediction(prob, file_name):
    """Create the image showing the RI probability from one model"""
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 22}
    font1 = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 15}

    fig = plt.figure(figsize=(10,10))
    pred_ax = plt.subplot2grid((10, 10), (3, 0), colspan=9, rowspan=4)
    prob_ax = plt.subplot2grid((10, 10), (9, 0), colspan=9, rowspan=1)


    prob_ax.text(-0.01, 0.5, 'non-RI', fontdict=font, horizontalalignment='right',
                verticalalignment='center', transform=prob_ax.transAxes)
    pred_ax.text(-0.01, 0.5, 'non-RI', fontdict=font, color='white', horizontalalignment='right',
                verticalalignment='center', transform=pred_ax.transAxes)
    prob_ax.text(1.01, 0.5, 'RI', fontdict=font, horizontalalignment='left',
                verticalalignment='center', transform=prob_ax.transAxes)

    #non RI
    prob_ax.axvspan(0, 1-prob-0.001,0,1, alpha=0.5, color='green')
    prob_ax.text((1-prob)/2, 1, f'{round((1-prob)*100)}%',  fontdict=font1, horizontalalignment='center',
                verticalalignment='bottom', transform=prob_ax.transAxes)
    #RI
    prob_ax.axvspan(1-prob, 1, 0,1, alpha=0.5, color='red')
    prob_ax.text((2-prob)/2, 1, f'{round((prob)*100)}%',  fontdict=font1, horizontalalignment='center',
                verticalalignment='bottom', transform=prob_ax.transAxes)
    # prob_ax.axvspan(1-prob, 1,0,1, alpha=0.5, color='red')

    color ='green'
    prediction_text = 'non-RI'
    if prob>0.5:
        color='red'
        prediction_text = 'RI'

    pred_ax.axvspan(0, 0.4 ,0, 1, alpha=1, color=color)
    pred_ax.axvspan(0.4, 1 ,0, 1, alpha=1, color='white')

    pred_ax.text(0.2, 0.5, prediction_text, fontdict=font, horizontalalignment='center',
                verticalalignment='center', transform=pred_ax.transAxes)

    prob_ax.axis('off'); prob_ax.set_xlim(0,1)
    pred_ax.axis('off'); pred_ax.set_xlim(0,1)


    plt.savefig(file_name, bbox_inches='tight', dpi=500)
    return 


def xception():
    """Load the xception model with the trained weights"""
    model_weights_file = './xception_weights.h5'
        
    model_factory = ModelFactory()
    model, preprocess_input, input_size, pi_str = model_factory.get_model(
        ['RI'],
        use_base_weights=False,
        weights_path=model_weights_file,
        input_shape=(224, 224, 1),
        )

    return model, preprocess_input, input_size, pi_str


def vgg():
    """Load the vgg model with the trained weights."""
    model_weights_file = './vgg_weights.h5'
    net = model(0.1)
    net.load_weights(model_weights_file)
    return net
