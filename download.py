import os
from urllib.request import urlopen
from streamlit import secrets
xception_weight_file = secrets['xception_url']
vgg_weight_file      = secrets['vgg_url']
xception_model_file  = secrets['xception_file']
vgg_model_file       = secrets['vgg_file']


def download_requirements() -> None:
    """Download all required documents/data here"""
    try:
        if not os.path.exists('./vgg_weights.h5'):
            u = urlopen(vgg_weight_file)
            data = u.read()
            u.close()
            with open('vgg_weights.h5', 'wb') as f:
                f.write(data)
        
        if not os.path.exists('./xception_weights.h5'):
            u = urlopen(xception_weight_file)
            data = u.read()
            u.close()
            with open('xception_weights.h5', 'wb') as f:
                f.write(data)
        
        if not os.path.exists('./xception_model.py'):
            u = urlopen(xception_model_file)
            data = u.read()
            u.close()
            with open('xception_model.py', 'wb') as f:
                f.write(data)
        
        if not os.path.exists('./vgg_model.py'):
            u = urlopen(vgg_model_file)
            data = u.read()
            u.close()
            with open('vgg_model.py', 'wb') as f:
                f.write(data)
    except:
        error = """<p style='font-family:sans-serif; color:Red; font-size: 24px;'>
        One or more file(s) are not properly downloaded. Please contact the owner.
        </p>"""
        st.markdown(error, unsafe_allow_html=True)
        st.stop()


