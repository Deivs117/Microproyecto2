from flask import Flask, request, jsonify
import mxnet as mx
from mxnet import nd, gluon
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.presets.imagenet import transform_eval
from PIL import Image
import numpy as np

app = Flask(__name__)

# Configuración sugerida por el profesor: Uso de modelo pre-entrenado 
net = get_model('cifar_resnet20_v1', classes=10, pretrained=True)

# Clases del dataset CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/predict', methods=['POST'])
def predict():
    if 'img' not in request.files:
        return "No image found", 400
    
    # Procesamiento de la imagen
    file = request.files['img']
    img_pil = Image.open(file.stream).convert('RGB')
    img_nd = mx.nd.array(np.array(img_pil))
    
    # Transformación para CIFAR (ajustar según necesidad del modelo)
    transformed_img = transform_eval(img_nd)
    
    # Inferencia
    pred = net(transformed_img)
    ind = nd.argmax(pred, axis=1)
    
    # Cadena de predicción con el formato de paréntesis corregido 
    prediction = ('The input picture is classified as [%s], with probability %.3f.' %
                  (class_names[int(ind.asscalar())], nd.softmax(pred)[0][int(ind.asscalar())].asscalar()))
    
    return prediction

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)