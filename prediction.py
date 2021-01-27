from Masking import apply_mask
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def prediction(model, img_128_128_3, img_name=None):
    import pickle
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    img_128_128_3 = keras.preprocessing.image.img_to_array(img_128_128_3)
    img_128_128_3 = img_128_128_3/255.
    img_128_128_3 = img_128_128_3.reshape(1, 128, 128, 3)

    name = 'querry'
    img_list = apply_mask(img_128_128_3, name)
    #img_list = pickle.load(open('/content/sample_data/img_files/img_' + name + '.pkl', 'rb'))

    mask_pred = []
    for img in img_list:
        mask_pred.append(model.predict(img)[0])

    real_pred = model.predict(img_128_128_3)[0]
    print(real_pred.shape)
    real_indx = list(real_pred).index(real_pred.max())

    p = ''
    if real_indx == 0:
        p = 'The chest is NORMAL'
        print('The chest is NORMAL')
    elif real_indx == 1:
        print('The chest is having BACTERIAL PNEUMANIA!')
        p = 'The chest is having BACTERIAL PNEUMANIA!'
    else:
        print('The chest is having VIRAL PNEUMONIA!')
        p = 'The chest is having VIRAL PNEUMONIA!'
    indx = [i[real_indx] for i in mask_pred]
    min_indx = indx.index(np.array(indx).max())
    path = 'C:/Users/amiti/Desktop/academics/MasterProject/Project_vis/static/'+str(img_name)
    keras.preprocessing.image.save_img(path,img_list[min_indx].reshape(128, 128, 3))

    #consent = input('Do you want more information? y/n? : ')
    '''
    fig, ax = plt.subplots(1, 2)
    
    if consent == 'y':
        indx = [i[real_indx] for i in mask_pred]
        min_indx = indx.index(np.array(indx).min())

        fig.set_figheight(20)
        fig.set_figwidth(20)
        ax[0].set_title('region leading to classification')
        ax[0].imshow(img_list[min_indx].reshape(128, 128, 3))
        ax[1].set_title('Prediction = ' + p)
        ax[1].imshow(img_128_128_3.reshape(128, 128, 3))
    '''
    return real_pred, p


