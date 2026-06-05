{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "da3c46ae-e1a9-4efa-b071-685470f3c231",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "from sklearn.model_selection import train_test_split\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b34d4b19-fb54-4949-aa5c-be28df80d34e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz\n",
      "\u001b[1m11490434/11490434\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m9s\u001b[0m 1us/step\n"
     ]
    }
   ],
   "source": [
    "(X_train,y_train),(X_test,y_test)=keras.datasets.mnist.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "290ec5a4-6272-4544-9e4f-45540b2ed44f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "60000"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "7aba2f11-582d-4bca-9ae9-e960b780a1ae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "10000"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "5b3ad5da-40d5-4cfe-a459-786bad785e51",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(28, 28)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train[0].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5099673d-7ed7-482b-84f0-728f74c3b065",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,\n",
       "         18,  18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170,\n",
       "        253, 253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253,\n",
       "        253, 253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253,\n",
       "        253, 198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253,\n",
       "        205,  11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,\n",
       "         90,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253,\n",
       "        190,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190,\n",
       "        253,  70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,\n",
       "        241, 225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "         81, 240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,  45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,  46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39,\n",
       "        148, 229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221,\n",
       "        253, 253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253,\n",
       "        253, 253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253,\n",
       "        195,  80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,\n",
       "         11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0],\n",
       "       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
       "          0,   0]], dtype=uint8)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "977aee4b-f627-48ac-ba75-fc3fd305f943",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x237e2ec82e0>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 480x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.matshow(X_train[0])"
   ]
  },
  {
   "cell_type": "raw",
   "id": "1ff3e454-708a-4c31-9183-6453ec3e8631",
   "metadata": {},
   "source": [
    "plt.matshow(X_train[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a57b2c75-1308-4e38-9c11-2e873d2aa9e7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.uint8(4)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train[2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "c6589f6b-0a93-4fc2-8b28-e6168060bf6f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 0, 4, 1, 9], dtype=uint8)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "7c23700d-54ad-4d7c-b42d-62a21b632665",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 28, 28)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "80c9495d-7184-4831-8540-2b092a06a16d",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_flat_train=X_train.reshape(len(X_train),28*28)\n",
    "X_flat_test=X_test.reshape(len(X_test),28*28)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "98a5befb-8fd8-4751-ad58-1728656b4ecc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 784)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_flat_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "3cd5f17f-83ba-422d-9cb0-ccdcca1902a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train=X_train/255\n",
    "X_test=X_test/255"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "d2d60501-dc58-42e2-bd71-8a02aa6d3bd8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 2ms/step - accuracy: 0.5530 - loss: 2.1879\n",
      "Epoch 2/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 2ms/step - accuracy: 0.7122 - loss: 1.9769\n",
      "Epoch 3/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 2ms/step - accuracy: 0.7394 - loss: 1.7901\n",
      "Epoch 4/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 2ms/step - accuracy: 0.7609 - loss: 1.6263\n",
      "Epoch 5/5\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 2ms/step - accuracy: 0.7759 - loss: 1.4835\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.history.History at 0x237e25b5090>"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model=keras.Sequential([\n",
    "    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')\n",
    "])\n",
    "model.compile(\n",
    "    optimizer='adam',\n",
    "    loss='sparse_categorical_crossentropy',\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "model.fit(X_flat_train,y_train, epochs=5)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "fdd3a3c5-800b-40ef-8ded-756529a663db",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8022 - loss: 1.4038\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[1.4037566184997559, 0.8022000193595886]"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.evaluate(X_flat_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "f25d990a-bb58-4a94-a9ec-a08ee4aaa7dd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x237e6950130>"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 480x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.matshow(X_test[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "64777890-fe65-4e34-af65-1d525598c0dd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([0.31445104, 0.39814413, 0.3540254 , 0.4183406 , 0.46987697,\n",
       "       0.3824351 , 0.28527582, 0.80618644, 0.36890543, 0.5841981 ],\n",
       "      dtype=float32)"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_predicted=model.predict(X_flat_test)\n",
    "y_predicted[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "07b7a26f-27c3-40cb-a6f2-a7828a6d45b1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.int64(7)"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.argmax(y_predicted[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c0eb3ed5-88df-4a00-8e56-c67b41f7a6c5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
