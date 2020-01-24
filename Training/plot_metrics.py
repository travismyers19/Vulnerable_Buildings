import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


training = np.load('training_loss.npy')
validatn = np.load('validatn_loss.npy')

epochs = [x for x in range(len(training))]

plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.21, left=0.21)
plt.plot(epochs, training, epochs, validatn)
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.xlim([0, len(training)])
plt.ylim([0, max(max(training), max(validatn))])
ax.legend(['Training', 'Validation'])
st.pyplot()