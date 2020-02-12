import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    metrics = np.load('Models/test_metrics.npy')
    #metrics = np.concatenate((metrics_1_5, metrics_6_10, metrics_11_15, metrics_16_35), axis=1)
    accuracy = metrics[0, :]
    loss = metrics[1, :]

    epochs = [x for x in range(len(accuracy))]

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    plt.gcf().subplots_adjust(bottom=0.21, left=0.21)
    plt.plot(epochs, accuracy, epochs, loss)
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.xlim([0, len(accuracy)])
    plt.ylim([0, max(max(accuracy), max(loss))])
    ax.legend(['Accuracy', 'Loss'])
    st.pyplot()