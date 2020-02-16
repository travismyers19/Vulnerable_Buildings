import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--metrics_filename", type=str, default='Models/metrics.npy',
        help = "The file location of the saved npy file containing the metrics.  Default is 'Models/metrics.npy'.")
    flags = parser.parse_args()

    metrics_filename = flags.metrics_filename
    metrics = np.load(metrics_filename)
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