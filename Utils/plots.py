
import numpy as np
import Utils.metrics as metrics
from pylab import rcParams
import matplotlib.pyplot as plt
import Utils.utils as utils


def show(_sting, content):
    print(_sting + str(content))

def display_loss_vs_lr( plt, trace, N_EPOCHS):
    learning_rates = 1e-3 * (10 ** (np.arange(N_EPOCHS) / 30))
    plt.semilogx(learning_rates, trace.history['loss'], lw=3, color='#000')
    plt.title('Learning rate vs. loss', size=20)
    plt.xlabel('Learning rate', size=14)
    plt.ylabel('Loss', size=14);

def display_loss_epochs(plt, trace):
    from pylab import rcParams
    rcParams['figure.figsize'] = (8, 8)
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False

    plt.plot(
        trace.history['loss'],
        label='Loss', lw=3
    )
    plt.plot(
        trace.history['lr'],
        label='Learning rate', color='#000', lw=3, linestyle='--'
    )
    plt.title('Evaluation metrics', size=20)
    plt.xlabel('Epoch', size=14)
    plt.legend()


# tensorflow
def displayTraining_ValidationLoss(plt, trace):
    trace_dico = trace.history
    loss_values = trace_dico['loss']  # coût durant l'apprentissage
    val_loss_values = trace_dico['val_loss']  # coût durant la validation
    epochs = range(1, len(loss_values) + 1)  # res -> range(1, 6)

    # tracer les points (bo)
    plt.plot(epochs, loss_values, 'b', label="Training loss")
    # tracer la courbe (b)
    plt.plot(epochs, val_loss_values, 'g', label="Validation loss")
    plt.title('Training & Validation loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()

def dispaly_losses(plt, train_epochs_losses, val_epochs_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_epochs_losses, label='Training Loss')
    plt.plot(val_epochs_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.close()

    #plt.show()

#plot & save figures
def _plot(y_true, y_pred, dim_x, dix_y, plt, rcParams, figure_name="", save_plot=False,
          x_label_plt="Grownd Truth", y_label_plt="Predictions"):
    # len_points = np.asarray(y_test_dataset).shape[1]
    # rcParams['figure.figsize'] = dim_x, dix_y
    # trace_dico = trace.history
    plt.plot(y_true, 'r', label=x_label_plt)
    plt.plot(y_pred, 'g', label=y_label_plt)
    plt.title('Prediction plot')
    r2_score_index = figure_name.split("R2_", 1)[1][:-4]
    plt.title('R2 = ' + str(r2_score_index), fontsize=15)
    plt.xlabel('Timestaps')
    plt.ylabel('Values')
    plt.legend()
    if save_plot is True:
        # plt.savefig(figure_name,format="eps")
        plt.savefig(figure_name + ".png")

    #plt.show()
    plt.close()


#plot and save figures
def plot_results(args,ground_truth,y_test_preds):
    for i in range(args.num_series):
        series_index = i
        # data_store._plot(y_true=ground_truth.reshape(ground_truth.shape[0],ground_truth.shape[1]*ground_truth.shape[2])[series_index],y_pred=y_test_preds.reshape(y_test_preds.shape[0],y_test_preds.shape[1]*y_test_preds.shape[2])[series_index],dim_x= 8, dix_y=8,plt=plt,rcParams=rcParams )
        s_R2 = metrics.r2_score(
            ground_truth.reshape(ground_truth.shape[0], ground_truth.shape[1] * ground_truth.shape[2])[series_index],
            y_test_preds.reshape(y_test_preds.shape[0], y_test_preds.shape[1] * y_test_preds.shape[2])[series_index])

        location = "Results/figures/"+args.dataset_name+"/"+args.selected_model+"/"+args.exec_context
        utils.create_dir(location)
        _plot(y_true=ground_truth.reshape(ground_truth.shape[0], ground_truth.shape[1] * ground_truth.shape[2])[
            series_index],
                    y_pred=y_test_preds.reshape(y_test_preds.shape[0], y_test_preds.shape[1] * y_test_preds.shape[2])[
                        series_index], dim_x=8, dix_y=8, plt=plt, rcParams=rcParams,
                    figure_name=location+"/"+'iter_'+str(args.current_exec_iter)+"_"+args.approach+"_index_" + str(series_index) + "_ph_fh_" + str(
                        args.past_history) + "_" + str(args.forecast_horizon) + "_R2_" + str(s_R2) + ".eps", save_plot=True)


def generateRandomTS(timestep=121, filename='randomTS.xlsx'):
    import matplotlib.pyplot as plt

    values = np.random.randn(timestep).cumsum()
    lst = [i for i in range(len(values))]
    #print(values)
    #print("range ", lst)

    # Create & display TS plot
    plt.plot(range(len(values)), values, linestyle='solid')
    plt.title('Time Series Plot')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.show()

    values = pd.DataFrame(values)
    values.to_excel(filename, index=False)
