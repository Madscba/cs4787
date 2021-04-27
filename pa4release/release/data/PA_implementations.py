from pa4release.release.main import *

from tqdm import tqdm
batch_size = 128
num_classes = 10
epochs = 10
mnist_input_shape = (28, 28, 1)
d1 = 1024
d2 = 256
alpha = 0.1
beta = 0.9
alpha_adam = 0.001
rho1 = 0.99
rho2 = 0.999

def part_1(Xs, Ys,Xs_te, Ys_te):
    #part 1.1
    MLP_sgd, train_history_sgd = train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, 0, batch_size, epochs)
    score_sgd = evaluate_model(Xs_te, Ys_te, MLP_sgd)
    # print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    #part 1.2
    MLP_sgd_mom, train_history_sgd_mom = train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs)
    score_sgd_mom = evaluate_model(Xs_te, Ys_te, MLP_sgd_mom)
    #
    # #part 1.3
    MLP_adam, train_history_adam = train_fully_connected_adam(Xs, Ys, d1, d2, alpha, rho1, rho2, batch_size, epochs)
    score_adam = evaluate_model(Xs_te, Ys_te, MLP_adam)
    #
    # # part 1.4
    MLP_bn_sgd, train_history_bn_sgd = train_fully_connected_bn_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs)
    score_bn_sgd = evaluate_model(Xs_te, Ys_te, MLP_bn_sgd)


    histories = [train_history_sgd,train_history_sgd_mom,train_history_adam,train_history_bn_sgd]
    legends = ["SGD no momentum","SGD with momentum","Adam","SGD momentum & batch_norm"]
    scores_loss = [score_sgd[0],score_sgd_mom[0],score_adam[0],score_bn_sgd[0]]
    scores_acc = [score_sgd[1],score_sgd_mom[1],score_adam[1],score_bn_sgd[1]]

    plot_train_val(histories, legends,scores_loss,scores_acc,part=1)

    #walltimes
    # num_runs = 5
    # t1 = time.time()
    # for run in range(num_runs):
    #     train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs, momentum=False)
    # sum_time1 = time.time() - t1
    # t1 = time.time()
    # for run in range(num_runs):
    #     train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs, momentum=True)
    # sum_time2 = time.time() - t1
    #
    # t1 = time.time()
    # for run in range(num_runs):
    #     train_fully_connected_adam(Xs, Ys, d1, d2, alpha, rho1, rho2, batch_size, epochs)
    # sum_time3 = time.time() - t1
    #
    # t1 = time.time()
    # for run in range(num_runs):
    #     train_fully_connected_bn_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs)
    # sum_time4 = time.time() - t1

    # print(f"alg1: {sum_time1/num_runs}, alg2: {sum_time2/num_runs}, alg3: {sum_time3/num_runs}, alg4: {sum_time4/num_runs}")




def part_3(Xs_tr, Ys_tr, Xs_te, Ys_te):
    model, history = train_CNN_sgd(Xs_tr, Ys_tr, alpha_adam, rho1, rho2, batch_size, epochs)
    train_plot(Xs_te, Ys_te, model, history, "Convolutional Neural Network", 10)

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # part_1(Xs_tr, Ys_tr,Xs_te, Ys_te)
    part_2(Xs_tr, Ys_tr,Xs_te, Ys_te)