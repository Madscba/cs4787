from pa4release.release.main_mads import *

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
    MLP_sgd, train_history_sgd = train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs,momentum=False)
    score_sgd = evaluate_model(Xs_te, Ys_te, MLP_sgd)
    # print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    #part 1.2
    MLP_sgd_mom, train_history_sgd_mom = train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs,momentum=True)
    score_sgd_mom = evaluate_model(Xs_te, Ys_te, MLP_sgd_mom)

    #part 1.3
    MLP_adam, train_history_adam = train_fully_connected_adam(Xs, Ys, d1, d2, alpha, rho1, rho2, batch_size, epochs)
    score_adam = evaluate_model(Xs_te, Ys_te, MLP_adam)

    #part 1.4
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
def part_2(Xs, Ys,Xs_te, Ys_te):
    #Part 2.1
    # best_alpha = 100
    # best_acc = 0
    #
    # alpha_values = [1.0,0.3,0.1,0.03,0.01,0.003,0.001]
    # val_acc = []
    # val_loss = []
    # for alpha in alpha_values:
    #     MLP_sgd_mom, train_history_sgd_mom = train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs,momentum=True)
    #     val_acc.append(train_history_sgd_mom.history["val_sparse_categorical_accuracy"][-1])
    #     val_loss.append(train_history_sgd_mom.history["val_loss"][-1])
    #     if val_acc[-1] > best_acc:
    #         best_alpha = alpha
    #         best_acc = val_acc[-1]
    # print("val_acc: ",val_acc)
    # print("val_loss: ",val_loss)
    # print("Best alpha and acc: {} {}".format(best_alpha,best_acc))
    #Part 2.2
    # best_vaL_acc = 0
    #
    # additional_layers = [1,2,4]
    # additional_layers_depth = [128,256,512]
    # beta_values = [0.5,0.9,0.99,0.999]
    # for additional_layer in additional_layers:
    #     for additional_layer_depth in additional_layers_depth:
    #         for beta in beta_values:
    #             MLP_bn_sgd, train_history_bn_sgd = train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size, epochs,addi_layer=additional_layer,depth=additional_layer_depth)
    #             score_bn_sgd = evaluate_model(Xs_te, Ys_te, MLP_bn_sgd)
    #             if train_history_bn_sgd.history["val_sparse_categorical_accuracy"][-1] > best_vaL_acc:
    #                 best_hyperparams = {"extra_layers":additional_layer,"depth":additional_layer_depth,"beta":beta}
    #                 best_vaL_acc = train_history_bn_sgd.history["val_sparse_categorical_accuracy"][-1]
    #                 best_val_loss = train_history_bn_sgd.history["val_loss"][-1]
    #                 best_test_loss = score_bn_sgd[0]
    #                 best_test_acc = score_bn_sgd[1]
    #
    # print("Best hyperparams: ",best_hyperparams)
    # print("val acc: {}, val loss: {}".format(best_vaL_acc,best_val_loss))
    # print("test acc: {}, test loss: {}".format(best_test_acc, best_test_loss))


    #part 2.3 random search
    best_vaL_acc = 0
    combinations_of_hyperparams =  []
    for i in range(3):
        additional_layer = int(np.random.uniform(low=1,high=5,size=1)[0])
        for j in range(3):
            additional_layer_depth = int(np.random.normal(loc=256,scale=128,size=1)[0])
            for k in range(4):
                beta = np.random.normal(loc=.5,scale=0.2,size=1)[0]
                MLP_bn_sgd, train_history_bn_sgd = train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, batch_size,
                                                                             epochs, addi_layer=additional_layer,
                                                                             depth=additional_layer_depth)
                score_bn_sgd = evaluate_model(Xs_te, Ys_te, MLP_bn_sgd)
                combinations_of_hyperparams.append((additional_layer,additional_layer_depth,beta))
                if train_history_bn_sgd.history["val_sparse_categorical_accuracy"][-1] > best_vaL_acc:
                    best_hyperparams = {"extra_layers": additional_layer, "depth": additional_layer_depth, "beta": beta}
                    best_vaL_acc = train_history_bn_sgd.history["val_sparse_categorical_accuracy"][-1]
                    best_val_loss = train_history_bn_sgd.history["val_loss"][-1]
                    best_test_loss = score_bn_sgd[0]
                    best_test_acc = score_bn_sgd[1]

    print("Best hyperparams: ", best_hyperparams)
    print("val acc: {}, val loss: {}".format(best_vaL_acc, best_val_loss))
    print("test acc: {}, test loss: {}".format(best_test_acc, best_test_loss))

    print("Tested combinations:",combinations_of_hyperparams)

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # part_1(Xs_tr, Ys_tr,Xs_te, Ys_te)
    part_2(Xs_tr, Ys_tr,Xs_te, Ys_te)