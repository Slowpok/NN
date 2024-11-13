import torch
import NN_training
import Datasets
import NN_init
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import torch_geometric

torch_geometric.nn.LightGCN

def model_predict(word, name_model, RM=False):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet101 = torch.load("whole_best_model" + name_model + ".pth", map_location=torch.device(NN_init.device))
    resnet101.eval()
    resnet101.to(NN_init.device)

    list1 = []
    ww = Datasets.string_to_seq(word)
    list1.append(ww)
    tensor_word = torch.Tensor(pad_sequences(list1, NN_init.size_of_array))

    tensor_word.to(NN_init.device)
    if RM:
        hidden = resnet101.init_hidden(NN_init.device)
        result, hl = resnet101(tensor_word.to(NN_init.device), hidden)

    else:
        result_list = resnet101(tensor_word.to(NN_init.device))

    result = torch.reshape(result, (-1,))[0].cpu().detach().numpy()
    return 1 if result > 0.5 else 0


def mass_model_predict(list_of_words, name_model, RM=False):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet101 = torch.load("whole_best_model" + name_model + ".pth", map_location=torch.device(NN_init.device))
    resnet101.eval()
    resnet101.to(NN_init.device)
    list_seq = Datasets.string_list_to_sequence(list_of_words)

    tensor_word = torch.Tensor(pad_sequences(list_seq, NN_init.size_of_array))

    if RM:
        hidden = resnet101.init_hidden(NN_init.device)
        result_list, hl = resnet101(tensor_word.to(NN_init.device), hidden)

    else:
        result_list = resnet101(tensor_word.to(NN_init.device))

    if resnet101.num_classes == 1:
        y_pred = result_list.cpu().detach().numpy()
        y_pred = [1 if x > 0.5 else 0 for x in y_pred]
    else:
        y_pred = torch.argmax(result_list, dim=1)
        y_pred = y_pred.cpu().detach().numpy()

    result = {key: value for key, value in zip(list_of_words, y_pred)}
    return result



