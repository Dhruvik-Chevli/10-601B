import numpy as np
import sys
import learnhmm as lh

def calcAlphas(combo, tag_length, word_length, hmmprior, hmmemit, hmmtrans):
    length = len(combo)
    alpha = np.zeros((tag_length, length))

    x_0 = combo[0][0]
    for k in range(0, tag_length):
        alpha[k,0] = hmmprior[k] * hmmemit[k,x_0]

    for t in range(1, length):
        x_t = combo[t][0]
        for k in range(0, tag_length):
            alpha[k,t] = hmmemit[k,x_t] * sum([alpha[j,t-1]*hmmtrans[j,k] for j in range(0, tag_length)])

    unrefined_alpha = alpha
    alpha = alpha/np.sum(alpha, axis=0)
    return alpha, unrefined_alpha

def calcBetas(combo, tag_length, word_length, hmmprior, hmmemit, hmmtrans):
    length = len(combo)
    beta = np.zeros((tag_length, length))

    for k in range(0, tag_length):
        beta[k, length-1] = 1.0

    for t in range(length-2, -1, -1):
        x_t = combo[t+1][0]
        for k in range(0, tag_length):
            beta[k,t] = sum([hmmtrans[k,j]*hmmemit[j,x_t]*beta[j, t+1] for j in range(0, tag_length)])

    beta = beta/np.sum(beta, axis=0)
    # print (beta)
    return beta

def getPredictionAndLikelihood(combo, alpha, unrefined_alpha, beta, tag_length):
    predicted_combo = []
    correct_count = 0
    length = len(combo)
    
    for t in range(0, length):
        alpha_t = alpha[:,t]
        beta_t = beta[:,t]
        y_t = np.argmax(np.multiply(alpha_t, beta_t))
        if y_t == combo[t][1]:
            correct_count += 1
        predicted_combo.append((combo[t][0], y_t))

    acc_list = (correct_count, length)
    # print (alpha, unrefined_alpha, unrefined_alpha[:,length-1])
    ll = np.log(np.sum(unrefined_alpha[:,length-1]))
    # ll = np.sum(unrefined_alpha[:,length-1])
    print (ll)

    return predicted_combo, acc_list, ll

def writePredictions(index_to_word, index_to_tag, final_predictions, predicted):
    word_lines = []
    with open(index_to_word) as infile:
        word_lines = infile.readlines()

    tag_lines = []
    with open(index_to_tag) as infile:
        tag_lines = infile.readlines()

    # print (index_to_word, index_to_tag)
    predicted_string = ""
    for predict in final_predictions:
        temp_string = ""
        for c in predict:
            temp_string += word_lines[c[0]].strip() + "_" + tag_lines[c[1]].strip() + " "
        temp_string = temp_string.rstrip()
        predicted_string += temp_string + "\n"
    
    predicted_string = predicted_string.rstrip()

    with open(predicted, 'w') as outfile:
        outfile.writelines(predicted_string)


if __name__ == "__main__":
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted = sys.argv[7]
    metric_file = sys.argv[8]

    combos,tag_length,word_length = lh.tokenizer(test_input, index_to_word, index_to_tag)

    avg_ll = 0.0
    accuracy = 0.0

    hmmprior = np.loadtxt(hmmprior)
    hmmtrans = np.loadtxt(hmmtrans)
    hmmemit = np.loadtxt(hmmemit)

    final_predictions = []
    final_accuracies = []
    final_ll = []
    for combo in combos:
        alpha, unrefined_alpha = calcAlphas(combo, tag_length, word_length, hmmprior, hmmemit, hmmtrans)
        beta = calcBetas(combo, tag_length, word_length, hmmprior, hmmemit, hmmtrans)
        prediction, acc_list, ll = getPredictionAndLikelihood(combo, alpha, unrefined_alpha, beta, tag_length)
        final_predictions.append(prediction)
        final_accuracies.append(acc_list)
        final_ll.append(ll)

    total_count = 0
    correct_count = 0
    for a,b in enumerate(final_accuracies):
        correct_count += b[0]
        total_count += b[1]

    accuracy = float(correct_count)/total_count
    avg_ll = sum(final_ll)/len(final_ll)
    # print (accuracy, avg_ll)

    metric_string = "Average Log-LikeLikhood: " + str(avg_ll) + "\n" + "Accuracy: " + str(accuracy)
    with open(metric_file, 'w') as outfile:
        outfile.writelines(metric_string)

    writePredictions(index_to_word, index_to_tag, final_predictions, predicted)
    