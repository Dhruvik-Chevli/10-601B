import numpy as np
import sys

def tokenizer(train_input, index_to_word, index_to_tag):
    dict_lines = []
    with open(train_input) as infile:
        dict_lines = infile.readlines()

    word_lines = []
    with open(index_to_word) as infile:
        word_lines = infile.readlines()

    tag_lines = []
    with open(index_to_tag) as infile:
        tag_lines = infile.readlines()

    word_lines = [i.strip() for i in word_lines]
    tag_lines = [i.strip() for i in tag_lines]
    dict_lines = [i.strip() for i in dict_lines]   

    tag_length = len(tag_lines)
    word_length = len(word_lines)

    combo = []
    length = len(dict_lines)
    for i in range(0, length):
        line = dict_lines[i]
        line = line.split(' ')
        temp_combo = []
        for l in line:
            word, tag = l.split('_')
            word_index = word_lines.index(word)
            tag_index = tag_lines.index(tag)
            temp_combo.append((word_index, tag_index))
        combo.append(temp_combo)

    return combo,tag_length,word_length

def getAndWriteParams(combos, tag_length ,word_length, hmmprior, hmmtrans, hmmemit):
    pi_matrix = np.zeros(tag_length)
    a_matrix = np.zeros((tag_length, tag_length))
    b_matrix = np.zeros((tag_length, word_length))

    for combo in combos:
        # getting pi matrix
        pi_matrix[combo[0][1]] += 1
        
        # getting A matrix
        tags = [i[1] for i in combo]
        for a,b in zip(tags[1:], tags):
            a_matrix[b][a] += 1

        # getting B matrix
        for c in combo:
            b_matrix[c[1]][c[0]] += 1 

    pi_matrix = (pi_matrix+1)/np.sum(pi_matrix+1)
    a_matrix = (a_matrix+1)/np.sum(a_matrix+1, axis=1)[:,None]
    b_matrix = (b_matrix+1)/np.sum(b_matrix+1, axis=1)[:,None]

    NEWLINE_SIZE_IN_BYTES = -1
    with open(hmmprior, 'wb') as outfile:
        np.savetxt(hmmprior, pi_matrix)
        outfile.seek(NEWLINE_SIZE_IN_BYTES, 2)
        outfile.truncate()

    with open(hmmtrans, 'wb') as outfile:
        np.savetxt(hmmtrans, a_matrix)
        outfile.seek(NEWLINE_SIZE_IN_BYTES, 2)
        outfile.truncate()

    with open(hmmemit, 'wb') as outfile:
        np.savetxt(hmmemit, b_matrix)
        outfile.seek(NEWLINE_SIZE_IN_BYTES, 2)
        outfile.truncate()

if __name__ == "__main__":
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    combo, tag_length, word_length = tokenizer(train_input, index_to_word, index_to_tag)
    getAndWriteParams(combo, tag_length, word_length, hmmprior, hmmtrans, hmmemit)