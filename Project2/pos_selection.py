import nltk 


set_exclude = {'CC','DT','IN','PRP','PRP$','MD','TO','UH','WDT','WP','WP$','WRB'}
#set_include = {'NN','NNS','NNP','VB','VBD','VBG','VBN','VBP','VBZ'}
set_include = {'NN'}
ind = []
cc = 0
with open("features_idx.txt",'r') as r:
    for row in r:
        line = row.split('\n')[0].split(' ')
        text = nltk.word_tokenize(line[1])
        tag = nltk.pos_tag(text)[0][1]
        if tag in set_include:
            ind.append([])
            ind[cc] = line[0]
            cc += 1
        else:
            continue
            

with open('pos_selection.txt', 'w') as w:
    for item in ind:
        w.write(item+'\n')

print(len(ind))