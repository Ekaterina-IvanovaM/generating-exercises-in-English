import pandas as pd 
import re 
import random 
import numpy as np 
import spacy
import pyinflect
import gensim.downloader as api
import contractions 
from sentence_splitter import SentenceSplitter



# Вспомогательная функция для подсчёта количества слов в предложении
def get_number_of_words(row):
    return len(re.findall(r'\w+', row))

# Вспомогательная функция для определения того, какие части речи присутствуют в предложении
def part_of_speech_in_sent(sentence, nlp):    
    # Неопредеделённые местоимения (some/any/no/every+body/one/thing/)
    pronouns_ind = ['something', 'somebody', 'someone', 'anything', 'anybody', 'anyone', 'nothing', 'nobody',
                    'everything', 'everybody', 'everyone']
    
    doc = nlp(sentence)
    part_of_speech = set()
    for token in doc:
        part_of_speech.add(token.pos_)
        
        if token.lower_ in ['a', 'an', 'the']:
            part_of_speech.add('ART')
        elif token.lower_ in pronouns_ind or (token.lower_ == 'no' and token.i < len(doc)-1 and token.nbor(1).lower_ == 'one'):
            part_of_speech.add('PRONind')
                   
    return part_of_speech



class genTasksEng:
    # Функция инициализации класса
    def __init__(self, str_text, used_task_type, percent_used_sent):
        # Удаление пустых строк и приведение всех сокращений к полной форме ('re -> are, n't -> not и т.д)
        read_text = contractions.fix(str_text.replace('\n\n', '\n'))
        
        # Разделение текста на отдельные предложения
        splitter = SentenceSplitter(language='en')
        sentences = splitter.split(text=read_text) 
        
        # Создание датафрейма, из данных которого будут генерироваться упражнения
        data = pd.DataFrame(columns=['source_text', 'type_task', 'options', 'answer', 'output_text', 'list_type_task'])
        data['source_text'] = sentences
        
        # Добавляем в датафрейм столбцец, который содержит количество слов в предложении
        data['word_count'] = data['source_text'].apply(get_number_of_words)
        
        # Загружаем предобученную модель из библиотеки spacy.
        nlp = spacy.load('en_core_web_sm')
        
        # Добавляем в датафрейм столбец, в котором содержатся списки частей речи, которые присутствуют в предложении
        data['part_of_speech_in_sent'] = data['source_text'].apply(part_of_speech_in_sent, nlp=nlp)
        
        # Атрибуты класса
        self.df = data
        self.percent_used_sent = percent_used_sent
        self.nlp = nlp
        self.used_task_type = used_task_type
        
        if used_task_type['select_word_by_meaning'] == 1:
            self.gensim_model = api.load("glove-wiki-gigaword-100")
    
    
    
    # Функция для определения типа упражнения в каждом предложении
    def set_task_type(self): 
        num_types_of_tasks = {k: 0 for k in self.used_task_type.keys()}
        
        # ВАЖНО: названия упражнений в список row['list_type_task'] добавляем в том порядке, 
        # в котором они находятся в словарях types_of_tasks и num_types_of_tasks
        possible_type_task = []    
        for i, row in self.df.iterrows():
            if row['word_count'] > 5:            
                if len(row['part_of_speech_in_sent'].intersection(set(['VERB', 'AUX', 'ADP', 'ADV']))) > 0 and \
                self.used_task_type['select_word_by_meaning'] == 1:
                    possible_type_task.append('select_word_by_meaning')
                    num_types_of_tasks['select_word_by_meaning'] += 1

                if len(row['part_of_speech_in_sent'].intersection(set(['AUX', 'VERB']))) > 0 and \
                self.used_task_type['select_verb_by_form'] == 1:
                    possible_type_task.append('select_verb_by_form')
                    num_types_of_tasks['select_verb_by_form'] += 1

                if 'ART' in row['part_of_speech_in_sent'] and self.used_task_type['select_articles'] == 1:
                    possible_type_task.append('select_articles')
                    num_types_of_tasks['select_articles'] += 1
                    
                if 'PRONind' in row['part_of_speech_in_sent'] and self.used_task_type['select_indefinite_pronouns'] == 1:
                    possible_type_task.append('select_indefinite_pronouns')
                    num_types_of_tasks['select_indefinite_pronouns'] += 1

                if 'ADP' in row['part_of_speech_in_sent'] and self.used_task_type['write_preposition'] == 1:
                    possible_type_task.append('write_preposition')
                    num_types_of_tasks['write_preposition'] += 1

                if 'VERB' in row['part_of_speech_in_sent'] and self.used_task_type['write_verbs'] == 1:
                    possible_type_task.append('write_verbs')
                    num_types_of_tasks['write_verbs'] += 1

                if row['word_count'] > 10 and self.used_task_type['correct_word_order'] == 1:
                    possible_type_task.append('correct_word_order')
                    num_types_of_tasks['correct_word_order'] += 1

                if len(possible_type_task) > 0:
                    self.df.at[i, 'list_type_task'] = possible_type_task.copy()
                    possible_type_task.clear()

        there_are_sent_with_type = len(list(filter(lambda x: x is not np.nan, self.df['list_type_task'])))
        need_sent_with_type = round(self.df.shape[0] * self.percent_used_sent)

        if there_are_sent_with_type > need_sent_with_type:
            type_to_no_tasks = random.sample(list(self.df.index), k=there_are_sent_with_type-need_sent_with_type)
            self.df.loc[type_to_no_tasks, 'list_type_task'] = np.nan

        for i, row in self.df.iterrows():
            if row['list_type_task'] is not np.nan:
                weights = [v for k, v in num_types_of_tasks.items() if k in row['list_type_task']]
                temp_min = min(filter(lambda x: x != 0, weights))
                weights = [temp_min * x ** (-1) for x in weights]        
                self.df.at[i, 'type_task'] = random.choices(row['list_type_task'], weights=weights, k=1)[0]
            else:
                self.df.at[i, 'type_task'] = 'no_tasks'   
    
        
        self.df = self.df.drop(['list_type_task', 'word_count', 'part_of_speech_in_sent'], axis=1)
    
    
    
    # Задание: выбрать наиболее подходящее по смыслу слово.
    # В качестве объектов (выбранные для задания слова) функция может выбрать 
    # предлоги (ADP), вспосомагетльные глаголы (AUX), например, is, has, will, should), глаголы (VERB) и наречия (ADV).
    # - top_words - ТОП-слов, подобранных как синонимы к верному ответу функцией most_similar.
    # - num_words_to_choose - количество вариантов ответа, помимо верного слова.
    # - num_missing_words - количество заданий в предложении, в данном случае равно количеству пропущенных слов. Количество заданий 
    #       в предложении зависит от количества слов в нём: на каждые 10 слов в предложении приходится 1 задание,
    #       например, если слов в предложении 7, то задание будет всего 1, если слов 15, то заданий 2, если 40, то 4, при условии, 
    #       что такое количество возможных слов-объектов среди глаголов, наречий, предлогов найдётся в предложении, 
    #       иначе количество заданий будет равно количеству найденных слов-объектов.    
    def select_word_by_meaning(self, sentence):
        top_words = 7
        num_words_to_choose = 2
        num_missing_words = int(round(get_number_of_words(sentence), -1) / 10)
        
        answers_options = {}
        doc = self.nlp(sentence)

        for token in doc:
            if token.pos_ in ['VERB', 'AUX', 'ADP', 'ADV']: 
                try:
                    opt_word_sum = self.gensim_model.most_similar(token.text.lower(), topn=top_words) 
                except KeyError:
                    continue

                opt_word = []
                for word in opt_word_sum:
                    # Проверяем, что слова из списка вариантов не равны предыдущему или следующему слову
                    if self.nlp(word[0])[0].pos_ != 'PUNCT' and ((token.i == 0 and word[0] != token.nbor(1).text) or \
                    (token.i == len(doc)-1 and word[0] != token.nbor(-1).text) or \
                    (0 < token.i < len(doc)-1 and word[0] != token.nbor(1).text and word[0] != token.nbor(-1).text)):
                        if word[0] == "n't":
                            opt_word.append('not')
                        elif  word[0] == "'m":
                            opt_word.append('am')
                        elif  word[0] == "'s":
                            opt_word.append('is')
                        elif  word[0] == "'re":
                            opt_word.append('are')
                        elif  word[0] == "'ll":
                            opt_word.append('will')
                        else:
                            opt_word.append(word[0])

                # Оставляем только уникальные значения списка
                opt_word = list(set(opt_word))

                if len(opt_word) != 0:
                    opt_word = [w.title() if token.text.istitle() or w == 'i' else w for w in opt_word] 

                # Проверяем, что исходное слово не входит в список вариантов
                if token.text in opt_word:
                    opt_word.remove(token.text)

                # Если список пустой, переходим к следующей итерации цикла for
                if len(opt_word) == 0:
                    continue

                opt_word = random.sample(opt_word, k=num_words_to_choose) + [token.text]

                random.shuffle(opt_word)
                answers_options[token] = opt_word

        # Если количество найденных слов-объектов в предложении меньше, чем расчитанное количество слов-объектов для задания, 
        # то для задания берём найденное количество слов-объектов
        num_missing_words = len(answers_options) if len(answers_options) < num_missing_words else num_missing_words        

        answers_options = dict(random.sample(list(answers_options.items()), k=num_missing_words))
        # Сортируем по порядку следования токенов в предложении
        answers_options = dict(sorted(answers_options.items(), key=lambda x: x[0].i))

        # Формируем предложение с пропусками вместо выбранных для задания слов
        sent = ''
        answers_list = list(answers_options.keys())
        for token in doc: 
            if token in answers_list:
                sent += '__'
                sent += token.whitespace_
            else:
                sent += token.text_with_ws
            
        answers = [token.text for token in list(answers_options.keys())] 
        options = list(answers_options.values())     
        
        return pd.Series([options, answers, sent], index=['options', 'answer', 'output_text'])
        
        

    # Задание: выбрать верную форму глагола из предложенных.
    # В качестве объектов (выбранные для задания слова) функция может выбрать 
    # вспосомагетльные глаголы (AUX), например, is, has, will, should) и глаголы (VERB).
    # - num_words_to_choose - количество вариантов ответа, помимо верного слова.
    # - num_missing_words - количество заданий в предложении, в данном случае равно количеству пропущенных слов. Количество заданий 
    #       в предложении зависит от количества слов в нём: на каждые 10 слов в предложении приходится 1 задание,
    #       например, если слов в предложении 7, то задание будет всего 1, если слов 15, то заданий 2, если 40, то 4, при условии, 
    #       что такое количество возможных слов-объектов среди глаголов, наречий, предлогов найдётся в предложении, 
    #       иначе количество заданий будет равно количеству найденных слов-объектов.
    def select_verb_by_form(self, sentence):
        num_missing_words = int(round(get_number_of_words(sentence), -1) / 10)        
        num_words_to_choose = 2

        answers_options = {}
        doc = self.nlp(sentence)

        for token in doc:
            if token.pos_ in ['AUX', 'VERB']:
                if token.pos_ == 'AUX': 
                    opt_word = []
                    opt_word.append(token._.inflect('VB'))
                    opt_word.append(token._.inflect('VBZ'))
                    opt_word.append(token._.inflect('VBP'))
                    opt_word.append(token._.inflect('VBD'))
                    opt_word.append(token._.inflect('MD'))
                elif token.pos_ == 'VERB':
                    opt_word = []
                    opt_word.append(token._.inflect('VB'))
                    opt_word.append(token._.inflect('VBP'))
                    opt_word.append(token._.inflect('VBZ'))
                    opt_word.append(token._.inflect('VBG'))
                    opt_word.append(token._.inflect('VBD'))
                    opt_word.append(token._.inflect('VBN'))


                # Оставляем только уникальные значения списка
                opt_word = list(set(opt_word))   
                # У глагола могут существовать не все указанные формы, поэтому проверям, что в списке нет None значений
                opt_word = list(filter(lambda x: x is not None, opt_word))

                opt_word = [w.title() if token.text.istitle() else w for w in opt_word] 

                # Проверяем, что исходное слово не входит в список вариантов
                if token.text in opt_word:
                    opt_word.remove(token.text)

                # Если список пустой, переходим к следующей итерации цикла for
                if len(opt_word) == 0:
                    continue

                opt_word = random.sample(opt_word, k=num_words_to_choose) + [token.text]

                random.shuffle(opt_word)
                answers_options[token] = opt_word

        # Если количество найденных слов-объектов в предложении меньше, чем расчитанное количество слов-объектов для задания, 
        # то для задания берём найденное количество слов-объектов
        num_missing_words = len(answers_options) if len(answers_options) < num_missing_words else num_missing_words

        answers_options = dict(random.sample(list(answers_options.items()), k=num_missing_words))
        # Сортируем по порядку следования токенов в предложении
        answers_options = dict(sorted(answers_options.items(), key=lambda x: x[0].i))
        
        # Формируем предложение с пропусками вместо выбранных для задания слов
        sent = ''
        answers_list = list(answers_options.keys())
        for token in doc: 
            if token in answers_list:
                sent += '__'
                sent += token.whitespace_
            else:
                sent += token.text_with_ws

        answers = [token.text for token in list(answers_options.keys())] 
        options = list(answers_options.values())
        
        return pd.Series([options, answers, sent], index=['options', 'answer', 'output_text'])  
    
    
    
    # Задание: выбрать верный артикль.
    # В качестве объектов (выбранные для задания слова) функция рассматривает только артикли "a", "an", "the".
    def select_articles(self, sentence):
        articles = ['a', 'an', 'the']

        answers = []
        options = []
        doc = self.nlp(sentence)

         # Формируем предложение с пропусками вместо выбранных для задания слов и добавляем эти слова в список ответов
        sent = ''
        for token in doc:
            if token.text in articles:
                answers.append(token.text)
                options.append(articles)
                sent += '___'
                sent += token.whitespace_
            else:
                sent += token.text_with_ws
        
        return pd.Series([options, answers, sent], index=['options', 'answer', 'output_text'])
    
    
    
    # Задание: выбрать подходящее по смыслу неопределённое местоимение.
    # В качестве объектов (выбранные для задания слова) функция рассматривает только неопределённые местоимения местоимения 
    # num_missing_words - количество заданий в предложении, в данном случае равно количеству пропущенных слов. Количество заданий 
    #     в предложении зависит от количества слов в нём: на каждые 10 слов в предложении приходится 1 задание,
    #     например, если слов в предложении 7, то задание будет всего 1, если слов 15, то заданий 2, если 40, то 4, при условии, 
    #     что такое количество возможных слов-объектов среди глаголов, наречий, предлогов найдётся в предложении, 
    #     иначе количество заданий будет равно количеству найденных слов-объектов.   
    def select_indefinite_pronouns(self, sentence):
        num_missing_words = int(round(get_number_of_words(sentence), -1) / 10)

        selection_list = [] 
        doc = self.nlp(sentence)
                
        pronouns = {'something': ['somebody', 'someone', 'anything', 'nothing', 'everything'],
                    'somebody': ['something', 'anybody', 'nobody', 'everybody'],
                    'someone': ['something', 'anyone', 'no one', 'everyone'],
                    'anything': ['anybody', 'anyone', 'something', 'nothing', 'everything'],
                    'anybody': ['anything', 'somebody', 'nobody', 'everybody'],
                    'anyone': ['anything', 'someone', 'no one', 'everyone'],
                    'nothing': ['nobody', 'no one', 'something', 'anything', 'everything'],
                    'nobody': ['nothing', 'somebody', 'anybody', 'everybody'],
                    'no one': ['nothing', 'someone', 'anyone', 'everyone'],
                    'everything': ['everybody', 'everyone', 'something', 'anything', 'nothing'],
                    'everybody': ['everything', 'somebody', 'anybody', 'nobody'],
                    'everyone': ['everything', 'someone', 'anyone', 'no one']
                   }

        for token in doc:        
            if token.lower_ in pronouns.keys() or (token.lower_ == 'no' and token.i < len(doc)-1 and token.nbor(1).lower_ == 'one'):
                selection_list.append(token.i)

        num_missing_words = len(selection_list) if len(selection_list) < num_missing_words else num_missing_words
        selection_list = random.sample(selection_list, k=num_missing_words)   
        
        # Формируем предложение с пропусками вместо выбранных для задания слов и добавляем эти слова в список ответов
        answers = []
        options = []
        sent = ''
        for token in doc:
            if token.i in selection_list:
                if token.lower_ == 'no' and token.i < len(doc)-1 and token.nbor(1).text == 'one':                    
                    pron_lower = token.lower_ + ' ' + token.nbor(1).text
                    pron_real = token.text_with_ws + token.nbor(1).text
                    answers.append(pron_real)
                else:
                    pron_lower = token.lower_
                    pron_real = token.text
                    answers.append(pron_real)
                    
                opt_word = random.sample(pronouns[pron_lower], k=2)
                opt_word = [w[0].title() + w[1:] if token.text.istitle() else w for w in opt_word]
                opt_word += [pron_real]
                random.shuffle(opt_word)
                options.append(opt_word)
                sent += '___'
                sent += token.whitespace_
            else:
                if token.lower_ == 'one' and token.i > 0 and token.nbor(-1).lower_ == 'no':
                    pass
                else:
                    sent += token.text_with_ws

        return pd.Series([options, answers, sent], index=['options', 'answer', 'output_text'])



    # Задание: написать пропущеный предлог (учитывая регистр).
    # В качестве объектов (выбранные для задания слова) функция рассматривает только предлоги (ADP)
    # - num_missing_words - количество заданий в предложении, в данном случае равно количеству пропущенных слов. Количество заданий 
    #       в предложении зависит от количества слов в нём: на каждые 10 слов в предложении приходится 1 задание,
    #       например, если слов в предложении 7, то задание будет всего 1, если слов 15, то заданий 2, если 40, то 4, при условии, 
    #       что такое количество возможных слов-объектов среди глаголов, наречий, предлогов найдётся в предложении, 
    #       иначе количество заданий будет равно количеству найденных слов-объектов.
    def write_preposition(self, sentence):
        num_missing_words = int(round(get_number_of_words(sentence), -1) / 10)

        selection_list = [] 
        doc = self.nlp(sentence)

        for token in doc:        
            if token.pos_ in ['ADP']: 
                selection_list.append(token.i)

        num_missing_words = len(selection_list) if len(selection_list) < num_missing_words else num_missing_words
        selection_list = random.sample(selection_list, k=num_missing_words)   

        # Формируем предложение с пропусками вместо выбранных для задания слов и добавляем эти слова в список ответов
        answers = []
        sent = ''
        for token in doc:
            if token.i in selection_list:
                answers.append(token.text)
                sent += '___'
                sent += token.whitespace_
            else:
                sent += token.text_with_ws

        options = []
        
        return pd.Series([options, answers, sent], index=['options', 'answer', 'output_text'])
    
    
    
    # Задание: раскрыть скобки и написать глагол в верной временной форме (учитывая регистр).
    # - num_missing_words - количество заданий в предложении, в данном случае равно количеству пропущенных слов. Количество заданий 
    #       в предложении зависит от количества слов в нём: на каждые 10 слов в предложении приходится 1 задание,
    #       например, если слов в предложении 7, то задание будет всего 1, если слов 15, то заданий 2, если 40, то 4, при условии, 
    #       что такое количество возможных слов-объектов среди глаголов, наречий, предлогов найдётся в предложении, 
    #       иначе количество заданий будет равно количеству найденных слов-объектов.
    def write_verbs(self, sentence):
        num_missing_words = int(round(get_number_of_words(sentence), -1) / 10) 

        selection_dict = {}
        doc = self.nlp(sentence)

        for token in doc:     
            if token.pos_ == 'VERB':            
                child_list = []           
                child_contains_neg = False
                child_contains_modal = ''
                child_contains_neg = ''
                for child in token.children:
                    if child.dep_ in ['aux', 'auxpass', 'neg']:
                        child_list.append(child.i)
                        if child.dep_ == 'neg':
                            child_contains_neg = child.lemma_
                            #child_contains_neg = True
                        if child.tag_ == 'MD' and child.lemma_ != 'will' and child.lemma_ != 'shall':
                            child_contains_modal = child.lemma_

                selection_dict[token.i] = {
                    # Есть ли в зависимыс словах глагола отрицание, например, частица "not" или слово "never"
                    'child_contains_neg': child_contains_neg, 
                    # Модальный глагол, если он есть среди зависимых слов глагола
                    'child_contains_modal': child_contains_modal,
                    # Список индекстов зависимых слов глагола, если они являются всмомогательными глаголами или частицей "not"
                    'child_list': child_list,
                    # Список индекстов всех слов, которые войдут в ответ и будут являться объектами задания. 
                    # Включает в себя глагол, всмомогательные глаголы или частицу "not" при их наличии, 
                    # а также подлежащее, относящееся к выбранному сказуемому (глаголу), если оно находится между
                    # вспомогательным и основным глаголами
                    'object_list': list(range(min(child_list + [token.i]), max(child_list + [token.i]) + 1))
                }    

        num_missing_words = len(selection_dict) if len(selection_dict) < num_missing_words else num_missing_words
        selection_dict = dict(random.sample(list(selection_dict.items()), k=num_missing_words))

        # Добавляем индексы всех пропущенных слов в один список
        ind_lost_word = []
        for elem in selection_dict.items():
            ind_lost_word.append(elem[0])
            ind_lost_word += elem[1]['child_list']   

        # Добавляем в один список индексы всех слов, которые будут являться объектами и верными ответами в этом задании
        ind_obj_word = []
        for elem in selection_dict.items():
             ind_obj_word += elem[1]['object_list']       


        # Формируем предложение с пропусками вместо выбранных для задания слов и добавляем эти слова в список ответов
        answers = []
        answer_str = ''
        sent = ''
        for token in doc:
            if token.i in ind_obj_word:
                answer_str += token.text
                answer_str += token.whitespace_

                if token.i in ind_lost_word:
                    sent += '___'
                    sent += token.whitespace_
                else:
                    sent += token.text_with_ws

                if token.i in selection_dict.keys():
                    answers.append(answer_str if answer_str[-1] != ' ' else answer_str[:-1])
                    answer_str = ''

                    if selection_dict[token.i]['child_contains_modal'] != '':
                        sent += f"({selection_dict[token.i]['child_contains_modal']}, "
                        if selection_dict[token.i]['child_contains_neg'] != '':
                            sent += f"{selection_dict[token.i]['child_contains_neg']}, {token.lemma_})"
                            #sent += f'not, {token.lemma_})'
                        else:
                            sent += f'{token.lemma_})'
                    else:                   
                        if selection_dict[token.i]['child_contains_neg'] != '':
                            sent += f"({selection_dict[token.i]['child_contains_neg']}, {token.lemma_})"
                            #sent += f'(not, {token.lemma_})'
                        else:
                            sent += f'({token.lemma_})'
                    sent += token.whitespace_
            else:
                sent += token.text_with_ws

        options = []
        
        return pd.Series([options, answers, sent], index=['options', 'answer', 'output_text'])
    
    
    
    # Задание: выбрать правильную последовательность слов в предложении.
    def correct_word_order(self, sentence):
        num_word = 0
        split_sent = []
        part_sent = ''

        doc = self.nlp(sentence) 

        for token in doc:
            part_sent += token.text_with_ws
            num_word += 1        

            if num_word == 3 and token.i < len(doc) - 2 and token.nbor(1).pos_ != 'PUNCT' or num_word == 4:
                split_sent.append(part_sent)
                num_word = 0
                part_sent = ''
            elif token.i == len(doc) - 1:
                split_sent.append(part_sent)

        answers = split_sent.copy()
        random.shuffle(split_sent)
        options = split_sent
        sent = ''
        
        return pd.Series([options, answers, sent], index=['options', 'answer', 'output_text'])

    
    # Функция для создания упражнений исходя из присвоенных этим придложениям типов 
    def generating_tasks(self):
        if self.used_task_type['select_word_by_meaning'] == 1:
            self.df.loc[(self.df['type_task'] == 'select_word_by_meaning'), ['options', 'answer', 'output_text']] = \
            self.df[self.df['type_task'] == 'select_word_by_meaning']['source_text'].apply(self.select_word_by_meaning)

        if self.used_task_type['select_verb_by_form'] == 1:
            self.df.loc[(self.df['type_task'] == 'select_verb_by_form'), ['options', 'answer', 'output_text']] = \
            self.df[self.df['type_task'] == 'select_verb_by_form']['source_text'].apply(self.select_verb_by_form)

        if self.used_task_type['select_articles'] == 1:
            self.df.loc[(self.df['type_task'] == 'select_articles'), ['options', 'answer', 'output_text']] = \
            self.df[self.df['type_task'] == 'select_articles']['source_text'].apply(self.select_articles)
            
        if self.used_task_type['select_indefinite_pronouns'] == 1:
            self.df.loc[(self.df['type_task'] == 'select_indefinite_pronouns'), ['options', 'answer', 'output_text']] = \
            self.df[self.df['type_task'] == 'select_indefinite_pronouns']['source_text'].apply(self.select_indefinite_pronouns) 

        if self.used_task_type['write_verbs'] == 1:
            self.df.loc[(self.df['type_task'] == 'write_verbs'), ['options', 'answer', 'output_text']] = \
            self.df[self.df['type_task'] == 'write_verbs']['source_text'].apply(self.write_verbs)

        if self.used_task_type['write_preposition'] == 1:
            self.df.loc[(self.df['type_task'] == 'write_preposition'), ['options', 'answer', 'output_text']] = \
            self.df[self.df['type_task'] == 'write_preposition']['source_text'].apply(self.write_preposition)

        if self.used_task_type['correct_word_order'] == 1:
            self.df.loc[(self.df['type_task'] == 'correct_word_order'), ['options', 'answer', 'output_text']] = \
            self.df[self.df['type_task'] == 'correct_word_order']['source_text'].apply(self.correct_word_order)
            
        return self.df