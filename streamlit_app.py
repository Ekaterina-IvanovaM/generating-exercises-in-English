import streamlit as st
from io import StringIO
from class_genTasksEng import genTasksEng


def click_button(key, value):
    st.session_state[key] = value
    
            
def check_answer(user_answer, right_answer, type_task):
    if type_task == 'correct_word_order':
        if len(user_answer) < len(right_answer):
            pass
        elif user_answer == right_answer:
            st.success('', icon="✅")
        else:
            st.error('', icon="❌")
    elif 'select' in type_task:
        if user_answer == '–––':
            pass
        elif user_answer == right_answer:
            st.success('', icon="✅")
        else:
            st.error('', icon="❌")   
    elif 'write' in type_task:
        if user_answer == '':
            pass                           
        elif user_answer == right_answer:
            st.success('', icon="✅")
        else:
            st.error('', icon="❌")            


st.title('Генератор упражнений по английскому языку')
        
with st.sidebar:
    st.subheader('Загрузка текста для создания упраженений')

    str_file = st.file_uploader('Выберете и загрузите файл в формате TXT', type=['txt'])
    str_text = st.text_area('или вставьте текст в поле ниже')

    st.subheader('Настройка параметров генерации упражнений')
    st.caption('Выберете типы упражнений, с которыми хотите поработать.')
    select_word_by_meaning = int(st.checkbox('Выбрать подходящеее по смыслу слово'))
    select_verb_by_form = int(st.checkbox('Выбрать глагол в правильной форме'))
    select_articles = int(st.checkbox('Выбрать верный артикль'))
    select_indefinite_pronouns = int(st.checkbox('Выбрать подходящее по смыслу неопределённое местоимение'))
    write_adp_word = int(st.checkbox('Написать верный предлог'))
    write_verbs = int(st.checkbox('Раскрыть скобки и написать глагол в правильной форме'))
    correct_word_order = int(st.checkbox('Выбрать правильную последовательность слов в предложении'))

    st.caption('Укажите процент предложений, в которых должны быть задания. Учтите, что в одном предложении может быть несколько ' + 
               'заданий. Фактический процент предложений с заданиями может быть меньше указанного, так как некоторые предложения ' +
               'не подходят для генерации в них заданий.')
    percent_used_sent = st.slider('Процент предложений с заданиями', 10, 100, step=5) / 100
  
    gen_tasks_button = st.button('Сгенерировать упраженения')
    

    
text_not_uploaded = str_file is None and len(str_text) == 0
exercises_not_selected = select_word_by_meaning == 0 and select_verb_by_form == 0 and select_articles == 0 and \
    select_indefinite_pronouns == 0 and write_adp_word == 0 and write_verbs == 0 and correct_word_order == 0
    
if gen_tasks_button and text_not_uploaded:
    st.subheader('**:red[Для создания упражнений загрузите файл или вставьте текст в поле на боковой панеле]**')
    
elif gen_tasks_button and exercises_not_selected:
    st.subheader('**:red[Выберете хотя бы один тип упражнения на боковой панеле]**')
    
elif gen_tasks_button: 
    with st.spinner('Идёт генерация упражнений...'):
        if str_file is not None:
            text = StringIO(str_file.getvalue().decode("utf-8")).read()
        elif len(str_text) > 0:
            text = str_text

        used_task_type = {
            'select_word_by_meaning': select_word_by_meaning, 
            'select_verb_by_form': select_verb_by_form,
            'select_articles': select_articles,
            'select_indefinite_pronouns': select_indefinite_pronouns,
            'write_preposition': write_adp_word,
            'write_verbs': write_verbs,
            'correct_word_order': correct_word_order
        }
        
        
        for key in st.session_state.keys():
            del st.session_state[key]
        
        class_genTasksEng = genTasksEng(text, used_task_type, percent_used_sent)
        class_genTasksEng.set_task_type()
        st.session_state.df = class_genTasksEng.generating_tasks()
         

    
if 'df' in st.session_state:
    for ind, row in st.session_state.df.iterrows():
               
        if row['type_task'] == 'no_tasks':
            st.write(row['source_text'])
        else:
            if ind > 0 and st.session_state.df.loc[ind-1, 'type_task'] == 'no_tasks':
                st.write('---')
                
            col1, col2, col3 = st.columns([0.25, 0.35, 0.4])

            with col1:
                if row['type_task'] == 'select_word_by_meaning':
                    st.write('Выберете подходящее по смылу слово')
                elif row['type_task'] == 'select_verb_by_form':
                    st.write('Выберете правильную форму глагола')
                elif row['type_task'] == 'select_articles':
                    st.write('Выберете правильный артикль')
                elif row['type_task'] == 'select_indefinite_pronouns':
                    st.write('Выберете подходящее по смыслу неопределённое местоимение')
                elif row['type_task'] == 'write_preposition':
                    st.write('Напишите верный предлог')
                elif row['type_task'] == 'write_verbs':
                    st.write('Раскройте скобки и напишите глагол в правильной форме')
                elif row['type_task'] == 'correct_word_order':
                    st.write('Выберете правильную последовательность слов в предложении')
                             
            with col3:                
                if row['type_task'] == 'correct_word_order':
                    widget_key = 'co' + str(ind)
                    options_correct_order = st.multiselect('nolabel', 
                                                           row['options'], 
                                                           key=widget_key, 
                                                           label_visibility="collapsed")
                    check_answer(options_correct_order, row['answer'], row['type_task'])                    
                    st.button('Показать ответ', key='but'+widget_key, on_click=click_button, args=(widget_key, row['answer']))
                        
                elif 'select' in row['type_task']:
                    for i in range(len(row['answer'])):
                        widget_key = ind*10 + i
                        answer = st.selectbox('nolabel', ['–––'] + row['options'][i], key=widget_key, label_visibility="collapsed")
                        check_answer(answer, row['answer'][i], row['type_task'])
                            
                elif 'write' in row['type_task']:
                    for i in range(len(row['answer'])):
                        widget_key = ind*10 + i
                        if row['type_task'] == 'write_verbs':
                            answer = st.text_input(' ', key=widget_key, label_visibility="visible",
                                                  help='Если ответ содержит вспомогательный глагол, и между ним и основным глаголом ' + 
                                                   'есть другие слова, например, подлежащее, то ответ должен их включать.')
                        else:
                            answer = st.text_input('nolabel', key=widget_key, label_visibility="collapsed")
                            
                        check_answer(answer, row['answer'][i], row['type_task']) 
                        st.button('Показать ответ', key='but'+str(widget_key), on_click=click_button, 
                                  args=(str(widget_key), row['answer'][i])) 
            
            with col2:               
                if row['type_task'] == 'correct_word_order':
                    st.write(''.join(options_correct_order))                   
                else:
                    st.write(row['output_text'])                
            
            st.write('---')