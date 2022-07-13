
### Вектора и эмбединги для текстов

- Текст на естественном языке, который нужно обрабатывать в задачах машинного обучения, сильно зависит от источника. 
- В связи с этим, возникает задача предобработки (или нормализации) текста, то есть приведения к некоторому единому виду.

#### 1. Приведение текста к нижнему регистру

```python
text = 'Способность эффективно обрабатывать данные с  \n большим числом признаков и классов... '
```

```python
text = text.lower()
text
```

```
'способность эффективно обрабатывать данные с  \n большим числом признаков и классов... '
```

#### 2. Удаление неинформативных символов

```python
import re

regex = re.compile(r'(\W)\1+')
regex.sub(r'\1', text)
```

```
'способность эффективно обрабатывать данные с \n большим числом признаков и классов. '
```

```python
regex = re.compile(r'[^\w\s]')
text=regex.sub(r' ', text).strip()
```

```
'способность эффективно обрабатывать данные с  \n большим числом признаков и классов'
```

```python
re.sub('\s+', ' ', text)
```

```
'способность эффективно обрабатывать данные с большим числом признаков и классов'
```
#### 3. Разбиение текста на смысловые единицы (токенизация)

- Самый простой подход к токенизации
  - это разбиение по текста по **пробельным** символам

```python
text = 'Купите кружку-термос на 0.5л (64см³) за 3 рубля. До 01.01.2050.'
```

- (a) <code>.split()</code> подход
  - По дефолту использует **" "**

```python
text.split()
```

```
['Купите',
 'кружку-термос',
 'на',
 '0.5л',
 '(64см³)',
 'за',
 '3',
 'рубля.',
 'До',
 '01.01.2050.']
 ```

- (b) Библиотека <code>ntlk</code>
  - В библиотеке для морфологического анализа для русского языка
  - >code>pymorphy2</code> есть простая вспомогательная функция <code>simple_word_tokenize</code> для токенизации

```python
from pymorphy2.tokenizers import simple_word_tokenize

simple_word_tokenize(text)

```

```
['Купите',
 'кружку-термос',
 'на',
 '0',
 '.',
 '5л',
 '(',
 '64см³',
 ')',
 'за',
 '3',
 'рубля',
 '.',
 'До',
 '01',
 '.',
 '01',
 '.',
 '2050',
 '.']
 ```
 
- (c) Библиотека <code>ntlk</code>
  - Более сложной метод токенизации представлен в <code>nltk</code> (общего NLP)
  - Используем метод <code>sent_tokenize</code>

```python
from nltk import sent_tokenize, word_tokenize, wordpunct_tokenize

sentences=sent_tokenize(text)
```

```python
[word_tokenize(sentence) for sentence in sentences]
```

```
[['Купите',
  'кружку-термос',
  'на',
  '0.5л',
  '(',
  '64см³',
  ')',
  'за',
  '3',
  'рубля',
  '.'],
 ['До', '01.01.2050', '.']]
 ```

```
[wordpunct_tokenize(sentence) for sentence in sentences]
```

```
[['Купите',
  'кружку',
  '-',
  'термос',
  'на',
  '0',
  '.',
  '5л',
  '(',
  '64см³',
  ')',
  'за',
  '3',
  'рубля',
  '.'],
 ['До', '01', '.', '01', '.', '2050', '.']]
 ```
 
 - (d) Библиотека <code>razdel</code>
   - Для русского языка также есть новая специализированная библиотека <code>razdel</code>
 
 ```python
 import razdel

sents=[]
for sentence in razdel.sentenize(text):
    sents.append(sentence.text)
    
sents
```

```
['Купите кружку-термос на 0.5л (64см³) за 3 рубля.', 'До 01.01.2050.']
```

```python
sentences = [sentence.text for sentence in razdel.sentenize(text)]
tokens = [ [token.text for token in razdel.tokenize(sentence)] for sentence in sentences ]
```

```
[['Купите',
  'кружку-термос',
  'на',
  '0.5',
  'л',
  '(',
  '64',
  'см³',
  ')',
  'за',
  '3',
  'рубля',
  '.'],
 ['До', '01.01.2050', '.']]
```

```python
razdel.tokenize
```

```
TokenSegmenter(TokenSplitter(),
               [DashRule(),
                UnderscoreRule(),
                FloatRule(),
                FractionRule(),
                FunctionRule('punct'),
                FunctionRule('other'),
                FunctionRule('yahoo')])
```

```python
import razdel

def tokenize_with_razdel(text):
    sentences = [sentence.text for sentence in razdel.sentenize(text)]
    tokens = [ [token.text for token in razdel.tokenize(sentence)] for sentence in sentences ]
    
    return tokens

tokenize_with_razdel(text)
```

```
[['Купите',
  'кружку-термос',
  'на',
  '0.5',
  'л',
  '(',
  '64',
  'см³',
  ')',
  'за',
  '3',
  'рубля',
  '.'],
 ['До', '01.01.2050', '.']]
 ```
 
