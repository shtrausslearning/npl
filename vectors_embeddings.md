
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

- Рассмотрим 4 способа:
  -  <code>.split()</code> подход
  -  Библиотека <code>pymorphy2</code> 
  -  Библиотека <code>ntlk</code> 
  -  Библиотека <code>razdel</code>

```python
text = 'Купите кружку-термос на 0.5л (64см³) за 3 рубля. До 01.01.2050.'
```

- <code>.split()</code> подход
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

- Библиотека <code>pymorphy2</code>
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
 
- Библиотека <code>ntlk</code>
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
 
 - Библиотека <code>razdel</code>
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
 
#### 4. Приведение слов к нормальной форме (стемминг/лемматизация)

- <code>Стемминг</code> - это нормализация слова путём отбрасывания окончания по правилам языка
  - Такая нормализация хорошо подходит для языков с небольшим разнообразием словоформ, (английского).
  - В библиотеке <code>nltk</code> есть несколько реализаций стеммеров:
    - Porter stemmer
    - Snowball stemmer
    - Lancaster stemmer
  
```python
from nltk.stem.snowball import SnowballStemmer
SnowballStemmer(language='english').stem('running')
```

```
'run'
```
  
- Для русского языка этот подход не очень подходит, поскольку в русском есть:
  - **падежные формы**, **время у глаголов** и т.д.
    
```python
SnowballStemmer(language='russian').stem('бежать')
```

```
'бежа'
```
   
- <code>Лемматизация</code> - приведение слов к начальной **морфологической форме** (с помощью **словаря** и **грамматики** языка)
  - Самый простой подход к лемматизации <code>словарный</code>.
  - Здесь не учитывается контекст слова, поэтому для омонимов такой подход работает не всегда.
  - Такой подход применяет библиотека <code>pymorphy2</code>

```
from pymorphy2 import MorphAnalyzer

pymorphy = MorphAnalyzer()
pymorphy.parse('бежал')
```

```
[Parse(word='бежал', tag=OpencorporaTag('VERB,perf,intr masc,sing,past,indc'), normal_form='бежать', score=0.5, methods_stack=((DictionaryAnalyzer(), 'бежал', 392, 1),)),
 Parse(word='бежал', tag=OpencorporaTag('VERB,impf,intr masc,sing,past,indc'), normal_form='бежать', score=0.5, methods_stack=((DictionaryAnalyzer(), 'бежал', 392, 49),))]
```

```python
def lemmatize_with_pymorphy(tokens):
    lemms = [pymorphy.parse(token)[0].normal_form for token in tokens]
    return lemms
```

```python
lemmatize_with_pymorphy(['бегут', 'бежал', 'бежите'])
```

```
['бежать', 'бежать', 'бежать']
```

```python
pymorphy.normal_forms('на заводе стали увидел виды стали')
```

```
['на заводе стали увидел виды стать', 'на заводе стали увидел виды сталь']
```

```python
lemmatize_with_pymorphy(['на', 'заводе', 'стали', 'увидел', 'виды', 'стали'])
```

```
['на', 'завод', 'стать', 'увидеть', 'вид', 'стать']
```

```python
pymorphy.parse('директора')
```

```
[Parse(word='директора', tag=OpencorporaTag('NOUN,anim,masc sing,gent'), normal_form='директор', score=0.632653, methods_stack=((DictionaryAnalyzer(), 'директора', 837, 1),)),
 Parse(word='директора', tag=OpencorporaTag('NOUN,anim,masc plur,nomn'), normal_form='директор', score=0.204081, methods_stack=((DictionaryAnalyzer(), 'директора', 837, 6),)),
 Parse(word='директора', tag=OpencorporaTag('NOUN,anim,masc sing,accs'), normal_form='директор', score=0.163265, methods_stack=((DictionaryAnalyzer(), 'директора', 837, 3),))]
 ```
 
 - Библиотека от Яндекса `mystem3` обходит это ограничение и рассматривает контекст слова, используя статистику и правила.
