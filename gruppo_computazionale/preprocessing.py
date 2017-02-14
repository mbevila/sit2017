import csv
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import re


def parse_paisa(source_fn, target_fn, cols=[2], joiner="|", forbidden_starts=["#","\n","<"]):
    """
    La funzione prende quattro argomenti:
    
      :source_fn:str
          il file di cui fare il parsing
          
      :target_fn:str  
          il file da creare, in cui il corpus sarà trasformato
          nella forma (ogni riga corrisponde a una sentence):
              word word word
              word word word word
          oppure:
              lemma lemma lemma
              lemma lemma
          oppure:
              word|pos word|pos word|pos
              word|pos word|pos
          a seconda del valore assegnato a cols.
      
      :cols:list[int]  (default=[2])
          le colonne da cui prendere i dati
      
      :joiner:str     
          in caso len(cols) > 1 i dati saranno aggregati con
          joiner.join(data). Per evitare per ogni dato sarà chiamato
          data.replace(joiner, "")
      
      :forbidden_starts:list[str]  (default=["#","\n","<"])
          skippa una riga se comincia con le stringhe seguenti
          
    """
    
    cols = set(cols)
    
    source = open(source_fn)
    target = open(target_fn, "w")

    readr = csv.reader(
        (line for line in source if not any(line.startswith(x) for x in forbidden_starts)), 
        delimiter="\t",
    )

    previous_place_in_sentence = 0
    for row in readr:
        token = joiner.join([col.replace(joiner, "") for i, col in enumerate(row) if i in cols]) 
        place_in_sentence = int(row[0])
        if place_in_sentence < previous_place_in_sentence:
            target.write("\n")
        target.write(token + " ")
        previous_place_in_sentence = place_in_sentence

    source.close(); target.close()
    
    
    
class CorpusIterator():
    """
    La classe all'inizializzazione prende i seguenti argomenti:
    Args:
    
     :fn:str    
            il file contenente il corpus
            
     :tokenizer:func  (default=lambda x: [token for token in x.split()])
            la funzione da utilizzare per tokenizzare ciascuna linea nel corpus

     :validation_function:None or func  (default=None)
            la funzione deve prendere come argomento una str e ritornare True o
            False. Nel caso ritorni False, la linea sarà saltata. Se 
            validation_function=None la procedura viene saltata.
            
     :remove_stopwords: (default="italian")
            l'argomento può essere sia False che una stringa contenente
            la lingua per la quale si vuole effettuare la rimozione delle 
            stopwords
            
     :joiner:str   (default="|")
            se più livelli di annotazione sono stati aggregati in un'unica string
            (es: "maiale|maiale|NOUN"), inserire il carattere utilizzato come
            separatore. Importante quando si vuole usare lo stopword filtering.
 
     :stopword_index:int   (default=0)
            se più livelli di annotazione sono stati aggregati in un'unica string
            (es: "maiale|maiale|NOUN"), inserire l'indice del livello di annotazione
            per cui si vuole usare lo stopword filtering.
     
     :remove_chars:str (default=r"[^a-zA-ZÀÈÉÌÒÙàèéìòù |]")
            l'argomento può essere sia False che una stringa (preferibilmente
            una rawstring) contenente una regex. Su ogni riga del corpus è
            chiamato re.sub(remove_chars, "", riga). Fare attenzione a non
            rimuovere il carattere joiner dei livelli di annotazione!
            
     :remove_identical_lines:bool  (default=True)
            se True per ogni riga nel corpus viene chiamato hash(riga), e il
            risultato aggiunto a un set self.hash_set. Se l'hash è gia presente
            nel set la riga viene saltata.
    
    una volta inizializzata la classe può essere iterata - e dunque passata 
    direttamente come argomento a gensim.models.Word2Vec(), senza mai caricare 
    in memoria l'intero corpus.
    Es:
        sentences = RAMEasyCorpusLoader("pincopallino.txt")
        model = Word2Vec(sentences, size=300, window=5)
    """
    
    def __init__(self, fn:str,
                 tokenizer=lambda x: [token.lower() for token in x.split()],
                 validation_function=None,
                 remove_stopwords="italian",
                 joiner="|",
                 stopword_index=0,
                 remove_chars=r"[^a-zA-ZÀÈÉÌÒÙàèéìòù |]",
                 remove_identical_lines=True):
        
        self.fn = fn
        if remove_chars:
            self.remove_chars = re.compile(remove_chars) #regex compilata per maggiore velocità
            
        if remove_stopwords:
            self.stopwords = set(stopwords.words("italian") + [""]) # operatore in più veloce così
        else:                                           
            self.stopwords = set()
        self.remove_identical_lines = remove_identical_lines
        self.__create_hash_set()
        self.tokenizer = tokenizer
        self.joiner = joiner
        self.stopword_index = stopword_index
 
    def __iter__(self):
        
        return self.__generator()
    
    
    def __create_hash_set(self):
        
        if self.remove_identical_lines:
            self.hash_set = set()
        else:
            pass
    
    
    def __generator(self):
        
        self.__create_hash_set()
        for line in open(self.fn):
            if self.is_valid(line):
                line = self.pipeline(line)
                if (bool(line) != False) and (isinstance(line[0], str)):
                    yield line
    
    def is_valid(self, string:str):
        
        if self.remove_identical_lines:
            hash_ = hash(string)
            if hash_ in self.hash_set:
                return False
            else:
                self.hash_set.add(hash_)
                return True
        elif bool(self.validation_function):
            return self.validation_function(string)
        else:
            return True
        
    
    def tokenize(self, string:str):
        
        tokens = []
        for token in self.tokenizer(string):
            if bool(token) and (token.split(self.joiner)[self.stopword_index] not in self.stopwords):
                tokens.append(token)
        return tokens
    
    
    def pipeline(self, string:str):
        if self.remove_chars:
            string = self.remove_chars.sub("", string)

        return self.tokenize(string)
