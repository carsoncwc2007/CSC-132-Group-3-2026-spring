import os 
from datetime import datetime

class SignTranslator:
    def __init__(self):
        """initializes the sign translator with word dictionary"""
        self.word_dictionary=self._load_dictionary()
        self.transcripts=[]
    def _load_dictionary(self):
        """load ASL word dictionary
        format:{"HELLO":"hello, how are you?","THANK":"thank you",...}
        can be expanded with more words and phrases
        """
        return{
            #common single words
            "HELLO":"hello",
            "GOODBYE":"goodbye",
            "THANK":"thank you",
            "PLEASE":"please",
            "YES":"yes",
            "NO":"no",
            "SORRY":"i'm sorry",
            "GOOD":"good",
            "BAD":"bad",
            "WATER":"water",
            "FOOD":"food",
            "HELP":"help",
            "LOVE":"love",
            "FAMILY":"family",
            "FRIEND":"friend",
            }
    def letter_sequence_to_words(self,letter_sequence):
        """convert letter sequence to words using dictionary lookup
        Args: letter_sequence (str): sequence of predicted letters, e.g. "H E L L O"
        returns:
        list of words with confidence
        """
        words=letter_sequence.split()
        translated_words=[]
        for word in words:
            if word in self.word_dictionary:
                translated_words.append({
                    "original": word,
                    'translation': self.word_dictionary[word]
                    "found":true
                })
            else:
                translated_words.append({
                    "original": word,
                    "translation": word,#fallback to original
                    "found":False
                })
        return translated_words
    def apply_spell_correction(self,letter_sequence):
        """basic spell correction using directionary 
        handles common confusion pairs from confusion matrix
        args:
        letter_sequence:detected letter sequence
        returns:
        corrected letter sequence
        """
        #common confusion pairs based on confusion matrix analysis
        corrections={
            'L':'i',
            'o':'0',
            '1':'i',
        }
        return letter_sequence
    def format_transcript(self,letter_sequence):
        """
        Format transcript with proper punction and capialization

        Args:
            letter_sequence:raw detected letter sequence
        Returns:
        formatted text 
        """
        formatted=letter_sequence.strip()
        if formatted:
            formatted=formatted[0].upper() + formatted[1:].lower()
        if not formatted.endswith('.','!','?'):
            formatted += '.'
        return formatted
    def displaty_transcript(self,letter_sequence):
        """display formatted transcript"""
        print("\n"+ "="*60)
        print("translation results:")
        print("="*60)
        print(f"raw letter sequence: {letter_sequence}")
        formatted=self.format_transcript(letter_sequence)
        print(f"formatted transcript: {formatted}")
        print(f"\n word by word")
        translated=self.letter_sequence_to_words(letter_sequence)
        for item in translated:
            status="(found)" if item['found'] else "(not found)"
            print(f"{item['original']} -> {item['translation']} {status}")
        print("="*60+"\n")
if __name__ == "__main__":
    translator=SignTranslator()
