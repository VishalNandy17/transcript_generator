import os
import streamlit as st
import sys
import json
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import wordnet, stopwords
import spacy
import pandas as pd
import numpy as np
import re
from random import shuffle, choice, sample
import base64
import logging
from typing import List, Dict, Union, Tuple
from youtube_transcript_api import YouTubeTranscriptApi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create data directory for NLTK
if not os.path.exists("nltk_data"):
    os.makedirs("nltk_data")

# Set NLTK data path
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))


class NLTKDownloader:
    """Handle NLTK resource downloads and verification."""
    
    @staticmethod
    def download_nltk_resources() -> None:
        """Download required NLTK resources with proper error handling."""
        resources = [
            'punkt',
            'averaged_perceptron_tagger',
            'maxent_ne_chunker',
            'words',
            'wordnet',
            'stopwords',
            'omw-1.4'
        ]
        
        for resource in resources:
            try:
                nltk.data.find(resource)
                logger.info(f"Resource {resource} already downloaded")
            except LookupError:
                try:
                    with st.spinner(f"Downloading {resource}..."):
                        nltk.download(resource, quiet=True, download_dir="nltk_data")
                    logger.info(f"Successfully downloaded {resource}")
                except Exception as e:
                    error_msg = f"Failed to download {resource}: {str(e)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

class TextPreprocessor:
    """Handle text preprocessing and cleaning operations."""
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess input text."""
        # Remove timestamps
        text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
        text = re.sub(r'\(\d{2}:\d{2}\)', '', text)
        
        # Remove speaker labels
        text = re.sub(r'[A-Za-z]+\s*:', '', text)
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        text = ' '.join(text.split())
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'\.+', '.', text)
        
        return text.strip()
    
    def extract_key_concepts(self, text: str) -> List[Dict]:
        """Extract key concepts and their importance from text."""
        doc = self.nlp(text)
        concepts = []
        
        # Extract named entities with context
        for ent in doc.ents:
            concepts.append({
                'text': ent.text,
                'type': ent.label_,
                'importance': self._calculate_importance(ent.text, doc),
                'context': self._get_context(ent, doc)
            })
        
        # Extract important noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:
                concepts.append({
                    'text': chunk.text,
                    'type': 'NOUN_PHRASE',
                    'importance': self._calculate_importance(chunk.text, doc),
                    'context': self._get_context(chunk, doc)
                })
        
        return sorted(concepts, key=lambda x: x['importance'], reverse=True)
    
    def _calculate_importance(self, text: str, doc) -> float:
        """Calculate importance score for a concept."""
        frequency = doc.text.count(text)
        length_score = len(text.split()) / 10
        return frequency * (1 + length_score)
    
    def _get_context(self, span, doc) -> str:
        """Get surrounding context for a span of text."""
        start = max(0, span.start - 5)
        end = min(len(doc), span.end + 5)
        return doc[start:end].text

class MCQGenerator:
    """Generate Multiple Choice Questions from text."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.question_patterns = [
            ("What is", "DEFINITION"),
            ("Who", "PERSON"),
            ("When did", "DATE"),
            ("Where is", "LOCATION"),
            ("Which", "CHOICE"),
            ("How does", "PROCESS")
        ]
    
    def generate_mcqs(self, text: str, num_questions: int) -> List[Dict]:
        """Generate MCQs from input text."""
        try:
            # Preprocess text
            clean_text = self.preprocessor.clean_text(text)
            concepts = self.preprocessor.extract_key_concepts(clean_text)
            sentences = sent_tokenize(clean_text)
            
            # Generate questions
            mcqs = []
            used_concepts = set()
            
            for concept in concepts:
                if len(mcqs) >= num_questions:
                    break
                
                if concept['text'] in used_concepts:
                    continue
                
                relevant_sentence = self._find_best_sentence(concept, sentences)
                if not relevant_sentence:
                    continue
                
                question = self._generate_question(concept, relevant_sentence)
                if not question:
                    continue
                
                distractors = self._generate_distractors(concept, concepts)
                if len(distractors) < 3:
                    continue
                
                mcq = self._create_mcq(question, concept['text'], distractors, relevant_sentence)
                mcqs.append(mcq)
                used_concepts.add(concept['text'])
            
            return mcqs
        
        except Exception as e:
            logger.error(f"Error generating MCQs: {str(e)}")
            raise
    
    def _find_best_sentence(self, concept: Dict, sentences: List[str]) -> str:
        """Find the most suitable sentence for creating a question."""
        relevant_sentences = [s for s in sentences if concept['text'] in s]
        if not relevant_sentences:
            return None
        
        scored_sentences = [
            (s, self._score_sentence(s, concept))
            for s in relevant_sentences
        ]
        return max(scored_sentences, key=lambda x: x[1])[0]
    
    def _score_sentence(self, sentence: str, concept: Dict) -> float:
        """Score a sentence's suitability for question generation."""
        length_score = 1 if 10 <= len(sentence.split()) <= 25 else 0.5
        context_score = 1 if concept['context'] in sentence else 0.5
        clarity_score = 1 if ',' in sentence or ';' in sentence else 0.8
        return length_score * context_score * clarity_score
    
    def _generate_question(self, concept: Dict, sentence: str) -> str:
        """Generate a question based on concept type and sentence."""
        question_type = concept['type']
        sentence = sentence.replace(concept['text'], "_____")
        
        for pattern, q_type in self.question_patterns:
            if question_type in q_type:
                return f"{pattern} {sentence}"
        
        return f"Which of the following correctly completes this statement: {sentence}"
    
    def _generate_distractors(self, concept: Dict, all_concepts: List[Dict], num_distractors: int = 3) -> List[str]:
        """Generate plausible distractors for the correct answer."""
        distractors = set()
        
        similar_concepts = [
            c['text'] for c in all_concepts 
            if c['type'] == concept['type'] and c['text'] != concept['text']
        ]
        
        if similar_concepts:
            distractors.update(sample(similar_concepts, min(2, len(similar_concepts))))
        
        for word in word_tokenize(concept['text']):
            if word.lower() not in self.preprocessor.stop_words:
                synsets = wordnet.synsets(word)
                for syn in synsets:
                    for lemma in syn.lemmas():
                        if lemma.name() != word:
                            distractors.add(lemma.name().replace('_', ' ').capitalize())
        
        context_words = [
            word for word in word_tokenize(concept['context'])
            if word.lower() not in self.preprocessor.stop_words
        ]
        if context_words:
            distractors.update(sample(context_words, min(2, len(context_words))))
        
        return list(distractors)[:num_distractors]
    
    def _create_mcq(self, question: str, answer: str, distractors: List[str], context: str) -> Dict:
        """Create MCQ dictionary with all required information."""
        options = [answer] + distractors
        shuffle(options)
        
        return {
            'question': question,
            'options': {chr(65 + i): opt for i, opt in enumerate(options)},
            'correct_answer': chr(65 + options.index(answer)),
            'context': context,
            'explanation': f"The correct answer is '{answer}' based on the context: {context}"
        }

class TranscriptGenerator:
    def __init__(self):
        self.language_map = {
            'English': 'en',
            'Hindi': 'hi',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it'
        }
    
    def extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from URL."""
        video_id_match = re.search(
            r'(?:v=|\/videos\/|embed\/|youtu.be\/|\/v\/|\/e\/|watch\?v%3D|watch\?feature=player_embedded&v=|%2Fvideos%2F|embed%\u200C\u200B2F|youtu.be%2F|%2Fv%2F)([^#\&\?\n]*)',
            url
        )
        return video_id_match.group(1) if video_id_match else None

    def generate_transcript(self, url: str, language: str) -> str:
        """Generate transcript from YouTube video."""
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=[self.language_map[language]]
            )
            
            transcript_text = ""
            for entry in transcript_list:
                timestamp = entry['start']
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                text = entry['text']
                transcript_text += f"[{minutes:02d}:{seconds:02d}] {text}\n"
            
            return transcript_text
            
        except Exception as e:
            raise Exception(f"Failed to generate transcript: {str(e)}")

# [Previous StreamlitUI class and main() remain the same...]
class StreamlitUI:
    def __init__(self):
        self.transcript_generator = TranscriptGenerator()
        self.mcq_generator = MCQGenerator()
        self.history = self.load_history()

    def load_history(self) -> List[Dict]:
        """Load transcript history from session state."""
        if 'history' not in st.session_state:
            st.session_state.history = []
        return st.session_state.history

    def save_to_history(self, entry: Dict) -> None:
        """Save transcript to history."""
        st.session_state.history.insert(0, entry)
        if len(st.session_state.history) > 100:
            st.session_state.history = st.session_state.history[:100]

    def render_transcript_tab(self):
        """Render transcript generator tab."""
        st.header("YouTube Transcript Generator")
        
        # URL input
        url = st.text_input("Video URL:", placeholder="Enter YouTube URL")
        
        # Options
        col1, col2, col3 = st.columns(3)
        with col1:
            language = st.selectbox("Language:", [
                'English', 'Hindi', 'Spanish', 'French', 'German', 'Italian'
            ])
        with col2:
            include_timestamps = st.checkbox("Include Timestamps", value=True)
        with col3:
            format_paragraphs = st.checkbox("Format as Paragraphs")
        
        if st.button("Generate Transcript", type="primary"):
            if not url:
                st.error("Please enter a YouTube URL")
                return
            
            try:
                with st.spinner("Generating transcript..."):
                    transcript = self.transcript_generator.generate_transcript(url, language)
                    
                    if format_paragraphs:
                        transcript = self.format_as_paragraphs(transcript)
                    if not include_timestamps:
                        transcript = re.sub(r'\[\d{2}:\d{2}\]\s*', '', transcript)
                    
                    st.session_state.current_transcript = transcript
                    
                    # Save to history
                    self.save_to_history({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'url': url,
                        'language': language,
                        'transcript': transcript
                    })
                    
                    st.success("Transcript generated successfully!")
                    
                    # Display transcript
                    st.text_area("Generated Transcript:", value=transcript, height=400)
                    
                    # Download and MCQ buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        self.download_button(transcript, "transcript.txt", "Download Transcript")
                    with col2:
                        if st.button("Generate MCQs"):
                            st.session_state.active_tab = "MCQ Generator"
                            st.experimental_rerun()
            
            except Exception as e:
                st.error(str(e))

    def render_history_tab(self):
        """Render history tab."""
        st.header("Transcript History")
        
        if not self.history:
            st.info("No transcripts in history")
            return
        
        for entry in self.history:
            with st.expander(f"{entry['timestamp']} - {entry['url']}"):
                st.text(f"Language: {entry['language']}")
                st.text_area("Transcript:", value=entry['transcript'], height=200)
                
                col1, col2 = st.columns(2)
                with col1:
                    self.download_button(
                        entry['transcript'],
                        f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "Download"
                    )
                with col2:
                    if st.button("Generate MCQs", key=f"mcq_{entry['timestamp']}"):
                        st.session_state.current_transcript = entry['transcript']
                        st.session_state.active_tab = "MCQ Generator"
                        st.experimental_rerun()

    def render_mcq_tab(self):
        """Render MCQ generator tab."""
        st.header("MCQ Generator")
        
        # Configuration
        num_questions = st.slider("Number of questions:", 1, 20, 5)
        
        # Get transcript from session state or allow new input
        transcript = st.session_state.get('current_transcript', '')
        transcript = st.text_area(
            "Enter text or use generated transcript:",
            value=transcript,
            height=200
        )
        
        if st.button("Generate MCQs", type="primary"):
            if not transcript:
                st.error("Please enter some text first!")
                return
            
            try:
                with st.spinner("Generating MCQs..."):
                    # Download NLTK resources if needed
                    NLTKDownloader.download_nltk_resources()
                    
                    # Generate MCQs
                    mcqs = self.mcq_generator.generate_mcqs(transcript, num_questions)
                    
                    if not mcqs:
                        st.warning("Couldn't generate enough quality questions. Please try with different text or adjust the number of questions.")
                        return
                    
                    # Display MCQs
                    st.success(f"Generated {len(mcqs)} questions!")
                    
                    for i, mcq in enumerate(mcqs, 1):
                        self.render_mcq(mcq, i)
                        st.markdown("---")
                    
                    # Create download link
                    df = pd.DataFrame([{
                        'Question Number': i,
                        'Question': mcq['question'],
                        'Correct Answer': mcq['correct_answer'],
                        'Explanation': mcq['explanation'],
                        **{f'Option {k}': v for k, v in mcq['options'].items()}
                    } for i, mcq in enumerate(mcqs, 1)])
                    
                    self.download_button(
                        df.to_csv(index=False),
                        f"mcqs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "Download MCQs as CSV"
                    )
            
            except Exception as e:
                logger.error(f"Application error: {str(e)}")
                st.error(f"An error occurred: {str(e)}\n\nPlease try again or contact support if the problem persists.")

    @staticmethod
    def format_as_paragraphs(transcript: str) -> str:
        """Format transcript text as paragraphs."""
        lines = transcript.split('\n')
        current_paragraph = []
        paragraphs = []
        
        for line in lines:
            if line.strip():
                current_paragraph.append(line)
            elif current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            
        return '\n\n'.join(paragraphs)

    @staticmethod
    def download_button(content: str, filename: str, label: str) -> None:
        """Create a download button for content."""
        st.download_button(
            label=label,
            data=content,
            file_name=filename,
            mime='text/plain'
        )

    @staticmethod
    def render_mcq(mcq: Dict, index: int) -> None:
        """Render a single MCQ."""
        st.subheader(f"Question {index}")
        st.write(mcq['question'])
        
        # Display options in columns
        cols = st.columns(2)
        for i, (option, text) in enumerate(mcq['options'].items()):
            cols[i % 2].write(f"{option}. {text}")
        
        # Show explanation in expander
        with st.expander("Show Answer and Explanation"):
            st.write(f"Correct Answer: {mcq['correct_answer']}")
            st.write(f"Explanation: {mcq['explanation']}")

    def main(self):
        """Main Streamlit application."""
        st.set_page_config(
            page_title="YouTube Transcript & MCQ Generator",
            page_icon="📚",
            layout="wide"
        )
        
        # Add custom CSS
        st.markdown("""
        <style>
        .stButton > button {
            width: 100%;
        }
        .stTextArea > div > div > textarea {
            background-color: #f0f2f6;
        }
        .main > div {
            padding-top: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize session state for active tab
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "Transcript Generator"
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        st.session_state.active_tab = st.sidebar.radio(
            "Go to",
            ["Transcript Generator", "MCQ Generator", "History"]
        )
        
        # Render appropriate tab
        if st.session_state.active_tab == "Transcript Generator":
            self.render_transcript_tab()
        elif st.session_state.active_tab == "MCQ Generator":
            self.render_mcq_tab()
        else:
            self.render_history_tab()

if __name__ == "__main__":
    app = StreamlitUI()
    app.main()