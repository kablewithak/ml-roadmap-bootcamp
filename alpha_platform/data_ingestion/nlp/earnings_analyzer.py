"""
Earnings call transcript analysis using NLP and FinBERT.

Extracts sentiment, key metrics, management confidence, and forward-looking
statements from earnings call transcripts.
"""

import re
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    import spacy
except ImportError:
    warnings.warn("spacy not installed. Install with: pip install spacy")
    spacy = None

from alpha_platform.utils.config import get_config
from alpha_platform.utils.logger import get_logger

logger = get_logger(__name__)


class EarningsCallAnalyzer:
    """
    Analyze earnings call transcripts for alpha signals.

    Uses FinBERT for sentiment analysis, linguistic analysis for
    management confidence, and entity extraction for key metrics.
    """

    def __init__(
        self,
        finbert_model: str = "ProsusAI/finbert",
        device: Optional[str] = None,
    ):
        """
        Initialize earnings call analyzer.

        Args:
            finbert_model: FinBERT model name from HuggingFace
            device: Device to run models on ('cuda', 'cpu', or None for auto)
        """
        self.config = get_config()

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing earnings analyzer on device: {self.device}")

        # Load FinBERT for sentiment analysis
        logger.info(f"Loading FinBERT model: {finbert_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(finbert_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(finbert_model)
        self.model.to(self.device)
        self.model.eval()

        # Sentiment labels
        self.sentiment_labels = ["positive", "negative", "neutral"]

        # Load spaCy for NER and linguistic analysis
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "spaCy model not found. Run: python -m spacy download en_core_web_sm"
                )
                self.nlp = None
        else:
            self.nlp = None

        logger.info("Earnings analyzer initialized")

    def analyze_transcript(
        self, transcript: str, ticker: str, earnings_date: datetime
    ) -> Dict[str, Any]:
        """
        Analyze complete earnings call transcript.

        Args:
            transcript: Full transcript text
            ticker: Stock ticker symbol
            earnings_date: Date of earnings call

        Returns:
            Comprehensive analysis results
        """
        logger.info(f"Analyzing earnings transcript for {ticker}")

        # Split into sections
        sections = self._split_transcript(transcript)

        # Analyze each section
        prepared_remarks = sections.get("prepared_remarks", "")
        qa_session = sections.get("qa", "")

        prepared_analysis = self._analyze_section(prepared_remarks, "prepared_remarks")
        qa_analysis = self._analyze_section(qa_session, "qa")

        # Extract key metrics and guidance
        metrics = self._extract_financial_metrics(transcript)
        guidance = self._extract_guidance(transcript)

        # Linguistic complexity analysis
        complexity = self._analyze_linguistic_complexity(prepared_remarks)

        # Management confidence indicators
        confidence = self._analyze_management_confidence(prepared_remarks)

        # Forward-looking statements
        forward_looking = self._extract_forward_looking_statements(transcript)

        # Overall sentiment
        overall_sentiment = self._calculate_overall_sentiment(
            prepared_analysis, qa_analysis
        )

        analysis = {
            "ticker": ticker,
            "earnings_date": earnings_date.isoformat(),
            "overall_sentiment": overall_sentiment,
            "prepared_remarks": prepared_analysis,
            "qa_session": qa_analysis,
            "financial_metrics": metrics,
            "guidance": guidance,
            "linguistic_complexity": complexity,
            "management_confidence": confidence,
            "forward_looking_statements": forward_looking,
            "analysis_timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Analysis complete for {ticker}: "
            f"sentiment={overall_sentiment['label']} "
            f"(score={overall_sentiment['score']:.3f})"
        )

        return analysis

    def _split_transcript(self, transcript: str) -> Dict[str, str]:
        """Split transcript into prepared remarks and Q&A sections."""
        # Common section headers
        qa_patterns = [
            r"(?i)question.{0,10}answer",
            r"(?i)q\s*&\s*a",
            r"(?i)questions?\s+and\s+answers?",
        ]

        for pattern in qa_patterns:
            match = re.search(pattern, transcript)
            if match:
                split_pos = match.start()
                return {
                    "prepared_remarks": transcript[:split_pos],
                    "qa": transcript[split_pos:],
                }

        # If no Q&A section found, treat entire transcript as prepared remarks
        return {"prepared_remarks": transcript, "qa": ""}

    def _analyze_section(
        self, text: str, section_name: str
    ) -> Dict[str, Any]:
        """Analyze a section of the transcript."""
        if not text.strip():
            return {"sentiment": None, "entities": [], "key_phrases": []}

        # Split into sentences for analysis
        sentences = self._split_into_sentences(text)

        # Analyze sentiment for each sentence
        sentiments = []
        for sentence in sentences[:100]:  # Limit to first 100 sentences
            if len(sentence.split()) > 5:  # Skip very short sentences
                sent_result = self._analyze_sentiment(sentence)
                sentiments.append(sent_result)

        # Aggregate sentiment
        if sentiments:
            avg_scores = {
                "positive": np.mean([s["scores"]["positive"] for s in sentiments]),
                "negative": np.mean([s["scores"]["negative"] for s in sentiments]),
                "neutral": np.mean([s["scores"]["neutral"] for s in sentiments]),
            }
            dominant_label = max(avg_scores, key=avg_scores.get)

            section_sentiment = {
                "label": dominant_label,
                "score": avg_scores[dominant_label],
                "scores": avg_scores,
                "n_sentences_analyzed": len(sentiments),
            }
        else:
            section_sentiment = None

        # Extract entities
        entities = self._extract_entities(text)

        # Extract key phrases
        key_phrases = self._extract_key_phrases(text)

        return {
            "sentiment": section_sentiment,
            "entities": entities,
            "key_phrases": key_phrases,
            "word_count": len(text.split()),
        }

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using FinBERT.

        Args:
            text: Input text

        Returns:
            Sentiment analysis results
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

        # Convert to scores
        scores = {
            label: float(prob) for label, prob in zip(self.sentiment_labels, probs)
        }

        # Get dominant sentiment
        dominant_idx = torch.argmax(probs).item()
        dominant_label = self.sentiment_labels[dominant_idx]

        return {
            "label": dominant_label,
            "score": scores[dominant_label],
            "scores": scores,
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using spaCy."""
        if self.nlp is None:
            return []

        doc = self.nlp(text[:100000])  # Limit text length

        entities = []
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "GPE", "MONEY", "PERCENT"]:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )

        return entities

    def _extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key phrases using noun chunks."""
        if self.nlp is None:
            return []

        doc = self.nlp(text[:100000])

        # Extract noun chunks
        noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]

        # Count frequencies
        from collections import Counter

        phrase_counts = Counter(noun_chunks)

        # Get top phrases
        top_phrases = [phrase for phrase, _ in phrase_counts.most_common(top_n)]

        return top_phrases

    def _extract_financial_metrics(self, transcript: str) -> Dict[str, List[str]]:
        """Extract financial metrics and numbers from transcript."""
        metrics = {
            "revenue": [],
            "earnings": [],
            "growth": [],
            "margin": [],
        }

        # Revenue patterns
        revenue_pattern = r"revenue[s]?\s+(?:of\s+)?[\$]?[\d,.]+ (?:million|billion)"
        metrics["revenue"] = re.findall(revenue_pattern, transcript, re.IGNORECASE)

        # Earnings patterns
        earnings_pattern = r"(?:earnings|EPS)\s+(?:of\s+)?[\$]?[\d,.]+"
        metrics["earnings"] = re.findall(earnings_pattern, transcript, re.IGNORECASE)

        # Growth patterns
        growth_pattern = r"(?:grew|growth|increase[d]?)\s+(?:by\s+)?[\d.]+%"
        metrics["growth"] = re.findall(growth_pattern, transcript, re.IGNORECASE)

        # Margin patterns
        margin_pattern = r"margin[s]?\s+(?:of\s+)?[\d.]+%"
        metrics["margin"] = re.findall(margin_pattern, transcript, re.IGNORECASE)

        return metrics

    def _extract_guidance(self, transcript: str) -> Dict[str, List[str]]:
        """Extract forward guidance statements."""
        guidance = {
            "full_year": [],
            "next_quarter": [],
            "long_term": [],
        }

        # Full year guidance
        fy_pattern = r"(?:full[- ]?year|FY\d{2,4}).*?(?:expect|forecast|guidance).*?[\$\d%]+"
        guidance["full_year"] = re.findall(fy_pattern, transcript, re.IGNORECASE)[:5]

        # Next quarter guidance
        q_pattern = r"(?:next quarter|Q\d).*?(?:expect|forecast|anticipate).*?[\$\d%]+"
        guidance["next_quarter"] = re.findall(q_pattern, transcript, re.IGNORECASE)[:5]

        # Long-term guidance
        lt_pattern = r"(?:long[- ]?term|multi[- ]?year).*?(?:target|goal|objective).*?[\$\d%]+"
        guidance["long_term"] = re.findall(lt_pattern, transcript, re.IGNORECASE)[:5]

        return guidance

    def _analyze_linguistic_complexity(self, text: str) -> Dict[str, float]:
        """
        Analyze linguistic complexity.

        Research shows management uses more complex language when
        trying to obfuscate poor performance.
        """
        sentences = self._split_into_sentences(text)

        if not sentences:
            return {"avg_sentence_length": 0.0, "avg_word_length": 0.0}

        # Average sentence length
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = np.mean(sentence_lengths)

        # Average word length
        words = text.split()
        word_lengths = [len(w) for w in words]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0.0

        # Flesch reading ease score (simplified)
        total_syllables = sum([self._count_syllables(w) for w in words[:1000]])
        total_words = min(len(words), 1000)
        total_sentences = len(sentences)

        if total_sentences > 0 and total_words > 0:
            flesch_score = (
                206.835
                - 1.015 * (total_words / total_sentences)
                - 84.6 * (total_syllables / total_words)
            )
        else:
            flesch_score = 50.0  # Default

        return {
            "avg_sentence_length": float(avg_sentence_length),
            "avg_word_length": float(avg_word_length),
            "flesch_reading_ease": float(flesch_score),
        }

    def _count_syllables(self, word: str) -> int:
        """Simple syllable counter."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e"):
            syllable_count -= 1

        return max(1, syllable_count)

    def _analyze_management_confidence(self, text: str) -> Dict[str, Any]:
        """
        Analyze management confidence indicators.

        Confident management uses more certain language, first-person pronouns.
        """
        text_lower = text.lower()

        # Confident words
        confident_words = [
            "confident",
            "strong",
            "excellent",
            "outstanding",
            "robust",
            "solid",
            "pleased",
        ]
        confident_count = sum([text_lower.count(w) for w in confident_words])

        # Uncertain words
        uncertain_words = [
            "may",
            "might",
            "could",
            "perhaps",
            "possibly",
            "uncertain",
            "challenging",
        ]
        uncertain_count = sum([text_lower.count(w) for w in uncertain_words])

        # First-person pronouns (confidence indicator)
        first_person = ["we", "our", "us"]
        first_person_count = sum([text_lower.count(w) for w in first_person])

        # Passive voice (lack of confidence)
        passive_indicators = ["was", "were", "been", "being"]
        passive_count = sum([text_lower.count(w) for w in passive_indicators])

        total_words = len(text.split())

        return {
            "confident_words_ratio": confident_count / max(total_words, 1),
            "uncertain_words_ratio": uncertain_count / max(total_words, 1),
            "first_person_ratio": first_person_count / max(total_words, 1),
            "passive_voice_ratio": passive_count / max(total_words, 1),
            "confidence_score": (confident_count - uncertain_count)
            / max(total_words, 1),
        }

    def _extract_forward_looking_statements(self, text: str) -> List[str]:
        """Extract forward-looking statements."""
        # Pattern for forward-looking language
        forward_patterns = [
            r"(?:we\s+)?(?:expect|anticipate|believe|plan|intend|project|estimate).*?[.!?]",
            r"going forward.*?[.!?]",
            r"in the (?:coming|next).*?[.!?]",
        ]

        statements = []
        for pattern in forward_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            statements.extend(matches[:10])  # Limit per pattern

        return statements[:30]  # Total limit

    def _calculate_overall_sentiment(
        self, prepared_analysis: Dict, qa_analysis: Dict
    ) -> Dict[str, Any]:
        """Calculate overall sentiment from section analyses."""
        sentiments = []

        if prepared_analysis.get("sentiment"):
            sentiments.append(
                (prepared_analysis["sentiment"], 0.6)
            )  # Weight prepared remarks more

        if qa_analysis.get("sentiment"):
            sentiments.append((qa_analysis["sentiment"], 0.4))

        if not sentiments:
            return {"label": "neutral", "score": 0.5}

        # Weighted average
        weighted_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        for sentiment, weight in sentiments:
            for label in weighted_scores:
                weighted_scores[label] += sentiment["scores"][label] * weight

        # Normalize
        total_weight = sum([w for _, w in sentiments])
        weighted_scores = {k: v / total_weight for k, v in weighted_scores.items()}

        dominant_label = max(weighted_scores, key=weighted_scores.get)

        return {
            "label": dominant_label,
            "score": weighted_scores[dominant_label],
            "scores": weighted_scores,
        }

    def generate_earnings_signal(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal from earnings call analysis.

        Args:
            analysis: Earnings call analysis results

        Returns:
            Trading signal
        """
        ticker = analysis["ticker"]

        # Extract key components
        sentiment = analysis["overall_sentiment"]
        confidence = analysis["management_confidence"]
        complexity = analysis["linguistic_complexity"]

        # Sentiment score (-1 to 1)
        if sentiment["label"] == "positive":
            sentiment_score = sentiment["score"]
        elif sentiment["label"] == "negative":
            sentiment_score = -sentiment["score"]
        else:
            sentiment_score = 0.0

        # Confidence score (0 to 1)
        conf_score = confidence.get("confidence_score", 0)

        # Complexity penalty (high complexity = obfuscation = negative)
        # Normal Flesch score is 60-70, lower is harder to read
        flesch = complexity.get("flesch_reading_ease", 60)
        complexity_penalty = max(0, (60 - flesch) / 60)  # 0 to 1

        # Combine signals
        signal_strength = (
            sentiment_score * 0.5  # Sentiment is main signal
            + conf_score * 0.3  # Confidence reinforces
            - complexity_penalty * 0.2  # Complexity is negative
        )

        # Clip to [-1, 1]
        signal_strength = max(min(signal_strength, 1.0), -1.0)

        signal = {
            "ticker": ticker,
            "signal_strength": float(signal_strength),
            "direction": "long" if signal_strength > 0 else "short",
            "confidence": float(abs(signal_strength)),
            "components": {
                "sentiment_score": float(sentiment_score),
                "confidence_score": float(conf_score),
                "complexity_penalty": float(complexity_penalty),
            },
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Generated earnings signal for {ticker}: "
            f"{signal['direction']} with strength {signal_strength:.3f}"
        )

        return signal
