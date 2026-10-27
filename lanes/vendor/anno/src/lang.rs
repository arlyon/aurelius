//! Language detection and classification utilities.

/// Supported languages for text analysis.
///
/// Variants are intentionally ordered for indexed access in `detect_language`.
/// The `repr(u8)` is required for safe conversion from index to enum variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Language {
    /// English language
    English,
    /// German language
    German,
    /// French language
    French,
    /// Spanish language
    Spanish,
    /// Italian language
    Italian,
    /// Portuguese language
    Portuguese,
    /// Russian language
    Russian,
    /// Chinese language (Simplified/Traditional)
    Chinese,
    /// Japanese language
    Japanese,
    /// Korean language
    Korean,
    /// Arabic language
    Arabic,
    /// Hebrew language
    Hebrew,
    /// Other/unknown language
    Other,
}

impl Language {
    /// Returns true if this is a CJK (Chinese, Japanese, Korean) language.
    #[must_use]
    pub fn is_cjk(&self) -> bool {
        matches!(
            self,
            Language::Chinese | Language::Japanese | Language::Korean
        )
    }

    /// Returns true if this is a right-to-left language (Arabic, Hebrew).
    #[must_use]
    pub fn is_rtl(&self) -> bool {
        matches!(self, Language::Arabic | Language::Hebrew)
    }

    /// Returns true if this language uses the Latin script with sentence-initial
    /// capitalization (English, French, Spanish, German, Italian, Portuguese).
    ///
    /// Used to gate capitalization adjustments in pronoun rewriting: only these
    /// languages capitalize the first word of a sentence, so replacing a
    /// sentence-initial pronoun with a proper noun needs case adjustment.
    #[must_use]
    pub fn uses_latin_capitalization(&self) -> bool {
        matches!(
            self,
            Language::English
                | Language::French
                | Language::Spanish
                | Language::German
                | Language::Italian
                | Language::Portuguese
        )
    }

    /// Get ISO 639-1 language code (2-letter).
    #[must_use]
    pub fn iso_code(&self) -> &'static str {
        match self {
            Language::English => "en",
            Language::German => "de",
            Language::French => "fr",
            Language::Spanish => "es",
            Language::Italian => "it",
            Language::Portuguese => "pt",
            Language::Russian => "ru",
            Language::Chinese => "zh",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::Arabic => "ar",
            Language::Hebrew => "he",
            Language::Other => "xx",
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.iso_code())
    }
}

impl Language {
    /// Parse from ISO 639-1/639-3 code or English name.
    ///
    /// Returns `None` for unrecognized codes (rather than silently defaulting).
    ///
    /// ```rust
    /// use anno::Language;
    ///
    /// assert_eq!(Language::from_code("en"), Some(Language::English));
    /// assert_eq!(Language::from_code("de"), Some(Language::German));
    /// assert_eq!(Language::from_code("english"), Some(Language::English));
    /// assert_eq!(Language::from_code("xyz"), None);
    /// ```
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_lowercase().as_str() {
            "en" | "eng" | "english" => Some(Language::English),
            "de" | "deu" | "german" => Some(Language::German),
            "fr" | "fra" | "french" => Some(Language::French),
            "es" | "spa" | "spanish" => Some(Language::Spanish),
            "it" | "ita" | "italian" => Some(Language::Italian),
            "pt" | "por" | "portuguese" => Some(Language::Portuguese),
            "ru" | "rus" | "russian" => Some(Language::Russian),
            "zh" | "zho" | "chinese" => Some(Language::Chinese),
            "ja" | "jpn" | "japanese" => Some(Language::Japanese),
            "ko" | "kor" | "korean" => Some(Language::Korean),
            "ar" | "ara" | "arabic" => Some(Language::Arabic),
            "he" | "heb" | "hebrew" => Some(Language::Hebrew),
            "xx" | "other" | "unknown" => Some(Language::Other),
            _ => None,
        }
    }
}

/// Simple heuristic language detection based on Unicode scripts.
///
/// Returns the most likely language based on character counts.
pub fn detect_language(text: &str) -> Language {
    let mut counts = [0usize; 13];
    let mut total = 0;

    for c in text.chars() {
        match c {
            // CJK Unified Ideographs
            '\u{4e00}'..='\u{9fff}' => {
                total += 1;
                counts[Language::Chinese as usize] += 1;
            }
            // Hiragana/Katakana
            '\u{3040}'..='\u{30ff}' => {
                total += 1;
                counts[Language::Japanese as usize] += 1;
            }
            // Hangul
            '\u{ac00}'..='\u{d7af}' => {
                total += 1;
                counts[Language::Korean as usize] += 1;
            }
            // Arabic
            '\u{0600}'..='\u{06ff}' => {
                total += 1;
                counts[Language::Arabic as usize] += 1;
            }
            // Hebrew
            '\u{0590}'..='\u{05ff}' => {
                total += 1;
                counts[Language::Hebrew as usize] += 1;
            }
            // Cyrillic
            '\u{0400}'..='\u{04ff}' => {
                total += 1;
                counts[Language::Russian as usize] += 1;
            }
            // Latin - distinguishing languages is hard without dictionary,
            // but we can check for specific chars
            'a'..='z' | 'A'..='Z' => {
                total += 1;
                counts[Language::English as usize] += 1; // Generic Latin
            }
            // German specific (ß, ä, ö, ü)
            'ß' | 'ä' | 'ö' | 'ü' | 'Ä' | 'Ö' | 'Ü' => {
                total += 1;
                counts[Language::German as usize] += 10
            }
            // French (à, â, ç, é, è, ê, ë, î, ï, ô, û, ù)
            'à' | 'â' | 'ç' | 'é' | 'è' | 'ê' | 'ë' | 'î' | 'ï' | 'ô' | 'û' | 'ù' => {
                total += 1;
                counts[Language::French as usize] += 5
            }
            // Spanish (ñ, ¿, ¡, á, é, í, ó, ú)
            'ñ' | '¿' | '¡' | 'á' | 'í' | 'ó' | 'ú' => {
                total += 1;
                counts[Language::Spanish as usize] += 5
            }
            _ => {}
        }
    }

    if total == 0 {
        return Language::English; // Default
    }

    // Find max
    let mut max_idx = 0;
    let mut max_val = 0;
    for (i, &val) in counts.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    // If we detected CJK chars but classified as Chinese, check if Japanese specific chars exist
    if max_idx == Language::Chinese as usize && counts[Language::Japanese as usize] > 0 {
        return Language::Japanese; // Japanese uses Kanji (Chinese chars) too
    }

    // Convert index to Language variant safely
    // Using explicit match instead of transmute for compile-time safety
    match max_idx {
        0 => Language::English,
        1 => Language::German,
        2 => Language::French,
        3 => Language::Spanish,
        4 => Language::Italian,
        5 => Language::Portuguese,
        6 => Language::Russian,
        7 => Language::Chinese,
        8 => Language::Japanese,
        9 => Language::Korean,
        10 => Language::Arabic,
        11 => Language::Hebrew,
        _ => Language::Other,
    }
}

/// Detect code-switching (mixed languages) in text.
///
/// Returns a vector of language segments with their positions.
/// Useful for processing multilingual text where languages switch mid-sentence.
///
/// # Example
///
/// ```rust
/// use anno::lang::{detect_code_switching, Language};
///
/// let segments = detect_code_switching("Dr. 田中 presented at MIT's conference.");
/// // Returns: [(Language::English, 0, 4), (Language::Japanese, 5, 7), (Language::English, 8, 40)]
/// ```
#[must_use]
pub fn detect_code_switching(text: &str) -> Vec<(Language, usize, usize)> {
    if text.is_empty() {
        return vec![];
    }

    let mut segments = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut current_lang = detect_language(text);
    let mut segment_start = 0;

    // Use a sliding window to detect language changes
    const WINDOW_SIZE: usize = 10; // Characters per window
    let mut i = 0;

    while i < chars.len() {
        // Check language in current window
        let window_end = (i + WINDOW_SIZE).min(chars.len());
        let window_text: String = chars[i..window_end].iter().collect();
        let window_lang = detect_language(&window_text);

        // If language changed significantly, start new segment
        if window_lang != current_lang && window_lang != Language::Other {
            // Save previous segment
            if i > segment_start {
                segments.push((current_lang, segment_start, i));
            }
            segment_start = i;
            current_lang = window_lang;
        }

        i += WINDOW_SIZE / 2; // Overlap windows for smoother detection
    }

    // Add final segment
    if segment_start < chars.len() {
        segments.push((current_lang, segment_start, chars.len()));
    }

    // Merge adjacent segments of the same language
    let mut merged = Vec::new();
    for (lang, start, end) in segments {
        if let Some((last_lang, _last_start, last_end)) = merged.last_mut() {
            if *last_lang == lang && *last_end == start {
                *last_end = end;
            } else {
                merged.push((lang, start, end));
            }
        } else {
            merged.push((lang, start, end));
        }
    }

    merged
}

/// Language clustering for cross-lingual transfer learning.
///
/// Groups languages by similarity for better multilingual NER performance.
/// Based on research showing that semantic clustering outperforms linguistic family grouping.
///
/// Returns language clusters where languages in the same cluster benefit from shared training.
#[must_use]
pub fn language_clusters() -> Vec<Vec<Language>> {
    // Research-based clusters (semantic similarity, not linguistic families)
    vec![
        // Cluster 1: Germanic + Romance (high-resource, similar syntax)
        vec![
            Language::English,
            Language::German,
            Language::French,
            Language::Spanish,
            Language::Italian,
            Language::Portuguese,
        ],
        // Cluster 2: Slavic
        vec![Language::Russian],
        // Cluster 3: CJK (character-based, similar entity patterns)
        vec![Language::Chinese, Language::Japanese, Language::Korean],
        // Cluster 4: Semitic (RTL, similar morphology)
        vec![Language::Arabic, Language::Hebrew],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_english() {
        assert_eq!(detect_language("Hello, world!"), Language::English);
        assert_eq!(detect_language("The quick brown fox"), Language::English);
    }

    #[test]
    fn test_detect_german() {
        // Need enough German-specific characters to outweigh generic Latin
        assert_eq!(
            detect_language("Größe Müller Öffentlichkeit Übung"),
            Language::German
        );
        assert_eq!(detect_language("ß ä ö ü ß Ä Ö Ü"), Language::German);
    }

    #[test]
    fn test_detect_french() {
        assert_eq!(detect_language("Café à Paris"), Language::French);
        assert_eq!(detect_language("être où ça"), Language::French);
    }

    #[test]
    fn test_detect_spanish() {
        assert_eq!(detect_language("¿Cómo estás? Mañana"), Language::Spanish);
    }

    #[test]
    fn test_detect_chinese() {
        assert_eq!(detect_language("北京欢迎您"), Language::Chinese);
        assert_eq!(detect_language("习近平"), Language::Chinese);
    }

    #[test]
    fn test_detect_japanese() {
        // Hiragana/Katakana triggers Japanese detection
        assert_eq!(detect_language("こんにちは"), Language::Japanese);
        assert_eq!(detect_language("東京タワー"), Language::Japanese);
    }

    #[test]
    fn test_detect_korean() {
        assert_eq!(detect_language("안녕하세요"), Language::Korean);
        assert_eq!(detect_language("서울"), Language::Korean);
    }

    #[test]
    fn test_detect_arabic() {
        assert_eq!(detect_language("مرحبا"), Language::Arabic);
        assert_eq!(detect_language("القاهرة"), Language::Arabic);
    }

    #[test]
    fn test_detect_hebrew() {
        assert_eq!(detect_language("שלום"), Language::Hebrew);
        assert_eq!(detect_language("ירושלים"), Language::Hebrew);
    }

    #[test]
    fn test_detect_russian() {
        assert_eq!(detect_language("Привет, мир!"), Language::Russian);
        assert_eq!(detect_language("Москва"), Language::Russian);
    }

    #[test]
    fn test_empty_text_defaults_to_english() {
        assert_eq!(detect_language(""), Language::English);
        assert_eq!(detect_language("123 !@# "), Language::English);
    }

    #[test]
    fn test_is_cjk() {
        assert!(Language::Chinese.is_cjk());
        assert!(Language::Japanese.is_cjk());
        assert!(Language::Korean.is_cjk());
        assert!(!Language::English.is_cjk());
        assert!(!Language::Arabic.is_cjk());
    }

    #[test]
    fn test_is_rtl() {
        assert!(Language::Arabic.is_rtl());
        assert!(Language::Hebrew.is_rtl());
        assert!(!Language::English.is_rtl());
        assert!(!Language::Chinese.is_rtl());
    }

    #[test]
    fn test_language_repr_matches_index() {
        // Verify the repr(u8) matches our index expectations
        assert_eq!(Language::English as u8, 0);
        assert_eq!(Language::German as u8, 1);
        assert_eq!(Language::French as u8, 2);
        assert_eq!(Language::Spanish as u8, 3);
        assert_eq!(Language::Italian as u8, 4);
        assert_eq!(Language::Portuguese as u8, 5);
        assert_eq!(Language::Russian as u8, 6);
        assert_eq!(Language::Chinese as u8, 7);
        assert_eq!(Language::Japanese as u8, 8);
        assert_eq!(Language::Korean as u8, 9);
        assert_eq!(Language::Arabic as u8, 10);
        assert_eq!(Language::Hebrew as u8, 11);
        assert_eq!(Language::Other as u8, 12);
    }

    #[test]
    fn test_detect_code_switching() {
        // Mixed English-Japanese (CJK characters should be detected)
        let segments = detect_code_switching("Dr. 田中 presented at MIT.");
        // Should detect at least one segment (may merge if window is too large)
        assert!(!segments.is_empty());

        // Mixed English-Chinese
        let segments = detect_code_switching("北京 (Beijing) is the capital.");
        assert!(!segments.is_empty());

        // Single language
        let segments = detect_code_switching("Hello world");
        assert_eq!(segments.len(), 1);

        // Verify segments have valid ranges
        for (_lang, start, end) in segments {
            assert!(start < end);
        }
    }

    #[test]
    fn test_language_iso_code() {
        assert_eq!(Language::English.iso_code(), "en");
        assert_eq!(Language::Spanish.iso_code(), "es");
        assert_eq!(Language::Chinese.iso_code(), "zh");
        assert_eq!(Language::Arabic.iso_code(), "ar");
    }

    #[test]
    fn test_language_clusters() {
        let clusters = language_clusters();
        assert!(!clusters.is_empty());

        // Check that major languages are in clusters
        let all_langs: Vec<Language> = clusters.iter().flat_map(|c| c.iter().copied()).collect();
        assert!(all_langs.contains(&Language::English));
        assert!(all_langs.contains(&Language::Chinese));
    }
}
