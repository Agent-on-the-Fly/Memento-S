
ZH_STOPWORDS = frozenset({
    "", "", "", "", "", "",
    "", "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "",
    "", "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "", "", "", "",
    "", "", "",
    "", "", "", "", "", "", "",
})

EN_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "into", "over", "under",
    "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need",
    "i", "me", "my", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "we", "us", "our", "they", "them", "their",
    "this", "that", "these", "those",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very", "just",
    "about", "above", "after", "again", "all", "also", "any", "because",
    "before", "between", "both", "each", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "here", "there",
    "when", "where", "how", "what", "which", "who", "whom", "why",
    "def", "return", "import", "class", "function", "method",
    "true", "false", "none", "null", "var", "let", "const",
})

ALL_STOPWORDS = ZH_STOPWORDS | EN_STOPWORDS

QUERY_STOPWORDS = ALL_STOPWORDS | frozenset({
    "use", "get", "set", "run", "make", "create", "help", "please",
    "send", "show", "find", "list", "open", "close", "start", "stop",
    "new", "add", "delete", "remove", "skill", "tool",
    "", "", "", "", "", "", "",
})

NAME_STOPWORDS = frozenset({
    "the", "get", "set", "new", "old", "all", "doc", "test",
    "skill", "tool", "helper", "util", "utils", "builder",
    "creator", "generator", "manager", "handler",
    "main", "core", "base", "default", "common", "simple",
    "my", "custom", "example", "demo", "sample", "template",
})
