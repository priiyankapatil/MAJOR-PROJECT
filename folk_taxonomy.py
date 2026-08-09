try:
    from rapidfuzz import process, fuzz
except Exception:
    # Provide a lightweight fallback using difflib when rapidfuzz
    # isn't installed or importable in the environment.
    import difflib

    class _SimpleScorer:
        WRatio = None

    def _extractOne_fallback(query, choices, scorer=None):
        # Use difflib to find the closest match and score (0-100)
        matches = difflib.get_close_matches(query, choices, n=1, cutoff=0.0)
        if not matches:
            return None
        best = matches[0]
        ratio = int(difflib.SequenceMatcher(None, query, best).ratio() * 100)
        return (best, ratio, None)

    process = type("proc", (), {"extractOne": staticmethod(_extractOne_fallback)})
    fuzz = _SimpleScorer()


# ─── STOPWORDS TO NEVER MATCH ───
# Common English and Hindi words that must never be fuzzy-matched
STOPWORDS = {
    # English stopwords
    "my", "is", "the", "a", "an", "in", "on", "at", "to", "do",
    "has", "have", "what", "how", "why", "when", "where", "which",
    "can", "will", "for", "of", "and", "or", "but", "not", "be",
    "are", "was", "were", "it", "its", "this", "that", "with",
    "from", "by", "as", "if", "so", "up", "out", "get", "use",
    "crop", "plant", "problem", "question", "tell", "give", "help",
    "about", "any", "some", "all", "more", "please", "need", "want",
    # Hindi stopwords (romanized)
    "mein", "se", "ke", "ko", "ka", "ki", "hai", "hain", "ho",
    "kya", "kaise", "kyun", "kab", "kahan", "kaun", "aur", "ya",
    "nahi", "nahin", "bhi", "sirf", "liye", "wala", "wali",
    "yeh", "woh", "iska", "uska", "aap", "hum", "tum", "main",
    "kuch", "sab", "bahut", "thoda", "jab", "tab", "agar", "toh",
    "par", "lekin", "magar", "phir", "ab", "pehle", "baad",
    "accha", "theek", "bilkul", "zaroor", "bahut", "bohot"
}

# ─── FOLK TAXONOMY DICTIONARY ───
# Structure: "folk_term": {
#     "scientific": "scientific name",
#     "english": "common English name",
#     "category": "crop/disease/pest/soil/fertilizer",
#     "aliases": ["variant1", "variant2", ...]
# }

FOLK_TAXONOMY = {

    # ─── CROPS ───
    "arhar": {
        "scientific": "Cajanus cajan",
        "english": "Pigeon Pea",
        "category": "crop",
        "aliases": ["arhar dal", "tur", "toor", "tuvar", "togari", "thogari",
                    "togri", "red gram", "lal arhar"]
    },
    "chana": {
        "scientific": "Cicer arietinum",
        "english": "Chickpea",
        "category": "crop",
        "aliases": ["chane", "channa", "gram", "desi chana", "kala chana",
                    "kabuli chana", "kadale", "kadalai", "Bengal gram"]
    },
    "gehun": {
        "scientific": "Triticum aestivum",
        "english": "Wheat",
        "category": "crop",
        "aliases": ["gehu", "gahu", "wheat", "gehoon"]
    },
    "dhan": {
        "scientific": "Oryza sativa",
        "english": "Rice / Paddy",
        "category": "crop",
        "aliases": ["paddy", "chawal", "bhat", "dhan fasal", "nel",
                    "nellu", "akki", "bhatta"]
    },
    "makka": {
        "scientific": "Zea mays",
        "english": "Maize / Corn",
        "category": "crop",
        "aliases": ["makai", "makkai", "bhutta", "corn", "maize",
                    "makki", "cholam", "jola"]
    },
    "bajra": {
        "scientific": "Pennisetum glaucum",
        "english": "Pearl Millet",
        "category": "crop",
        "aliases": ["bajri", "bajara", "sajje", "kambu", "pearl millet",
                    "spiked millet"]
    },
    "jowar": {
        "scientific": "Sorghum bicolor",
        "english": "Sorghum",
        "category": "crop",
        "aliases": ["jwari", "juar", "jolada", "cholam", "great millet",
                    "sorghum", "jola"]
    },
    "ragi": {
        "scientific": "Eleusine coracana",
        "english": "Finger Millet",
        "category": "crop",
        "aliases": ["mandua", "nachni", "kezhvaragu", "keppai",
                    "finger millet", "african millet"]
    },
    "sarson": {
        "scientific": "Brassica napus",
        "english": "Mustard / Rapeseed",
        "category": "crop",
        "aliases": ["sarso", "mustard", "rape", "toria", "rai",
                    "canola", "brasica"]
    },
    "moong": {
        "scientific": "Vigna radiata",
        "english": "Green Gram / Mung Bean",
        "category": "crop",
        "aliases": ["mung", "moong dal", "green gram", "hesaru",
                    "pachai payaru", "mung bean"]
    },
    "urad": {
        "scientific": "Vigna mungo",
        "english": "Black Gram",
        "category": "crop",
        "aliases": ["urad dal", "black gram", "uddu", "ulundu",
                    "black lentil", "kali dal"]
    },
    "masoor": {
        "scientific": "Lens culinaris",
        "english": "Lentil",
        "category": "crop",
        "aliases": ["masur", "lentil", "red lentil", "masoor dal",
                    "mysore dal"]
    },
    "mungfali": {
        "scientific": "Arachis hypogaea",
        "english": "Groundnut / Peanut",
        "category": "crop",
        "aliases": ["moongfali", "groundnut", "peanut", "shengdana",
                    "verkadalai", "kadalekai"]
    },
    "ganna": {
        "scientific": "Saccharum officinarum",
        "english": "Sugarcane",
        "category": "crop",
        "aliases": ["ikh", "sugarcane", "eekh", "karumbu",
                    "kabbu", "ukh"]
    },
    "kapas": {
        "scientific": "Gossypium hirsutum",
        "english": "Cotton",
        "category": "crop",
        "aliases": ["cotton", "karpas", "rui", "paruthi",
                    "hubbu", "banaas"]
    },
    "til": {
        "scientific": "Sesamum indicum",
        "english": "Sesame",
        "category": "crop",
        "aliases": ["tilli", "sesame", "gingelly", "ellu",
                    "nuvvulu", "til tel"]
    },
    "nariyal": {
        "scientific": "Cocos nucifera",
        "english": "Coconut",
        "category": "crop",
        "aliases": ["coconut", "nariyel", "thengai", "thengai maram",
                    "tengu", "narel"]
    },
    "tamatar": {
        "scientific": "Solanum lycopersicum",
        "english": "Tomato",
        "category": "crop",
        "aliases": ["tomato", "tamaatar", "tomatar", "tameta",
                    "thakkali"]
    },
    "aloo": {
        "scientific": "Solanum tuberosum",
        "english": "Potato",
        "category": "crop",
        "aliases": ["potato", "batata", "alu", "aaloo",
                    "urulaikizhangu"]
    },
    "pyaz": {
        "scientific": "Allium cepa",
        "english": "Onion",
        "category": "crop",
        "aliases": ["onion", "pyaaz", "dungri", "eerulli",
                    "vengayam", "kanda"]
    },

    # ─── DISEASES ───
    "jhulsa": {
        "scientific": "Alternaria blight / Phytophthora infestans",
        "english": "Blight Disease",
        "category": "disease",
        "aliases": ["jhulaas", "patta jhulasna", "leaf blight",
                    "blight", "late blight", "early blight",
                    "patti jhulasna"]
    },
    "khari bimari": {
        "scientific": "Sclerospora graminicola",
        "english": "Downy Mildew",
        "category": "disease",
        "aliases": ["khari rog", "mosaru roga", "downy mildew",
                    "safed rogi", "safed bimari"]
    },
    "karwa rog": {
        "scientific": "Ustilago spp.",
        "english": "Smut Disease",
        "category": "disease",
        "aliases": ["kanda rog", "smut", "loose smut", "covered smut",
                    "kuppe roga", "kari rog"]
    },
    "patti marodna": {
        "scientific": "Begomovirus / Tomato Leaf Curl Virus",
        "english": "Leaf Curl Virus",
        "category": "disease",
        "aliases": ["patta curl", "leaf curl", "ele chukke",
                    "chukke roga", "patti ghuma", "curl virus"]
    },
    "safed rog": {
        "scientific": "Erysiphe spp.",
        "english": "Powdery Mildew",
        "category": "disease",
        "aliases": ["safed powder", "powdery mildew", "bili roga",
                    "podi noi", "sufaid rog", "sada rog"]
    },
    "jad sadan": {
        "scientific": "Fusarium solani / Rhizoctonia solani",
        "english": "Root Rot",
        "category": "disease",
        "aliases": ["jad galana", "root rot", "bhoomi sutte",
                    "jad ki bimari", "mudichal ari"]
    },
    "phal sadna": {
        "scientific": "Phytophthora spp.",
        "english": "Fruit Rot",
        "category": "disease",
        "aliases": ["phal galana", "fruit rot", "kayi kolla",
                    "kai sadal", "phal ki bimari"]
    },
    "lal rog": {
        "scientific": "Colletotrichum falcatum",
        "english": "Red Rot",
        "category": "disease",
        "aliases": ["kempu roga", "lal bimari", "red rot",
                    "red stripe", "laal rog"]
    },
    "peela rog": {
        "scientific": "Yellow Mosaic Virus (MYMV)",
        "english": "Yellow Mosaic Virus",
        "category": "disease",
        "aliases": ["peeli bimari", "mosaic", "yellow mosaic",
                    "peela paan", "yellow virus", "peeli patti"]
    },
    "tikka": {
        "scientific": "Cercospora arachidicola",
        "english": "Tikka / Leaf Spot Disease",
        "category": "disease",
        "aliases": ["tikka rog", "leaf spot", "patti daag",
                    "cercospora", "brown spot"]
    },

    # ─── PESTS ───
    "kambali keeda": {
        "scientific": "Amsacta moorei",
        "english": "Hairy Caterpillar",
        "category": "pest",
        "aliases": ["kambali hulu", "hairy caterpillar", "baal wala keeda",
                    "rome caterpillar", "balu keeda"]
    },
    "tana borer": {
        "scientific": "Chilo partellus",
        "english": "Stem Borer",
        "category": "pest",
        "aliases": ["stem borer", "dappa hulu", "tana chedak",
                    "tana keeda", "boring insect", "tutdu hulu"]
    },
    "maahu": {
        "scientific": "Aphis craccivora / Myzus persicae",
        "english": "Aphid",
        "category": "pest",
        "aliases": ["mahu", "aphid", "tuppa hachi", "moyla",
                    "chipchipa keeda", "juice sucker"]
    },
    "tambaku keeda": {
        "scientific": "Spodoptera litura",
        "english": "Tobacco Caterpillar / Fall Armyworm",
        "category": "pest",
        "aliases": ["thamaku hachi", "sena keeda", "army worm",
                    "fall armyworm", "spodoptera"]
    },
    "safed makkhi": {
        "scientific": "Bemisia tabaci",
        "english": "Whitefly",
        "category": "pest",
        "aliases": ["whitefly", "thambitu hachi", "vellai eegai",
                    "safed keeda", "white insect"]
    },
    "thrips": {
        "scientific": "Thrips tabaci / Scirtothrips dorsalis",
        "english": "Thrips",
        "category": "pest",
        "aliases": ["trip", "chota keeda", "thrip insect",
                    "flower thrips", "chilli thrips"]
    },
    "ghun": {
        "scientific": "Sitophilus oryzae / Tribolium castaneum",
        "english": "Grain Weevil / Storage Pest",
        "category": "pest",
        "aliases": ["anaj keeda", "storage weevil", "grain pest",
                    "bhandar keeda", "godown keeda"]
    },
    "lal keeda": {
        "scientific": "Dysdercus koenigii",
        "english": "Red Cotton Bug",
        "category": "pest",
        "aliases": ["lal bug", "cotton bug", "kempu hachi",
                    "lal beg", "red bug"]
    },

    # ─── SOIL TYPES ───
    "kali mitti": {
        "scientific": "Vertisol (Black Cotton Soil)",
        "english": "Black Cotton Soil",
        "category": "soil",
        "aliases": ["kali bhumi", "kaali mitti", "black soil",
                    "regur", "cotton soil", "kari Nadu"]
    },
    "lal mitti": {
        "scientific": "Alfisol (Red Loamy Soil)",
        "english": "Red Loamy Soil",
        "category": "soil",
        "aliases": ["laal mitti", "red soil", "kempu Nadu",
                    "lal bhumi", "red loam"]
    },
    "retili mitti": {
        "scientific": "Entisol (Sandy Soil)",
        "english": "Sandy Soil",
        "category": "soil",
        "aliases": ["baluee mitti", "sandy soil", "manu Nadu",
                    "ret wali mitti", "halki mitti"]
    },
    "domat mitti": {
        "scientific": "Inceptisol (Loamy Soil)",
        "english": "Loam Soil",
        "category": "soil",
        "aliases": ["domat", "loam", "mixed soil", "madhyam mitti",
                    "bharbhari mitti"]
    },

    # ─── FERTILIZERS / INPUTS ───
    "gobar khad": {
        "scientific": "Farm Yard Manure (FYM)",
        "english": "Farmyard Manure",
        "category": "fertilizer",
        "aliases": ["gobar", "FYM", "farm manure", "dung manure",
                    "kadaru", "kode sappu", "organic manure"]
    },
    "hara khad": {
        "scientific": "Green Manure",
        "english": "Green Manure",
        "category": "fertilizer",
        "aliases": ["hari khad", "green manure", "hasi sappu",
                    "pachchi uri", "harit khad"]
    },
    "neem khali": {
        "scientific": "Neem Cake (Azadirachta indica seed cake)",
        "english": "Neem Cake",
        "category": "fertilizer",
        "aliases": ["neem cake", "neem ki khali", "neem powder",
                    "veppam pindhi", "bevu hinde"]
    },
    "yurea": {
        "scientific": "Urea (CO(NH2)2) — 46% N",
        "english": "Urea",
        "category": "fertilizer",
        "aliases": ["urea", "yuria", "halu sappu", "nitrogen fertilizer",
                    "N fertilizer", "carbamide"]
    },
    "DAP": {
        "scientific": "Diammonium Phosphate (DAP) — 18% N, 46% P2O5",
        "english": "DAP Fertilizer",
        "category": "fertilizer",
        "aliases": ["di ammonium phosphate", "dap khad", "phosphate fertilizer",
                    "dap fertilizer", "18:46"]
    },
}

# ─── FLAT ALIAS INDEX ───
# Build a fast lookup: any alias → canonical folk key
ALIAS_INDEX = {}
for key, data in FOLK_TAXONOMY.items():
    ALIAS_INDEX[key.lower()] = key
    for alias in data.get("aliases", []):
        ALIAS_INDEX[alias.lower()] = key


# ─── STOPWORDS TO NEVER MATCH ───
# Common English and Hindi words that must never be fuzzy-matched
STOPWORDS = {
    # English stopwords
    "my", "is", "the", "a", "an", "in", "on", "at", "to", "do",
    "has", "have", "what", "how", "why", "when", "where", "which",
    "can", "will", "for", "of", "and", "or", "but", "not", "be",
    "are", "was", "were", "it", "its", "this", "that", "with",
    "from", "by", "as", "if", "so", "up", "out", "get", "use",
    "crop", "plant", "problem", "question", "tell", "give", "help",
    "about", "any", "some", "all", "more", "please", "need", "want",
    # Hindi stopwords (romanized)
    "mein", "se", "ke", "ko", "ka", "ki", "hai", "hain", "ho",
    "kya", "kaise", "kyun", "kab", "kahan", "kaun", "aur", "ya",
    "nahi", "nahin", "bhi", "sirf", "liye", "wala", "wali",
    "yeh", "woh", "iska", "uska", "aap", "hum", "tum", "main",
    "kuch", "sab", "bahut", "thoda", "jab", "tab", "agar", "toh",
    "par", "lekin", "magar", "phir", "ab", "pehle", "baad",
    "accha", "theek", "bilkul", "zaroor", "bahut", "bohot"
}


def get_scientific(folk_term: str) -> dict | None:
    """Look up scientific info for a folk term or alias."""
    term = folk_term.lower().strip()
    canonical = ALIAS_INDEX.get(term)
    if canonical:
        return {"folk_term": folk_term, **FOLK_TAXONOMY[canonical]}
    return None


def fuzzy_match(term: str, threshold: int = 93) -> dict | None:
    """Fuzzy match — only for terms not in stopwords and length > 3.

    Returns a dict with match details if a match >= threshold is found,
    otherwise None.
    """
    term_clean = (term or "").lower().strip()

    # Never match stopwords
    if term_clean in STOPWORDS:
        return None

    # Never match very short tokens (length <= 3 chars)
    if len(term_clean) <= 3:
        return None

    all_aliases = list(ALIAS_INDEX.keys())
    result = process.extractOne(
        term_clean,
        all_aliases,
        scorer=fuzz.WRatio
    )
    if result and result[1] >= threshold:
        matched_alias = result[0]
        canonical = ALIAS_INDEX[matched_alias]
        return {
            "folk_term": term,
            "matched_alias": matched_alias,
            "match_score": result[1],
            **FOLK_TAXONOMY[canonical]
        }
    return None


def get_all_crops() -> list:
    return [k for k, v in FOLK_TAXONOMY.items() if v["category"] == "crop"]


def get_all_diseases() -> list:
    return [k for k, v in FOLK_TAXONOMY.items() if v["category"] == "disease"]


def get_all_pests() -> list:
    return [k for k, v in FOLK_TAXONOMY.items() if v["category"] == "pest"]