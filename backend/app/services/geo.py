COUNTY_MAP = {
    # --- BUCUREȘTI & ILFOV ---
    "bucuresti": "B", "bucurești": "B", "capitala": "B", 
    "ilfov": "IF", "buftea": "IF", "otopeni": "IF", "voluntari": "IF",

    # --- TRANSILVANIA ---
    "cluj": "CJ", "cluj-napoca": "CJ", "cluj napoca": "CJ",
    "bihor": "BH", "oradea": "BH",
    "bistrita-nasaud": "BN", "bistrița-năsăud": "BN", "bistrita": "BN", "bistrița": "BN",
    "brasov": "BV", "brașov": "BV",
    "alba": "AB", "alba iulia": "AB",
    "covasna": "CV", "sfantu gheorghe": "CV", "sfântu gheorghe": "CV",
    "harghita": "HR", "miercurea ciuc": "HR",
    "hunedoara": "HD", "deva": "HD",
    "mures": "MS", "mureș": "MS", "targu mures": "MS", "târgu mureș": "MS",
    "salaj": "SJ", "sălaj": "SJ", "zalau": "SJ", "zalău": "SJ",
    "sibiu": "SB",

    # --- BANAT ---
    "timis": "TM", "timiș": "TM", "timisoara": "TM", "timișoara": "TM",
    "caras-severin": "CS", "caraș-severin": "CS", "resita": "CS", "reșița": "CS",
    "arad": "AR",

    # --- MARAMUREȘ & CRIȘANA ---
    "maramures": "MM", "maramureș": "MM", "baia mare": "MM",
    "satu mare": "SM",

    # --- MOLDOVA ---
    "bacau": "BC", "bacău": "BC",
    "botosani": "BT", "botoșani": "BT",
    "galati": "GL", "galați": "GL",
    "iasi": "IS", "iași": "IS",
    "neamt": "NT", "neamț": "NT", "piatra neamt": "NT", "piatra neamț": "NT",
    "suceava": "SV",
    "vaslui": "VS",
    "vrancea": "VN", "focsani": "VN", "focșani": "VN",

    # --- OLTENIA ---
    "dolj": "DJ", "craiova": "DJ",
    "gorj": "GJ", "targu jiu": "GJ", "târgu jiu": "GJ",
    "mehedinti": "MH", "mehedinți": "MH", "drobeta-turnu severin": "MH", "severin": "MH",
    "olt": "OT", "slatina": "OT",
    "valcea": "VL", "vâlcea": "VL", "ramnicu valcea": "VL", "râmnicu vâlcea": "VL",

    # --- MUNTENIA ---
    "arges": "AG", "argeș": "AG", "pitesti": "AG", "pitești": "AG",
    "braila": "BR", "brăila": "BR",
    "buzau": "BZ", "buzău": "BZ",
    "calarasi": "CL", "călărași": "CL",
    "dambovita": "DB", "dâmbovița": "DB", "targoviste": "DB", "târgoviște": "DB",
    "giurgiu": "GR",
    "ialomita": "IL", "ialomița": "IL", "slobozia": "IL",
    "prahova": "PH", "ploiesti": "PH", "ploiești": "PH",
    "teleorman": "TR", "alexandria": "TR",

    # --- DOBROGEA ---
    "constanta": "CT", "constanța": "CT", "litoral": "CT", "mamaia": "CT",
    "tulcea": "TL", "delta dunarii": "TL", "delta dunării": "TL"
}

def resolve_location(loc_entities):
    """Returns the most frequent county code mentioned in the article."""
    counts = {}
    for loc in loc_entities:
        # Normalize: lowercase, remove prefixes, strip spaces
        norm = loc.lower().replace("municipiul ", "").replace("judetul ", "").replace("județul ", "").strip()
        
        code = COUNTY_MAP.get(norm)
        if code:
            counts[code] = counts.get(code, 0) + 1
    
    if not counts: return None
    # Returns the code with the highest count (e.g., "CJ")
    return max(counts, key=counts.get)