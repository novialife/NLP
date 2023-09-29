def has_number(token):
    for elem in token:
        try:
            if elem in ["en", "ett", "två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio", "tio"]:
                return 1
            float(elem)
            return 1
        except ValueError:
            pass
    return 0


def age_feature(token):
    if any(x in token for x in ["gammal", "ung", "år", "månader", "gamla", "årsåldern", "åldern", "års"]):
        return 1
    else:
        return 0


def distance_feature(token):
    if any(x in token for x in ["km", "m", "meter", "kilometer", "mil", "cm", "mm", "centimeter"]):
        return 1
    else:
        return 0


def date_feature(token):
    if any(x in ["-", "_", "/", "talet", "januari", "februari", "mars", "april", "maj", "juni", "juli", "augusti",
                 "september", "oktober", "november", "december", "månad", "år"] for x in token):
        return 1
    else:
        return 0


def time_feature(token):
    if any(x in [",", ":", "timme", "minuter", "timmar", "minuter"] for x in token):
        return 1
    else:
        return 0


def quantity_feature(token):
    if any(x in token for x in ["gång", "gånger", "gången", "stycken", "par", "per"]):
        return 1
    else:
        return 0


def money_feature(token):
    if any(x in token for x in
           ["Kronor", "kronor", "tusen", "spänn", "miljoner", "miljarder", "SEK", "sek", "kr", "Kr", "öre", "rubel",
            "dollar", "euro"]):
        return 1
    else:
        return 0


def parseSentence(sentence):
    sentence = sentence.split(" ")
    datapoint = [has_number(sentence), age_feature(sentence), date_feature(sentence),
                 time_feature(sentence),
                 distance_feature(sentence),
                 quantity_feature(sentence), money_feature(sentence)]
    return datapoint
