from src.chatkbqa.adapt_chatkbqa_pred import (
    transform_compact_sexpr,
    transform_normed_sexpr,
    expand_entity_map,
)

def test_transform_compact_sexpr():
    tests: list[tuple[str, str]] = [
        (
            "(ARGMAX (JOIN (R sports.sports_team.championships) (JOIN sports.sports_team.team_mascot m.03_dwn)) time.event.start_date)",
            "(ARGMAX (JOIN (R <http://rdf.freebase.com/ns/sports.sports_team.championships>) (JOIN <http://rdf.freebase.com/ns/sports.sports_team.team_mascot> <http://rdf.freebase.com/ns/m.03_dwn>)) <http://rdf.freebase.com/ns/time.event.start_date>)"
        ),
        (
            "(JOIN (R location.country.languages_spoken) (JOIN location.country.administrative_divisions m.02g__4))",
            "(JOIN (R <http://rdf.freebase.com/ns/location.country.languages_spoken>) (JOIN <http://rdf.freebase.com/ns/location.country.administrative_divisions> <http://rdf.freebase.com/ns/m.02g__4>))"
        )
    ]
    
    for inp, out in tests:
        assert transform_compact_sexpr(inp) == out
        
        
def test_transform_normed_sexpr():
    tests: list[tuple[str, str]] = [
        (
            "( AND ( JOIN [ location , location , contains ] ( JOIN [ aviation , airport , serves ] [ Nijmegen ] ) ) ( AND ( JOIN [ common , topic , notable types ] [ Country ] ) ( JOIN ( R [ location , adjoining relationship , adjoins ] ) ( JOIN ( R [ location , location , adjoin s ] ) [ France ] ) ) ) )",
            "( AND ( JOIN fbp:location.location.contains ( JOIN fbp:aviation.airport.serves fb:Nijmegen ) ) ( AND ( JOIN fbp:common.topic.notable_types fb:Country ) ( JOIN (R fbp:location.adjoining_relationship.adjoins) ( JOIN (R fbp:location.location.adjoin_s) fb:France ) ) ) )"
        ),
        (
            "( JOIN ( R [ location , country , languages spoken ] ) ( JOIN [ location , country , administrative divisions ] [ Nord-Ouest Department ] ) )",
            "( JOIN (R fbp:location.country.languages_spoken) ( JOIN fbp:location.country.administrative_divisions fb:Nord-Ouest_Department ) )"
        )
    ]
    
    for inp, out in tests:
        assert transform_normed_sexpr(inp) == out
        
        
def test_expand_entity_map():
    tests: list[tuple[dict, dict]] = [
        (
            {
                "m.04pk3f": "The Maneater",
                "m.01y2hnl": "College/University",
                "m.0c4y8": "Tennessee Williams"
            },
            {
                "http://rdf.freebase.com/ns/m.04pk3f": "The Maneater",
                "http://rdf.freebase.com/ns/m.01y2hnl": "College/University",
                "http://rdf.freebase.com/ns/m.0c4y8": "Tennessee Williams"
            }
        )
    ]

    for inp, out in tests:
        assert expand_entity_map(inp) == out