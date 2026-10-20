"""Spanish fixtures written for these tests (fictional places and people)."""

CLEAN_ES = (
    "La biblioteca municipal de San Esteban abrió sus puertas hace más de un siglo y desde "
    "entonces ha sido un punto de encuentro para estudiantes, trabajadores y vecinos del "
    "barrio. Durante el invierno, las salas de lectura se llenan de personas que buscan un "
    "lugar tranquilo para estudiar o simplemente leer el diario.\n"
    "\n"
    "El edificio actual fue restaurado después de un incendio. La restauración respetó la "
    "fachada original, pero incorporó una rampa de acceso, un ascensor y una sala infantil "
    "con libros en español y en mapudungun.\n"
    "\n"
    "—¿Vienes a la biblioteca mañana? —preguntó Camila.\n"
    "—Sí, tengo que terminar el informe de historia antes del viernes.\n"
    "\n"
    "Los bibliotecarios organizan talleres de escritura todos los sábados por la mañana, y "
    "la inscripción es gratuita para cualquier persona que viva en la comuna."
)

CLEAN_PT = (
    "A biblioteca municipal de São Estêvão abriu as portas há mais de um século e desde "
    "então tem sido um ponto de encontro para estudantes, trabalhadores e moradores do "
    "bairro. Durante o inverno, as salas de leitura ficam cheias de pessoas que procuram "
    "um lugar tranquilo para estudar ou simplesmente ler o jornal. Os bibliotecários "
    "organizam oficinas de escrita todos os sábados pela manhã, e a inscrição é gratuita "
    "para qualquer pessoa que more na cidade."
)

# An ordinary English news paragraph. Of the Spanish stopwords it contains only
# "a", so it fails the Spanish rule, which counts distinct stopwords.
ENGLISH = (
    "The town council approved next year's budget on Tuesday after a long debate about "
    "library funding. Under the new plan, the central library will extend its opening hours "
    "on weekends and convert a storage area into a reading room for students. Several "
    "residents asked the council to keep a small cafe near the entrance, and the mayor said "
    "the proposal would be reviewed in the spring. The budget also includes repairs to two "
    "bridges and a pilot program for electric buses on the main avenue."
)

# English that happens to contain two Spanish stopwords ("a", "no"). It passes
# the Spanish stopword rule: the rule removes junk, it does not identify the
# language, so language identification has to run upstream.
ENGLISH_SHARED_STOPWORDS = (
    "There is no easy way to describe the new railway station. It has a glass roof, a "
    "market on the lower floor and a long platform that bends along the river. Commuters "
    "say that trains are more punctual than last year, although there are no plans yet to "
    "add evening services on Sundays. The city expects the station to handle twice as many "
    "passengers by the end of the decade."
)
